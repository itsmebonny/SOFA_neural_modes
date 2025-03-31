import os
import time
import torch
import numpy as np
import logging
from abc import ABC, abstractmethod
import pyvista
from dolfinx import plot
import traceback
from dolfinx.io import gmshio
from mpi4py import MPI

# Abstract base classes for modular desig, help me move boundary condition
class EnergyModel(ABC):
    """Abstract energy model defining material behavior"""
    @abstractmethod
    def compute_energy(self, displacement_batch):
        pass
    
    @abstractmethod
    def compute_gradient(self, displacement_batch):
        """Returns internal forces"""
        pass

class LinearSolver(ABC):
    """Abstract solver for linear systems (e.g., CG, direct)"""
    @abstractmethod
    def solve(self, matrix_operator, rhs_batch, **kwargs):
        pass

class Preconditioner(ABC):
    """Abstract preconditioner interface"""
    @abstractmethod
    def compute(self, displacement_batch):
        pass
    
    @abstractmethod
    def apply(self, residual_batch):
        pass

class ModularNeoHookeanEnergy(torch.nn.Module):
    """
    Modular Neo-Hookean energy model.
    
    Implements the Neo-Hookean formulation:
    W = 0.5 * mu * (IC - 3 - 2 * log(J)) + 0.25 * lambda * (J^2 - 1 - 2 * log(J))
    """
    
    def __init__(self, domain, degree, E, nu, precompute_matrices=True, device=None, dtype=torch.float64):
        """
        Initialize with DOLFINx domain
        
        Args:
            domain: DOLFINx domain
            degree: FEM degree
            E: Young's modulus
            nu: Poisson's ratio
            precompute_matrices: Whether to precompute FEM matrices
            device: Computation device
            dtype: Data type for computation
        """
        super(ModularNeoHookeanEnergy, self).__init__()
        
        # Set device and precision
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        
        # Material properties 
        self.E = E
        self.nu = nu
        self.mu = self.E / (2 * (1 + self.nu))  # Shear modulus
        self.lmbda = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))  # First Lamé parameter
        
        # Store domain and parameters for mesh operations
        self.domain = domain
        self.degree = degree
        
        # Extract mesh information from domain
        self._extract_mesh_data()
        
        # Create element data structure
        self._setup_elements(precompute_matrices)
        
        # Save configuration parameters
        self.precompute_matrices = precompute_matrices
        
    def _extract_mesh_data(self):
        """Extract mesh data from DOLFINx domain"""
        # Get coordinates as tensor
        x = self.domain.geometry.x
        self.coordinates = torch.tensor(x, device=self.device, dtype=self.dtype)
        self.num_nodes = self.coordinates.shape[0]
        self.dim = self.coordinates.shape[1]  # Should be 3 for 3D
        
        # Extract element connectivity
        tdim = self.domain.topology.dim
        elements_list = []
        for cell in range(self.domain.topology.index_map(tdim).size_local):
            elements_list.append(self.domain.topology.connectivity(tdim, 0).links(cell))
        
        self.elements = torch.tensor(np.array(elements_list), device=self.device, dtype=torch.long)
        self.num_elements = len(self.elements)
        self.nodes_per_element = self.elements.shape[1]
        
        # Print info
        print(f"Mesh has {self.num_nodes} nodes and {self.num_elements} elements")
        print(f"Element type: {self.nodes_per_element}-node")
        
    def _setup_elements(self, precompute=True):
        """Setup element data and potentially precompute matrices"""
        # Generate quadrature points based on element type
        self._generate_quadrature()
        
        if precompute:
            print("Precomputing element matrices...")
            self._precompute_derivatives()
            self.precomputed = True
        else:
            self.precomputed = False
    
    def _generate_quadrature(self):
        """Generate quadrature rules based on element type"""
        if self.nodes_per_element == 4:  # Tetrahedron
            # 4-point quadrature for tetrahedron
            self.quadrature_points = torch.tensor([
                [0.58541020, 0.13819660, 0.13819660],
                [0.13819660, 0.58541020, 0.13819660],
                [0.13819660, 0.13819660, 0.58541020],
                [0.13819660, 0.13819660, 0.13819660]
            ], dtype=self.dtype, device=self.device)
            self.quadrature_weights = torch.tensor([0.25, 0.25, 0.25, 0.25], 
                                                  dtype=self.dtype, 
                                                  device=self.device) / 6.0
        else:  # Hexahedron or other
            # 2×2×2 Gaussian quadrature for hexahedra
            gp = 1.0 / torch.sqrt(torch.tensor(3.0, device=self.device, dtype=self.dtype))
            self.quadrature_points = torch.tensor([
                [-gp, -gp, -gp], [gp, -gp, -gp], [gp, gp, -gp], [-gp, gp, -gp],
                [-gp, -gp, gp], [gp, -gp, gp], [gp, gp, gp], [-gp, gp, gp]
            ], dtype=self.dtype, device=self.device)
            self.quadrature_weights = torch.ones(8, dtype=self.dtype, device=self.device)
    
    def _precompute_derivatives(self):
        """Precompute shape function derivatives for all elements and quadrature points"""
        num_qp = len(self.quadrature_points)
        
        # Allocate tensors
        self.dN_dx_all = torch.zeros((self.num_elements, num_qp, self.nodes_per_element, 3), 
                                    dtype=self.dtype, device=self.device)
        self.detJ_all = torch.zeros((self.num_elements, num_qp), 
                                   dtype=self.dtype, device=self.device)
        
        # Process elements in batches for memory efficiency
        batch_size = 512  # Adjust based on memory
        for batch_start in range(0, self.num_elements, batch_size):
            batch_end = min(batch_start + batch_size, self.num_elements)
            batch_elements = self.elements[batch_start:batch_end]
            
            for e, element_nodes in enumerate(batch_elements):
                element_coords = self.coordinates[element_nodes]
                
                for q_idx, qp in enumerate(self.quadrature_points):
                    dN_dx, detJ = self._compute_element_derivatives(element_coords, qp)
                    self.dN_dx_all[batch_start+e, q_idx] = dN_dx
                    self.detJ_all[batch_start+e, q_idx] = detJ
    
    def _compute_element_derivatives(self, element_coords, qp):
        """Compute shape function derivatives for an element at quadrature point"""
        if self.nodes_per_element == 4:  # Tetrahedron
            # Shape functions for tetrahedron (linear)
            # dN/d(xi) matrix - derivatives of shape functions w.r.t. reference coordinates
            dN_dxi = torch.tensor([
                [-1.0, -1.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=self.dtype, device=element_coords.device)
            
            # Jacobian matrix: J = dx/d(xi) = ∑(x_e * dN/d(xi))
            J = torch.matmul(element_coords.transpose(0, 1), dN_dxi)
            detJ = torch.det(J)
            invJ = torch.linalg.inv(J)
            
            # Shape function derivatives w.r.t. physical coordinates: dN/dx = dN/d(xi) * d(xi)/dx
            dN_dx = torch.matmul(dN_dxi, invJ)
            
        else:  # Hexahedron
            # For 8-node hexahedron, we need to compute derivatives at the specific quadrature point
            xi, eta, zeta = qp
            
            # Precompute terms for efficiency
            xim = 1.0 - xi
            xip = 1.0 + xi
            etam = 1.0 - eta
            etap = 1.0 + eta
            zetam = 1.0 - zeta
            zetap = 1.0 + zeta
            
            # Shape function derivatives at this point
            dN_dxi = torch.zeros((8, 3), dtype=self.dtype, device=element_coords.device)
            
            # dN/d(xi) for each shape function
            dN_dxi[0, 0] = -0.125 * etam * zetam
            dN_dxi[1, 0] =  0.125 * etam * zetam
            dN_dxi[2, 0] =  0.125 * etap * zetam
            dN_dxi[3, 0] = -0.125 * etap * zetam
            dN_dxi[4, 0] = -0.125 * etam * zetap
            dN_dxi[5, 0] =  0.125 * etam * zetap
            dN_dxi[6, 0] =  0.125 * etap * zetap
            dN_dxi[7, 0] = -0.125 * etap * zetap
            
            # dN/d(eta) for each shape function
            dN_dxi[0, 1] = -0.125 * xim * zetam
            dN_dxi[1, 1] = -0.125 * xip * zetam
            dN_dxi[2, 1] =  0.125 * xip * zetam
            dN_dxi[3, 1] =  0.125 * xim * zetam
            dN_dxi[4, 1] = -0.125 * xim * zetap
            dN_dxi[5, 1] = -0.125 * xip * zetap
            dN_dxi[6, 1] =  0.125 * xip * zetap
            dN_dxi[7, 1] =  0.125 * xim * zetap
            
            # dN/d(zeta) for each shape function
            dN_dxi[0, 2] = -0.125 * xim * etam
            dN_dxi[1, 2] = -0.125 * xip * etam
            dN_dxi[2, 2] = -0.125 * xip * etap
            dN_dxi[3, 2] = -0.125 * xim * etap
            dN_dxi[4, 2] =  0.125 * xim * etam
            dN_dxi[5, 2] =  0.125 * xip * etam
            dN_dxi[6, 2] =  0.125 * xip * etap
            dN_dxi[7, 2] =  0.125 * xim * etap
            
            # Jacobian calculation
            J = torch.matmul(element_coords.transpose(0, 1), dN_dxi)
            detJ = torch.det(J)
            invJ = torch.linalg.inv(J)
            
            # Shape function derivatives w.r.t. physical coordinates
            dN_dx = torch.matmul(dN_dxi, invJ)
        
        return dN_dx, detJ
    
    def forward(self, u_tensor):
        """
        PyTorch forward method - computes total energy
        
        Args:
            u_tensor: Displacement field [num_nodes*dim] or [batch_size, num_nodes*dim]
            
        Returns:
            Total strain energy
        """
        return self.compute_energy(u_tensor)
    
    def compute_energy(self, displacement_batch):
        """
        Compute total elastic energy for displacement field(s) - EnergyModel interface
        
        Args:
            displacement_batch: Displacement tensor [batch_size, num_nodes*dim] or [num_nodes*dim]
            
        Returns:
            Total strain energy per batch sample
        """
        # Handle input dimensionality
        is_batch = displacement_batch.dim() > 1
        if not is_batch:
            displacement_batch = displacement_batch.unsqueeze(0)
            
        batch_size = displacement_batch.shape[0]
        
        # Reshape displacement tensor to [batch_size, num_nodes, dim]
        u_reshaped = displacement_batch.reshape(batch_size, self.num_nodes, self.dim)
        
        # Initialize energy tensor
        energy = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
        
        # Process in batches for memory efficiency
        chunk_size = min(1024, self.num_elements)
        
        for chunk_start in range(0, self.num_elements, chunk_size):
            chunk_end = min(chunk_start + chunk_size, self.num_elements)
            chunk_elements = self.elements[chunk_start:chunk_end]
            chunk_size_actual = chunk_end - chunk_start
            
            # Gather element coordinates and displacements
            element_coords_batch = self.coordinates[chunk_elements]
            element_indices = chunk_elements.view(chunk_size_actual, self.nodes_per_element)
            element_disps_batch = u_reshaped[:, element_indices]
            
            # Initialize chunk energy
            chunk_energy = torch.zeros(batch_size, chunk_size_actual, 
                                     dtype=self.dtype, device=self.device)
            
            # Process quadrature points
            for q_idx, (qp, qw) in enumerate(zip(self.quadrature_points, self.quadrature_weights)):
                if self.precomputed:
                    # Use precomputed derivatives
                    dN_dx_batch = self.dN_dx_all[chunk_start:chunk_end, q_idx]
                    detJ_batch = self.detJ_all[chunk_start:chunk_end, q_idx]
                else:
                    # Compute on-the-fly (not implemented for efficiency)
                    raise NotImplementedError("Non-precomputed derivatives not supported")
                
                # Compute deformation gradient F for all elements
                F_batch = torch.eye(3, dtype=self.dtype, device=self.device)
                F_batch = F_batch.view(1, 1, 3, 3).expand(batch_size, chunk_size_actual, 3, 3).clone()
                
                # Compute gradient using einsum: ∑ u_nj * dN_n/dx_i
                grad_u_batch = torch.einsum('benj,eni->beij', element_disps_batch, dN_dx_batch)
                
                # F = I + grad(u)
                F_batch += grad_u_batch
                
                # Reshape for energy density calculation
                F_batch_flat = F_batch.reshape(-1, 3, 3)
                
                # Compute energy density 
                energy_density_flat = self._compute_neohookean_energy_density(F_batch_flat)
                
                # Reshape back and add weighted contribution
                energy_density_batch = energy_density_flat.reshape(batch_size, chunk_size_actual)
                chunk_energy += energy_density_batch * detJ_batch * qw
            
            # Sum contributions from all elements in chunk
            energy += chunk_energy.sum(dim=1)
        
        # Return single value if input wasn't batched
        if not is_batch:
            energy = energy.squeeze(0)
            
        return energy
    
    def _compute_neohookean_energy_density(self, F):
        """
        Compute Neo-Hookean strain energy density 
        
        Args:
            F: Deformation gradient tensor [batch_size, 3, 3]
            
        Returns:
            Energy density per batch sample
        """
        # Original implementation:
        # return 0.5 * mu * (IC - 3 - 2 * torch.log(J)) \
        #    + 0.25 * lmbd * (J ** 2 - 1 - 2 * torch.log(J))
        
        # Extract determinant J
        J = torch.linalg.det(F)
        
        # First invariant of right Cauchy-Green deformation tensor C = F^T·F
        # IC = tr(C) = F_ji·F_ji
        IC = torch.einsum('...ji,...ji->...', F, F)
        
        # Safe log for stability 
        safe_J = torch.clamp(J, min=1e-10)
        log_J = torch.log(safe_J)
        
        # Neo-Hookean energy 
        W = 0.5 * self.mu * (IC - 3.0 - 2.0 * log_J) + \
            0.25 * self.lmbda * (J ** 2 - 1.0 - 2.0 * log_J)
        
        return W
    
    def compute_gradient(self, displacement_batch):
        """
        Compute internal forces (negative gradient of energy) - EnergyModel interface
        
        Args:
            displacement_batch: Displacement tensor [batch_size, num_nodes*dim] or [num_nodes*dim]
            
        Returns:
            Internal forces
        """
        # For batch handling
        is_batch = displacement_batch.dim() > 1
        if not is_batch:
            displacement_batch = displacement_batch.unsqueeze(0)
            
        batch_size = displacement_batch.shape[0]
        
        # Create output tensor for internal forces
        internal_forces = torch.zeros_like(displacement_batch)
        
        # For each sample, compute gradient separately
        for i in range(batch_size):
            u_i = displacement_batch[i:i+1].detach().clone().requires_grad_(True)
            
            # Compute energy for this sample
            energy_i = self.compute_energy(u_i)
            
            # Compute gradient
            grad_i = torch.autograd.grad(
                energy_i, u_i,
                create_graph=torch.is_grad_enabled(),
                retain_graph=True
            )[0]
            
            # Store result
            internal_forces[i] = grad_i
        
        # Return single tensor if input wasn't batched
        if not is_batch:
            internal_forces = internal_forces.squeeze(0)
            
        return internal_forces
    
    def compute_div_p(self, displacement_batch):
        """
        Compute the divergence of the first Piola-Kirchhoff stress tensor
        
        Args:
            displacement_batch: Displacement tensor [batch_size, num_nodes*dim] or [num_nodes*dim]
            
        Returns:
            Divergence of P tensor [batch_size, num_nodes, 3] or [num_nodes, 3]
        """
        
        # Handle input dimensionality
        is_batch = displacement_batch.dim() > 1
        if not is_batch:
            displacement_batch = displacement_batch.unsqueeze(0)
            
        batch_size = displacement_batch.shape[0]
        
        # Initialize output tensor
        div_p = torch.zeros((batch_size, self.num_nodes, 3), 
                           dtype=self.dtype, device=self.device)
        
        # Process each sample individually
        for i in range(batch_size):
            # Make displacement tensor require grad
            u_i = displacement_batch[i].clone().detach().requires_grad_(True)
            
            # Compute energy for this sample
            energy_i = self.compute_energy(u_i)
            
            # Compute internal forces (negative gradient of energy)
            internal_forces = -torch.autograd.grad(
                energy_i, u_i, create_graph=False, retain_graph=False
            )[0]
            
            # Reshape to [num_nodes, 3] - this is the divergence of P
            div_p_i = internal_forces.reshape(self.num_nodes, 3)
            
            # Store in output tensor
            div_p[i] = div_p_i
        
        # Return without batch dimension if input wasn't batched
        if not is_batch:
            div_p = div_p.squeeze(0)
            
        return div_p
    
    def compute_PK1(self, displacement_batch):
        """
        Compute the First Piola-Kirchhoff stress tensor (for compatibility)
        
        Args:
            displacement_batch: Displacement tensor
            
        Returns:
            First Piola-Kirchhoff stress tensor
        """
        # Handle input dimensionality
        is_batch = displacement_batch.dim() > 1
        if not is_batch:
            displacement_batch = displacement_batch.unsqueeze(0)
            
        batch_size = displacement_batch.shape[0]
        
        # Reshape displacement tensor
        u_reshaped = displacement_batch.reshape(batch_size, self.num_nodes, self.dim)
        
        # Initialize PK1 tensor at quadrature points
        # For FEM, we compute PK1 at quadrature points
        pk1 = torch.zeros((batch_size, self.num_elements, len(self.quadrature_points), 3, 3), 
                         dtype=self.dtype, device=self.device)
        
        # Process elements in batches
        for batch_start in range(0, self.num_elements, 1024):
            batch_end = min(batch_start + 1024, self.num_elements)
            batch_elements = self.elements[batch_start:batch_end]
            batch_size_actual = batch_end - batch_start
            
            # Get element data
            element_coords = self.coordinates[batch_elements]
            element_disps = u_reshaped[:, batch_elements]
            
            # Process each quadrature point
            for q_idx, qp in enumerate(self.quadrature_points):
                if self.precomputed:
                    dN_dx = self.dN_dx_all[batch_start:batch_end, q_idx]
                else:
                    # Not implemented for efficiency
                    raise NotImplementedError("Non-precomputed derivatives not supported")
                
                # Compute deformation gradient F at quadrature point
                F = torch.eye(3, dtype=self.dtype, device=self.device)
                F = F.view(1, 1, 3, 3).expand(batch_size, batch_size_actual, 3, 3).clone()
                
                # Add displacement gradient
                grad_u = torch.einsum('benj,eni->beij', element_disps, dN_dx)
                F += grad_u
                
                # Compute PK1 tensor 
                for b in range(batch_size):
                    for e in range(batch_size_actual):
                        # Determinant and inverse
                        J = torch.linalg.det(F[b, e])
                        inv_F = torch.linalg.inv(F[b, e])
                        
                        # First Piola-Kirchhoff stress tensor 
                        # P = mu * (F - F^-T) + 0.5 * lambda * (J^2 - 1) * F^-T
                        P = self.mu * (F[b, e] - inv_F.transpose(0, 1)) + \
                            0.5 * self.lmbda * (J**2 - 1.0) * inv_F.transpose(0, 1)
                        
                        # Store in output tensor
                        pk1[b, batch_start + e, q_idx] = P
    
        return pk1
    
    def compute_volume_comparison(self, u_linear, u_total):
        """
        Compare volumes between original mesh, linear modes prediction, and neural network prediction.
        
        Args:
            u_linear: Linear displacement field
            u_total: Total displacement field
            
        Returns:
            Dictionary with volume information
        """
        # Handle batch dimension if present
        if len(u_linear.shape) > 1 and u_linear.shape[0] == 1:
            u_linear = u_linear.squeeze(0)
        if len(u_total.shape) > 1 and u_total.shape[0] == 1:
            u_total = u_total.squeeze(0)
        
        # Calculate volumes
        original_volume = self._compute_mesh_volume()
        linear_volume = self._compute_deformed_volume(u_linear)
        neural_volume = self._compute_deformed_volume(u_total)
        
        # Calculate volume ratios
        linear_volume_ratio = linear_volume / original_volume
        neural_volume_ratio = neural_volume / original_volume
        
        # Calculate improvement ratio
        if abs(linear_volume_ratio - 1.0) > 1e-10:
            improvement_ratio = abs(neural_volume_ratio - 1.0) / abs(linear_volume_ratio - 1.0)
        else:
            improvement_ratio = 1.0
        
        # Create result dictionary
        volume_info = {
            'original_volume': original_volume,
            'linear_volume': linear_volume,
            'neural_volume': neural_volume,
            'linear_volume_ratio': linear_volume_ratio,
            'neural_volume_ratio': neural_volume_ratio,
            'volume_preservation_improvement': improvement_ratio
        }
        
        return volume_info
    
    def _compute_mesh_volume(self):
        """Calculate the total volume of the original mesh"""
        total_volume = 0.0
        
        if self.nodes_per_element == 4:  # Tetrahedron
            for element_nodes in self.elements:
                vertices = self.coordinates[element_nodes].cpu().numpy()
                edges = vertices[1:] - vertices[0]
                volume = abs(np.linalg.det(edges)) / 6.0
                total_volume += volume
        else:  # Hexahedron or other
            # Compute volume using quadrature
            for e in range(self.num_elements):
                for q_idx, qw in enumerate(self.quadrature_weights):
                    detJ = self.detJ_all[e, q_idx].item()
                    total_volume += detJ * qw.item()
        
        return total_volume
    
    def _compute_deformed_volume(self, displacement):
        """Calculate the total volume of the mesh after applying a displacement field"""
        # Ensure displacement is in CPU numpy format
        if isinstance(displacement, torch.Tensor):
            displacement = displacement.detach().cpu().numpy()
        
        # Reshape if needed
        if len(displacement.shape) == 1:
            displacement = displacement.reshape(-1, 3)
        
        # Get deformed coordinates
        coordinates = self.coordinates.cpu().numpy()
        deformed_coords = coordinates + displacement
        
        # Calculate volume
        total_volume = 0.0
        
        if self.nodes_per_element == 4:  # Tetrahedron
            for element_nodes in self.elements.cpu().numpy():
                vertices = deformed_coords[element_nodes]
                edges = vertices[1:] - vertices[0]
                volume = abs(np.linalg.det(edges)) / 6.0
                total_volume += volume
        else:
            # For hexahedra, compute using quadrature and deformation gradient
            # This is more complex - would need to recompute Jacobian determinants
            # for the deformed configuration
            pass
        
        return total_volume


class UFLNeoHookeanModel(torch.nn.Module):
    """
    Modular Neo-Hookean energy model (UFL-Equivalent Formulation).

    Implements the Neo-Hookean formulation commonly derived from multiplicative
    decomposition, matching standard UFL implementations:
    W = (μ/2) * (I_C - 3) - μ * ln(J) + (λ/2) * (ln(J))²
    where:
        μ = Shear modulus
        λ = First Lamé parameter
        F = Deformation gradient
        J = det(F)
        C = Fᵀ F (Right Cauchy-Green tensor)
        I_C = tr(C) (First invariant of C)
    """

    def __init__(self, domain, degree, E, nu, precompute_matrices=True, device=None, dtype=torch.float64):
        """
        Initialize with DOLFINx domain

        Args:
            domain: DOLFINx domain (used for mesh info extraction)
            degree: FEM degree (influences nodes_per_element if not explicit)
            E: Young's modulus
            nu: Poisson's ratio
            precompute_matrices: Whether to precompute FEM matrices (highly recommended)
            device: Computation device (e.g., 'cpu', 'cuda:0')
            dtype: Data type for computation (e.g., torch.float32, torch.float64)
        """
        super(UFLNeoHookeanModel, self).__init__()

        # Set device and precision
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        print(f"Using device: {self.device}, dtype: {self.dtype}")

        # Material properties
        self.E = torch.tensor(E, dtype=self.dtype, device=self.device)
        self.nu = torch.tensor(nu, dtype=self.dtype, device=self.device)
        # Prevent division by zero or instability for nu close to 0.5
        if torch.abs(1.0 - 2.0 * self.nu) < 1e-9:
             print("Warning: nu is close to 0.5. Using a large value for lambda.")
             # Adjust lambda calculation or handle appropriately
             # For practical purposes, maybe cap nu slightly below 0.5
             safe_nu = torch.min(self.nu, torch.tensor(0.49999, dtype=self.dtype, device=self.device))
             self.lmbda = self.E * safe_nu / ((1 + safe_nu) * (1 - 2 * safe_nu))
        else:
             self.lmbda = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu)) # First Lamé parameter

        self.mu = self.E / (2 * (1 + self.nu))  # Shear modulus

        print(f"Material Properties: E={self.E.item():.2f}, nu={self.nu.item():.2f}, mu={self.mu.item():.2f}, lambda={self.lmbda.item():.2f}")

        # Store domain info (or extracted data)
        self.domain = domain # Keep for potential future use, but extract data now
        self.degree = degree # May influence element type/nodes

        # Extract mesh information from domain
        self._extract_mesh_data() # Needs dolfinx

        # Create element data structure
        self._setup_elements(precompute_matrices)

        # Save configuration parameters
        self.precompute_matrices = precompute_matrices

        # Small value for safe log/sqrt
        self.eps = torch.tensor(1e-10, dtype=self.dtype, device=self.device)


    def _extract_mesh_data(self):
        """Extract mesh data from DOLFINx domain"""
        # This part still requires dolfinx to be installed and run once
        try:
            import dolfinx
            import numpy as np
            from dolfinx.fem import FunctionSpace
            from dolfinx.mesh import CellType
        except ImportError:
            raise ImportError("dolfinx is required for mesh data extraction during initialization.")

        # Get coordinates as tensor
        x = self.domain.geometry.x
        self.coordinates = torch.tensor(x, device=self.device, dtype=self.dtype)
        self.num_nodes = self.coordinates.shape[0]
        self.dim = self.coordinates.shape[1]
        if self.dim != 3:
            print(f"Warning: Expected 3D coordinates, but got {self.dim}D.")

        # --- Determine element type and nodes per element ---
        # Create a dummy function space to query element info
        try:
            # Try vector space with updated API
            element_family = "Lagrange"  # Use Lagrange instead of CG for newer DOLFINx
            element = dolfinx.fem.VectorElement(element_family, self.domain.ufl_cell(), self.degree, self.dim)
            V_dummy = dolfinx.fem.FunctionSpace(self.domain, element)
            dolfin_element = V_dummy.ufl_element()
            self.nodes_per_element = dolfin_element.space_dimension() // self.dim
            cell_type_str = dolfin_element.cell().cellname()

        except Exception as e_vec:
            try:
                # Fallback to scalar space if vector fails
                print(f"Vector space query failed ({e_vec}), trying scalar space...")
                element = dolfinx.fem.FiniteElement(element_family, self.domain.ufl_cell(), self.degree)
                V_dummy = dolfinx.fem.FunctionSpace(self.domain, element)
                dolfin_element = V_dummy.ufl_element()
                self.nodes_per_element = dolfin_element.space_dimension()
                cell_type_str = dolfin_element.cell().cellname()
            except Exception as e_scalar:
                # If both approaches fail, try direct inference based on degree and element type
                print(f"Failed to create function spaces: Vector: {e_vec}, Scalar: {e_scalar}")
                print("Attempting to infer nodes per element from cell type and degree...")
                
                # Fallback to determine nodes_per_element based on common patterns
                cell_type = self.domain.topology.cell_type
                if cell_type == CellType.tetrahedron:
                    cell_type_str = "tetrahedron"
                    # For tetrahedra: degree 1 → 4 nodes, degree 2 → 10 nodes
                    if self.degree == 1:
                        self.nodes_per_element = 4
                    elif self.degree == 2:
                        self.nodes_per_element = 10
                    else:
                        raise ValueError(f"Unsupported degree {self.degree} for tetrahedron")
                elif cell_type == CellType.hexahedron:
                    cell_type_str = "hexahedron"
                    # For hexahedra: degree 1 → 8 nodes, degree 2 → 27 nodes
                    if self.degree == 1:
                        self.nodes_per_element = 8
                    elif self.degree == 2:
                        self.nodes_per_element = 27
                    else:
                        raise ValueError(f"Unsupported degree {self.degree} for hexahedron")
                else:
                    raise ValueError(f"Unsupported cell type: {cell_type}")

        print(f"Detected cell type: {cell_type_str}, Degree: {self.degree}")

        # Continue with the rest of the method...
        # Extract element connectivity
        tdim = self.domain.topology.dim
        
        # Extract connectivity from topology directly if function space approach fails
        try:
            if 'V_dummy' in locals():
                elements_list = V_dummy.dofmap.list.array.reshape(-1, self.nodes_per_element)
            else:
                # Direct topology extraction
                elements_list = []
                for cell in range(self.domain.topology.index_map(tdim).size_local):
                    elements_list.append(self.domain.topology.connectivity(tdim, 0).links(cell))
        except Exception as e_conn:
            print(f"Error extracting element connectivity: {e_conn}")
            print("Attempting manual connectivity extraction...")
            elements_list = []
            for cell in range(self.domain.topology.index_map(tdim).size_local):
                try:
                    # Try different methods to get connectivity
                    conn = self.domain.topology.connectivity(tdim, 0)
                    if hasattr(conn, 'links'):
                        elements_list.append(conn.links(cell))
                    elif hasattr(conn, 'array'):
                        # Some versions use a different API
                        cell_nodes = conn.array[cell * self.nodes_per_element:(cell + 1) * self.nodes_per_element]
                        elements_list.append(cell_nodes)
                    else:
                        raise ValueError("Cannot determine connectivity method")
                except Exception as e_cell:
                    print(f"Failed to extract connectivity for cell {cell}: {e_cell}")
                    # Use placeholder connectivity for robustness
                    elements_list.append(np.arange(self.nodes_per_element, dtype=np.int32))

        self.elements = torch.tensor(np.array(elements_list), device=self.device, dtype=torch.long)
        self.num_elements = len(self.elements)

        print(f"Mesh info extracted: {self.num_nodes} nodes, {self.num_elements} elements")
        print(f"Nodes per element: {self.nodes_per_element} (degree={self.degree}, cell='{cell_type_str}')")

    def _setup_elements(self, precompute=True):
        """Setup element data and potentially precompute matrices"""
        self._generate_quadrature() # Depends on self.nodes_per_element

        if precompute:
            print("Precomputing element shape function derivatives...")
            self._precompute_derivatives()
            self.precomputed = True
            print("Precomputation finished.")
        else:
            print("Derivatives will be computed on-the-fly (less efficient).")
            self.precomputed = False
            # Initialize placeholders if needed later, or handle errors
            self.dN_dx_all = None
            self.detJ_all = None

    def _generate_quadrature(self):
        """Generate quadrature rules based on element type and degree"""
        # Basic quadrature rules - may need refinement for higher accuracy/order
        if self.nodes_per_element == 4:  # Linear Tetrahedron
            # 1-point rule (exact for constant, often sufficient for linear tet volume)
            # self.quadrature_points = torch.tensor([[0.25, 0.25, 0.25]], dtype=self.dtype, device=self.device)
            # self.quadrature_weights = torch.tensor([1.0], dtype=self.dtype, device=self.device) / 6.0
            # 4-point rule (degree 3 exactness) - Often preferred for accuracy
            self.quadrature_points = torch.tensor([
                [0.58541020, 0.13819660, 0.13819660],
                [0.13819660, 0.58541020, 0.13819660],
                [0.13819660, 0.13819660, 0.58541020],
                [0.13819660, 0.13819660, 0.13819660]
            ], dtype=self.dtype, device=self.device)
            self.quadrature_weights = torch.tensor([0.25, 0.25, 0.25, 0.25],
                                                  dtype=self.dtype,
                                                  device=self.device) / 6.0 # Volume of reference tet = 1/6
            print(f"Using 4-point quadrature for {self.nodes_per_element}-node elements.")
        elif self.nodes_per_element == 8: # Linear Hexahedron
            # 2x2x2 Gaussian quadrature (8 points)
            gp = 1.0 / torch.sqrt(torch.tensor(3.0, device=self.device, dtype=self.dtype))
            self.quadrature_points = torch.tensor([
                [-gp, -gp, -gp], [ gp, -gp, -gp], [ gp,  gp, -gp], [-gp,  gp, -gp],
                [-gp, -gp,  gp], [ gp, -gp,  gp], [ gp,  gp,  gp], [-gp,  gp,  gp]
            ], dtype=self.dtype, device=self.device)
            # Weights for [-1, 1] interval are 1.0 for 2-point Gauss. Volume is 8.
            self.quadrature_weights = torch.ones(8, dtype=self.dtype, device=self.device)
            print(f"Using 8-point (2x2x2 Gauss) quadrature for {self.nodes_per_element}-node elements.")
        else:
            # Fallback or error for other element types/degrees
            # For simplicity, using a single point quadrature (centroid) - **likely inaccurate**
            print(f"Warning: Using simple centroid quadrature for {self.nodes_per_element}-node elements. Accuracy may be low.")
            if "tetrahedron" in self._get_cell_type_str(): # Requires _get_cell_type_str helper
                 self.quadrature_points = torch.tensor([[0.25, 0.25, 0.25]], dtype=self.dtype, device=self.device)
                 self.quadrature_weights = torch.tensor([1.0], dtype=self.dtype, device=self.device) / 6.0
            elif "hexahedron" in self._get_cell_type_str():
                 self.quadrature_points = torch.tensor([[0.0, 0.0, 0.0]], dtype=self.dtype, device=self.device)
                 self.quadrature_weights = torch.tensor([8.0], dtype=self.dtype, device=self.device) # Ref volume = 8
            else: # Default guess
                 self.quadrature_points = torch.zeros((1, 3), dtype=self.dtype, device=self.device)
                 self.quadrature_weights = torch.ones((1,), dtype=self.dtype, device=self.device)


        self.num_quad_points = len(self.quadrature_points)

    # Helper to get cell type string robustly after init
    def _get_cell_type_str(self):
         try:
            import dolfinx
            from dolfinx.fem import FunctionSpace
            V_dummy = FunctionSpace(self.domain, ("CG", self.degree))
            return V_dummy.ufl_element().cell().cellname()
         except:
            # Fallback if dolfinx not available later or error
            if self.nodes_per_element == 4: return "tetrahedron"
            if self.nodes_per_element == 8: return "hexahedron"
            return "unknown"


    def _precompute_derivatives(self):
        """Precompute shape function derivatives w.r.t. physical coords (dN/dx)
           and Jacobian determinants (detJ) for all elements and quadrature points."""
        num_qp = self.num_quad_points

        # Allocate tensors
        # Shape: [num_elements, num_quad_points, nodes_per_element, dim]
        self.dN_dx_all = torch.zeros((self.num_elements, num_qp, self.nodes_per_element, self.dim),
                                    dtype=self.dtype, device=self.device)
        # Shape: [num_elements, num_quad_points]
        self.detJ_all = torch.zeros((self.num_elements, num_qp),
                                   dtype=self.dtype, device=self.device)

        # Process elements - can be done element-wise or batched if memory allows
        # Simpler element-wise loop for clarity:
        for e_idx in range(self.num_elements):
            element_node_indices = self.elements[e_idx]
            element_coords = self.coordinates[element_node_indices] # Coords of nodes for this element

            for q_idx, qp in enumerate(self.quadrature_points):
                # Compute dN/dξ (derivatives in reference element) and then dN/dx
                dN_dxi = self._shape_function_derivatives_ref(qp) # Shape: [nodes_per_element, dim]
                # Compute Jacobian J = dX/dξ = ∑ (Coords_n * dNn/dξ)
                # J_ij = ∑_n (Coords_ni * dNn/dξ_j)
                J = torch.einsum('ni,nj->ij', element_coords, dN_dxi) # Shape: [dim, dim]

                try:
                    detJ = torch.linalg.det(J)
                    invJ = torch.linalg.inv(J)
                except torch.linalg.LinAlgError as err:
                     print(f"Error computing inv/det(J) for element {e_idx} at QP {q_idx}: {err}")
                     print(f"Jacobian Matrix J:\n{J.cpu().numpy()}")
                     # Handle degenerate element - maybe set derivatives to zero or raise error
                     detJ = torch.tensor(0.0, dtype=self.dtype, device=self.device)
                     # Assign zero derivatives to prevent NaN propagation, but this element won't contribute correctly
                     dN_dx = torch.zeros_like(dN_dxi) # Zeros for dN/dx
                     # Or could try pseudo-inverse? invJ = torch.linalg.pinv(J)

                if detJ <= 0:
                     print(f"Warning: Non-positive Jacobian determinant ({detJ.item():.4e}) for element {e_idx} at QP {q_idx}. Check mesh quality.")
                     # Set to small positive value to avoid issues with weights? Or handle upstream.


                # Compute shape function derivatives w.r.t. physical coordinates:
                # dN/dx = dN/dξ * dξ/dX = dN/dξ * J⁻¹
                # dNn/dx_k = ∑_j (dNn/dξ_j * invJ_jk)
                dN_dx = torch.einsum('nj,jk->nk', dN_dxi, invJ) # Shape: [nodes_per_element, dim]

                # Store results
                self.dN_dx_all[e_idx, q_idx] = dN_dx
                self.detJ_all[e_idx, q_idx] = detJ

            # Optional: Add progress indicator for large meshes
            # if (e_idx + 1) % 1000 == 0:
            #     print(f"Precomputing derivatives: {e_idx + 1}/{self.num_elements} elements processed.")

    def _shape_function_derivatives_ref(self, qp_ref):
        """Compute shape function derivatives w.r.t. reference coordinates (ξ, η, ζ)
           at a given reference quadrature point qp_ref."""
        # qp_ref is a tensor [dim] with coordinates in the reference element

        if self.nodes_per_element == 4: # Linear Tetrahedron (Ref coords: L1, L2, L3, L4 sum to 1)
             # Derivatives are constant for linear tet
             # Using barycentric derivatives: dNi/dξj where ξ = (ξ, η, ζ) mapped from barycentric
             # dN1/dξ = -1, dN1/dη = -1, dN1/dζ = -1
             # dN2/dξ = 1,  dN2/dη = 0,  dN2/dζ = 0
             # dN3/dξ = 0,  dN3/dη = 1,  dN3/dζ = 0
             # dN4/dξ = 0,  dN4/dη = 0,  dN4/dζ = 1
             dN_dxi = torch.tensor([
                 [-1.0, -1.0, -1.0],
                 [ 1.0,  0.0,  0.0],
                 [ 0.0,  1.0,  0.0],
                 [ 0.0,  0.0,  1.0]
             ], dtype=self.dtype, device=self.device)

        elif self.nodes_per_element == 8: # Linear Hexahedron (Ref coords: ξ, η, ζ in [-1, 1])
            xi, eta, zeta = qp_ref[0], qp_ref[1], qp_ref[2]

            # Precompute terms
            one = torch.tensor(1.0, dtype=self.dtype, device=self.device)
            xim = one - xi
            xip = one + xi
            etam = one - eta
            etap = one + eta
            zetam = one - zeta
            zetap = one + zeta

            # Shape function derivatives at this point (dN/dξ, dN/dη, dN/dζ)
            # Shape: [nodes_per_element=8, dim=3]
            dN_dxi = torch.zeros((8, 3), dtype=self.dtype, device=self.device)
            eighth = 0.125

            # Derivatives w.r.t. xi (d/dξ)
            dN_dxi[0, 0] = -eighth * etam * zetam
            dN_dxi[1, 0] =  eighth * etam * zetam
            dN_dxi[2, 0] =  eighth * etap * zetam
            dN_dxi[3, 0] = -eighth * etap * zetam
            dN_dxi[4, 0] = -eighth * etam * zetap
            dN_dxi[5, 0] =  eighth * etam * zetap
            dN_dxi[6, 0] =  eighth * etap * zetap
            dN_dxi[7, 0] = -eighth * etap * zetap

            # Derivatives w.r.t. eta (d/dη)
            dN_dxi[0, 1] = -eighth * xim * zetam
            dN_dxi[1, 1] = -eighth * xip * zetam
            dN_dxi[2, 1] =  eighth * xip * zetam
            dN_dxi[3, 1] =  eighth * xim * zetam
            dN_dxi[4, 1] = -eighth * xim * zetap
            dN_dxi[5, 1] = -eighth * xip * zetap
            dN_dxi[6, 1] =  eighth * xip * zetap
            dN_dxi[7, 1] =  eighth * xim * zetap

            # Derivatives w.r.t. zeta (d/dζ)
            dN_dxi[0, 2] = -eighth * xim * etam
            dN_dxi[1, 2] = -eighth * xip * etam
            dN_dxi[2, 2] = -eighth * xip * etap
            dN_dxi[3, 2] = -eighth * xim * etap
            dN_dxi[4, 2] =  eighth * xim * etam
            dN_dxi[5, 2] =  eighth * xip * etam
            dN_dxi[6, 2] =  eighth * xip * etap
            dN_dxi[7, 2] =  eighth * xim * etap

        else:
            # Needs implementation for other element types (e.g., quadratic tet/hex)
            raise NotImplementedError(f"Shape function derivatives not implemented for {self.nodes_per_element}-node elements.")

        return dN_dxi


    def forward(self, u_tensor):
        """
        PyTorch forward method - computes total energy.

        Args:
            u_tensor: Displacement field [num_nodes*dim] or [batch_size, num_nodes*dim]

        Returns:
            Total strain energy (scalar for single input, tensor [batch_size] for batch)
        """
        return self.compute_energy(u_tensor)

    def compute_energy(self, displacement_batch):
        """
        Compute total elastic energy for displacement field(s).

        Args:
            displacement_batch: Displacement tensor [batch_size, num_nodes*dim] or [num_nodes*dim]

        Returns:
            Total strain energy per batch sample [batch_size] or scalar
        """
        if not self.precomputed:
             raise RuntimeError("Energy computation requires precomputed matrices. Initialize with precompute_matrices=True.")

        # Handle input dimensionality
        is_batch = displacement_batch.dim() > 1
        if not is_batch:
            displacement_batch = displacement_batch.unsqueeze(0)

        batch_size = displacement_batch.shape[0]

        # Reshape displacement tensor to [batch_size, num_nodes, dim]
        u_reshaped = displacement_batch.view(batch_size, self.num_nodes, self.dim)

        # --- Vectorized Energy Computation ---
        # Gather displacements for all elements: [batch_size, num_elements, nodes_per_element, dim]
        # Need efficient gathering. self.elements gives indices [num_elements, nodes_per_element]
        # u_reshaped is [batch_size, num_nodes, dim]
        # We can use advanced indexing, but need to handle batch dimension.
        # Option 1: Loop over batch (simple, might be okay if batch_size is small)
        # Option 2: Expand elements indices and gather (more complex, potentially faster)

        # Option 1: Loop over batch (simpler to implement/debug)
        total_energy = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
        for b in range(batch_size):
            u_sample = u_reshaped[b] # [num_nodes, dim]
            element_disps = u_sample[self.elements] # [num_elements, nodes_per_element, dim]

            # Initialize energy for this sample
            energy_sample = torch.tensor(0.0, dtype=self.dtype, device=self.device)

            # Loop over quadrature points
            for q_idx in range(self.num_quad_points):
                # Get precomputed data for this quad point:
                dN_dx_q = self.dN_dx_all[:, q_idx, :, :] # [num_elements, nodes_per_element, dim]
                detJ_q = self.detJ_all[:, q_idx]         # [num_elements]
                qw_q = self.quadrature_weights[q_idx]    # scalar

                # Compute gradient of displacement for all elements at this qp:
                # grad(u)_ij = ∑_n (u_nj * dNn/dx_i)
                # Input element_disps: [E, N, D]
                # Input dN_dx_q:      [E, N, D]
                # Output grad_u:      [E, D, D]
                grad_u = torch.einsum('enj,enk->ejk', element_disps, dN_dx_q)

                # Compute deformation gradient F = I + grad(u)
                # Need identity matrix shaped [E, D, D]
                I = torch.eye(self.dim, dtype=self.dtype, device=self.device).expand(self.num_elements, -1, -1)
                F = I + grad_u # [num_elements, dim, dim]

                # Compute energy density at this quadrature point for all elements
                # Need F with shape [*, 3, 3] for the density function
                energy_density_q = self._compute_neohookean_energy_density(F) # [num_elements]

                # Add weighted contribution to total energy for this sample
                # dEnergy = W * detJ * quad_weight
                energy_sample += torch.sum(energy_density_q * detJ_q * qw_q)

            total_energy[b] = energy_sample

        # Return single value if input wasn't batched
        if not is_batch:
            total_energy = total_energy.squeeze(0)

        return total_energy


    def _compute_neohookean_energy_density(self, F):
        """
        Compute Neo-Hookean strain energy density (UFL-equivalent formulation).

        Args:
            F: Deformation gradient tensor [*, 3, 3] (batch dimension handled by *)

        Returns:
            Energy density W [*]
        """
        # Ensure F has the correct dimensions (at least 3x3)
        if F.shape[-2:] != (3, 3):
             raise ValueError(f"Input F must have shape [..., 3, 3], but got {F.shape}")

        # Determinant J = det(F)
        J = torch.linalg.det(F)

        # Check for non-positive J (potential element inversion)
        if torch.any(J <= self.eps): # Use small epsilon threshold
             # print(f"Warning: Non-positive Jacobian detected (min J = {torch.min(J).item()}). Clamping J for log.")
             # Clamp J to a small positive value for stability in log
             J_safe = torch.clamp(J, min=self.eps)
        else:
             J_safe = J

        # Logarithm of J
        log_J = torch.log(J_safe)

        # First invariant of Right Cauchy-Green C = FᵀF
        # I_C = tr(C) = ∑_i,j (F_ji * F_ji)
        IC = torch.einsum('...ji,...ji->...', F, F) # Sum over last two dimensions

        # Neo-Hookean energy density (W₂)
        # W = (μ/2) * (I_C - 3) - μ * ln(J) + (λ/2) * (ln(J))²
        W = 0.5 * self.mu * (IC - 3.0) - self.mu * log_J + 0.5 * self.lmbda * (log_J ** 2)

        return W

    # --- Methods for Gradients and Stress ---

    def compute_gradient(self, displacement_batch):
        """
        Compute internal forces (negative gradient of energy w.r.t. displacements).

        Uses torch.autograd for automatic differentiation.

        Args:
            displacement_batch: Displacement tensor [batch_size, num_nodes*dim] or [num_nodes*dim]

        Returns:
            Internal forces (nodal forces) [-dE/du] with the same shape as input.
        """
        # Ensure input requires grad
        if not displacement_batch.requires_grad:
            displacement_batch = displacement_batch.detach().clone().requires_grad_(True)
        elif not torch.is_grad_enabled():
             # If called within torch.no_grad(), autograd won't work.
             # Re-enable grad temporarily for this computation.
             # This might be needed if this function is called during evaluation loops.
             with torch.enable_grad():
                 displacement_batch = displacement_batch.detach().clone().requires_grad_(True)
                 energy = self.compute_energy(displacement_batch)
                 # Compute gradient
                 grad = torch.autograd.grad(
                     outputs=energy.sum(), # Sum needed if energy is batched
                     inputs=displacement_batch,
                     create_graph=torch.is_grad_enabled(), # Preserve graph if needed for higher derivatives
                     retain_graph=True # Keep graph if energy or grad might be used again
                 )[0]
             return grad # Return directly from within the enable_grad block

        # Handle batching within compute_energy
        energy = self.compute_energy(displacement_batch)

        # Compute gradient: dE/du
        # Need to sum energy if it's batched, as grad expects scalar output
        grad = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=displacement_batch,
            create_graph=torch.is_grad_enabled(), # Create graph if we need grad of grad (Hessian)
            retain_graph=True # Allows calling backward or grad multiple times if needed elsewhere
        )[0]

        # Return the gradient (often used as -gradient for internal forces)
        return grad


    def compute_div_p(self, displacement_batch):
        """
        Compute the divergence of the first Piola-Kirchhoff (PK1) stress tensor,
        which corresponds to the internal forces (negative gradient of energy).

        Effectively computes -dE/du and reshapes it.

        Args:
            displacement_batch: Displacement tensor [batch_size, num_nodes*dim] or [num_nodes*dim]

        Returns:
            Divergence of P tensor reshaped to [batch_size, num_nodes, 3] or [num_nodes, 3]
        """
         # Handle input dimensionality
        is_batch = displacement_batch.dim() > 1
        if not is_batch:
            original_shape_was_flat = True
            displacement_batch = displacement_batch.unsqueeze(0)
        else:
             original_shape_was_flat = False

        batch_size = displacement_batch.shape[0]

        # Compute the gradient dE/du using autograd
        # Ensure requires_grad=True for the input to compute_gradient
        if not displacement_batch.requires_grad:
             u_for_grad = displacement_batch.detach().clone().requires_grad_(True)
        else:
             u_for_grad = displacement_batch

        internal_forces_flat = self.compute_gradient(u_for_grad) # Shape: [batch_size, num_nodes*dim]

        # The internal forces are -dE/du. The divergence of PK1 is also -dE/du (in weak form).
        # So, we just need to reshape the negative gradient.
        div_p = internal_forces_flat.view(batch_size, self.num_nodes, self.dim)

        # Return without batch dimension if input wasn't batched
        if original_shape_was_flat:
            div_p = div_p.squeeze(0)

        return div_p


    def compute_PK1(self, displacement_batch):
        """
        Compute the First Piola-Kirchhoff (PK1) stress tensor P at each
        quadrature point for each element.

        P = ∂W/∂F = μ * (F - F⁻ᵀ) + λ * ln(J) * F⁻ᵀ

        Args:
            displacement_batch: Displacement tensor [batch_size, num_nodes*dim] or [num_nodes*dim]

        Returns:
            PK1 tensor P [batch_size, num_elements, num_quad_points, 3, 3]
            or [num_elements, num_quad_points, 3, 3] if not batched.
        """
        if not self.precomputed:
             raise RuntimeError("PK1 computation requires precomputed matrices.")

        # Handle input dimensionality
        is_batch = displacement_batch.dim() > 1
        if not is_batch:
            original_shape_was_flat = True
            displacement_batch = displacement_batch.unsqueeze(0)
        else:
            original_shape_was_flat = False

        batch_size = displacement_batch.shape[0]

        # Reshape displacement tensor: [batch_size, num_nodes, dim]
        u_reshaped = displacement_batch.view(batch_size, self.num_nodes, self.dim)

        # --- Vectorized PK1 Computation ---
        # Initialize PK1 tensor: [batch_size, num_elements, num_quad_points, 3, 3]
        pk1 = torch.zeros((batch_size, self.num_elements, self.num_quad_points, self.dim, self.dim),
                         dtype=self.dtype, device=self.device)

        # Gather displacements for all elements: [batch_size, num_elements, nodes_per_element, dim]
        # element_indices = self.elements # Shape: [num_elements, nodes_per_element]
        # Use broadcasting and indexing for efficiency
        # Expand elements indices for batch dim: [batch_size, num_elements, nodes_per_element]
        # This step can be memory intensive if num_elements * nodes_per_element is huge.
        # Consider chunking elements if needed.
        element_indices_expanded = self.elements.unsqueeze(0).expand(batch_size, -1, -1) # B x E x N
        # Gather displacements using batch_gather or equivalent logic
        # Need to index u_reshaped [B, num_nodes, D] using indices [B, E, N] -> result [B, E, N, D]
        # This requires gather along node dimension (dim=1)
        # element_disps = torch.gather(u_reshaped.unsqueeze(2).expand(-1, -1, self.nodes_per_element, -1),
        #                              1, element_indices_expanded.unsqueeze(-1).expand(-1, -1, -1, self.dim)) # This seems overly complex
        # Alternative: Replicate u_reshaped and use advanced indexing?
        # Easier approach: Iterate through batch dim, as done in compute_energy

        # Loop over batch samples
        for b in range(batch_size):
             u_sample = u_reshaped[b] # [num_nodes, dim]
             element_disps = u_sample[self.elements] # [num_elements, nodes_per_element, dim]

             # Loop over quadrature points
             for q_idx in range(self.num_quad_points):
                 # Get precomputed shape derivatives for this qp: [E, N, D]
                 dN_dx_q = self.dN_dx_all[:, q_idx, :, :]

                 # Compute grad(u) = ∑ u_n * dNn/dx : [E, D, D]
                 grad_u = torch.einsum('enj,enk->ejk', element_disps, dN_dx_q)

                 # Compute deformation gradient F = I + grad(u) : [E, D, D]
                 I = torch.eye(self.dim, dtype=self.dtype, device=self.device).expand(self.num_elements, -1, -1)
                 F = I + grad_u

                 # --- Calculate PK1 for all elements at this qp ---
                 # Compute J = det(F) : [E]
                 J = torch.linalg.det(F)

                 # Clamp J for log stability
                 J_safe = torch.clamp(J, min=self.eps)
                 log_J = torch.log(J_safe)

                 # Compute F inverse transpose: F⁻ᵀ : [E, D, D]
                 try:
                     # Compute inverse first, then transpose for potentially better stability
                     inv_F = torch.linalg.inv(F)
                     inv_F_T = inv_F.transpose(-1, -2)
                 except torch.linalg.LinAlgError:
                      # Handle singular F (e.g., due to element inversion)
                      # print(f"Warning: Singular F encountered in PK1 calculation (batch {b}, qp {q_idx}). Using pseudo-inverse.")
                      # Fallback to pseudo-inverse (slower)
                      inv_F = torch.linalg.pinv(F)
                      inv_F_T = inv_F.transpose(-1, -2)
                      # Set log_J based on pseudo-determinant or handle otherwise?
                      # For simplicity, maybe set PK1 contribution to zero here if J is near zero?
                      zero_pk1_mask = torch.abs(J) < self.eps
                      log_J = torch.log(torch.clamp(J, min=self.eps)) # Still need logJ for non-singular ones

                 # Compute PK1: P = μ * (F - F⁻ᵀ) + λ * ln(J) * F⁻ᵀ
                 # Need to reshape log_J for broadcasting: [E, 1, 1]
                 log_J_reshaped = log_J.unsqueeze(-1).unsqueeze(-1)

                 P_q = self.mu * (F - inv_F_T) + self.lmbda * log_J_reshaped * inv_F_T

                 # Handle potentially singular elements where we used pseudo-inverse
                 # If mask exists, apply it (optional, depends on desired behavior)
                 # if 'zero_pk1_mask' in locals() and torch.any(zero_pk1_mask):
                 #     P_q[zero_pk1_mask] = 0.0

                 # Store result for this batch sample and quad point
                 pk1[b, :, q_idx, :, :] = P_q


        # Return result, removing batch dim if input was flat
        if original_shape_was_flat:
            pk1 = pk1.squeeze(0)

        return pk1


    # --- Volume Comparison (assuming it uses only coordinates and detJ_all) ---
    # Keep the volume comparison methods as they were, assuming they rely on
    # precomputed detJ or calculate volume geometrically.
    # Note: _compute_deformed_volume might need adjustment if it relies on F
    # or if the geometric calculation needs refinement for hex elements.

    def compute_volume_comparison(self, u_linear, u_total):
        """
        Compare volumes between original mesh, linear modes prediction, and neural network prediction.

        Args:
            u_linear: Linear displacement field [num_nodes*dim] or [batch=1, num_nodes*dim]
            u_total: Total displacement field [num_nodes*dim] or [batch=1, num_nodes*dim]

        Returns:
            Dictionary with volume information (volumes and ratios)
        """
        # Handle batch dimension if present (assuming batch=1 for comparison)
        if u_linear.dim() > 1 and u_linear.shape[0] == 1:
            u_linear = u_linear.squeeze(0)
        if u_total.dim() > 1 and u_total.shape[0] == 1:
            u_total = u_total.squeeze(0)

        # Ensure displacements are flat [N*D] for _compute_deformed_volume if needed
        # (Current implementation reshapes inside)

        # Calculate volumes
        original_volume = self._compute_mesh_volume()
        # Note: _compute_deformed_volume currently only works well for Tets.
        # Needs update for Hex using quadrature and J from deformed state.
        linear_volume = self._compute_deformed_volume(u_linear)
        neural_volume = self._compute_deformed_volume(u_total)

        # Calculate volume ratios
        # Add epsilon to prevent division by zero if original volume is tiny
        vol_eps = 1e-12
        linear_volume_ratio = linear_volume / (original_volume + vol_eps)
        neural_volume_ratio = neural_volume / (original_volume + vol_eps)

        # Calculate improvement ratio (how much closer neural is to 1 than linear)
        # Avoid division by zero if linear is already perfect
        linear_deviation = abs(linear_volume_ratio - 1.0)
        neural_deviation = abs(neural_volume_ratio - 1.0)
        if linear_deviation > self.eps: # Use class epsilon
            improvement_ratio = neural_deviation / linear_deviation
        elif neural_deviation <= self.eps : # Both are very close to 1
             improvement_ratio = 0.0 # Perfect preservation or improvement
        else: # Linear was perfect, neural is not
             improvement_ratio = float('inf') # Indicates worsening

        # Create result dictionary
        volume_info = {
            'original_volume': original_volume.item() if isinstance(original_volume, torch.Tensor) else original_volume,
            'linear_volume': linear_volume.item() if isinstance(linear_volume, torch.Tensor) else linear_volume,
            'neural_volume': neural_volume.item() if isinstance(neural_volume, torch.Tensor) else neural_volume,
            'linear_volume_ratio': linear_volume_ratio.item() if isinstance(linear_volume_ratio, torch.Tensor) else linear_volume_ratio,
            'neural_volume_ratio': neural_volume_ratio.item() if isinstance(neural_volume_ratio, torch.Tensor) else neural_volume_ratio,
            'volume_preservation_improvement_ratio': improvement_ratio # Lower is better (0=perfect neural, 1=same as linear)
        }

        return volume_info

    def _compute_mesh_volume(self):
        """Calculate the total volume of the original undeformed mesh using precomputed detJ."""
        if not self.precomputed or self.detJ_all is None:
             print("Warning: Precomputed detJ not available for volume calculation. Falling back to geometric (Tet only).")
             # Fallback geometric calculation (Tet only)
             if self.nodes_per_element != 4:
                 return torch.tensor(float('nan'), device=self.device) # Indicate failure
             total_volume = torch.tensor(0.0, dtype=self.dtype, device=self.device)
             for element_nodes in self.elements:
                 vertices = self.coordinates[element_nodes[:4]] # Use first 4 nodes for linear tet volume
                 # Using formula V = |det(v1-v0, v2-v0, v3-v0)| / 6
                 mat = (vertices[1:] - vertices[0]).T # Shape [3, 3]
                 volume = torch.abs(torch.linalg.det(mat)) / 6.0
                 total_volume += volume
             return total_volume

        # Preferred method: Integrate 1 over the domain using quadrature
        # Volume = ∑_elements ∑_qp (1 * detJ * quad_weight)
        # Reshape quad_weights for broadcasting: [1, num_qp]
        quad_weights_r = self.quadrature_weights.unsqueeze(0)
        # Element volumes: sum over quad points: [num_elements]
        element_volumes = torch.sum(self.detJ_all * quad_weights_r, dim=1)
        # Total volume: sum over elements
        total_volume = torch.sum(element_volumes)

        # Ensure positive volume (detJ should be positive for valid meshes)
        return torch.clamp(total_volume, min=0.0)


    def _compute_deformed_volume(self, displacement):
        """
        Calculate the total volume of the mesh after applying a displacement field.
        Uses integration of the determinant of the deformation gradient (J).

        Args:
            displacement: Displacement tensor [num_nodes*dim] or [num_nodes, dim]

        Returns:
            Total deformed volume (scalar tensor).
        """
        if not self.precomputed:
             raise RuntimeError("Deformed volume computation requires precomputed matrices.")

        # Ensure displacement is in the correct shape [num_nodes, dim]
        if displacement.dim() == 1:
             u_sample = displacement.view(self.num_nodes, self.dim)
        elif displacement.dim() == 2 and displacement.shape[0] == self.num_nodes:
             u_sample = displacement
        else:
             raise ValueError(f"Invalid displacement shape: {displacement.shape}")

        # Get element displacements [num_elements, nodes_per_element, dim]
        element_disps = u_sample[self.elements]

        # Initialize volume
        total_volume = torch.tensor(0.0, dtype=self.dtype, device=self.device)

        # Integrate J over the reference domain: ∫_Ω₀ J dV = Volume(Ω)
        # Loop over quadrature points
        for q_idx in range(self.num_quad_points):
            # Get precomputed data for this quad point:
            dN_dx_q = self.dN_dx_all[:, q_idx, :, :] # [E, N, D]
            detJ_ref_q = self.detJ_all[:, q_idx]     # [E] (Jacobian of reference map)
            qw_q = self.quadrature_weights[q_idx]    # scalar

            # Compute grad(u) = ∑ u_n * dNn/dx : [E, D, D]
            grad_u = torch.einsum('enj,enk->ejk', element_disps, dN_dx_q)

            # Compute deformation gradient F = I + grad(u) : [E, D, D]
            I = torch.eye(self.dim, dtype=self.dtype, device=self.device).expand(self.num_elements, -1, -1)
            F = I + grad_u

            # Compute J = det(F) for the deformed state : [E]
            J_deformed = torch.linalg.det(F)

            # Add contribution to total volume: ∫ J dV₀ = ∑ J * detJ_ref * quad_weight
            # We integrate J over the *reference* volume elements
            total_volume += torch.sum(J_deformed * detJ_ref_q * qw_q)

        return total_volume



class BoundaryConditionManager:
    """Manages boundary conditions for FEM problems"""
    
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fixed_dofs = torch.tensor([], dtype=torch.long, device=self.device)
        self.fixed_values = torch.tensor([], dtype=torch.float, device=self.device)
        
    def set_fixed_dofs(self, indices, values):
        """Set fixed DOFs with their values"""
        self.fixed_dofs = indices.to(self.device) if isinstance(indices, torch.Tensor) else torch.tensor(indices, dtype=torch.long, device=self.device)
        self.fixed_values = values.to(self.device) if isinstance(values, torch.Tensor) else torch.tensor(values, dtype=torch.float, device=self.device)
        
    def apply(self, displacement_batch):
        """Apply boundary conditions to displacement field"""
        # If no fixed DOFs, return original displacement
        if self.fixed_dofs.numel() == 0:
            return displacement_batch
            
        # Clone displacement to avoid modifying the input
        u_batch_fixed = displacement_batch.clone()
        
        # Get batch size
        batch_size = displacement_batch.shape[0]
        
        # Apply fixed boundary conditions using advanced indexing
        # Create batch indices that match the fixed DOFs for each sample in the batch
        batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, self.fixed_dofs.numel())
        flat_batch_indices = batch_indices.reshape(-1)
        repeated_dofs = self.fixed_dofs.repeat(batch_size)
        
        # Set fixed values for all samples in the batch
        u_batch_fixed[flat_batch_indices, repeated_dofs] = self.fixed_values.repeat(batch_size)
        
        return u_batch_fixed
    

class SmoothBoundaryConditionManager:
    """Manages boundary conditions for FEM problems with smooth enforcement"""
    
    def __init__(self, device=None, penalty_strength=1e3):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fixed_dofs = torch.tensor([], dtype=torch.long, device=self.device)
        self.fixed_values = torch.tensor([], dtype=torch.float, device=self.device)
        self.penalty_strength = penalty_strength
        
    def set_fixed_dofs(self, indices, values):
        """Set fixed DOFs with their values"""
        self.fixed_dofs = indices.to(self.device) if isinstance(indices, torch.Tensor) else torch.tensor(indices, dtype=torch.long, device=self.device)
        self.fixed_values = values.to(self.device) if isinstance(values, torch.Tensor) else torch.tensor(values, dtype=torch.float, device=self.device)
    
    def apply(self, displacement_batch):
        """
        Apply boundary conditions with smooth penalty rather than hard enforcement
        This returns the original displacements for gradient calculation
        """
        # If no fixed DOFs, return original displacement
        if self.fixed_dofs.numel() == 0:
            return displacement_batch
            
        # In this approach, we don't modify the displacements
        # Instead we'll add a penalty term to the energy
        return displacement_batch
        
    def compute_penalty_energy(self, displacement_batch):
        """
        Compute penalty energy for boundary condition enforcement
        
        Args:
            displacement_batch: Displacement tensor [batch_size, num_nodes*dim]
            
        Returns:
            Penalty energy per batch sample
        """
        # If no fixed DOFs, return zero energy
        if self.fixed_dofs.numel() == 0:
            return torch.zeros(displacement_batch.shape[0], device=self.device)
            
        # Get batch size
        batch_size = displacement_batch.shape[0]
        
        # Create batch indices that match the fixed DOFs for each sample
        batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, self.fixed_dofs.numel())
        flat_batch_indices = batch_indices.reshape(-1)
        repeated_dofs = self.fixed_dofs.repeat(batch_size)
        repeated_values = self.fixed_values.repeat(batch_size)
        
        # Get actual displacement values at constrained DOFs
        actual_values = displacement_batch[flat_batch_indices, repeated_dofs]
        
        # Compute squared differences
        squared_diff = torch.pow(actual_values - repeated_values, 2)
        
        # Reshape to [batch_size, num_fixed_dofs]
        squared_diff = squared_diff.reshape(batch_size, -1)
        
        # Sum over fixed DOFs and apply penalty strength
        penalty_energy = self.penalty_strength * squared_diff.sum(dim=1)
        
        return penalty_energy


class ModernFEMSolver(torch.nn.Module):
    """
    Modern Finite Element Method (FEM) solver using PyTorch for automatic differentiation.
    This solver is designed for nonlinear solid mechanics problems with Neo-Hookean materials.
    
    Args:
        energy_model: Energy model for the material behavior
        max_iterations: Maximum number of iterations for the nonlinear solver
        tolerance: Convergence tolerance for the nonlinear solver
        energy_tolerance: Energy convergence tolerance for the nonlinear solver
        verbose: Whether to print detailed solver information
        visualize: Whether to visualize the solver progress
        filename: Filename for mesh visualization (optional)
    """
    def __init__(self, energy_model, max_iterations=20, tolerance=1e-8,
                energy_tolerance=1e-8, verbose=True, visualize=False, filename=None):
        super().__init__()

        # Store energy model
        self.energy_model = energy_model
        
        # Store original parameters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.energy_tolerance = energy_tolerance
        self.verbose = verbose
        
        # Store device and problem size
        self.device = energy_model.device
        self.dtype = energy_model.dtype
        self.num_nodes = energy_model.num_nodes
        self.dim = energy_model.dim
        self.dof_count = self.num_nodes * self.dim
        
        # Create boundary condition manager
        self.bc_manager = SmoothBoundaryConditionManager(device=self.device)
        
        # Boundary conditions
        self.fixed_dofs = torch.tensor([], dtype=torch.long, device=self.device)
        self.fixed_values = torch.tensor([], dtype=self.dtype, device=self.device)

        # Visualization flag
        self.visualize = visualize
        if self.visualize:
            self._setup_visualization(filename)
    
    def forward(self, external_forces, u0=None):
        """Solve nonlinear system for a batch of forces"""
    
        # Try to solve with standard approach
        return self._solve_with_torch_lbfgs(external_forces, u0)
    
    def apply_boundary_conditions(self, u_batch):
        """Apply boundary conditions to displacement field"""
        return self.bc_manager.apply(u_batch)

    # Add methods to set boundary conditions
    def set_fixed_dofs(self, indices, values):
        """Set fixed DOFs with their values"""
        self.bc_manager.set_fixed_dofs(indices, values)
    
    
    def _solve_with_torch_lbfgs(self, external_forces, u0=None, history_size=50, max_iter=50):
        """Use PyTorch's built-in L-BFGS optimizer for FEM solving"""
        # Initialize displacement
        batch_size = external_forces.shape[0]
        
        # Process each sample individually for better control
        solutions = []
        
        for i in range(batch_size):
            # Get single sample (keeping batch dimension)
            f_i = external_forces[i:i+1]
            
            # Initialize displacement for this sample
            if u0 is None:
                # Initialize with small random displacements instead of zeros
                u_i = torch.randn_like(f_i) * 0.1
                u_i.requires_grad_(True)
            else:
                u_i = u0[i:i+1].clone().detach().requires_grad_(True)
            
            # Apply boundary conditions
            u_i = self.apply_boundary_conditions(u_i)
            
            # Create optimizer
            optimizer = torch.optim.LBFGS([u_i], 
                            lr=1,
                            max_iter=max_iter,
                            history_size=history_size,
                            line_search_fn='strong_wolfe',
                            max_eval=100,  # Add this - limits function evaluations
                            tolerance_grad=self.tolerance,  # Relax tolerance
                            tolerance_change=self.energy_tolerance)  # Relax tolerance
            
            # Convergence tracking
            initial_energy = None
            iter_count = 0
            
            # Optimization closure
            # Inside _solve_with_torch_lbfgs method, in the closure function:
            def closure():
                nonlocal iter_count, initial_energy
                optimizer.zero_grad()
                
                # Compute strain energy
                strain_energy = self.energy_model.compute_energy(u_i)
                
                # Compute external work
                external_work = torch.sum(f_i * u_i, dim=1)
                
                # Compute BC penalty
                bc_penalty = self.bc_manager.compute_penalty_energy(u_i)
                
                # CORRECT: Direct potential energy - no squaring, no artificial penalties
                energy_functional = strain_energy - external_work

                # compute the gradient of the total energy
                energy_grad = torch.autograd.grad(energy_functional, u_i, create_graph=True)[0]

                grad_norm = torch.norm(energy_grad)
                
                # Compute force-based convergence metrics
                internal_forces = torch.autograd.grad(energy_functional, u_i, create_graph=True)[0]
                residual = f_i - internal_forces

                fixed_dofs_mask = torch.zeros_like(residual, dtype=torch.bool)
                for dof in self.bc_manager.fixed_dofs:
                    fixed_dofs_mask[:, dof] = True

                # Apply mask (keep only free DOF residuals)
                filtered_residual = residual * (~fixed_dofs_mask)

                # Compute proper norm (only considering free DOFs)
                free_dof_count = residual.numel() - self.bc_manager.fixed_dofs.numel()
                free_dof_count = torch.tensor(free_dof_count, device=self.device, dtype=self.dtype)
                filtered_residual_norm = torch.norm(filtered_residual) / torch.sqrt(free_dof_count)
                                    
                objective = 1000 * torch.sum(filtered_residual**2) + bc_penalty

             
                
                # Store original energy for logging
                original_energy = energy_functional + bc_penalty 
               

                # Compute energy ratio with tensor operations (absolute values)
                if iter_count == 0:
                    initial_energy = energy_functional.clone()
                else:
                    # Add small epsilon to prevent division by zero
                    energy_ratio = energy_functional.abs() / (initial_energy + 1e-10)
                    # Scale large ratios using tensor operations
                    energy_ratio = torch.where(
                        energy_ratio > 10.0,
                        torch.log10(energy_ratio) + 1.0,
                        energy_ratio
                    )
                
                
                # For logging
                if self.verbose and iter_count % 1 == 0:
                   print(f"Sample {i+1}: iter {iter_count}, residual={filtered_residual_norm.item():.2e}, "
                        f"orig_energy={original_energy.item():.2e}, energy={energy_functional.item():.2e}, "
                        f"energy_ratio={energy_ratio.item():.4f}, ext_work={external_work.item():.2e}")
                    
                # Update visualization if enabled
                if self.visualize and iter_count % 5 == 0:
                    self._update_visualization(
                        u_i, f_i, i, iter_count,
                        filtered_residual_norm.item(), energy_functional.item(), grad_norm.item(),
                        strain_energy.item(), external_work.item()  # Pass energy components
                    )
                                
                # Compute gradient
                objective.backward(retain_graph=True)
                
                iter_count += 1
                return objective


            optimizer.step(closure)
            
            
            # Store solution
            solutions.append(u_i.detach())
        
        # Stack solutions back into batch
        return torch.cat(solutions, dim=0)
        
    
    def _setup_visualization(self, filename=None):
        """Set up real-time visualization for solver progress"""
        # Convert mesh to PyVista format
        domain, _, _ = gmshio.read_from_msh(filename, MPI.COMM_WORLD, gdim=3)
        topology, cell_types, x = plot.vtk_mesh(domain)
        self.viz_grid = pyvista.UnstructuredGrid(topology, cell_types, x)
        
        # Create plotter with two viewports
        self.viz_plotter = pyvista.Plotter(shape=(1, 2), 
                                          title="FEM Solver Visualization",
                                          window_size=[1200, 600], 
                                          off_screen=False)
        
        # Initialize initial and deformed mesh actors
        self.viz_plotter.subplot(0, 0)
        self.viz_plotter.add_text("Applied Forces", position="upper_right", font_size=12)
        self.mesh_actor_left = self.viz_plotter.add_mesh(self.viz_grid, color='lightblue', show_edges=True)
        
        self.viz_plotter.subplot(0, 1)
        self.viz_plotter.add_text("Deformed Configuration", position="upper_right", font_size=10)
        self.mesh_actor_right = self.viz_plotter.add_mesh(self.viz_grid, color='lightblue', show_edges=True)
        
        # Add info text for solver status
        self.info_actor = self.viz_plotter.add_text("Initializing solver...", position=(0.02, 0.02), font_size=10)
        
        # Link camera views
        self.viz_plotter.link_views()
        
        # Show the window without blocking
        self.viz_plotter.show(interactive=False, auto_close=False)
        
    def _update_visualization(self, u_i, f_i, i, iter_count, residual_norm, energy, energy_ratio, strain_energy=None, external_work=None):
        """Update real-time visualization with current solver state and force field"""
        if not hasattr(self, 'viz_plotter') or not self.visualize:
            return
            
        try:
            # Cache CPU arrays once at initialization time
            if not hasattr(self, 'viz_points_cpu'):
                # Store original points as numpy array once
                self.viz_points_cpu = self.viz_grid.points.copy()
                
            # Move tensors to CPU only once per update
            with torch.no_grad():  # Avoid tracking history
                u_cpu = u_i.detach().cpu()
                f_cpu = f_i.detach().cpu()
                
                # Reshape once
                u_array = u_cpu.numpy().reshape(-1, 3)
                f_array = f_cpu.numpy().reshape(-1, 3)
            
            # Create a new grid for forces (left viewport)
            force_grid = self.viz_grid.copy()
            force_grid.point_data["Forces"] = f_array
            force_mag = np.linalg.norm(f_array, axis=1)
            force_grid["force_magnitude"] = force_mag
            
            # Create a deformed grid for displacements (right viewport)
            deformed_grid = self.viz_grid.copy()
            # Use cached points instead of accessing self.viz_grid.points again
            deformed_grid.points = self.viz_points_cpu + u_array
            
            # Compute displacement magnitude for coloring
            displacement_magnitude = np.linalg.norm(u_array, axis=1)
            deformed_grid["displacement"] = displacement_magnitude

            if hasattr(self, 'info_actor'):
                self.viz_plotter.remove_actor(self.info_actor)
            
            # Update left viewport with force visualization and strain energy data
            self.viz_plotter.subplot(0, 0)
            self.viz_plotter.remove_actor(self.mesh_actor_left)
            self.mesh_actor_left = self.viz_plotter.add_mesh(
                force_grid, 
                scalars="force_magnitude",
                cmap="plasma",
                show_edges=True,
                clim=[0, np.max(force_mag) if np.max(force_mag) > 0 else 1.0]
            )
            
            # Create strain energy text for left subplot
            if strain_energy is not None:
                strain_energy_text = (
                    f"Strain Energy: {strain_energy:.2e}\n"
                    f"External Work: {external_work:.2e}\n"
                    f"Force Magnitude: {np.max(force_mag):.2e}\n"
                    f"SE/EW: {strain_energy / external_work:.2e}\n"
                    f"Sample: {i+1}, Iter: {iter_count}\n"
                )
                
                # Remove old text if it exists
                if hasattr(self, 'strain_text_actor'):
                    self.viz_plotter.remove_actor(self.strain_text_actor)
                    
                # Add strain energy text to left subplot
                self.strain_text_actor = self.viz_plotter.add_text(
                    strain_energy_text, 
                    position="upper_left",
                    font_size=10, 
                    color='black',
                    shadow=True
                )
            
            # Add vector field visualization using glyphs more efficiently
            self.viz_plotter.subplot(0, 0)
            
            # Remove previous arrows if they exist
            if hasattr(self, 'arrow_actor'):
                self.viz_plotter.remove_actor(self.arrow_actor)
            
            # Only calculate vectors if forces are non-zero
            if np.max(force_mag) > 0:
                # Get indices of significant forces - more efficient filtering
                threshold = np.max(force_mag) * 0.1
                mask = force_mag > threshold
                
                if np.sum(mask) > 0:
                    # Create subset of points and vectors for glyphs
                    points = self.viz_points_cpu[mask]  # Use cached points
                    
                    # Scale vectors for visualization
                    scale = 0.2 * self.viz_grid.length / (np.max(force_mag) + 1e-10)
                    vectors = f_array[mask] * scale
                    
                    # Create point cloud for glyph source
                    point_cloud = pyvista.PolyData(points)
                    point_cloud["vectors"] = vectors
                    
                    # Create glyphs (arrows) with simpler parameters
                    arrows = point_cloud.glyph(
                        orient="vectors",
                        scale="vectors", 
                        factor=1.0,
                        geom=pyvista.Arrow(shaft_radius=0.02, tip_length=0.25)
                    )
                    
                    # Store the actor for future removal
                    self.arrow_actor = self.viz_plotter.add_mesh(arrows, color='red')
            
            # Update right viewport with deformed mesh and external work data
            self.viz_plotter.subplot(0, 1)
            self.viz_plotter.remove_actor(self.mesh_actor_right)
            self.mesh_actor_right = self.viz_plotter.add_mesh(
                deformed_grid, 
                scalars="displacement",
                cmap="viridis",
                show_edges=True,
                clim=[0, np.max(displacement_magnitude) if np.max(displacement_magnitude) > 0 else 1.0]
            )
            
            # Create external work text for right subplot
            if external_work is not None:
                external_work_text = (
                    f"Max Displacement: {np.max(displacement_magnitude):.2e}\n"
                    f"Energy Functional: {energy:.2f}\n"
                    f"Energy Gradient: {energy_ratio:.4f}\n"
                    f"Residual: {residual_norm:.2e}\n"
                )
                
                # Remove old text if it exists
                if hasattr(self, 'work_text_actor'):
                    self.viz_plotter.remove_actor(self.work_text_actor)
                    
                # Add external work text to right subplot
                self.work_text_actor = self.viz_plotter.add_text(
                    external_work_text, 
                    position="upper_left",
                    font_size=10,
                    color='black',
                    shadow=True
                )
            
            # Render the updated scene
            self.viz_plotter.update()
            
        except Exception as e:
            print(f"Visualization error: {str(e)}")
            import traceback
            print(traceback.format_exc())
  

    def close_visualization(self):
        """Close the visualization window"""
        if hasattr(self, 'viz_plotter') and self.viz_plotter is not None:
            self.viz_plotter.close()


    """
    High-performance Newton-method FEM solver with full vectorization.
    Implements direct solution of the nonlinear equilibrium equation using
    Newton's method with line search and PCG for the linear substeps.
    
    Args:
        energy_model: Energy model for the material behavior
        max_iterations: Maximum number of Newton iterations
        tolerance: Residual convergence tolerance 
        verbose: Whether to print detailed solver information
        visualize: Whether to visualize the solver progress
        filename: Mesh filename for visualization (optional)
    """
    def __init__(self, energy_model, max_iterations=20, tolerance=1e-8, 
                 verbose=True, visualize=False, filename=None):
        super().__init__()
        
        # Store energy model and parameters
        self.energy_model = energy_model
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        
        # Store device and problem size
        self.device = energy_model.device
        self.dtype = energy_model.dtype
        self.num_nodes = energy_model.num_nodes
        self.dim = energy_model.dim
        self.dof_count = self.num_nodes * self.dim
        
        # Create boundary condition manager
        self.bc_manager = SmoothBoundaryConditionManager(device=self.device)
        
        # Visualization settings
        self.visualize = visualize
        if self.visualize:
            self._setup_visualization(filename)
    
    def forward(self, external_forces, u0=None):
        """Solve nonlinear system for a batch of external forces"""
        return self._solve_with_newton(external_forces, u0)
    
    def set_fixed_dofs(self, indices, values):
        """Set fixed DOFs with their values"""
        self.bc_manager.set_fixed_dofs(indices, values)
        
    def _solve_with_newton(self, external_forces, u0=None):
        """
        Solve the nonlinear FEM problem using Newton's method.
        
        Args:
            external_forces: External force tensor [batch_size, num_nodes*dim]
            u0: Initial displacement guess (optional)
            
        Returns:
            Equilibrium displacement field
        """
        # Initialize displacement
        batch_size = external_forces.shape[0]
        
        # Process each sample in the batch (but still leverage vectorization)
        solutions = []
        
        for i in range(batch_size):
            # Get single sample (keeping batch dimension)
            f_i = external_forces[i:i+1]
            
            # Initialize displacement for this sample
            if u0 is None:
                # Initialize with small random displacements
                u_i = torch.randn_like(f_i) * 0.01
            else:
                u_i = u0[i:i+1].clone().detach()
            
            # Ensure gradients are tracked
            u_i.requires_grad_(True)
            
            # Apply boundary conditions to initial guess
            u_i = self.bc_manager.apply(u_i)
            
            # Prepare mask for fixed DOFs
            fixed_dofs_mask = torch.zeros_like(u_i, dtype=torch.bool)
            for dof in self.bc_manager.fixed_dofs:
                fixed_dofs_mask[:, dof] = True
            
            # Newton iteration tracking
            iter_count = 0
            converged = False
            
            # For line search
            alpha_min = 0.05
            alpha_max = 1.0
            
            # Initial residual computation for convergence check
            strain_energy = self.energy_model.compute_energy(u_i)
            external_work = torch.sum(f_i * u_i, dim=1)
            internal_forces = self.energy_model.compute_gradient(u_i)
            residual = f_i - internal_forces
            filtered_residual = residual * (~fixed_dofs_mask)
            residual_norm = torch.norm(filtered_residual)
            
            # Initial logging
            if self.verbose:
                energy = strain_energy - external_work
                print(f"Initial state: residual={residual_norm.item():.2e}, energy={energy.item():.2e}")
            
            # Main Newton iteration loop
            while iter_count < self.max_iterations and not converged:
                # 1. Compute the gradient (residual)
                u_i.requires_grad_(True)
                strain_energy = self.energy_model.compute_energy(u_i)
                external_work = torch.sum(f_i * u_i, dim=1)
                
                # Get internal forces (gradient of strain energy)
                internal_forces = self.energy_model.compute_gradient(u_i)
                
                # Compute residual: R = f_ext - f_int
                residual = f_i - internal_forces
                
                # Zero out residual at fixed DOFs
                filtered_residual = residual * (~fixed_dofs_mask)
                
                # Compute residual norm for convergence check
                residual_norm = torch.norm(filtered_residual)
                
                # Check convergence
                if residual_norm < self.tolerance:
                    converged = True
                    if self.verbose:
                        print(f"Converged at iteration {iter_count}, residual={residual_norm.item():.2e}")
                    break
                
                # 2. Compute the tangent stiffness matrix (Hessian)
                # We use Hessian-vector products for efficiency
                def hessian_vector_product(v):
                    """Compute Hessian-vector product via autodiff"""
                    Rv = torch.autograd.grad(
                        filtered_residual, u_i, 
                        grad_outputs=v, 
                        retain_graph=True, 
                        create_graph=False
                    )[0]
                    return -Rv  # Negative because residual = f_ext - f_int
                
                # 3. Solve for displacement update using Conjugate Gradient
                delta_u = self._solve_newton_system(
                    hessian_vector_product, filtered_residual, 
                    fixed_dofs_mask, max_iter=100, tol=1e-6
                )
                
                # 4. Line search for step size
                alpha = self._line_search(
                    u_i, delta_u, f_i, filtered_residual, fixed_dofs_mask,
                    alpha_min=alpha_min, alpha_max=alpha_max
                )
                
                # 5. Update displacement
                with torch.no_grad():
                    u_i = u_i + alpha * delta_u
                
                # Log progress
                if self.verbose:
                    energy = strain_energy - external_work
                    print(f"Iter {iter_count+1}: residual={residual_norm.item():.2e}, "
                          f"energy={energy.item():.2e}, alpha={alpha:.3f}")
                
                # Visualize if enabled
                if self.visualize and iter_count % 2 == 0:
                    energy = strain_energy - external_work
                    self._update_visualization(
                        u_i, f_i, i, iter_count, 
                        residual_norm.item(), energy.item(), 0.0,
                        strain_energy.item(), external_work.item()
                    )
                
                # Update iteration counter
                iter_count += 1
            
            # Final state
            if not converged and self.verbose:
                print(f"Warning: Newton solver did not converge in {self.max_iterations} iterations")
            
            # Store solution
            solutions.append(u_i.detach())
        
        # Stack solutions back into batch
        return torch.cat(solutions, dim=0)
    
    def _solve_newton_system(self, hessian_vector_product, residual, fixed_dofs_mask, max_iter=100, tol=1e-6):
        """
        Solve the Newton system K(u)·Δu = -R(u) using Conjugate Gradient.
        
        Args:
            hessian_vector_product: Function that computes H·v
            residual: Current residual vector
            fixed_dofs_mask: Mask for fixed DOFs
            max_iter: Maximum CG iterations
            tol: Convergence tolerance for CG
            
        Returns:
            Displacement update vector Δu
        """
        # Initialize solution vector
        x = torch.zeros_like(residual)
        
        # Initial residual for CG: r = -R - K·0 = -R
        r = -residual.clone()
        
        # Zero out fixed DOFs in the residual
        r = r * (~fixed_dofs_mask)
        
        # Initialize direction vector
        p = r.clone()
        
        # Initial residual norm
        r_norm_sq = torch.sum(r * r)
        initial_r_norm_sq = r_norm_sq
        
        # CG iteration loop
        for j in range(max_iter):
            # Apply Hessian to direction vector: Ap = K·p
            Ap = hessian_vector_product(p)
            Ap = Ap * (~fixed_dofs_mask)
            
            # Compute step size
            pAp = torch.sum(p * Ap)
            if pAp.item() == 0:
                # If direction is a null vector, stop iteration
                break
            
            alpha = r_norm_sq / pAp
            
            # Update solution
            x = x + alpha * p
            
            # Update residual
            r = r - alpha * Ap
            
            # Check convergence
            new_r_norm_sq = torch.sum(r * r)
            if new_r_norm_sq < tol * tol * initial_r_norm_sq:
                break
            
            # Update direction
            beta = new_r_norm_sq / r_norm_sq
            r_norm_sq = new_r_norm_sq
            p = r + beta * p
        
        # Zero out fixed DOF components
        x = x * (~fixed_dofs_mask)
        
        return x
    
    def _line_search(self, u, delta_u, f_ext, residual, fixed_dofs_mask, 
                     alpha_min=0.05, alpha_max=1.0, max_trials=8):
        """
        Backtracking line search to find a step size that decreases the residual.
        
        Args:
            u: Current displacement
            delta_u: Computed update direction
            f_ext: External forces
            residual: Current residual
            fixed_dofs_mask: Mask for fixed DOFs
            alpha_min: Minimum step size
            alpha_max: Initial (maximum) step size
            max_trials: Maximum number of step size reductions
            
        Returns:
            Step size alpha
        """
        # Compute initial residual norm
        initial_residual_norm = torch.norm(residual)
        
        # Start with maximum step size
        alpha = alpha_max
        
        # Initial energy for reference
        with torch.no_grad():
            initial_energy = self.energy_model.compute_energy(u)
        
        # Try decreasing step sizes
        for _ in range(max_trials):
            # Trial displacement
            u_trial = u + alpha * delta_u
            
            # Compute trial residual
            with torch.no_grad():
                internal_forces = self.energy_model.compute_gradient(u_trial)
                trial_residual = f_ext - internal_forces
                trial_residual = trial_residual * (~fixed_dofs_mask)
                trial_residual_norm = torch.norm(trial_residual)
            
            # Accept if residual decreased
            if trial_residual_norm < initial_residual_norm:
                return alpha
            
            # Otherwise reduce step size
            alpha *= 0.5
            
            # Don't go below minimum step size
            if alpha < alpha_min:
                return alpha_min
        
        # Return minimum step size if we couldn't find a better one
        return alpha_min
    
    def _setup_visualization(self, filename=None):
        """Set up visualization similar to ModernFEMSolver"""
        # This would be similar to the implementation in ModernFEMSolver
        if filename is None:
            self.visualize = False
            return
            
        # Convert mesh to PyVista format
        domain, _, _ = gmshio.read_from_msh(filename, MPI.COMM_WORLD, gdim=3)
        topology, cell_types, x = plot.vtk_mesh(domain)
        self.viz_grid = pyvista.UnstructuredGrid(topology, cell_types, x)
        
        # Create plotter with two viewports
        self.viz_plotter = pyvista.Plotter(shape=(1, 2), 
                                          title="Newton FEM Solver", 
                                          window_size=[1200, 600],
                                          off_screen=False)
        
        # Left viewport - forces
        self.viz_plotter.subplot(0, 0)
        self.viz_plotter.add_text("Applied Forces", position="upper_right", font_size=12)
        self.mesh_actor_left = self.viz_plotter.add_mesh(self.viz_grid, color='lightblue', show_edges=True)
        
        # Right viewport - deformed shape
        self.viz_plotter.subplot(0, 1)
        self.viz_plotter.add_text("Deformed Configuration", position="upper_right", font_size=10)
        self.mesh_actor_right = self.viz_plotter.add_mesh(self.viz_grid, color='lightblue', show_edges=True)
        
        # Add info text
        self.info_actor = self.viz_plotter.add_text("Initializing Newton solver...", position=(0.02, 0.02), font_size=10)
        
        # Link camera views
        self.viz_plotter.link_views()
        
        # Show window without blocking
        self.viz_plotter.show(interactive=False, auto_close=False)
        
        # Cache points for faster visualization
        self.viz_points_cpu = self.viz_grid.points.copy()
    
    def _update_visualization(self, u_i, f_i, i, iter_count, residual_norm, energy, energy_ratio, strain_energy=None, external_work=None):
        """Update visualization with current solver state"""
        # This would be similar to the implementation in ModernFEMSolver
        if not hasattr(self, 'viz_plotter') or not self.visualize:
            return
            
        try:
            # Move tensors to CPU
            with torch.no_grad():
                u_cpu = u_i.detach().cpu()
                f_cpu = f_i.detach().cpu()
                
                # Reshape
                u_array = u_cpu.numpy().reshape(-1, 3)
                f_array = f_cpu.numpy().reshape(-1, 3)
            
            # Create force grid for left viewport
            force_grid = self.viz_grid.copy()
            force_grid.point_data["Forces"] = f_array
            force_mag = np.linalg.norm(f_array, axis=1)
            force_grid["force_magnitude"] = force_mag
            
            # Create deformed grid for right viewport
            deformed_grid = self.viz_grid.copy()
            deformed_grid.points = self.viz_points_cpu + u_array
            
            # Displacement magnitude for coloring
            displacement_magnitude = np.linalg.norm(u_array, axis=1)
            deformed_grid["displacement"] = displacement_magnitude
            
            # Update force visualization (left)
            self.viz_plotter.subplot(0, 0)
            self.viz_plotter.remove_actor(self.mesh_actor_left)
            self.mesh_actor_left = self.viz_plotter.add_mesh(
                force_grid,
                scalars="force_magnitude",
                cmap="plasma",
                show_edges=True,
                clim=[0, np.max(force_mag) if np.max(force_mag) > 0 else 1.0]
            )
            
            # Update strain energy text
            if strain_energy is not None:
                strain_energy_text = (
                    f"Strain Energy: {strain_energy:.2e}\n"
                    f"External Work: {external_work:.2e}\n"
                    f"Force Magnitude: {np.max(force_mag):.2e}\n"
                    f"Sample: {i+1}, Iter: {iter_count}\n"
                )
                
                # Remove old text
                if hasattr(self, 'strain_text_actor'):
                    self.viz_plotter.remove_actor(self.strain_text_actor)
                    
                # Add strain energy text
                self.strain_text_actor = self.viz_plotter.add_text(
                    strain_energy_text,
                    position="upper_left",
                    font_size=10,
                    color='black',
                    shadow=True
                )
            
            # Update deformation visualization (right)
            self.viz_plotter.subplot(0, 1)
            self.viz_plotter.remove_actor(self.mesh_actor_right)
            self.mesh_actor_right = self.viz_plotter.add_mesh(
                deformed_grid,
                scalars="displacement",
                cmap="viridis",
                show_edges=True,
                clim=[0, np.max(displacement_magnitude) if np.max(displacement_magnitude) > 0 else 1.0]
            )
            
            # Update solution info text
            newton_text = (
                f"Newton Iteration: {iter_count}\n"
                f"Residual Norm: {residual_norm:.2e}\n"
                f"Energy: {energy:.2e}\n"
                f"Max Displacement: {np.max(displacement_magnitude):.2e}\n"
            )
            
            # Remove old text
            if hasattr(self, 'newton_text_actor'):
                self.viz_plotter.remove_actor(self.newton_text_actor)
                
            # Add newton info text
            self.newton_text_actor = self.viz_plotter.add_text(
                newton_text,
                position="upper_left",
                font_size=10,
                color='black',
                shadow=True
            )
            
            # Render the updated scene
            self.viz_plotter.update()
            
        except Exception as e:
            print(f"Visualization error: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    def close_visualization(self):
        """Close the visualization window"""
        if hasattr(self, 'viz_plotter') and self.viz_plotter is not None:
            self.viz_plotter.close()