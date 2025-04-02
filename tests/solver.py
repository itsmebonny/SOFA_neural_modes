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
import matplotlib.pyplot as plt

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
        self.gt_disp = None
        
        # Create boundary condition manager
        self.bc_manager = SmoothBoundaryConditionManager(device=self.device)
        
        # Boundary conditions
        self.fixed_dofs = torch.tensor([], dtype=torch.long, device=self.device)
        self.fixed_values = torch.tensor([], dtype=self.dtype, device=self.device)

        # Visualization flag
        self.visualize = visualize
        if self.visualize:
            self._setup_visualization(filename)
            
            # Initialize additional actors for ground truth visualization
            self.gt_actor = None
            self.error_text_actor = None
    
    def forward(self, external_forces, u0=None):
        """
        Solve nonlinear system for a batch of forces
        
        Args:
            external_forces: External forces to apply
            u0: Initial displacement guess (optional)
            gt_disp: Ground truth displacement for comparison visualization (optional)
        
        Returns:
            Displacement solution
        """
        # Try to solve with standard approach
        return self._solve_with_torch_lbfgs(external_forces, u0)

    
    def apply_boundary_conditions(self, u_batch):
        """Apply boundary conditions to displacement field"""
        return self.bc_manager.apply(u_batch)

    # Add methods to set boundary conditions
    def set_fixed_dofs(self, indices, values):
        """Set fixed DOFs with their values"""
        self.bc_manager.set_fixed_dofs(indices, values)

    def set_disp_gt(self, disp_gt):
        self.gt_disp = disp_gt
    
    
    def _solve_with_torch_lbfgs(self, external_forces, u0=None, history_size=50, max_iter=50):
        """Use PyTorch's built-in L-BFGS optimizer for FEM solving"""
        # Initialize displacement
        batch_size = external_forces.shape[0]
        
        # Process each sample individually for better control
        solutions = []
        
        for i in range(batch_size):
            # Get single sample (keeping batch dimension)
            f_i = external_forces[i:i+1]

            # Get corresponding ground truth if provided
            gt_disp_i = None
            if self.gt_disp is not None:
                gt_disp_i = self.gt_disp[i:i+1]
            
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
                        strain_energy.item(), external_work.item(),
                        gt_disp=gt_disp_i  # Pass ground truth to visualization
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
        
    def _update_visualization(self, u_i, f_i, i, iter_count, residual_norm, energy, energy_ratio, 
                        strain_energy=None, external_work=None, gt_disp=None):
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
                
                # Convert ground truth if provided
                gt_array = None
                if gt_disp is not None:
                    gt_cpu = gt_disp.detach().cpu()
                    gt_array = gt_cpu.numpy().reshape(-1, 3)
                
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
            
            # Add ground truth wireframe if provided
            if gt_array is not None:
                # Remove old ground truth if it exists
                if hasattr(self, 'gt_actor'):
                    self.viz_plotter.remove_actor(self.gt_actor)
                
                # Create ground truth mesh
                gt_grid = self.viz_grid.copy()
                gt_grid.points = self.viz_points_cpu + gt_array
                
                # Add as wireframe
                self.gt_actor = self.viz_plotter.add_mesh(
                    gt_grid,
                    style='wireframe',
                    color='red',
                    line_width=2,
                    opacity=0.7
                )
                
                # Calculate error between current and ground truth
                error = np.linalg.norm(u_array - gt_array, axis=1)
                mean_error = np.mean(error)
                max_error = np.max(error)
                
                # Add error information to the visualization
                error_text = (
                    f"GT Comparison:\n"
                    f"Mean Error: {mean_error:.2e}\n"
                    f"Max Error: {max_error:.2e}\n"
                )
                
                # Remove old error text if it exists
                if hasattr(self, 'error_text_actor'):
                    self.viz_plotter.remove_actor(self.error_text_actor)
                    
                # Add error text
                self.error_text_actor = self.viz_plotter.add_text(
                    error_text,
                    position="lower_right",
                    font_size=10,
                    color='red',
                    shadow=True
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

# --- FullFEMSolver Class ---
class FullFEMSolver(torch.nn.Module):


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
        self.gt_mesh_actor = None # Initialize ground truth actor reference
        if self.visualize:
            # Ensure filename is provided if visualize is True
            if filename is None:
                 print("Warning: Visualization enabled but no mesh filename provided. Disabling visualization.")
                 self.visualize = False
            else:
                # Use a dummy import block for visualization setup if dolfinx/gmshio aren't strictly needed elsewhere
                try:
                    from dolfinx import plot
                    from dolfinx.io import gmshio
                    self._setup_visualization(filename)
                except ImportError:
                    print("Warning: dolfinx or gmshio not found. Disabling visualization.")
                    self.visualize = False
        else:
            self.viz_plotter = None # Ensure plotter is None if not visualizing


    def forward(self, external_forces, u0=None, gt_disp=None): 
        """
        Solve nonlinear system for a batch of external forces.

        Args:
            external_forces: External force tensor [batch_size, num_nodes*dim]
            u0: Initial displacement guess (optional) [batch_size, num_nodes*dim]
            gt_disp: Ground truth displacement (optional) [batch_size, num_nodes*dim]

        Returns:
            Equilibrium displacement field [batch_size, num_nodes*dim]
        """
        return self._solve_with_newton(external_forces, u0, gt_disp) # PASS gt_disp

    def set_fixed_dofs(self, indices, values):
        """Set fixed DOFs with their values"""
        # Ensure values are compatible with the model's dtype
        values_tensor = torch.tensor(values, dtype=self.dtype)
        self.bc_manager.set_fixed_dofs(indices, values_tensor)

    def _solve_with_newton(self, external_forces, u0=None, gt_disp=None): # ADDED gt_disp
        """
        Solve the nonlinear FEM problem using Newton's method.

        Args:
            external_forces: External force tensor [batch_size, num_nodes*dim]
            u0: Initial displacement guess (optional)
            gt_disp: Ground truth displacement (optional) [batch_size, num_nodes*dim]

        Returns:
            Equilibrium displacement field
        """
        # Initialize displacement
        batch_size = external_forces.shape[0]

        # Process each sample in the batch
        solutions = []

        for i in range(batch_size):
            # Get single sample (keeping batch dimension of size 1)
            f_i = external_forces[i:i+1].detach().clone() # Ensure it's a fresh tensor

            # Extract ground truth for this sample, if available
            gt_disp_i = gt_disp[i:i+1].detach().clone() if gt_disp is not None else None

            # Initialize displacement for this sample
            if u0 is None:
                # Initialize with small random displacements or zeros
                u_i = torch.randn_like(f_i) * 0.001 # Small random
                #u_i = torch.zeros_like(f_i) # Start from zero
            else:
                # Use provided initial guess, ensure it's detached
                u_i = u0[i:i+1].detach().clone()

            # Apply boundary conditions to initial guess
            # Important: BC values should match the dtype
            u_i = self.bc_manager.apply(u_i)
            u_i.requires_grad_(True) # Enable gradient tracking AFTER BCs

            # Prepare mask for fixed DOFs (needs to match u_i's batch dim)
            fixed_dofs_mask = torch.zeros_like(u_i, dtype=torch.bool)
            if len(self.bc_manager.fixed_dofs) > 0:
                fixed_dofs_mask[:, self.bc_manager.fixed_dofs] = True

            # Newton iteration tracking
            iter_count = 0
            converged = False

            # For line search
            alpha_min = 0.1 # Allow smaller steps if needed
            alpha_max = 1.0

            # Initial residual computation for convergence check
            # Need try-except for potential initial state issues
            try:
                with torch.enable_grad(): # Ensure grad is enabled for first calculation
                    strain_energy = self.energy_model.compute_energy(u_i)
                    internal_forces = self.energy_model.compute_gradient(u_i)
                external_work = torch.sum(f_i * u_i, dim=1) # Work uses current u_i
                residual = f_i - internal_forces
                filtered_residual = residual * (~fixed_dofs_mask)
                residual_norm = torch.linalg.norm(filtered_residual) # Use linalg.norm
            except Exception as e:
                 print(f"Error during initial residual calculation for sample {i}: {e}")
                 residual_norm = torch.tensor(float('inf'), device=self.device) # Indicate failure
                 converged = False # Cannot start if initial state fails

            # Initial logging
            if self.verbose:
                energy = strain_energy - external_work if 'strain_energy' in locals() else torch.tensor(float('nan'))
                print(f"--- Sample {i+1}/{batch_size} ---")
                print(f"Initial state: residual={residual_norm.item():.2e}, energy={energy.item():.2e}")

            # Check if initial residual is already converged (or failed)
            if residual_norm < self.tolerance:
                converged = True
                if self.verbose: print(f"Already converged at iteration 0.")
            elif torch.isinf(residual_norm) or torch.isnan(residual_norm):
                converged = False # Failed initial state
                print(f"Warning: Failed to compute initial state for sample {i}. Skipping Newton iterations.")


            # Main Newton iteration loop
            while iter_count < self.max_iterations and not converged:
                iter_count += 1 # Increment at the start
                print(f"--- Iteration {iter_count} ---")

                # --- Try block for safety during iteration ---
                try:
                    # 1. Compute the gradient (residual) - Ensure requires_grad is True
                    u_i.requires_grad_(True)
                    # Recompute energy and forces for the *current* u_i
                    strain_energy = self.energy_model.compute_energy(u_i)
                    internal_forces = self.energy_model.compute_gradient(u_i) # grad of strain energy w.r.t u_i
                    external_work = torch.sum(f_i * u_i, dim=1) # External work depends on u_i

                    # Compute residual: R = f_ext - f_int
                    residual = f_i - internal_forces

                    # Zero out residual at fixed DOFs
                    filtered_residual = residual * (~fixed_dofs_mask)

                    # Compute residual norm for convergence check
                    residual_norm = torch.linalg.norm(filtered_residual)

                    # Check convergence BEFORE solving the system
                    if residual_norm < self.tolerance:
                        converged = True
                        if self.verbose:
                            print(f"Converged at iteration {iter_count}, residual={residual_norm.item():.2e}")
                        # Visualize final state before breaking
                        if self.visualize:
                           energy = strain_energy - external_work
                           self._update_visualization(
                               u_i, f_i, gt_disp_i, # Pass GT disp
                               i, iter_count,
                               residual_norm.item(), energy.item(), 0.0,
                               strain_energy.item(), external_work.item()
                           )
                        break

                    # If not converged and not exceeding max_iter yet
                    if iter_count > self.max_iterations:
                        break # Exit loop if max iterations reached

                    # 2. Compute the tangent stiffness matrix (Hessian) via Hessian-vector products
                    # Need to define HVP based on internal_forces (gradient) w.r.t u_i
                    def hessian_vector_product(v):
                        """Compute Hessian-vector product: (d(f_int)/du) @ v"""
                        # Ensure v doesn't require grad if it comes from CG
                        v_detached = v.detach()
                        # Compute gradient of (internal_forces ⋅ v) w.r.t u_i
                        # This requires internal_forces to be computed based on u_i requiring grad
                        grad_outputs_dot_v, = torch.autograd.grad(
                            internal_forces, u_i,
                            grad_outputs=v_detached,
                            retain_graph=True, # Keep graph for potential line search re-evals
                            create_graph=False # Don't need grad of Hessian itself
                        )
                        # Apply mask: HVP should be zero for fixed DOFs
                        grad_outputs_dot_v = grad_outputs_dot_v * (~fixed_dofs_mask)
                        return grad_outputs_dot_v

                    # 3. Solve for displacement update using Conjugate Gradient: K * delta_u = -residual
                    delta_u = self._solve_newton_system(
                        hessian_vector_product, filtered_residual, 
                        fixed_dofs_mask, max_iter=200, tol=1e-5
                    )
                    print(f"delta_u mean: {delta_u.mean().item():.2e}, std: {delta_u.std().item():.2e}")
                    # delta_u already has fixed DOFs zeroed out by _solve_newton_system

                    # 4. Line search for step size alpha
                    alpha = self._line_search(
                        u_i, delta_u, f_i, internal_forces, # Pass current int_forces for efficiency
                        fixed_dofs_mask,
                        alpha_min=alpha_min, alpha_max=alpha_max,
                        max_trials=50
                    )
                    print(f"Line search alpha_min: {alpha_min:.2e}, alpha_max: {alpha_max:.2e}, alpha: {alpha:.2e}")

                    # 5. Update displacement (use torch.no_grad for efficiency)
                    with torch.no_grad():
                        u_i += alpha * delta_u # In-place update might save memory if needed u_i.add_(delta_u, alpha=alpha)
                        # Re-apply boundary conditions strictly after update
                        u_i = self.bc_manager.apply(u_i)

                    # Log progress
                    if self.verbose:
                        energy = strain_energy - external_work
                        print(f"Iter {iter_count}: residual={residual_norm.item():.2e}, "
                              f"energy={energy.item():.2e}, alpha={alpha:.3f}")

                    # Visualize if enabled
                    if self.visualize and iter_count % 1 == 0: # Visualize every iteration
                        energy = strain_energy - external_work
                        self._update_visualization(
                            u_i, f_i, gt_disp_i, 
                            i, iter_count,
                            residual_norm.item(), energy.item(), 0.0, # energy_ratio placeholder
                            strain_energy.item(), external_work.item()
                        )

                # --- End of Try block ---
                except Exception as e:
                    print(f"\nError during Newton iteration {iter_count} for sample {i}: {e}")
                    import traceback
                    print(traceback.format_exc())
                    print(f"Stopping iterations for sample {i}.")
                    # Store the state *before* the error if possible
                    solutions.append(u_i.detach().clone()) # Store last known good state
                    # Skip to next sample in the outer loop
                    # Need a way to signal failure for this sample if desired
                    converged = False # Mark as not converged due to error
                    break # Break inner while loop

            # End of Newton while loop

            # Final state logging
            if not converged and iter_count >= self.max_iterations:
                if self.verbose:
                    print(f"Warning: Newton solver did not converge for sample {i} in {self.max_iterations} iterations. Final residual={residual_norm.item():.2e}")
            elif converged and self.verbose:
                 print(f"--- Sample {i+1} Converged ---")


            # Store solution (ensure it's detached)
            solutions.append(u_i.detach().clone())

        # End of batch loop

        # Stack solutions back into batch
        if not solutions: # Handle case where batch size was 0 or all failed early
             return torch.empty((0, self.dof_count), device=self.device, dtype=self.dtype)
        return torch.cat(solutions, dim=0)

    def _solve_newton_system(self, hessian_vector_product, residual, fixed_dofs_mask, max_iter=200, tol=1e-5):
        """
        Solve the Newton system K(u)·Δu = -R(u) using Conjugate Gradient.
        Handles fixed DOFs correctly.

        Args:
            hessian_vector_product: Function that computes H·v (where H = d(f_int)/du)
            residual: Current residual vector R = f_ext - f_int [1, N*D]
            fixed_dofs_mask: Mask for fixed DOFs [1, N*D]
            max_iter: Maximum CG iterations
            tol: Relative convergence tolerance for CG (||r_k|| < tol * ||r_0||)

        Returns:
            Displacement update vector Δu [1, N*D]
        """
        # System to solve: K * x = b, where x = delta_u, b = -residual
        x = torch.zeros_like(residual)
        b = -residual # Right hand side

        # Apply BCs to RHS: force corresponding to fixed DOFs is irrelevant
        b = b * (~fixed_dofs_mask)

        # Initial residual for CG: r = b - K*x (since x=0 initially)
        r = b.clone()
        p = r.clone() # Initial search direction

        rsold_tensor = torch.sum(r * r) # Squared norm of initial residual - Keep as tensor for beta calc
        rsinit = rsold_tensor.sqrt().item() # Initial residual norm for relative tolerance check

        if rsinit < 1e-15: # Already solved (or zero residual)
            return x

        best_x = x.clone()
        min_residual_norm = rsinit

        for i in range(max_iter):
            # Compute Ap = K * p
            Ap = hessian_vector_product(p)
            # Ensure Ap respects BCs (although HVP should already do this)
            Ap = Ap * (~fixed_dofs_mask)

            pAp_tensor = torch.sum(p * Ap)

            # Check for breakdown (negative curvature or zero direction)
            # Use a small tolerance relative to p norm squared
            p_norm_sq = torch.sum(p*p)
            if pAp_tensor <= 1e-12 * p_norm_sq : # Check relative value
                if self.verbose:
                    print(f"CG breakdown: pAp = {pAp_tensor.item():.2e} vs p^2 = {p_norm_sq.item():.2e} at iter {i}. Using previous best solution.")
                # Return the best solution found so far if breakdown occurs
                return best_x * (~fixed_dofs_mask) # Ensure BCs applied

            alpha_tensor = rsold_tensor / pAp_tensor # alpha remains tensor for now
            alpha_scalar = alpha_tensor.item() # Extract scalar for updates

            # Update solution and residual using the scalar alpha
            x.add_(p, alpha=alpha_scalar)      # x = x + alpha * p
            r.add_(Ap, alpha=-alpha_scalar)   # r = r - alpha * Ap

            rsnew_tensor = torch.sum(r * r)
            current_residual_norm = rsnew_tensor.sqrt().item()

            # Store the best solution found so far based on residual norm
            if current_residual_norm < min_residual_norm:
                min_residual_norm = current_residual_norm
                best_x = x.clone()

            # Check convergence: ||r_k|| / ||r_0|| < tol
            if current_residual_norm < tol * rsinit:
                break

            # Update search direction (using tensors for beta)
            beta_tensor = rsnew_tensor / rsold_tensor
            # p = r + beta * p
            p.mul_(beta_tensor).add_(r) # More efficient in-place: p = beta*p + r

            rsold_tensor = rsnew_tensor
        else: # Loop finished without break (max_iter reached)
            if self.verbose:
                print(f"CG warning: Max iterations ({max_iter}) reached. Final rel residual: {current_residual_norm / rsinit:.2e}")
            # Return the best solution found during iterations
            x = best_x


        # Ensure final solution strictly adheres to BCs
        x = x * (~fixed_dofs_mask)
        return x

    def _line_search(self, u, delta_u, f_ext, f_int_current, fixed_dofs_mask, # Pass f_int_current
                     alpha_min=0.01, alpha_max=1.0, max_trials=10, c1=1e-4):
        """
        Backtracking line search using Armijo condition (sufficient decrease).
        Finds alpha such that E(u + alpha*delta_u) <= E(u) + c1*alpha*(gradE(u) ⋅ delta_u)
        where gradE = -residual = f_int - f_ext

        Args:
            u: Current displacement [1, N*D]
            delta_u: Computed update direction [1, N*D]
            f_ext: External forces [1, N*D]
            f_int_current: Internal forces at current u [1, N*D]
            fixed_dofs_mask: Mask for fixed DOFs [1, N*D]
            alpha_min: Minimum step size
            alpha_max: Initial (maximum) step size
            max_trials: Maximum number of step size reductions
            c1: Parameter for Armijo condition (typically 1e-4)

        Returns:
            Step size alpha
        """
        alpha = alpha_max
        u = u.detach() # Ensure no grad tracking during line search calculations
        delta_u = delta_u.detach()

        with torch.no_grad(): # All calculations within line search should not track gradients
            # Current energy E(u)
            energy_current = self.energy_model.compute_energy(u) - torch.sum(f_ext * u)

            # Gradient of energy gradE = f_int - f_ext = -residual
            grad_energy = f_int_current - f_ext
            # Directional derivative: gradE ⋅ delta_u
            # Only consider non-fixed DOFs for the descent direction check
            descent_dot_product = torch.sum((grad_energy * (~fixed_dofs_mask)) * (delta_u * (~fixed_dofs_mask)))

            # Expect descent_dot_product to be negative if delta_u is a descent direction
            if descent_dot_product >= 0:
                 if self.verbose:
                     print(f"Line search warning: delta_u is not a descent direction (dot product = {descent_dot_product.item():.2e}). Using alpha_min.")
                 return alpha_min

            for _ in range(max_trials):
                # Trial displacement
                u_trial = u + alpha * delta_u
                # Strictly enforce BCs on trial point
                u_trial = self.bc_manager.apply(u_trial)

                # Energy at trial point E(u_trial)
                energy_trial = self.energy_model.compute_energy(u_trial) - torch.sum(f_ext * u_trial)

                # Armijo condition: E(u_trial) <= E(u) + c1 * alpha * (gradE ⋅ delta_u)
                if energy_trial <= energy_current + c1 * alpha * descent_dot_product:
                    # Sufficient decrease achieved
                    # print(f"LS found alpha={alpha:.4f}")
                    return alpha

                # Reduce step size (backtracking)
                alpha *= 0.5
                if alpha < alpha_min:
                    if self.verbose:
                         print(f"Line search hit alpha_min ({alpha_min}) after {max_trials} trials.")
                    return alpha_min

        # If loop finishes, max_trials reached without satisfying condition
        if self.verbose:
            print(f"Line search failed to satisfy Armijo after {max_trials} trials. Using alpha_min ({alpha_min}).")
        return alpha_min

    def _setup_visualization(self, filename):
        """Set up visualization using PyVista."""
        # --- Requires dolfinx and gmshio ---
        from dolfinx import plot
        from dolfinx.io import gmshio
        # ---
        if MPI.COMM_WORLD.rank == 0: # Only rank 0 reads mesh and sets up plotter
            print(f"Setting up visualization from mesh: {filename}")
            try:
                # Read mesh on rank 0
                domain, _, _ = gmshio.read_from_msh(filename, MPI.COMM_SELF, rank=0, gdim=3) # Use COMM_SELF for local read
                topology, cell_types, x = plot.vtk_mesh(domain)
                self.viz_grid = pyvista.UnstructuredGrid(topology, cell_types, x.copy()) # Use copy
                self.viz_points_cpu = x.copy() # Cache original points

                # Create plotter
                self.viz_plotter = pyvista.Plotter(shape=(1, 2),
                                                  title="Newton FEM Solver",
                                                  window_size=[1600, 800], # Increased size
                                                  off_screen=False) # Ensure interactive window

                # Left viewport - forces
                self.viz_plotter.subplot(0, 0)
                self.viz_plotter.add_text("Applied Forces / Initial", position="upper_edge", font_size=10)
                self.mesh_actor_left = self.viz_plotter.add_mesh(self.viz_grid, color='lightgrey', show_edges=True, opacity=0.5)
                # Add placeholder for force glyphs if needed later
                self.force_glyphs_actor = None

                # Right viewport - deformed shape
                self.viz_plotter.subplot(0, 1)
                self.viz_plotter.add_text("Deformed Configuration", position="upper_edge", font_size=10)
                self.mesh_actor_right = self.viz_plotter.add_mesh(self.viz_grid.copy(), color='lightblue', show_edges=True) # Plot copy
                self.gt_mesh_actor = None # Initialize ground truth actor reference

                # Add info text placeholders (will be updated)
                self.info_actor_left = self.viz_plotter.add_text("Left Info", position="lower_left", font_size=9, shadow=True)
                self.info_actor_right = self.viz_plotter.add_text("Right Info", position="lower_left", font_size=9, shadow=True)

                # Link camera views
                self.viz_plotter.link_views()
                # Set initial camera position (example)
                self.viz_plotter.camera_position = 'iso'
                self.viz_plotter.camera.zoom(1.2)

                # Show window interactively (does not block if interactive=True used later)
                self.viz_plotter.show(interactive_update=True, auto_close=False) # Use interactive_update

                print("Visualization setup complete on rank 0.")

            except Exception as e:
                print(f"Error during visualization setup: {e}")
                self.visualize = False
                self.viz_plotter = None
        else:
             # Other ranks don't set up visualization
             self.viz_plotter = None
             self.visualize = False # Ensure visualize is False on non-root ranks


    def _update_visualization(self, u_i, f_i, gt_disp_i, 
                        i, iter_count, residual_norm, energy, energy_ratio,
                        strain_energy=None, external_work=None):
        """Update visualization with current solver state."""
        # Only rank 0 performs visualization updates
        if MPI.COMM_WORLD.rank != 0 or not self.visualize or self.viz_plotter is None:
            return

        try:
            # --- Data Preparation ---
            with torch.no_grad():
                u_cpu = u_i[0].detach().cpu().numpy() # Get data for the single sample
                f_cpu = f_i[0].detach().cpu().numpy()
                if gt_disp_i is not None:
                    gt_disp_cpu = gt_disp_i[0].detach().cpu().numpy()
                else:
                    gt_disp_cpu = None

                u_array = u_cpu.reshape(-1, self.dim)
                f_array = f_cpu.reshape(-1, self.dim)
                if gt_disp_cpu is not None:
                    gt_disp_array = gt_disp_cpu.reshape(-1, self.dim)

            # --- Left Viewport (Forces) ---
            self.viz_plotter.subplot(0, 0)
            force_grid = self.viz_grid.copy() # Work on copies
            force_grid.point_data["Forces"] = f_array
            force_mag = np.linalg.norm(f_array, axis=1)
            force_grid["force_magnitude"] = force_mag

            # Update mesh color based on force magnitude (optional)
            self.viz_plotter.remove_actor(self.mesh_actor_left)
            self.mesh_actor_left = self.viz_plotter.add_mesh(
                force_grid,
                scalars="force_magnitude",
                cmap="plasma",
                show_edges=True,
                clim=[0, np.max(force_mag) if np.max(force_mag) > 0 else 1.0]
            )

            # Update or add force glyphs (optional)
            # if self.force_glyphs_actor: self.viz_plotter.remove_actor(self.force_glyphs_actor)
            # self.force_glyphs_actor = self.viz_plotter.add_arrows(force_grid.points, f_array, mag=0.1)

            # Update left info text
            left_text = (f"Sample: {i+1}\n"
                         f"Max Force Mag: {force_mag.max():.3e}")
            if strain_energy is not None:
                 left_text += f"\nStrain Energy: {strain_energy:.3e}"
            if external_work is not None:
                 left_text += f"\nExternal Work: {external_work:.3e}"
            
            # Remove old text actor if it exists
            if hasattr(self, 'info_actor_left'):
                self.viz_plotter.remove_actor(self.info_actor_left)
            # Add the updated text actor
            self.info_actor_left = self.viz_plotter.add_text(left_text, position="lower_left", font_size=9, shadow=True)

            # --- Right Viewport (Deformation) ---
            self.viz_plotter.subplot(0, 1)
            # Update deformed grid (current solution)
            deformed_grid = self.viz_grid.copy()
            deformed_grid.points = self.viz_points_cpu + u_array
            displacement_magnitude = np.linalg.norm(u_array, axis=1)
            deformed_grid["displacement"] = displacement_magnitude

            # Update the main deformed mesh actor
            self.viz_plotter.remove_actor(self.mesh_actor_right)
            self.mesh_actor_right = self.viz_plotter.add_mesh(
                deformed_grid,
                scalars="displacement",
                cmap="viridis",
                show_edges=True,
                clim=[0, np.max(displacement_magnitude) if np.max(displacement_magnitude) > 0 else 1.0]
            )


            # Update or add Ground Truth Wireframe
            if hasattr(self, 'gt_mesh_actor') and self.gt_mesh_actor is not None:
                self.viz_plotter.remove_actor(self.gt_mesh_actor, render=False) # Remove previous GT actor without rendering yet
                self.gt_mesh_actor = None # Clear reference

            if gt_disp_array is not None:
                gt_grid = self.viz_grid.copy()
                gt_grid.points = self.viz_points_cpu + gt_disp_array
                # Add GT wireframe
                self.gt_mesh_actor = self.viz_plotter.add_mesh(
                    gt_grid,
                    style="wireframe",
                    color='lime', # Use a bright color
                    opacity=0.7,
                    line_width=2
                )

            # Update right info text
            right_text = (f"Newton Iter: {iter_count}\n"
                         f"Residual: {residual_norm:.3e}\n"
                         f"Total Energy: {energy:.3e}\n"
                         f"Max Disp: {displacement_magnitude.max():.3e}")
            # Remove old text actor if it exists
            if hasattr(self, 'info_actor_right'):
                self.viz_plotter.remove_actor(self.info_actor_right)
            # Add the updated text actor
            self.info_actor_right = self.viz_plotter.add_text(right_text, position="lower_left", font_size=9, shadow=True)


            # --- Render ---
            # self.viz_plotter.render() # Render updates
            self.viz_plotter.update(1) # Process events briefly
            # Or use app.processEvents() if using Qt backend

        except Exception as e:
            # Catch errors during visualization update to prevent crashing the solver
            print(f"\nVisualization update error (iter {iter_count}, sample {i}): {str(e)}")
            # import traceback
            # print(traceback.format_exc())
            # Optionally disable further visualization attempts for this run
            # self.visualize = False


    def close_visualization(self):
        """Close the visualization window."""
        if MPI.COMM_WORLD.rank == 0 and hasattr(self, 'viz_plotter') and self.viz_plotter is not None:
            print("Closing visualization window...")
            self.viz_plotter.close()
            self.viz_plotter = None
            print("Visualization closed.")


# ==============================================================================
# Custom Autograd Function for Implicit Differentiation (REVISED)
# ==============================================================================

class FEMSolverFunction(torch.autograd.Function):
    """
    Custom PyTorch autograd Function for the FEM Solver using the Implicit Function Theorem (IFT).

    Solves the non-linear system R(u, f_ext) = f_int(u) - f_ext = 0 for u*(f_ext).
    The backward pass computes dL/df_ext = [dR/du |_(u*, f_ext)]^{-T} * dL/du*
    where dR/du = d(f_int)/du = K (tangent stiffness matrix).
    Assuming K is symmetric (K=K^T), we solve K*v = dL/du* and the result is v = dL/df_ext.
    """

    @staticmethod
    def forward(ctx, f_ext, disp_gt_subset, solver_instance):
        """
        Executes the forward pass: solves the non-linear FEM system R(u, f_ext) = 0.
        Args:
            f_ext: External forces tensor (batch_size, num_dofs).
            disp_gt_subset: Ground truth displacements for visualization (optional).
            solver_instance: Instance of the DifferentiableFEMSolverIFT class.
        Returns:
            u_star: Converged displacement tensor (batch_size, num_dofs).
        """
        if not f_ext.requires_grad:
            # If input doesn't require grad, skip autograd tracking
            with torch.no_grad():
                 f_ext_detached = f_ext.clone()
                 u_star = solver_instance._solve_with_newton_internal(f_ext_detached, u0=None, disp_gt_subset_for_viz=disp_gt_subset)
            return u_star # No need to save context if no grad needed
        else:
            # Detach f_ext for the internal solver (solver doesn't need graph from f_ext -> u)
            f_ext_detached = f_ext.detach().clone()
            # Run the internal solver to find the equilibrium displacement u*
            u_star = solver_instance._solve_with_newton_internal(f_ext_detached, u0=None, disp_gt_subset_for_viz=disp_gt_subset)

            # Save necessary items for backward pass
            ctx.solver_instance = solver_instance
            # Save the *solution* u_star and the *input* f_ext that produced it
            ctx.save_for_backward(u_star, f_ext_detached)
            # Return the solution u_star, maintaining its place in the overall computation graph
            return u_star

    @staticmethod
    def backward(ctx, grad_output):
        """
        Executes the backward pass using the Implicit Function Theorem (Adjoint Method).
        Computes gradient dL/df_ext.

        Args:
            grad_output: Gradient of the loss L w.r.t. the output of forward (dL/du*).
                         Shape: (batch_size, num_dofs).
        Returns:
            grad_f_ext: Gradient of the loss L w.r.t. the input f_ext (dL/df_ext).
                        Shape: (batch_size, num_dofs).
            None: Gradient for disp_gt_subset (not differentiable input).
            None: Gradient for solver_instance (not differentiable input).
        """
        solver_instance = ctx.solver_instance
        # u_star_saved is the converged displacement from forward.
        # f_ext_detached is the external force input used in forward.
        u_star_saved, f_ext_detached = ctx.saved_tensors

        if grad_output is None:
            # If no gradient flows back, no need to compute anything
            return None, None, None

        batch_size = grad_output.shape[0]
        adjoint_solutions = [] # To store dL/df_ext for each batch item

        # Prepare fixed DOFs mask (potentially different per batch item if needed,
        # but usually the same for a given problem setup)
        fixed_dofs_mask = torch.zeros_like(u_star_saved, dtype=torch.bool)
        if len(solver_instance.bc_manager.fixed_dofs) > 0:
            fixed_dofs_indices = solver_instance.bc_manager.fixed_dofs
            # Ensure mask applies correctly even if fixed_dofs are flat indices
            mask_view = fixed_dofs_mask.view(batch_size, -1)
            mask_view[:, fixed_dofs_indices] = True


        # --- HVP Function Definition ---
        # This function computes K(u*) @ v, where K = d(f_int)/du evaluated at u*.
        # It's needed for the CG solver to represent the action of K.
        def hvp_at_ustar(v, u_star_current_batch):
            # Ensure v doesn't require gradients for this internal HVP computation
            v_detached = v.detach()
            # We need to compute the gradient of f_int w.r.t. u, evaluated at u*.
            # Use torch.enable_grad and requires_grad_(True) temporarily.
            with torch.enable_grad():
                # Create a fresh input tensor for gradient calculation
                u_star_input = u_star_current_batch.detach().clone().requires_grad_(True)
                # Compute internal forces using the energy model (inside grad context)
                f_int_for_hvp = solver_instance.energy_model.compute_gradient(u_star_input)

            # Compute the HVP: (d(f_int)/du) @ v using autograd.grad
            # This calculates the gradient of sum(f_int_for_hvp * v_detached) w.r.t u_star_input
            hvp_result, = torch.autograd.grad(
                outputs=f_int_for_hvp,     # The vector function f_int(u)
                inputs=u_star_input,       # Differentiate w.r.t this u
                grad_outputs=v_detached,   # The vector 'v' to multiply by the Jacobian K
                retain_graph=False,        # No need to keep graph for HVP computation itself
                create_graph=False         # Not computing higher-order derivatives here
            )
            # Apply boundary conditions (zero out contributions related to fixed DOFs)
            # Assuming fixed_dofs_mask is (batch, dofs) for this HVP call operating on a batch slice
            hvp_result = hvp_result * (~fixed_dofs_mask[0:hvp_result.shape[0]]) # Mask matching current batch slice size
            return hvp_result

        # --- Solve Adjoint System for each batch element ---
        # We need to solve K(u*) @ v = grad_output, where v = dL/df_ext.
        # Assuming K = K^T, this is the system solved by CG below.
        # If K != K^T, we should be solving K^T @ v = grad_output.
        for i in range(batch_size):
            # Incoming gradient for this batch item
            grad_output_i = grad_output[i:i+1]
            # Fixed DOFs mask for this item
            fixed_dofs_mask_i = fixed_dofs_mask[i:i+1]
            # Converged displacement for this item
            u_star_i = u_star_saved[i:i+1]

            # Create the specific HVP function for this sample's u*
            # This lambda captures the current u_star_i
            hvp_func_for_cg = lambda vec: hvp_at_ustar(vec, u_star_i)

            # Solve the linear system K @ v = grad_output_i using Conjugate Gradient
            # The solution 'v' is the adjoint variable, which equals dL/df_ext
            v = solver_instance._solve_linear_system_cg(
                hvp_function=hvp_func_for_cg,   # Function K(u*) @ v
                rhs=grad_output_i,              # Right-hand side dL/du*
                fixed_dofs_mask=fixed_dofs_mask_i, # Mask for BCs
                max_iter=solver_instance.cg_max_iter_backward,
                tol=solver_instance.cg_tol_backward
            )
            adjoint_solutions.append(v)

        # Concatenate gradients from all batch elements
        grad_f_ext = torch.cat(adjoint_solutions, dim=0)

        # Return gradient w.r.t. f_ext, and None for other inputs
        return grad_f_ext, None, None


# ==============================================================================
# Differentiable Solver Class (REVISED)
# ==============================================================================

class DifferentiableFEMSolverIFT(torch.nn.Module):
    def __init__(self, energy_model, max_iterations=20, tolerance=1e-6,
                 cg_max_iter=200, cg_tol=1e-5,
                 cg_max_iter_backward=300, cg_tol_backward=1e-6,
                 verbose=False, line_search_params=None,
                 visualize=False, # Flag to ENABLE visualization feature
                 filename=None):  # Filename REQUIRED if visualize=True
        super().__init__()

        # --- Store Configuration & Parameters ---
        self.energy_model = energy_model
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.device = energy_model.device
        self.dtype = energy_model.dtype
        if not all(hasattr(energy_model, attr) for attr in ['num_nodes', 'dim', 'device', 'dtype', 'compute_gradient', 'compute_energy']):
             raise AttributeError("energy_model must have 'num_nodes', 'dim', 'device', 'dtype', 'compute_gradient', 'compute_energy' attributes/methods.")
        self.num_nodes = energy_model.num_nodes
        self.dim = energy_model.dim
        self.dof_count = self.num_nodes * self.dim
        # Ensure BC Manager uses the same device
        self.bc_manager = SmoothBoundaryConditionManager(device=self.device)
        self.cg_max_iter = cg_max_iter
        self.cg_tol = cg_tol
        self.cg_max_iter_backward = cg_max_iter_backward
        self.cg_tol_backward = cg_tol_backward
        self.line_search_params = line_search_params if line_search_params is not None else {}
        self.ls_alpha_min = self.line_search_params.get('alpha_min', 0.01)
        self.ls_alpha_max = self.line_search_params.get('alpha_max', 1.0)
        self.ls_max_trials = self.line_search_params.get('max_trials', 10)
        self.ls_c1 = self.line_search_params.get('c1', 1e-4) # Armijo condition constant

        # --- Visualization Attributes Initialization ---
        self.visualize_active = False # Flag to track if plotter is currently active
        self.viz_plotter = None
        self.viz_grid = None
        self.viz_points_cpu = None
        self.mesh_actor_left = None
        self.mesh_actor_right = None
        self.gt_mesh_actor = None
        self.info_actor_left = None
        self.info_actor_right = None
        self._should_visualize = visualize # Store intent

        # --- Setup Visualization ONCE during initialization if requested ---
        # Setup only on rank 0 if running with MPI
        self._setup_visualization(filename)


    def _setup_visualization(self, filename):
        """Handles visualization setup logic."""
        # Determine if eligible for visualization (rank 0 if MPI is used, or always if not)
        is_rank_zero = True
        try:
             if MPI.COMM_WORLD.size > 1:
                  is_rank_zero = (MPI.COMM_WORLD.rank == 0)
        except ImportError:
             pass # MPI not available, assume single process

        if self._should_visualize and is_rank_zero:
            if filename is None:
                print("Warning: Visualization requested but no filename provided. Visualization disabled.")
                self._should_visualize = False
                return

            try:
                # Try importing necessary libraries here
                import pyvista
                from dolfinx import plot
                from dolfinx.io import gmshio
            except ImportError as e:
                print(f"Warning: Missing library for visualization ({e}). Visualization disabled.")
                self._should_visualize = False
                return

            # Attempt setup (now imports are confirmed possible)
            try:
                self._setup_visualization_internal(filename) # Call the actual setup
                if self.viz_plotter is not None: # Check if plotter was created
                     self.visualize_active = True
                     print("Visualization successfully initialized.")
                else:
                     # Setup failed despite filename and imports
                     self._should_visualize = False
                     self.visualize_active = False # Ensure flag consistency
            except Exception as e:
                print(f"Error during visualization setup: {e}. Visualization disabled.")
                self._should_visualize = False
                self.visualize_active = False # Ensure flag consistency
                # import traceback; traceback.print_exc() # Uncomment for detailed debug
        else:
            # Ensure flags are False if not rank 0 or not requested initially
             self._should_visualize = False
             self.visualize_active = False


    def forward(self, external_forces, disp_gt_subset=None):
        """
        Performs the differentiable forward solve using the custom autograd Function.
        Args:
            external_forces (torch.Tensor): Tensor of external forces (batch_size, num_dofs).
                                            Requires grad if differentiation is needed.
            disp_gt_subset (torch.Tensor, optional): Ground truth displacements for viz.
        Returns:
            torch.Tensor: Converged displacements (batch_size, num_dofs).
        """
        # Pass GT disp for potential use in internal visualization during solve
        return FEMSolverFunction.apply(external_forces, disp_gt_subset, self)

    def set_fixed_dofs(self, indices, values):
        """Sets the fixed degrees of freedom and their values."""
        # Convert inputs to tensors on the correct device and dtype
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        else:
            indices = indices.to(device=self.device, dtype=torch.long)

        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, device=self.device, dtype=self.dtype)
        else:
            # Ensure value tensor is on the right device and dtype
            values = values.to(device=self.device, dtype=self.dtype)

        self.bc_manager.set_fixed_dofs(indices, values)

    # ==========================================================================
    # Internal Helper Methods (Used by FEMSolverFunction's Forward Pass)
    # ==========================================================================

    def _compute_internal_force_robustly(self, u_input):
        """
        Helper to compute internal force, handling potential grad requirements
        within the energy_model's compute_gradient method.
        Returns a detached force tensor.
        """
        try:
            # Try computing normally first (most efficient if it works)
            with torch.no_grad():
                f_int = self.energy_model.compute_gradient(u_input.detach())
            return f_int.detach() # Ensure detached output
        except RuntimeError as e:
            # Check if the error is the specific "requires grad" issue
            if "does not require grad and does not have a grad_fn" in str(e):
                if self.verbose: print("Verbose: Re-computing f_int with temporary grad context.")
                # If normal computation fails with the grad error, retry with temporary grad context
                with torch.enable_grad():
                    u_temp_grad = u_input.detach().clone().requires_grad_(True)
                    f_int_maybe_grad = self.energy_model.compute_gradient(u_temp_grad)
                return f_int_maybe_grad.detach() # Return detached result
            else:
                # If it's a different error, re-raise it
                raise e
        except Exception as e: # Catch other potential exceptions
             print(f"Warning: Unexpected error in _compute_internal_force_robustly: {e}")
             # Depending on desired robustness, you might return zeros or re-raise
             # Returning zeros might mask underlying issues in energy_model
             # raise e # Re-raising is safer to expose model issues
             return torch.zeros_like(u_input) # Or return zeros as a fallback





    def _solve_with_newton_internal(self, external_forces_detached, u0=None, disp_gt_subset_for_viz=None):
        """
        Internal non-linear solver using Newton's method. Operates on detached tensors.
        This method itself does NOT track gradients for autograd; that's handled by FEMSolverFunction.
        Includes calls to internal visualization update if active.
        Args:
            external_forces_detached (torch.Tensor): Detached external forces (batch, dofs).
            u0 (torch.Tensor, optional): Initial guess for displacement (batch, dofs).
            disp_gt_subset_for_viz (torch.Tensor, optional): GT displacements for viz.
        Returns:
            torch.Tensor: Converged displacement solution (batch, dofs).
        """
        batch_size = external_forces_detached.shape[0]
        solutions = [] # Store solution for each batch item

        # --- Process each sample in the batch ---
        for i in range(batch_size):
            if self.verbose and batch_size > 1: print(f"\n--- Sample {i+1}/{batch_size} ---")
            f_i = external_forces_detached[i:i+1]
            gt_disp_i_for_viz = None
            if disp_gt_subset_for_viz is not None and disp_gt_subset_for_viz.shape[0] > i:
                gt_disp_i_for_viz = disp_gt_subset_for_viz[i:i+1]

            # --- Initialize displacement (without requires_grad) ---
            if u0 is None:
                u_i = torch.randn_like(f_i) * 1e-5
            else:
                u_i = u0[i:i+1].detach().clone()

            u_i = self.bc_manager.apply(u_i)

            # --- Prepare fixed DOFs mask ---
            fixed_dofs_mask = torch.zeros_like(u_i, dtype=torch.bool)
            if self.bc_manager.fixed_dofs.numel() > 0:
                 mask_view = fixed_dofs_mask.view(1, -1)
                 mask_view[:, self.bc_manager.fixed_dofs] = True

            # --- Initial Residual Calculation (FIXED) ---
            iter_count = 0
            converged = False
            residual_norm = torch.tensor(float('inf'), device=self.device)
            residual = torch.zeros_like(u_i) # Initialize residual

            try:
                # Use the robust helper function to compute initial internal force
                f_int_i = self._compute_internal_force_robustly(u_i)

                # Compute residual (outside gradient tracking)
                with torch.no_grad():
                    residual = f_i - f_int_i
                    filtered_residual = residual * (~fixed_dofs_mask)
                    residual_norm = torch.linalg.norm(filtered_residual)

            except Exception as e:
                # Catch errors from _compute_internal_force_robustly or subsequent ops
                print(f"Sample {i}: Error during initial residual calculation: {e}")
                solutions.append(torch.zeros_like(f_i))
                continue

            if self.verbose: print(f"Sample {i}: Initial residual norm = {residual_norm.item():.3e}")

            # Check for immediate convergence or divergence
            if residual_norm < self.tolerance:
                if self.verbose: print(f"Sample {i}: Converged at iteration 0.")
                converged = True
            elif torch.isinf(residual_norm) or torch.isnan(residual_norm):
                print(f"Sample {i}: Diverged at iteration 0 (Residual is Inf/NaN).")
                solutions.append(torch.zeros_like(f_i))
                continue

            # --- Visualization of Initial State ---
            if self.visualize_active and i == 0:
                 with torch.no_grad():
                     # Use try-except for energy computation as well, just in case
                     try:
                         strain_e = self.energy_model.compute_energy(u_i).item()
                         ext_w = torch.sum(f_i * u_i).item()
                         total_e = strain_e - ext_w
                         self._update_visualization_internal(u_i, f_i, gt_disp_i_for_viz, i, iter_count, residual_norm.item(), total_e, 0.0, strain_e, ext_w)
                     except Exception as viz_e:
                          print(f"Warning: Could not compute energy for initial visualization: {viz_e}")


            u_i_last_successful = u_i.clone()

            # --- Newton Iteration Loop ---
            while iter_count < self.max_iterations and not converged:
                iter_count += 1
                if self.verbose: print(f"--- Sample {i}, Iter {iter_count} ---")

                try:
                    # --- Step 1: Define HVP Function (using current u_i) ---
                    u_i_for_hvp = u_i # Capture current u_i

                    def hvp_forward(v):
                        # (Keep the HVP implementation from the previous corrected version)
                        v_detached = v.detach()
                        with torch.enable_grad():
                            u_i_input = u_i_for_hvp.detach().clone().requires_grad_(True)
                            # NOTE: Assumes compute_gradient works correctly *within* enable_grad context
                            f_int_for_hvp = self.energy_model.compute_gradient(u_i_input)
                        hvp_result, = torch.autograd.grad(
                            outputs=f_int_for_hvp, inputs=u_i_input,
                            grad_outputs=v_detached,
                            retain_graph=False, create_graph=False
                        )
                        return hvp_result * (~fixed_dofs_mask)

                    # --- Step 2: Solve Linear System K * delta_u = -residual ---
                    if self.verbose: print(f"Iter {iter_count}: Solving K*du = -R with CG (||R|| = {residual_norm.item():.3e})")
                    delta_u = self._solve_linear_system_cg(
                        hvp_function=hvp_forward,
                        rhs=-residual, # Use residual from start of iteration
                        fixed_dofs_mask=fixed_dofs_mask,
                        max_iter=self.cg_max_iter,
                        tol=self.cg_tol
                    )
                    if self.verbose: print(f"Iter {iter_count}: CG finished, ||delta_u|| = {torch.linalg.norm(delta_u).item():.3e}")

                    # --- Step 3: Line Search ---
                    if self.verbose: print(f"Iter {iter_count}: Performing line search...")
                    # Pass detached tensors, including the *current* f_int_i used for the residual
                    alpha = self._line_search_internal(u_i.detach(), delta_u.detach(), f_i.detach(), f_int_i.detach(), fixed_dofs_mask)
                    if self.verbose: print(f"Iter {iter_count}: Line search found alpha = {alpha:.4f}")

                    # --- Step 4: Update Displacement (No Gradient Tracking) ---
                    with torch.no_grad():
                        u_i_new = u_i + alpha * delta_u
                        u_i_new = self.bc_manager.apply(u_i_new)
                        u_i = u_i_new # Accept update

                    # --- Step 5: Compute New Internal Force and Residual (FIXED) ---
                    # Use the robust helper again for the updated u_i
                    f_int_i = self._compute_internal_force_robustly(u_i)

                    with torch.no_grad():
                        residual = f_i - f_int_i # Update residual for next iteration/check
                        filtered_residual = residual * (~fixed_dofs_mask)
                        residual_norm_new = torch.linalg.norm(filtered_residual)

                    # --- Check for Divergence after Update ---
                    if torch.isnan(residual_norm_new) or torch.isinf(residual_norm_new):
                         print(f"Warning: Sample {i}, Iter {iter_count}: Residual became NaN/Inf after update. Reverting state and stopping.")
                         u_i = u_i_last_successful # Revert
                         converged = False
                         break # Exit Newton loop

                    # Update residual norm and store last successful state
                    residual_norm = residual_norm_new
                    u_i_last_successful = u_i.clone()

                    # --- Check Convergence ---
                    if residual_norm < self.tolerance:
                        converged = True
                        if self.verbose: print(f"Iter {iter_count}: Converged! Residual norm = {residual_norm.item():.3e}")

                    # --- Logging and Visualization ---
                    if self.verbose or (self.visualize_active and i == 0):
                        with torch.no_grad():
                            try:
                                strain_e = self.energy_model.compute_energy(u_i).item()
                                ext_w = torch.sum(f_i * u_i).item()
                                total_e = strain_e - ext_w
                                if self.verbose:
                                    print(f"Iter {iter_count}: Residual = {residual_norm.item():.3e}, Energy = {total_e:.3e}")
                                if self.visualize_active and i == 0: # Update every iteration
                                     self._update_visualization_internal(u_i, f_i, gt_disp_i_for_viz, i, iter_count, residual_norm.item(), total_e, 0.0, strain_e, ext_w)
                            except Exception as viz_e:
                                 print(f"Warning: Could not compute energy for viz/logging at iter {iter_count}: {viz_e}")


                except Exception as e:
                    print(f"\nError occurred in Newton iteration for Sample {i}, Iter {iter_count}: {e}")
                    import traceback; traceback.print_exc() # More detailed error
                    u_i = u_i_last_successful
                    converged = False
                    break # Exit Newton loop

            # --- End Newton Loop ---
            if not converged:
                if self.verbose: print(f"Warning: Sample {i}: Newton solver did NOT converge after {iter_count} iterations. Final residual norm = {residual_norm.item():.3e}")
                # Option: Return last successful state instead of potentially bad final 'u_i'
                solutions.append(u_i_last_successful.detach().clone())
            else:
                # Append the converged solution
                solutions.append(u_i.detach().clone())


        # --- End Batch Loop ---
        if not solutions:
            return torch.empty((0, self.dof_count), device=self.device, dtype=self.dtype)

        final_solutions = torch.cat(solutions, dim=0)
        return final_solutions


    def _solve_linear_system_cg(self, hvp_function, rhs, fixed_dofs_mask, max_iter, tol):
        """Solves the linear system A*x = b using Conjugate Gradient, where A is defined by hvp_function."""
        # Ensure operations inside CG do not track gradients
        with torch.no_grad():
            x = torch.zeros_like(rhs) # Initial guess (solution vector)

            # Apply mask to RHS: b_masked = b * (~mask)
            # Ensures that the system effectively solves for free DOFs only,
            # implicitly setting delta_u = 0 at fixed DOFs.
            b_masked = rhs * (~fixed_dofs_mask)

            r = b_masked.clone() # Initial residual: r = b_masked - A*x0 = b_masked
            p = r.clone() # Initial search direction

            # Calculate initial residual norm squared (use dot product for scalar result)
            rsold_sq = torch.dot(r.flatten(), r.flatten())
            rsinit_norm = torch.sqrt(rsold_sq).item()

            if self.verbose: print(f"CG Start: Initial ||Residual|| = {rsinit_norm:.3e}, Max Iter = {max_iter}, Tol = {tol:.1e}")

            if rsinit_norm < 1e-15: # If initial residual is already near zero
                if self.verbose: print("CG: Initial residual near zero. Returning zero solution.")
                return x # Return zero solution (already satisfies Ax=b)

            best_x = x.clone()
            min_residual_norm_sq = rsold_sq.item()

            for i in range(max_iter):
                # Compute Ap = A @ p using the provided HVP function
                try:
                    Ap = hvp_function(p)
                except Exception as e:
                    print(f"CG Error: HVP computation failed at iteration {i}: {e}")
                    # Return the best solution found so far in case of HVP error
                    return best_x * (~fixed_dofs_mask)

                # Check for NaN/Inf in Ap
                if torch.isnan(Ap).any() or torch.isinf(Ap).any():
                    print(f"CG Error: NaN/Inf detected in HVP result at iteration {i}.")
                    return best_x * (~fixed_dofs_mask)

                # Calculate alpha = r_k^T * r_k / (p_k^T * A * p_k)
                pAp_dot = torch.dot(p.flatten(), Ap.flatten())

                # Check for breakdown condition (denominator close to zero or negative)
                # Indicates loss of positive definiteness or numerical instability
                if pAp_dot <= 1e-12 * torch.dot(p.flatten(), p.flatten()):
                    print(f"CG Warning: Breakdown condition met at iteration {i} (p^T*A*p = {pAp_dot.item():.3e} <= tolerance). Returning best solution found.")
                    return best_x * (~fixed_dofs_mask)

                alpha = rsold_sq / pAp_dot

                # Update solution: x_{k+1} = x_k + alpha * p_k
                x.add_(p, alpha=alpha)

                # Update residual: r_{k+1} = r_k - alpha * A * p_k
                r.add_(Ap, alpha=-alpha)

                # Calculate new residual norm squared
                rsnew_sq = torch.dot(r.flatten(), r.flatten())
                current_residual_norm = torch.sqrt(rsnew_sq).item()

                # Store best solution found so far (robustness for non-convergence)
                if rsnew_sq.item() < min_residual_norm_sq:
                    min_residual_norm_sq = rsnew_sq.item()
                    best_x = x.clone()

                # Check convergence: ||r_{k+1}|| / ||r_0|| < tol
                if current_residual_norm < tol * rsinit_norm:
                    if self.verbose: print(f"CG Converged at iteration {i+1}. Final Rel Residual = {current_residual_norm / rsinit_norm:.3e}")
                    break # Exit loop

                # Update search direction: p_{k+1} = r_{k+1} + beta * p_k
                # beta = rsnew_sq / rsold_sq
                beta = rsnew_sq / rsold_sq
                p = r + beta * p # Fletcher-Reeves update

                # Update rsold_sq for the next iteration
                rsold_sq = rsnew_sq

                if self.verbose and (i + 1) % 50 == 0: # Print progress periodically
                     print(f"CG Iter {i+1}: Rel Residual = {current_residual_norm / rsinit_norm:.3e}")

            else: # Loop finished without break (max_iter reached)
                print(f"CG Warning: Max iterations ({max_iter}) reached. Final Rel Residual = {current_residual_norm / rsinit_norm:.3e}")
                x = best_x # Return the best solution found

            # Ensure solution respects fixed DOFs explicitly (zero out fixed components)
            x = x * (~fixed_dofs_mask)
            if self.verbose: print(f"CG End: ||Solution|| = {torch.linalg.norm(x).item():.3e}")
            return x


    def _line_search_internal(self, u, delta_u, f_ext, f_int_current, fixed_dofs_mask):
        """Performs backtracking line search to find suitable step size alpha."""
        # Operates on detached tensors - no gradient tracking needed here
        # Goal: Find alpha such that E(u + alpha*delta_u) < E(u) + c1*alpha*grad(E)^T*delta_u
        # where E(u) = StrainEnergy(u) - Work(u) = W(u) - f_ext^T * u
        # grad(E) = f_int(u) - f_ext
        # grad(E)^T * delta_u = (f_int(u) - f_ext)^T * delta_u
        alpha = self.ls_alpha_max
        with torch.no_grad():
            try:
                 # Calculate current total potential energy E(u)
                 energy_current = self.energy_model.compute_energy(u).item() # Use .item() for scalar
            except Exception as e:
                 print(f"Line Search Warning: Failed to compute initial energy. Error: {e}. Using alpha_min.")
                 return self.ls_alpha_min

            # Calculate gradient of potential energy at current u
            grad_energy = (f_int_current - f_ext) * (~fixed_dofs_mask) # Mask gradient

            # Calculate directional derivative: grad(E)^T * delta_u
            descent_dot_product = torch.sum(grad_energy * (delta_u * (~fixed_dofs_mask))).item() # Use .item()

            if descent_dot_product >= -1e-12:
                if self.verbose and descent_dot_product > 1e-9:
                     print(f"Line Search Warning: Not descent direction (g^T*p = {descent_dot_product:.3e} >= 0). Using alpha_min.")
                return self.ls_alpha_min

            required_decrease_slope = self.ls_c1 * descent_dot_product # Negative scalar

            for trial in range(self.ls_max_trials):
                u_trial = u + alpha * delta_u
                u_trial = self.bc_manager.apply(u_trial)

                try:
                    # Calculate energy at trial point: E(u_trial)
                    energy_trial = self.energy_model.compute_energy(u_trial).item() # Use .item()
                except Exception as e:
                     if self.verbose: print(f"Line Search Warning: Error computing energy at alpha={alpha:.3e}. Reducing alpha. Error: {e}")
                     alpha *= 0.5
                     if alpha < self.ls_alpha_min: return self.ls_alpha_min
                     continue # Try next smaller alpha

                # Armijo condition check (using scalar values)
                if energy_trial <= energy_current + alpha * required_decrease_slope + 1e-9: # Adjust tolerance if needed
                    if self.verbose and trial > 0: print(f"Line Search: Accepted alpha={alpha:.3e} after {trial+1} trials.")
                    return alpha

                alpha *= 0.5
                if alpha < self.ls_alpha_min:
                    if self.verbose: print(f"Line Search Warning: Alpha below minimum ({self.ls_alpha_min:.1e}). Using alpha_min.")
                    return self.ls_alpha_min

            if self.verbose: print(f"Line Search Warning: Max trials ({self.ls_max_trials}) reached. Using alpha_min.")
            return self.ls_alpha_min


    # ==========================================================================
    # Visualization Methods (Setup in init, Update called internally if active)
    # (Copied from original, assuming correctness as requested)
    # ==========================================================================

    def _setup_visualization_internal(self, filename):
        """Internal: Sets up PyVista plotter ONCE (rank 0 only)."""
        # --- Assumes dolfinx, gmshio, pyvista are available ---
        from dolfinx import plot
        from dolfinx.io import gmshio
        import pyvista

        if self.viz_plotter is not None: return # Already setup

        print(f"[Viz Setup] Setting up visualization from mesh: {filename}")
        try:
            # Use COMM_SELF for reading mesh on a single rank
            domain, _, _ = gmshio.read_from_msh(filename, MPI.COMM_SELF, rank=0, gdim=self.dim) # Use self.dim
            # Generate VTK mesh representation
            topology, cell_types, x = plot.vtk_mesh(domain, dim=self.dim) # Use self.dim
            self.viz_grid = pyvista.UnstructuredGrid(topology, cell_types, x.copy())

            # Pre-allocate scalar arrays needed for plotting
            self.viz_grid.point_data["force_magnitude"] = np.zeros(self.viz_grid.n_points, dtype=np.float32)
            self.viz_grid.point_data["displacement_magnitude"] = np.zeros(self.viz_grid.n_points, dtype=np.float32) # Renamed for clarity
            # Store original node coordinates
            self.viz_points_cpu = x.copy()

            # Create the plotter window
            self.viz_plotter = pyvista.Plotter(shape=(1, 2), title="Newton Solver Iterations", window_size=[1600, 800], off_screen=False)

            # --- Left subplot: Forces ---
            self.viz_plotter.subplot(0, 0)
            self.viz_plotter.add_text("Applied Forces / Reference", position="upper_edge", font_size=10)
            # Add the mesh showing force magnitude
            self.mesh_actor_left = self.viz_plotter.add_mesh(self.viz_grid.copy(), scalars="force_magnitude", cmap="plasma", show_edges=True, scalar_bar_args={'title': 'Force Mag.'})
            # Add reference wireframe
            self.viz_plotter.add_mesh(self.viz_grid, color='grey', style='wireframe', opacity=0.3)
            # Add text info display
            self.info_actor_left = self.viz_plotter.add_text("Initializing...", position="lower_left", font_size=9, shadow=True)

            # --- Right subplot: Deformed Configuration ---
            self.viz_plotter.subplot(0, 1)
            self.viz_plotter.add_text("Deformed Configuration", position="upper_edge", font_size=10)
            # Add the mesh showing displacement magnitude (will be updated)
            self.mesh_actor_right = self.viz_plotter.add_mesh(self.viz_grid.copy(), scalars="displacement_magnitude", cmap="viridis", show_edges=True, scalar_bar_args={'title': 'Disp Mag.'}) # Use renamed scalar
            # Placeholder for ground truth wireframe
            self.gt_mesh_actor = None
            # Add text info display
            self.info_actor_right = self.viz_plotter.add_text("Initializing...", position="lower_left", font_size=9, shadow=True)

            # Configure plotter view
            self.viz_plotter.link_views()
            self.viz_plotter.camera_position = 'iso'
            self.viz_plotter.camera.zoom(1.2)
            self.viz_plotter.show(interactive_update=True, auto_close=False) # Keep window open
            print(f"[Viz Setup] Visualization setup complete.")

        except Exception as e:
             print(f"[Viz Setup] Error during visualization setup: {e}")
             import traceback; traceback.print_exc()
             # Ensure flags reflect failure
             self.visualize_active = False
             self._should_visualize = False
             self.viz_plotter = None # Ensure plotter is None if setup fails

    def _update_visualization_internal(self, u_i, f_i, gt_disp_i, sample_idx, iter_count, residual_norm, energy, energy_ratio, strain_energy=None, external_work=None):
        """Internal update called during Newton iterations IF visualization is active."""
        # Double check flags before proceeding
        if not self.visualize_active or self.viz_plotter is None:
            return

        print(f"Updating visualization for iteration {iter_count}")  # Confirm function is called

        try:
            # --- Prepare Data (on CPU) ---
            with torch.no_grad():
                # Ensure data is on CPU and NumPy format for PyVista
                u_cpu = u_i[0].detach().cpu().numpy().reshape(-1, self.dim)
                f_cpu = f_i[0].detach().cpu().numpy().reshape(-1, self.dim)

                # Process ground truth displacement if available
                gt_disp_array = None
                if gt_disp_i is not None:
                    gt_disp_array = gt_disp_i[0].detach().cpu().numpy().reshape(-1, self.dim)

            # --- Left Plot Update (Forces) ---
            self.viz_plotter.subplot(0, 0)
            # Create a fresh copy of the grid for forces
            force_grid = self.viz_grid.copy()
            force_grid.point_data["Forces"] = f_cpu
            force_mag = np.linalg.norm(f_cpu, axis=1)
            force_grid["force_magnitude"] = force_mag
            
            # Remove old mesh actor and add new one
            self.viz_plotter.remove_actor(self.mesh_actor_left)
            self.mesh_actor_left = self.viz_plotter.add_mesh(
                force_grid,
                scalars="force_magnitude",
                cmap="plasma",
                show_edges=True,
                clim=[0, np.max(force_mag) if np.max(force_mag) > 0 else 1.0]
            )
            
            # Update left info text - REMOVE AND RE-ADD ACTOR
            left_text = f"Sample: {sample_idx+1}\nMax F: {force_mag.max():.2e}"
            if strain_energy is not None: left_text += f"\nStrain E: {strain_energy:.2e}"
            if external_work is not None: left_text += f"\nExt Work: {external_work:.2e}"
            
            # Remove old text actor before adding new one
            self.viz_plotter.remove_actor(self.info_actor_left)
            self.info_actor_left = self.viz_plotter.add_text(
                left_text, 
                position="lower_left", 
                font_size=9, 
                shadow=True
            )
            
            # --- Right Plot Update (Deformation) ---
            self.viz_plotter.subplot(0, 1)
            # Create a fresh copy for deformation
            deformed_grid = self.viz_grid.copy()
            deformed_grid.points = self.viz_points_cpu + u_cpu
            disp_mag = np.linalg.norm(u_cpu, axis=1)
            deformed_grid["displacement"] = disp_mag
            
            # Remove old mesh actor and add new one
            self.viz_plotter.remove_actor(self.mesh_actor_right)
            self.mesh_actor_right = self.viz_plotter.add_mesh(
                deformed_grid,
                scalars="displacement",
                cmap="viridis",
                show_edges=True,
                clim=[0, np.max(disp_mag) if np.max(disp_mag) > 0 else 1.0]
            )
            
            # Update Ground Truth Wireframe
            if self.gt_mesh_actor is not None:
                self.viz_plotter.remove_actor(self.gt_mesh_actor)
                self.gt_mesh_actor = None
                
            if gt_disp_array is not None:
                gt_grid = self.viz_grid.copy()
                gt_grid.points = self.viz_points_cpu + gt_disp_array
                self.gt_mesh_actor = self.viz_plotter.add_mesh(
                    gt_grid,
                    style="wireframe",
                    color='lime',
                    opacity=0.7,
                    line_width=2
                )
            
            # Update right info text - REMOVE AND RE-ADD ACTOR
            right_text = f"Iter: {iter_count}\nRes: {residual_norm:.2e}\nTotal Energy: {energy:.2e}\nMax U: {disp_mag.max():.2e}"
            if gt_disp_array is not None:
                right_text += f"\nMax GT U: {np.linalg.norm(gt_disp_array, axis=1).max():.2e}"
            
            # Remove old text actor before adding new one
            self.viz_plotter.remove_actor(self.info_actor_right)
            self.info_actor_right = self.viz_plotter.add_text(
                right_text, 
                position="lower_left", 
                font_size=9, 
                shadow=True
            )
            
            # --- Force Rendering ---
            self.viz_plotter.render()  # Explicit render call
            self.viz_plotter.update(1)  # Process events
            print(f"Visualization updated for iteration {iter_count}")

        except Exception as e:
            print(f"\n[Viz Update Error] Iter {iter_count}: {str(e)}")
            import traceback; traceback.print_exc()

    def close_visualization(self):
        """Closes the PyVista plotter window (must be called explicitly)."""
        # Check if eligible to close (rank 0 if MPI used)
        is_rank_zero = True
        try:
             if MPI.COMM_WORLD.size > 1:
                  is_rank_zero = (MPI.COMM_WORLD.rank == 0)
        except ImportError:
             pass # MPI not available, assume single process

        if is_rank_zero and self.viz_plotter is not None and self.visualize_active:
            print("Closing visualization window...")
            try:
                self.viz_plotter.close()
            except Exception as e:
                print(f"Error closing plotter: {e}")
            finally:
                # Reset visualization state variables regardless of close success
                self.viz_plotter = None
                self.visualize_active = False
                self.viz_grid = None # Release mesh data
                self.viz_points_cpu = None
                # Ensure actors are cleared
                self.mesh_actor_left = None
                self.mesh_actor_right = None
                self.gt_mesh_actor = None
                self.info_actor_left = None
                self.info_actor_right = None
            print("Visualization closed.")
        # Ensure flag is false even if not rank zero or plotter was already None
        self.visualize_active = False

    def __del__(self):
         # Attempt to close plotter when solver object is deleted/goes out of scope
         # Check attributes exist before accessing, in case __init__ failed early
         if hasattr(self, 'visualize_active') and self.visualize_active:
              if hasattr(self, 'viz_plotter') and self.viz_plotter is not None:
                   self.close_visualization()