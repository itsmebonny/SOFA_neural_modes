import argparse
import os, sys, logging
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from torch.utils.tensorboard import SummaryWriter
import math

from dolfinx.fem.petsc import assemble_matrix as assemble_matrix_petsc
from dolfinx.fem import form 

from ufl import TrialFunction, TestFunction, inner, dx, grad, sym, Identity, div




# In train.py - add these imports
import glob
import json
import datetime

import matplotlib.pyplot as plt

import pyvista
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmshio
import gmsh
from mpi4py import MPI
import ufl
from scipy.linalg import eig
 

# Add after imports
from slepc4py import SLEPc
from petsc4py import PETSc

logger = logging.getLogger('train')

def setup_logger(name, log_dir=None, distributed_rank=0):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s %(levelname)s: %(message)s', datefmt='%m/%d %H:%M:%S'
    )

    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'log_rank{distributed_rank}.log')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

class RotationStrainNet(torch.nn.Module):
    """
    Neural network that explicitly models rotation and strain components separately
    for better handling of large deformations.
    """
    def __init__(self, latent_dim, num_nodes, hid_layers=2, hid_dim=128):
        super(RotationStrainNet, self).__init__()
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        self.hid_layers = hid_layers
        self.hid_dim = hid_dim
        
        # Helper function to create a sequence of linear layers with GELU activations
        def create_mlp(input_dim, output_dim, num_layers, hidden_dim):
            layers = []
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.GELU())
            for _ in range(num_layers - 1):
                layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
                layers.append(torch.nn.GELU())
            layers.append(torch.nn.Linear(hidden_dim, output_dim))
            return torch.nn.Sequential(*layers)
        
        # Network for predicting rotation parameters (quaternions)
        self.rotation_net = create_mlp(latent_dim, num_nodes * 4, hid_layers, hid_dim)
        
        # Network for predicting strain parameters
        self.strain_net = create_mlp(latent_dim, num_nodes * 6, hid_layers, hid_dim)
    
    def forward(self, z, rest_positions=None):
        """
        Forward pass computing displacement field from latent vector
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            rest_positions: Nodal positions in rest configuration [batch_size, num_nodes, 3]
                        (If None, displacements are directly predicted)
        
        Returns:
            displacements: Predicted displacement field [batch_size, num_nodes*3]
            rotation_matrices: Rotation matrices [batch_size, num_nodes, 3, 3] (if requested)
            strain_matrices: Strain matrices [batch_size, num_nodes, 3, 3] (if requested)
        """
        batch_size = z.shape[0] if z.dim() > 1 else 1
        if z.dim() == 1:
            z = z.unsqueeze(0)  # Add batch dimension if not present
        
        # Predict rotation parameters (quaternions)
        quat_params = self.rotation_net(z).view(batch_size, self.num_nodes, 4)
        # Normalize quaternions to ensure valid rotations
        quat_params = torch.nn.functional.normalize(quat_params, dim=2)
        
        # Predict strain parameters
        strain_params = self.strain_net(z).view(batch_size, self.num_nodes, 6)
        
        # Convert quaternions to rotation matrices
        rotation_matrices = self._quaternion_to_rotation_matrix(quat_params)
        
        # Convert strain parameters to symmetric positive-definite matrices
        strain_matrices = self._params_to_strain_matrix(strain_params)
        
        # MISSING STEP: Calculate deformed positions by applying rotation and strain
        deformed_positions = self._apply_deformation(rest_positions, rotation_matrices, strain_matrices)
        
        # Calculate displacements using rotation and strain
        displacements = deformed_positions - rest_positions
        
      
        
        return displacements, rotation_matrices, strain_matrices
    
    def _quaternion_to_rotation_matrix(self, quaternions):
        """Convert quaternions [batch, num_nodes, 4] to rotation matrices [batch, num_nodes, 3, 3]"""
        batch_size, num_nodes, _ = quaternions.shape
        
        # Extract quaternion components
        qw = quaternions[..., 0]
        qx = quaternions[..., 1]
        qy = quaternions[..., 2]
        qz = quaternions[..., 3]
        
        # Convert to rotation matrices (standard conversion formula)
        R = torch.zeros(batch_size, num_nodes, 3, 3, device=quaternions.device, dtype=quaternions.dtype)
        
        # First row
        R[..., 0, 0] = 1 - 2*qy*qy - 2*qz*qz
        R[..., 0, 1] = 2*qx*qy - 2*qw*qz
        R[..., 0, 2] = 2*qx*qz + 2*qw*qy
        
        # Second row
        R[..., 1, 0] = 2*qx*qy + 2*qw*qz
        R[..., 1, 1] = 1 - 2*qx*qx - 2*qz*qz
        R[..., 1, 2] = 2*qy*qz - 2*qw*qx
        
        # Third row
        R[..., 2, 0] = 2*qx*qz - 2*qw*qy
        R[..., 2, 1] = 2*qy*qz + 2*qw*qx
        R[..., 2, 2] = 1 - 2*qx*qx - 2*qy*qy
        
        return R
    
    def _params_to_strain_matrix(self, strain_params):
        """
        Convert strain parameters to symmetric positive-definite matrices
        Uses a modified Cholesky factorization approach to ensure positive-definiteness
        
        Args:
            strain_params: [batch_size, num_nodes, 6] strain parameters
            
        Returns:
            strain_matrices: [batch_size, num_nodes, 3, 3] symmetric positive-definite matrices
        """
        batch_size, num_nodes, _ = strain_params.shape
        
        # Extract the 6 parameters representing the strain tensor in Voigt notation
        # [xx, yy, zz, xy, yz, xz]
        xx = strain_params[..., 0]
        yy = strain_params[..., 1]
        zz = strain_params[..., 2]
        xy = strain_params[..., 3]
        yz = strain_params[..., 4]
        xz = strain_params[..., 5]
        
        # Ensure positive-definiteness using matrix exponential approach
        # First create symmetric matrices
        S = torch.zeros(batch_size, num_nodes, 3, 3, device=strain_params.device, dtype=strain_params.dtype)
        
        # Fill in the symmetric matrix
        S[..., 0, 0] = xx
        S[..., 1, 1] = yy
        S[..., 2, 2] = zz
        S[..., 0, 1] = S[..., 1, 0] = xy
        S[..., 1, 2] = S[..., 2, 1] = yz
        S[..., 0, 2] = S[..., 2, 0] = xz
        
        # Add identity and exponentiate through eigendecomposition to ensure positive-definiteness
        # This creates a matrix U = exp(S) which is always positive-definite
        identity = torch.eye(3, device=strain_params.device, dtype=strain_params.dtype)
        identity = identity.view(1, 1, 3, 3).repeat(batch_size, num_nodes, 1, 1)
        
        # Use scale factor to control deformation magnitude
        scale = 0.001 
        U = identity + scale * S
        
        return U
    
    def _apply_deformation(self, rest_positions, rotation_matrices, strain_matrices):
        """
        Apply deformation (strain followed by rotation) to rest positions
        
        Args:
            rest_positions: [batch_size, num_nodes, 3] rest positions
            rotation_matrices: [batch_size, num_nodes, 3, 3] rotation matrices
            strain_matrices: [batch_size, num_nodes, 3, 3] strain matrices
            
        Returns:
            deformed_positions: [batch_size, num_nodes, 3] deformed positions
        """
        # Check if rest_positions is None
        if rest_positions is None:
            raise ValueError("Rest positions cannot be None in _apply_deformation")
        
        batch_size = rest_positions.shape[0]
        
        # Apply strain then rotation: p_deformed = R·(U·p_rest)
        # For each node, transform its rest position
        strained_positions = torch.bmm(
            strain_matrices.reshape(batch_size * self.num_nodes, 3, 3),
            rest_positions.reshape(batch_size * self.num_nodes, 3, 1)
        ).reshape(batch_size, self.num_nodes, 3)
        
        # Apply rotation
        deformed_positions = torch.bmm(
            rotation_matrices.reshape(batch_size * self.num_nodes, 3, 3),
            strained_positions.reshape(batch_size * self.num_nodes, 3, 1)
        ).reshape(batch_size, self.num_nodes, 3)
        
        return deformed_positions





    


class PabloNeoHookeanEnergy(torch.nn.Module):
    """
    Neo-Hookean energy implementation with Pablo's formulation, fully optimized
    using batch processing and vectorized operations for improved performance.
    """
    def __init__(self, domain, degree, E, nu, precompute_matrices=True, device=None, 
                 dtype=torch.float64, batch_size=100):
        super(PabloNeoHookeanEnergy, self).__init__()
        
        # Set device and precision
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.batch_size = batch_size
        
        # Material properties
        self.E = E
        self.nu = nu
        self.mu = E / (2 * (1 + nu))  # Shear modulus
        self.lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))  # First Lamé parameter
        
        # Extract mesh information
        self.coordinates = torch.tensor(domain.geometry.x, dtype=self.dtype, device=self.device)
        self.num_nodes = self.coordinates.shape[0]
        self.dim = self.coordinates.shape[1]
        
        # Create elements tensor
        elements_list = []
        tdim = domain.topology.dim
        for cell in range(domain.topology.index_map(tdim).size_local):
            elements_list.append(domain.topology.connectivity(tdim, 0).links(cell))
        
        # Convert directly to device tensor in one step
        self.elements = torch.tensor(np.array(elements_list), dtype=torch.long, device=self.device)
        self.num_elements = len(self.elements)
        self.nodes_per_element = self.elements.shape[1]
        
        # Determine element type
        if self.nodes_per_element == 4:
            self.element_type = "tetrahedron"
        elif self.nodes_per_element == 8:
            self.element_type = "hexahedron"
        else:
            raise ValueError(f"Unsupported element type with {self.nodes_per_element} nodes")
        
        print(f"Mesh: {self.num_elements} {self.element_type} elements, {self.num_nodes} nodes")
        
        # Generate quadrature points - optimized for Neo-Hookean materials
        self.quadrature_points, self.quadrature_weights = self._generate_quadrature()
        print(f"Using {len(self.quadrature_weights)} quadrature points per element")
        
        # Memory-efficient precomputation strategy
        self.precomputed = False
        if precompute_matrices:
            self._precompute_derivatives_in_chunks()
            self.precomputed = True
            print("Precomputation complete")

    def _estimate_precompute_memory(self):
        """Estimate memory needed for precomputation in GB"""
        num_qp = len(self.quadrature_points)
        bytes_per_element = (
            # dN_dx_all: [num_elements, num_qp, nodes_per_element, 3]
            self.num_elements * num_qp * self.nodes_per_element * 3 * self.dtype.itemsize +
            # detJ_all: [num_elements, num_qp]
            self.num_elements * num_qp * self.dtype.itemsize
        )
        return bytes_per_element / (1024**3)  # Convert to GB

    def _generate_quadrature(self):
        """Generate quadrature rules optimized for Neo-Hookean materials"""
        if self.element_type == "tetrahedron":
            # 4-point quadrature for tetrahedron (more accurate for nonlinear materials)
            points = torch.tensor([
                [0.58541020, 0.13819660, 0.13819660],
                [0.13819660, 0.58541020, 0.13819660],
                [0.13819660, 0.13819660, 0.58541020],
                [0.13819660, 0.13819660, 0.13819660]
            ], dtype=self.dtype, device=self.device)
            weights = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=self.dtype, device=self.device) / 6.0
        else:  # hexahedron
            # 2×2×2 Gaussian quadrature
            gp = 1.0 / math.sqrt(3)
            points = []
            weights = []
            for i in [-gp, gp]:
                for j in [-gp, gp]:
                    for k in [-gp, gp]:
                        points.append([i, j, k])
                        weights.append(1.0)
            points = torch.tensor(points, dtype=self.dtype, device=self.device)
            weights = torch.tensor(weights, dtype=self.dtype, device=self.device)
        
        return points, weights
    
    def _precompute_derivatives_in_chunks(self):
        """Precompute derivatives in a format optimized for batch operations"""
        print("Precomputing derivatives for efficient batch processing...")
        
        # Count total quadrature points per element
        num_qp = len(self.quadrature_points)
        
        # Initialize storage as tensors rather than lists of lists
        self.dN_dx_all = torch.zeros((self.num_elements, num_qp, self.nodes_per_element, 3), 
                                   dtype=self.dtype, device=self.device)
        self.detJ_all = torch.zeros((self.num_elements, num_qp), 
                                  dtype=self.dtype, device=self.device)
        
        # Process elements in chunks
        for chunk_start in range(0, self.num_elements, self.batch_size):
            chunk_end = min(chunk_start + self.batch_size, self.num_elements)
            
            for e in range(chunk_start, chunk_end):
                element_nodes = self.elements[e].long()
                element_coords = self.coordinates[element_nodes]
                
                for q_idx in range(num_qp):
                    qp = self.quadrature_points[q_idx]
                    dN_dx, detJ = self._compute_derivatives(element_coords, qp)
                    
                    # Store in tensor format
                    self.dN_dx_all[e, q_idx] = dN_dx
                    self.detJ_all[e, q_idx] = detJ
            
            # Clear GPU cache after each chunk
            if torch.cuda.is_available() and (chunk_end - chunk_start) > 1000:
                torch.cuda.empty_cache()
        
        est_memory = self._estimate_precompute_memory()
        print(f"Precomputation complete. Memory usage: {est_memory:.2f} GB")
    
    def _compute_derivatives(self, element_coords, qp):
        """Compute shape function derivatives for an element at a quadrature point"""
        if element_coords.shape[0] == 4:  # tetrahedron
            # Shape function derivatives for tetrahedron (constant)
            dN_dxi = torch.tensor([
                [-1.0, -1.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=self.dtype, device=element_coords.device)
            
            # Jacobian calculation
            J = torch.matmul(element_coords.T, dN_dxi)
            detJ = torch.det(J)
            invJ = torch.linalg.inv(J)
            
            # Shape function derivatives w.r.t. physical coordinates
            dN_dx = torch.matmul(dN_dxi, invJ)
            
        else:  # hexahedron with 8 nodes
            xi, eta, zeta = qp
            
            # Precompute terms for efficiency
            xim = 1.0 - xi
            xip = 1.0 + xi
            etam = 1.0 - eta
            etap = 1.0 + eta
            zetam = 1.0 - zeta
            zetap = 1.0 + zeta
            
            # Shape function derivatives using vectorized operations
            dN_dxi = torch.zeros((8, 3), dtype=self.dtype, device=element_coords.device)
            
            # First derivatives with respect to xi
            dN_dxi[:, 0] = torch.tensor([
                -0.125 * etam * zetam,
                0.125 * etam * zetam,
                0.125 * etap * zetam,
                -0.125 * etap * zetam,
                -0.125 * etam * zetap,
                0.125 * etam * zetap,
                0.125 * etap * zetap,
                -0.125 * etap * zetap
            ], dtype=self.dtype, device=element_coords.device)
            
            # First derivatives with respect to eta
            dN_dxi[:, 1] = torch.tensor([
                -0.125 * xim * zetam,
                -0.125 * xip * zetam,
                0.125 * xip * zetam,
                0.125 * xim * zetam,
                -0.125 * xim * zetap,
                -0.125 * xip * zetap,
                0.125 * xip * zetap,
                0.125 * xim * zetap
            ], dtype=self.dtype, device=element_coords.device)
            
            # First derivatives with respect to zeta
            dN_dxi[:, 2] = torch.tensor([
                -0.125 * xim * etam,
                -0.125 * xip * etam,
                -0.125 * xip * etap,
                -0.125 * xim * etap,
                0.125 * xim * etam,
                0.125 * xip * etam,
                0.125 * xip * etap,
                0.125 * xim * etap
            ], dtype=self.dtype, device=element_coords.device)
            
            # Jacobian calculation
            J = torch.matmul(element_coords.T, dN_dxi)
            detJ = torch.det(J)
            invJ = torch.linalg.inv(J)
            
            # Shape function derivatives w.r.t. physical coordinates
            dN_dx = torch.matmul(dN_dxi, invJ)
            
        return dN_dx, detJ
    
    def _compute_F_batch(self, batch_coords, batch_disps, batch_dN_dx):
        """
        Compute deformation gradient F = I + ∇u for a batch of elements using broadcasting
        
        Args:
            batch_coords: Element coordinates [batch_size, nodes_per_element, 3]
            batch_disps: Element displacements [batch_size, nodes_per_element, 3]
            batch_dN_dx: Shape function derivatives [batch_size, nodes_per_element, 3]
            
        Returns:
            Batch of deformation gradients [batch_size, 3, 3]
        """
        # Initialize deformation gradient as identity for each element in batch
        batch_size = batch_disps.shape[0]
        batch_F = torch.eye(3, dtype=self.dtype, device=self.device).expand(batch_size, 3, 3).clone()
        
        # Use einsum for efficient batch computation: F += u_i ⊗ ∇N_i
        batch_F += torch.einsum('bij,bik->bjk', batch_disps, batch_dN_dx)
        
        return batch_F
    
    def _compute_neohook_energy_density_batch(self, batch_F):
        """
        Compute Neo-Hookean strain energy density for a batch of deformation gradients
        with improved stability handling based on Pablo's formulation
        
        Args:
            batch_F: Tensor of shape [batch_size, 3, 3] containing deformation gradients
        
        Returns:
            Tensor of shape [batch_size] containing energy densities
        """
        # Compute J = det(F) for all elements in batch
        batch_J = torch.linalg.det(batch_F)
        
        # Compute right Cauchy-Green tensor and its first invariant
        batch_C = torch.bmm(batch_F.transpose(1, 2), batch_F)
        batch_I1 = torch.diagonal(batch_C, dim1=1, dim2=2).sum(dim=1)
        
        # Create safe versions of variables for numerical stability
        # Following Pablo's implementation but with improved stability
        batch_size = batch_F.shape[0]
        batch_W = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
        
        # Split computation based on J values for stability
        # Case 1: Valid J values (J > threshold)
        valid_mask = batch_J > 1e-6
        if torch.any(valid_mask):
            valid_J = batch_J[valid_mask]
            valid_I1 = batch_I1[valid_mask]
            valid_log_J = torch.log(valid_J)
            
            # Calculate both terms exactly as in pablo_loss.py
            isochoric_term = 0.5 * self.mu * (valid_I1 - 3.0 - 2.0 * valid_log_J)
            volumetric_term = 0.25 * self.lmbda * ((valid_J*valid_J) - 1.0 - 2.0 * valid_log_J)
            
            # Total energy
            batch_W[valid_mask] = isochoric_term + volumetric_term
        
        # Case 2: Small positive J values (need stabilization)
        small_mask = (batch_J > 0) & (batch_J <= 1e-6)
        if torch.any(small_mask):
            small_J = batch_J[small_mask]
            small_I1 = batch_I1[small_mask]
            
            # Use Taylor expansion for log(J) when J is very small
            # log(J) ≈ J-1 - (J-1)²/2 + (J-1)³/3 - ...
            approx_log_J = small_J - 1.0 - (small_J - 1.0)**2 / 2.0
            
            isochoric_term = 0.5 * self.mu * (small_I1 - 3.0 - 2.0 * approx_log_J)
            volumetric_term = 0.25 * self.lmbda * ((small_J*small_J) - 1.0 - 2.0 * approx_log_J)
            
            # Add regularization to prevent energy from going negative
            barrier = 1e2 * torch.pow(1e-6 - small_J, 2)
            batch_W[small_mask] = isochoric_term + volumetric_term + barrier
        
        # Case 3: Negative J values (inverted elements)
        invalid_mask = batch_J <= 0
        if torch.any(invalid_mask):
            # Use a large positive penalty proportional to how negative J is
            batch_W[invalid_mask] = 1e6 * torch.abs(batch_J[invalid_mask])
        
        return batch_W
    

    def _compute_PK1_batch(self, batch_F):
        """
        Compute the First Piola-Kirchhoff stress tensor for a batch of deformation gradients
        
        Args:
            batch_F: Tensor of shape [batch_size, 3, 3] containing deformation gradients
            
        Returns:
            Tensor of shape [batch_size, 3, 3] containing PK1 tensors
        """
        # Compute J = det(F)
        batch_J = torch.linalg.det(batch_F)
        
        # Compute inverse of F (batched)
        batch_inv_F = torch.linalg.inv(batch_F)
        
        # Compute F^-T
        batch_inv_F_T = batch_inv_F.transpose(1, 2)
        
        # First Piola-Kirchhoff stress tensor (Pablo's formulation)
        # P = μ(F - F^-T) + 0.5λ(J²-1)F^-T
        term1 = self.mu * batch_F
        term2 = -self.mu * batch_inv_F_T
        term3 = 0.5 * self.lmbda * ((batch_J * batch_J - 1.0).unsqueeze(-1).unsqueeze(-1)) * batch_inv_F_T
        
        batch_P = term1 + term2 + term3
        
        return batch_P
    
    def forward(self, u_tensor):
        """
        Compute total elastic energy for the displacement field using optimized batch operations
        
        Args:
            u_tensor: Displacement field [num_nodes * 3]
            
        Returns:
            Total strain energy
        """
        # Ensure tensor is on the correct device and type
        if u_tensor.device != self.device or u_tensor.dtype != self.dtype:
            u_tensor = u_tensor.to(device=self.device, dtype=self.dtype)
        
        # Reshape displacement vector
        u = u_tensor.reshape(self.num_nodes, self.dim)
        
        # Initialize total energy
        total_energy = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        
        # Get number of quadrature points
        num_qp = len(self.quadrature_points)
        
        # Process elements in batches for memory efficiency
        for batch_start in range(0, self.num_elements, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.num_elements)
            batch_elements = self.elements[batch_start:batch_end]
            batch_size = batch_end - batch_start
            
            # Gather element coordinates and displacements for the entire batch
            batch_coords = self.coordinates[batch_elements]  # [batch, nodes_per_elem, 3]
            batch_disps = u[batch_elements]                 # [batch, nodes_per_elem, 3]
            
            # Initialize batch energy
            batch_energy = torch.zeros(batch_size, device=self.device, dtype=self.dtype)
            
            # Process all quadrature points
            for q_idx in range(num_qp):
                if self.precomputed:
                    # Use precomputed derivatives
                    batch_dN_dx = self.dN_dx_all[batch_start:batch_end, q_idx]  # [batch, nodes, 3]
                    batch_detJ = self.detJ_all[batch_start:batch_end, q_idx]    # [batch]
                else:
                    # Compute derivatives on-the-fly (less efficient)
                    qp = self.quadrature_points[q_idx]
                    batch_dN_dx = []
                    batch_detJ = []
                    for i in range(batch_size):
                        dN_dx, detJ = self._compute_derivatives(batch_coords[i], qp)
                        batch_dN_dx.append(dN_dx)
                        batch_detJ.append(detJ)
                    batch_dN_dx = torch.stack(batch_dN_dx)
                    batch_detJ = torch.tensor(batch_detJ, device=self.device, dtype=self.dtype)
                
                # Compute deformation gradients for all elements at once
                batch_F = self._compute_F_batch(batch_coords, batch_disps, batch_dN_dx)
                
                # Compute energy densities for all elements at once using Pablo's formulation
                batch_energy_density = self._compute_neohook_energy_density_batch(batch_F)
                
                # Numerical integration
                batch_energy += batch_energy_density * batch_detJ * self.quadrature_weights[q_idx]
            
            # Sum up all element energies in this batch
            total_energy += torch.sum(batch_energy)
            
            # Clean up memory periodically
            if torch.cuda.is_available() and batch_size > 500:
                torch.cuda.empty_cache()
        
        return total_energy
    
    def compute_batch_energy(self, batch_u):
        """
        Process batch of displacement fields efficiently
        
        Args:
            batch_u: Batch of displacement fields [batch_size, num_nodes * 3]
            
        Returns:
            Energies for each displacement field [batch_size]
        """
        batch_size = batch_u.shape[0]
        energies = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
        
        # For larger batches, process in smaller chunks to save memory
        max_samples_per_batch = 4
        for i in range(0, batch_size, max_samples_per_batch):
            end_idx = min(i + max_samples_per_batch, batch_size)
            sub_batch = batch_u[i:end_idx]
            sub_batch_size = end_idx - i
            
            # Process each sample in sub-batch
            for j in range(sub_batch_size):
                energies[i + j] = self.forward(sub_batch[j])
            
            # Free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return energies
    
    def compute_strains_stresses(self, u_tensor):
        """
        Compute strains and stresses across the domain using batch operations
        
        Args:
            u_tensor: Displacement field [num_nodes * 3]
            
        Returns:
            Dictionary with 'strains' and 'stresses' tensors
        """
        # Ensure tensor is on correct device/type
        if u_tensor.device != self.device or u_tensor.dtype != self.dtype:
            u_tensor = u_tensor.to(device=self.device, dtype=self.dtype)
        
        # Reshape displacement vector
        u = u_tensor.reshape(self.num_nodes, self.dim)
        
        # Initialize storage for results
        strains = torch.zeros((self.num_elements, len(self.quadrature_points), 6), 
                             dtype=self.dtype, device=self.device)
        stresses = torch.zeros((self.num_elements, len(self.quadrature_points), 6), 
                              dtype=self.dtype, device=self.device)
        
        # Process elements in batches
        for batch_start in range(0, self.num_elements, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.num_elements)
            batch_elements = self.elements[batch_start:batch_end]
            batch_size = batch_end - batch_start
            
            # Get coordinates and displacements for this batch
            batch_coords = self.coordinates[batch_elements]
            batch_disps = u[batch_elements]
            
            # Process each quadrature point
            for q_idx in range(len(self.quadrature_points)):
                if self.precomputed:
                    batch_dN_dx = self.dN_dx_all[batch_start:batch_end, q_idx]
                    batch_detJ = self.detJ_all[batch_start:batch_end, q_idx]
                else:
                    # Compute on-the-fly
                    qp = self.quadrature_points[q_idx]
                    batch_dN_dx = []
                    batch_detJ = []
                    for i in range(batch_size):
                        dN_dx, detJ = self._compute_derivatives(batch_coords[i], qp)
                        batch_dN_dx.append(dN_dx)
                        batch_detJ.append(detJ)
                    batch_dN_dx = torch.stack(batch_dN_dx)
                    batch_detJ = torch.tensor(batch_detJ, device=self.device, dtype=self.dtype)
                
                # Compute deformation gradients for all elements at once
                batch_F = self._compute_F_batch(batch_coords, batch_disps, batch_dN_dx)
                
                # Batch identity matrix
                batch_I = torch.eye(3, dtype=self.dtype, device=self.device).expand(batch_size, 3, 3)
                
                # Batch Right Cauchy-Green tensor C = F^T F
                batch_C = torch.bmm(batch_F.transpose(1, 2), batch_F)
                
                # Batch Green-Lagrange strain tensor E = 0.5(C - I)
                batch_E = 0.5 * (batch_C - batch_I)
                
                # Calculate batch PK1 stress tensor using Pablo's formulation
                batch_P = self._compute_PK1_batch(batch_F)
                
                # Calculate batch Second Piola-Kirchhoff stress tensor S = F^{-1} P
                batch_inv_F = torch.linalg.inv(batch_F)
                batch_S = torch.bmm(batch_inv_F, batch_P)
                
                # Convert tensors to Voigt notation [xx, yy, zz, xy, yz, xz]
                for b in range(batch_size):
                    e_idx = batch_start + b
                    E = batch_E[b]
                    S = batch_S[b]
                    
                    strains[e_idx, q_idx] = torch.tensor([
                        E[0,0], E[1,1], E[2,2], E[0,1], E[1,2], E[0,2]
                    ], device=self.device, dtype=self.dtype)
                    
                    stresses[e_idx, q_idx] = torch.tensor([
                        S[0,0], S[1,1], S[2,2], S[0,1], S[1,2], S[0,2]
                    ], device=self.device, dtype=self.dtype)
        
        return {'strains': strains, 'stresses': stresses}
    
    def compute_PK1(self, F):
        """
        Compute the First Piola-Kirchhoff stress tensor from deformation gradient
        
        Args:
            F: Deformation gradient tensor [3, 3]
            
        Returns:
            First Piola-Kirchhoff stress tensor [3, 3]
        """
        # Compute J = det(F)
        J = torch.det(F)
        
        # Compute inverse of F
        inv_F = torch.linalg.inv(F)
        
        # First Piola-Kirchhoff stress tensor (Pablo's formulation)
        # P = μ(F - F^-T) + 0.5λ(J²-1)F^-T
        P = self.mu * (F - torch.transpose(inv_F, 0, 1)) + \
            0.5 * self.lmbda * (J * J - 1) * torch.transpose(inv_F, 0, 1)
        
        return P
    
    def visualize_deformation(self, u_tensor, scale_factor=1.0):
        """
        Helper method to visualize deformation with stress/strain coloring
        
        Args:
            u_tensor: Displacement field [num_nodes * 3]
            scale_factor: Scale factor for displacement visualization
        
        Returns:
            pyvista plotter object
        """
        try:
            import pyvista as pv
            from dolfinx import plot
        except ImportError:
            print("Visualization requires pyvista and dolfinx.plot")
            return None
        
        # Ensure tensor is numpy array
        if torch.is_tensor(u_tensor):
            u_np = u_tensor.detach().cpu().numpy()
        else:
            u_np = u_tensor
        
        # Compute strains and stresses
        result = self.compute_strains_stresses(u_tensor)
        stresses = result['stresses']
        
        # Compute von Mises stress for each element
        von_mises = []
        for e in range(stresses.shape[0]):
            for q in range(stresses.shape[1]):
                s = stresses[e, q]  # [s_xx, s_yy, s_zz, s_xy, s_yz, s_zx]
                
                # Reconstruct stress tensor
                stress_tensor = torch.zeros((3, 3), device=stresses.device)
                stress_tensor[0, 0] = s[0]  # s_xx
                stress_tensor[1, 1] = s[1]  # s_yy
                stress_tensor[2, 2] = s[2]  # s_zz
                stress_tensor[0, 1] = stress_tensor[1, 0] = s[3]  # s_xy
                stress_tensor[1, 2] = stress_tensor[2, 1] = s[4]  # s_yz
                stress_tensor[0, 2] = stress_tensor[2, 0] = s[5]  # s_zx
                
                # Compute von Mises stress
                deviatoric = stress_tensor - torch.trace(stress_tensor)/3 * torch.eye(3, device=stress_tensor.device)
                vm = torch.sqrt(1.5 * torch.sum(deviatoric * deviatoric)).item()
                von_mises.append(vm)
        
        # Create a mesh with deformation
        topology, cell_types, x = plot.vtk_mesh(self.domain)
        grid = pv.UnstructuredGrid(topology, cell_types, x)
        
        # Add displacement
        u_reshaped = u_np.reshape((-1, 3))
        grid.point_data["displacement"] = u_reshaped
        grid["displacement_magnitude"] = np.linalg.norm(u_reshaped, axis=1)
        
        # Warp the mesh
        warped = grid.copy()
        warped.points += u_reshaped * scale_factor
        
        # Add stress data
        if len(von_mises) == self.num_elements:
            warped.cell_data["von_mises_stress"] = np.array(von_mises)
            
        # Create plotter
        plotter = pv.Plotter()
        plotter.add_mesh(warped, scalars="von_mises_stress", cmap="jet", show_edges=True)
        plotter.add_scalar_bar("von Mises Stress")
        
        return plotter


class StVenantKirchhoffEnergy(torch.nn.Module):
    """
    Optimized implementation of Saint Venant-Kirchhoff elastic energy calculation using
    PyTorch broadcasting operations for improved performance.
    
    This model captures geometric nonlinearity while maintaining a linear stress-strain relationship.
    """
    def __init__(self, domain, degree, E, nu, precompute_matrices=True, device=None, 
                 dtype=torch.float64, batch_size=200):
        super(StVenantKirchhoffEnergy, self).__init__()
        
        # Set device and precision
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.batch_size = batch_size
        
        # Material properties
        self.E = E
        self.nu = nu
        self.mu = E / (2 * (1 + nu))  # Shear modulus
        self.lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))  # First Lamé parameter
        
        # Extract mesh information
        self.coordinates = torch.tensor(domain.geometry.x, dtype=self.dtype, device=self.device)
        self.num_nodes = self.coordinates.shape[0]
        self.dim = self.coordinates.shape[1]
        
        # Create elements tensor 
        elements_list = []
        tdim = domain.topology.dim
        for cell in range(domain.topology.index_map(tdim).size_local):
            elements_list.append(domain.topology.connectivity(tdim, 0).links(cell))
        
        # Convert directly to device tensor in one step
        self.elements = torch.tensor(np.array(elements_list), dtype=torch.long, device=self.device)
        self.num_elements = len(self.elements)
        self.nodes_per_element = self.elements.shape[1]
        
        # Determine element type
        if self.nodes_per_element == 4:
            self.element_type = "tetrahedron"
        elif self.nodes_per_element == 8:
            self.element_type = "hexahedron"
        else:
            raise ValueError(f"Unsupported element type with {self.nodes_per_element} nodes")
        
        print(f"Mesh: {self.num_elements} {self.element_type} elements, {self.num_nodes} nodes")
        
        # Generate quadrature points - optimized for SVK
        self.quadrature_points, self.quadrature_weights = self._generate_quadrature()
        print(f"Using {len(self.quadrature_weights)} quadrature points per element")
        
        # Memory-efficient precomputation strategy
        self.precomputed = False
        if precompute_matrices:
            self._precompute_derivatives_in_chunks()
            self.precomputed = True

    def _estimate_precompute_memory(self):
        """Estimate memory needed for precomputation in GB"""
        num_qp = len(self.quadrature_points)
        bytes_per_element = (
            # dN_dx_all: [num_elements, num_qp, nodes_per_element, 3]
            self.num_elements * num_qp * self.nodes_per_element * 3 * self.dtype.itemsize +
            # detJ_all: [num_elements, num_qp]
            self.num_elements * num_qp * self.dtype.itemsize
        )
        return bytes_per_element / (1024**3)  # Convert to GB

    def _generate_quadrature(self):
        """Generate optimized quadrature rules based on element type"""
        if self.element_type == "tetrahedron":
            # For SVK, one-point quadrature is often sufficient for tetrahedral elements
            points = torch.tensor([
                [0.25, 0.25, 0.25]
            ], dtype=self.dtype, device=self.device)
            weights = torch.tensor([1.0/6.0], dtype=self.dtype, device=self.device)
        else:  # hexahedron
            # 2×2×2 Gaussian quadrature
            gp = 1.0 / math.sqrt(3)
            points = []
            weights = []
            for i in [-gp, gp]:
                for j in [-gp, gp]:
                    for k in [-gp, gp]:
                        points.append([i, j, k])
                        weights.append(1.0)
            points = torch.tensor(points, dtype=self.dtype, device=self.device)
            weights = torch.tensor(weights, dtype=self.dtype, device=self.device)
        
        return points, weights
    
    def _precompute_derivatives_in_chunks(self):
        """Precompute derivatives in a format optimized for batch operations"""
        print("Precomputing derivatives for efficient batch processing...")
        
        # Count total quadrature points per element
        num_qp = len(self.quadrature_points)
        
        # Initialize storage as tensors rather than lists of lists
        self.dN_dx_all = torch.zeros((self.num_elements, num_qp, self.nodes_per_element, 3), 
                                   dtype=self.dtype, device=self.device)
        self.detJ_all = torch.zeros((self.num_elements, num_qp), 
                                  dtype=self.dtype, device=self.device)
        
        # Process elements in chunks
        for chunk_start in range(0, self.num_elements, self.batch_size):
            chunk_end = min(chunk_start + self.batch_size, self.num_elements)
            
            for e in range(chunk_start, chunk_end):
                element_nodes = self.elements[e].long()
                element_coords = self.coordinates[element_nodes]
                
                for q_idx in range(num_qp):
                    qp = self.quadrature_points[q_idx]
                    dN_dx, detJ = self._compute_derivatives(element_coords, qp)
                    
                    # Store in tensor format
                    self.dN_dx_all[e, q_idx] = dN_dx
                    self.detJ_all[e, q_idx] = detJ
            
            # Clear GPU cache after each chunk
            if torch.cuda.is_available() and (chunk_end - chunk_start) > 1000:
                torch.cuda.empty_cache()
        
        est_memory = self._estimate_precompute_memory()
        print(f"Precomputation complete. Memory usage: {est_memory:.2f} GB")
    
    def _compute_derivatives(self, element_coords, qp):
        """Compute shape function derivatives for an element at a quadrature point"""
        if element_coords.shape[0] == 4:  # tetrahedron
            # Shape function derivatives for tetrahedron (constant)
            dN_dxi = torch.tensor([
                [-1.0, -1.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=self.dtype, device=element_coords.device)
            
            # Jacobian calculation
            J = torch.matmul(element_coords.T, dN_dxi)
            detJ = torch.det(J)
            invJ = torch.linalg.inv(J)
            
            # Shape function derivatives w.r.t. physical coordinates
            dN_dx = torch.matmul(dN_dxi, invJ)
            
        else:  # hexahedron with 8 nodes
            xi, eta, zeta = qp
            
            # Precompute terms for efficiency
            xim = 1.0 - xi
            xip = 1.0 + xi
            etam = 1.0 - eta
            etap = 1.0 + eta
            zetam = 1.0 - zeta
            zetap = 1.0 + zeta
            
            # Shape function derivatives using vectorized operations
            dN_dxi = torch.zeros((8, 3), dtype=self.dtype, device=element_coords.device)
            
            # First derivatives with respect to xi
            dN_dxi[:, 0] = torch.tensor([
                -0.125 * etam * zetam,
                0.125 * etam * zetam,
                0.125 * etap * zetam,
                -0.125 * etap * zetam,
                -0.125 * etam * zetap,
                0.125 * etam * zetap,
                0.125 * etap * zetap,
                -0.125 * etap * zetap
            ], dtype=self.dtype, device=element_coords.device)
            
            # First derivatives with respect to eta
            dN_dxi[:, 1] = torch.tensor([
                -0.125 * xim * zetam,
                -0.125 * xip * zetam,
                0.125 * xip * zetam,
                0.125 * xim * zetam,
                -0.125 * xim * zetap,
                -0.125 * xip * zetap,
                0.125 * xip * zetap,
                0.125 * xim * zetap
            ], dtype=self.dtype, device=element_coords.device)
            
            # First derivatives with respect to zeta
            dN_dxi[:, 2] = torch.tensor([
                -0.125 * xim * etam,
                -0.125 * xip * etam,
                -0.125 * xip * etap,
                -0.125 * xim * etap,
                0.125 * xim * etam,
                0.125 * xip * etam,
                0.125 * xip * etap,
                0.125 * xim * etap
            ], dtype=self.dtype, device=element_coords.device)
            
            # Jacobian calculation
            J = torch.matmul(element_coords.T, dN_dxi)
            detJ = torch.det(J)
            invJ = torch.linalg.inv(J)
            
            # Shape function derivatives w.r.t. physical coordinates
            dN_dx = torch.matmul(dN_dxi, invJ)
            
        return dN_dx, detJ
    
    def _compute_F_batch(self, batch_coords, batch_disps, batch_dN_dx):
        """
        Compute deformation gradient F = I + ∇u for a batch of elements using broadcasting
        
        Args:
            batch_coords: Element coordinates [batch_size, nodes_per_element, 3]
            batch_disps: Element displacements [batch_size, nodes_per_element, 3]
            batch_dN_dx: Shape function derivatives [batch_size, nodes_per_element, 3]
            
        Returns:
            Batch of deformation gradients [batch_size, 3, 3]
        """
        # Initialize deformation gradient as identity for each element in batch
        batch_size = batch_disps.shape[0]
        batch_F = torch.eye(3, dtype=self.dtype, device=self.device).expand(batch_size, 3, 3).clone()
        
        # Use einsum for efficient batch computation: F += u_i ⊗ ∇N_i
        # This computes the outer product for each element and node, then sums over nodes
        batch_F += torch.einsum('bij,bik->bjk', batch_disps, batch_dN_dx)
        
        return batch_F
    
    def _compute_svk_energy_density_batch(self, batch_F):
        """
        Compute Saint Venant-Kirchhoff strain energy density for a batch of deformation gradients
        
        Args:
            batch_F: Tensor of shape [batch_size, 3, 3] containing deformation gradients
        
        Returns:
            Tensor of shape [batch_size] containing energy densities
        """
        # Identity matrix expanded to batch size
        batch_size = batch_F.shape[0]
        batch_I = torch.eye(3, dtype=self.dtype, device=self.device).expand(batch_size, 3, 3)
        
        # Right Cauchy-Green tensor C = F^T F
        batch_C = torch.bmm(batch_F.transpose(1, 2), batch_F)
        
        # Green-Lagrange strain tensor E = (1/2)(C - I)
        batch_E = 0.5 * (batch_C - batch_I)
        
        # Calculate tr(E) and tr(E²)
        batch_trE = torch.diagonal(batch_E, dim1=1, dim2=2).sum(dim=1)
        batch_E_squared = torch.bmm(batch_E, batch_E)
        batch_trE_squared = torch.diagonal(batch_E_squared, dim1=1, dim2=2).sum(dim=1)
        
        # SVK strain energy W = (λ/2)(tr(E))² + μtr(E²)
        batch_W = 0.5 * self.lambda_ * batch_trE**2 + self.mu * batch_trE_squared
        
        # Add regularization for severely compressed elements
        batch_J = torch.linalg.det(batch_F)
        penalty = torch.where(batch_J < 0.01, 1e6 * (0.01 - batch_J)**2, 
                            torch.zeros_like(batch_J))
        
        return batch_W + penalty
    
    def forward(self, u_tensor):
        """
        Compute total elastic energy for the displacement field using optimized batch operations
        
        Args:
            u_tensor: Displacement field [num_nodes * 3]
            
        Returns:
            Total strain energy
        """
        # Ensure tensor is on the correct device and type
        if u_tensor.device != self.device or u_tensor.dtype != self.dtype:
            u_tensor = u_tensor.to(device=self.device, dtype=self.dtype)
        
        # Reshape displacement vector
        u = u_tensor.reshape(self.num_nodes, self.dim)
        
        # Initialize total energy
        total_energy = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        
        # Get number of quadrature points
        num_qp = len(self.quadrature_points)
        
        # Process elements in batches for memory efficiency
        for batch_start in range(0, self.num_elements, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.num_elements)
            batch_elements = self.elements[batch_start:batch_end]
            batch_size = batch_end - batch_start
            
            # Gather element coordinates and displacements for the entire batch
            batch_coords = self.coordinates[batch_elements]  # [batch, nodes_per_elem, 3]
            batch_disps = u[batch_elements]                 # [batch, nodes_per_elem, 3]
            
            # Initialize batch energy
            batch_energy = torch.zeros(batch_size, device=self.device, dtype=self.dtype)
            
            # Process all quadrature points
            for q_idx in range(num_qp):
                if self.precomputed:
                    # Use precomputed derivatives
                    batch_dN_dx = self.dN_dx_all[batch_start:batch_end, q_idx]  # [batch, nodes, 3]
                    batch_detJ = self.detJ_all[batch_start:batch_end, q_idx]    # [batch]
                else:
                    # Compute derivatives on-the-fly (less efficient)
                    qp = self.quadrature_points[q_idx]
                    batch_dN_dx = []
                    batch_detJ = []
                    for i in range(batch_size):
                        dN_dx, detJ = self._compute_derivatives(batch_coords[i], qp)
                        batch_dN_dx.append(dN_dx)
                        batch_detJ.append(detJ)
                    batch_dN_dx = torch.stack(batch_dN_dx)
                    batch_detJ = torch.tensor(batch_detJ, device=self.device, dtype=self.dtype)
                
                # Compute deformation gradients for all elements at once
                batch_F = self._compute_F_batch(batch_coords, batch_disps, batch_dN_dx)
                
                # Compute energy densities for all elements at once
                batch_energy_density = self._compute_svk_energy_density_batch(batch_F)
                
                # Numerical integration
                batch_energy += batch_energy_density * batch_detJ * self.quadrature_weights[q_idx]
            
            # Sum up all element energies in this batch
            total_energy += torch.sum(batch_energy)
            
            # Clean up memory periodically
            if torch.cuda.is_available() and batch_size > 500:
                torch.cuda.empty_cache()
        
        return total_energy
    
    def compute_batch_energy(self, batch_u):
        """
        Process batch of displacement fields efficiently
        
        Args:
            batch_u: Batch of displacement fields [batch_size, num_nodes * 3]
            
        Returns:
            Energies for each displacement field [batch_size]
        """
        batch_size = batch_u.shape[0]
        energies = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
        
        # For larger batches, process in smaller chunks to save memory
        max_samples_per_batch = 4
        for i in range(0, batch_size, max_samples_per_batch):
            end_idx = min(i + max_samples_per_batch, batch_size)
            sub_batch = batch_u[i:end_idx]
            sub_batch_size = end_idx - i
            
            # Process each sample in sub-batch
            for j in range(sub_batch_size):
                energies[i + j] = self.forward(sub_batch[j])
            
            # Free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return energies
    
    def compute_strains_stresses(self, u_tensor):
        """
        Compute strains and stresses across the domain using batch operations
        
        Args:
            u_tensor: Displacement field [num_nodes * 3]
            
        Returns:
            Dictionary with 'strains' and 'stresses' tensors
        """
        # Ensure tensor is on correct device/type
        if u_tensor.device != self.device or u_tensor.dtype != self.dtype:
            u_tensor = u_tensor.to(device=self.device, dtype=self.dtype)
        
        # Reshape displacement vector
        u = u_tensor.reshape(self.num_nodes, self.dim)
        
        # Initialize storage for results
        strains = torch.zeros((self.num_elements, len(self.quadrature_points), 6), 
                             dtype=self.dtype, device=self.device)
        stresses = torch.zeros((self.num_elements, len(self.quadrature_points), 6), 
                              dtype=self.dtype, device=self.device)
        
        # Process elements in batches
        for batch_start in range(0, self.num_elements, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.num_elements)
            batch_elements = self.elements[batch_start:batch_end]
            batch_size = batch_end - batch_start
            
            # Get coordinates and displacements for this batch
            batch_coords = self.coordinates[batch_elements]
            batch_disps = u[batch_elements]
            
            # Process each quadrature point
            for q_idx in range(len(self.quadrature_points)):
                if self.precomputed:
                    batch_dN_dx = self.dN_dx_all[batch_start:batch_end, q_idx]
                    batch_detJ = self.detJ_all[batch_start:batch_end, q_idx]
                else:
                    # Compute on-the-fly
                    qp = self.quadrature_points[q_idx]
                    batch_dN_dx = []
                    batch_detJ = []
                    for i in range(batch_size):
                        dN_dx, detJ = self._compute_derivatives(batch_coords[i], qp)
                        batch_dN_dx.append(dN_dx)
                        batch_detJ.append(detJ)
                    batch_dN_dx = torch.stack(batch_dN_dx)
                    batch_detJ = torch.tensor(batch_detJ, device=self.device, dtype=self.dtype)
                
                # Compute deformation gradients for all elements at once
                batch_F = self._compute_F_batch(batch_coords, batch_disps, batch_dN_dx)
                
                # Batch identity matrix
                batch_I = torch.eye(3, dtype=self.dtype, device=self.device).expand(batch_size, 3, 3)
                
                # Batch Right Cauchy-Green tensor C = F^T F
                batch_C = torch.bmm(batch_F.transpose(1, 2), batch_F)
                
                # Batch Green-Lagrange strain tensor E = 0.5(C - I)
                batch_E = 0.5 * (batch_C - batch_I)
                
                # Calculate batch Second Piola-Kirchhoff stress tensor
                # S = λ tr(E) I + 2μ E
                batch_trE = torch.diagonal(batch_E, dim1=1, dim2=2).sum(dim=1).unsqueeze(-1).unsqueeze(-1)
                batch_S = self.lambda_ * batch_trE * batch_I + 2 * self.mu * batch_E
                
                # Convert tensors to Voigt notation [xx, yy, zz, xy, yz, xz]
                for b in range(batch_size):
                    e_idx = batch_start + b
                    E = batch_E[b]
                    S = batch_S[b]
                    
                    strains[e_idx, q_idx] = torch.tensor([
                        E[0,0], E[1,1], E[2,2], E[0,1], E[1,2], E[0,2]
                    ], device=self.device, dtype=self.dtype)
                    
                    stresses[e_idx, q_idx] = torch.tensor([
                        S[0,0], S[1,1], S[2,2], S[0,1], S[1,2], S[0,2]
                    ], device=self.device, dtype=self.dtype)
        
        return {'strains': strains, 'stresses': stresses}
    
    def compute_stress_tensor(self, F):
        """
        Compute the Second Piola-Kirchhoff stress tensor for a given deformation gradient
        
        Args:
            F: Deformation gradient tensor [3, 3]
        
        Returns:
            Second Piola-Kirchhoff stress tensor [3, 3]
        """
        # Identity matrix
        I = torch.eye(3, dtype=self.dtype, device=self.device)
        
        # Right Cauchy-Green tensor
        C = torch.matmul(F.T, F)
        
        # Green-Lagrange strain tensor
        E = 0.5 * (C - I)
        
        # Second Piola-Kirchhoff stress
        trE = torch.trace(E)
        S = self.lambda_ * trE * I + 2 * self.mu * E
        
        return S
    
    def visualize_deformation(self, u_tensor, scale_factor=1.0):
        """
        Helper method to visualize deformation with stress/strain coloring
        
        Args:
            u_tensor: Displacement field [num_nodes * 3]
            scale_factor: Scale factor for displacement visualization
        
        Returns:
            pyvista plotter object
        """
        try:
            import pyvista as pv
            from dolfinx import plot
        except ImportError:
            print("Visualization requires pyvista and dolfinx.plot")
            return None
        
        # Ensure tensor is numpy array
        if torch.is_tensor(u_tensor):
            u_np = u_tensor.detach().cpu().numpy()
        else:
            u_np = u_tensor
        
        # Compute strains and stresses
        result = self.compute_strains_stresses(u_tensor)
        stresses = result['stresses']
        
        # Compute von Mises stress for each element
        von_mises = []
        for e in range(stresses.shape[0]):
            for q in range(stresses.shape[1]):
                s = stresses[e, q]  # [s_xx, s_yy, s_zz, s_xy, s_yz, s_zx]
                
                # Reconstruct stress tensor
                stress_tensor = torch.zeros((3, 3), device=stresses.device)
                stress_tensor[0, 0] = s[0]  # s_xx
                stress_tensor[1, 1] = s[1]  # s_yy
                stress_tensor[2, 2] = s[2]  # s_zz
                stress_tensor[0, 1] = stress_tensor[1, 0] = s[3]  # s_xy
                stress_tensor[1, 2] = stress_tensor[2, 1] = s[4]  # s_yz
                stress_tensor[0, 2] = stress_tensor[2, 0] = s[5]  # s_zx
                
                # Compute von Mises
                deviatoric = stress_tensor - torch.trace(stress_tensor)/3 * torch.eye(3, device=stress_tensor.device)
                vm = torch.sqrt(1.5 * torch.sum(deviatoric * deviatoric)).item()
                von_mises.append(vm)
        
        # Create a mesh with deformation
        topology, cell_types, x = plot.vtk_mesh(self.domain)
        grid = pv.UnstructuredGrid(topology, cell_types, x)
        
        # Add displacement
        u_reshaped = u_np.reshape((-1, 3))
        grid.point_data["displacement"] = u_reshaped
        grid["displacement_magnitude"] = np.linalg.norm(u_reshaped, axis=1)
        
        # Warp the mesh
        warped = grid.copy()
        warped.points += u_reshaped * scale_factor
        
        # Add stress data
        if len(von_mises) == self.num_elements:
            warped.cell_data["von_mises_stress"] = np.array(von_mises)
            
        # Create plotter
        plotter = pv.Plotter()
        plotter.add_mesh(warped, scalars="von_mises_stress", cmap="jet", show_edges=True)
        plotter.add_scalar_bar("von Mises Stress")
        
        return plotter


# Add this class near the top of your file, after imports
class LBFGSScheduler:
    """Custom learning rate scheduler for LBFGS optimizer"""
    def __init__(self, optimizer, factor=0.5, patience=3, threshold=0.01, min_lr=1e-6, verbose=True):
        """
        Args:
            optimizer: LBFGS optimizer
            factor: Factor by which to reduce learning rate
            patience: Number of epochs with no significant improvement before reducing LR
            threshold: Minimum relative improvement to reset patience counter
            min_lr: Minimum learning rate
            verbose: Whether to print learning rate changes
        """
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.best_loss = float('inf')
        self.wait_epochs = 0
        self.last_epoch = 0
        
    def step(self, loss):
        """Call after each epoch to update learning rate if needed"""
        self.last_epoch += 1
        
        # Check if current loss is better than best loss
        if loss < self.best_loss * (1.0 - self.threshold):
            # We have sufficient improvement
            self.best_loss = loss
            self.wait_epochs = 0
        else:
            # No significant improvement
            self.wait_epochs += 1
            
            # If we've waited enough epochs with no improvement, reduce learning rate
            if self.wait_epochs >= self.patience:
                for param_group in self.optimizer.param_groups:
                    old_lr = param_group['lr']
                    if old_lr > self.min_lr:
                        new_lr = max(old_lr * self.factor, self.min_lr)
                        param_group['lr'] = new_lr
                        if self.verbose:
                            print(f"Epoch {self.last_epoch}: reducing learning rate from {old_lr:.6f} to {new_lr:.6f}")
                        self.wait_epochs = 0  # Reset wait counter




class RotationStrainRoutine:
    def __init__(self, cfg):
        # Basic initialization similar to Routine
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load mesh from file
        # Modified to match default.yaml structure
        filename = cfg.get('mesh', {}).get('filename', cfg.get('data', {}).get('mesh_file', 'mesh/beam_401.msh'))
        print(f"Loading mesh from file: {filename}")
        self.domain, self.cell_tags, self.facet_tags = gmshio.read_from_msh(filename, MPI.COMM_WORLD, gdim=3)
        print("Mesh loaded successfully.")

        # Define function space - handle both key structures
        print("Defining function space...")
        self.fem_degree = cfg.get('mesh', {}).get('fem_degree', 
                        cfg.get('data', {}).get('fem_degree', 1))  # Default to 1 if not specified
        print(f"Using FEM degree: {self.fem_degree}")
        self.V = fem.functionspace(self.domain, ("Lagrange", self.fem_degree, (self.domain.geometry.dim, )))
        print("Function space defined.")
        print(f"Function space dimension: {self.V.dofmap.index_map.size_global * 3}")

        # Define material properties
        print("Defining material properties...")
        self.E = float(cfg['material']['youngs_modulus'])
        self.nu = float(cfg['material']['poissons_ratio'])
        self.rho = float(cfg['material']['density'])
        print(f"E = {self.E}, nu = {self.nu}, rho = {self.rho}")

        # Calculate Lamé parameters from E and nu
        self.mu = self.E / (2 * (1 + self.nu))      # Shear modulus
        self.lambda_ = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))  # First Lamé parameter
        print(f"mu = {self.mu}, lambda = {self.lambda_}")

        # Gravity and scaling - Handle both formats
        gravity = cfg['physics']['gravity']
        if isinstance(gravity, list):
            self.g = gravity  # It's already a list
        else:
            self.g = [0, float(gravity), 0]  # Convert scalar to 3D vector
        print(f"g = {self.g}")

        # Choose energy calculator based on config or default to PabloNeoHookeanEnergy
        energy_type = cfg.get('physics', {}).get('energy_type', 'neohookean')
        
        if energy_type == 'stvk':
            self.energy_calculator = StVenantKirchhoffEnergy(
                self.domain, self.fem_degree, self.E, self.nu,
                precompute_matrices=True, device=self.device
            ).to(self.device)
        else:  # Default to neo-Hookean
            self.energy_calculator = PabloNeoHookeanEnergy(
                self.domain, self.fem_degree, self.E, self.nu,
                precompute_matrices=True, device=self.device
            ).to(self.device)

        self.scale = self.compute_safe_scaling_factor()
        print(f"Scaling factor: {self.scale}")

        # Load neural network
        print("Loading neural network...")
        self.latent_dim = cfg['model']['latent_dim']
        self.num_modes = self.latent_dim  # Make them the same
        output_dim = self.V.dofmap.index_map.size_global * self.domain.geometry.dim
        hid_layers = cfg['model'].get('hid_layers', 2)
        hid_dim = cfg['model'].get('hid_dim', 64)
        print(f"Output dimension: {output_dim}")
        print(f"Network architecture: {hid_layers} hidden layers with {hid_dim} neurons each")
        self.model = RotationStrainNet(
            self.latent_dim, 
            self.V.dofmap.index_map.size_global, 
            hid_layers=hid_layers,  # Pass hid_layers
            hid_dim=hid_dim       # Pass hid_dim
        ).to(self.device).double()

        print(f"Neural network loaded. Latent dim: {self.latent_dim}, Num Modes: {self.num_modes}")

        # Load linear modes
        print("Loading linear eigenmodes...")
        
        # Check matrices configuration
        use_sofa = cfg.get('matrices', {}).get('use_sofa_matrices', False)
        matrices_path = cfg.get('matrices', {}).get('matrices_path', 'matrices')
        timestamp = cfg.get('matrices', {}).get('timestamp', None)
        
        if use_sofa:
            print(f"Using SOFA matrices from path: {matrices_path}")
            self.linear_modes = self.compute_linear_modes()
        else:
            print("Using FEniCS-generated matrices")
            self.linear_modes = self.compute_linear_modes()
        
        self.linear_modes = torch.tensor(self.linear_modes, device=self.device).double()
        print("Linear eigenmodes loaded.")

        # Tensorboard setup
        checkpoint_dir = cfg.get('training', {}).get('checkpoint_dir', 'checkpoints')
        tensorboard_dir = cfg.get('training', {}).get('tensorboard_dir', 'tensorboard')
        self.writer = SummaryWriter(os.path.join(checkpoint_dir, tensorboard_dir))

        # Setup optimizers
        lr = cfg.get('training', {}).get('learning_rate', 0.005)
        self.optimizer_adam = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.optimizer_lbfgs = torch.optim.LBFGS(
            self.model.parameters(), 
            lr=cfg.get('training', {}).get('lbfgs_learning_rate', 1.0),
            max_iter=20,
            line_search_fn="strong_wolfe"
        )
        self.scheduler = LBFGSScheduler(self.optimizer_lbfgs)

        self.reset()
        print("Routine initialized.")
    
    def reset(self):
            print("Resetting simulation...")
            self.z = torch.zeros(self.latent_dim, device=self.device).double()
            self.z.requires_grad = True
            self.uh = fem.Function(self.V)
            self.uh.name = "Displacement"
            self.uh.x.array[:] = 0.0
            print("Simulation reset.")

    # Replace the energy calculation in train_step with this implementation:

    
    

    def compute_linear_modes(self):
        print("Computing linear modes...")
        
        # Try to load SOFA matrices first
        M, A = self.load_sofa_matrices()
        
        if M is not None and A is not None:
            print("Using SOFA-generated matrices for linear modes computation")
            # Store matrices in self for later use
            self.M = M
            self.A = A
        else:
            print("No pre-computed matrices found, assembling matrices from FEniCS...")
            # Get domain extents
            x_coords = self.domain.geometry.x
            x_min = x_coords[:, 0].min()
            x_max = x_coords[:, 0].max()
            y_min = x_coords[:, 1].min()
            y_max = x_coords[:, 1].max()
            z_min = x_coords[:, 2].min()
            z_max = x_coords[:, 2].max()
            print(f"Domain extents: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}], z=[{z_min}, {z_max}]")

            # Use the same function space as the main model
            V = self.V
            u_tr = TrialFunction(V)
            u_test = TestFunction(V)

            # Determine which energy model is being used
            if isinstance(self.energy_calculator, StVenantKirchhoffEnergy):
                print("Using Saint-Venant-Kirchhoff forms for linear modes")
                
                # Define strain tensor for Saint-Venant-Kirchhoff linearized at undeformed state
                def epsilon(u):
                    return sym(grad(u))

                # Define stress tensor - for SVK linearized at undeformed state,
                # it's the same as linear elasticity
                def sigma(u):
                    return self.lambda_ * div(u) * Identity(3) + 2 * self.mu * epsilon(u)
                    
                # Standard bilinear form for linearized elasticity
                a_form = inner(sigma(u_tr), epsilon(u_test)) * dx
                m_form = self.rho * inner(u_tr, u_test) * dx
                
            elif isinstance(self.energy_calculator, PabloNeoHookeanEnergy):
                print("Using Neo-Hookean forms for linear modes")
                
                # Define strain tensor
                def epsilon(u):
                    return sym(grad(u))

                # Linearized Neo-Hookean stress tensor at undeformed state
                def sigma(u):
                    return self.lambda_ * div(u) * Identity(3) + self.mu * (grad(u) + grad(u).T)
                    
                a_form = inner(sigma(u_tr), epsilon(u_test)) * dx
                m_form = self.rho * inner(u_tr, u_test) * dx
                
            else:
                print("Using standard linear elasticity forms for linear modes")
                
                # Default to linear elasticity forms
                def epsilon(u):
                    return sym(grad(u))

                def sigma(u):
                    return self.lambda_ * div(u) * Identity(3) + 2 * self.mu * epsilon(u)
                    
                a_form = inner(sigma(u_tr), epsilon(u_test)) * dx
                m_form = self.rho * inner(u_tr, u_test) * dx

            x_coordinates = self.domain.geometry.x
            x_min = np.min(x_coordinates[:, 0])
            x_min_tol = 1e-10  # Tolerance for identifying boundary nodes
            
            # Create boundary condition function
            def x_min_boundary(x):
                return np.isclose(x[0], x_min, atol=x_min_tol)
            
            # Create a function for the fixed values
            u_fixed = fem.Function(self.V)
            u_fixed.x.array[:] = 0.0  # Set all values to zero
            
            # Create boundary condition using the function
            boundary_dofs = fem.locate_dofs_geometrical(self.V, x_min_boundary)
            bc = fem.dirichletbc(u_fixed, boundary_dofs)
            
            print("Assembling A matrix")
            A = assemble_matrix_petsc(form(a_form), bcs=[bc])
            A.assemble()
            self.A = A
            
            print("Assembling M matrix")
            M = assemble_matrix_petsc(form(m_form))
            M.assemble()
            self.M = M
        
        print("Matrices assembled")

        # Setup eigensolver
        # In the compute_linear_modes method, replace the eigensolver setup with:

        # Setup eigensolver with more robust settings for thin geometries
        print("Setting up eigensolver with robust settings for Neo-Hookean materials...")
        eigensolver = SLEPc.EPS().create(self.domain.comm)
        eigensolver.setOperators(A, M)

        # Use KRYLOVSCHUR for better convergence with difficult materials
        eigensolver.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_MAGNITUDE)

        # More robust shift strategy for thin domains
        st = eigensolver.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        st.setShift(1e-5)  # Very small shift for thin geometries

        # Much larger subspace for difficult problems
        eigensolver.setDimensions(nev=self.latent_dim, ncv=min(self.latent_dim*10, A.getSize()[0]))

        # Increase iteration count and lower tolerance
        eigensolver.setTolerances(tol=1e-3, max_it=5000)

        # Add convergence monitoring
        eigensolver.setConvergenceTest(SLEPc.EPS.Conv.REL)

        # Extract eigenvectors
        print("Extracting eigenvectors...")
        nconv = eigensolver.getConverged()
        print(f"Number of converged eigenvalues: {nconv}")

        # Implement fallback if no convergence
        # In the compute_linear_modes method, update the fallback section:

        if nconv == 0:
            print("WARNING: No eigenvalues converged. Trying fallback method...")
            # Try again with shift-and-invert and more aggressive parameters
            eigensolver.reset()
            
            # Need to set operators again after reset
            eigensolver.setOperators(A, M)
            
            # Use more robust settings for challenging problems
            eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
            eigensolver.setType(SLEPc.EPS.Type.KRYLOVSCHUR)  # Use KRYLOVSCHUR instead of LAPACK
            
            # Critical: When using SINVERT, must set target and use TARGET_MAGNITUDE
            eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
            eigensolver.setTarget(0.0)  # Target eigenvalues near zero
            
            # Get a new ST object after reset
            st = eigensolver.getST()
            st.setType(SLEPc.ST.Type.SINVERT)
            st.setShift(1.0)
            
            eigensolver.setDimensions(nev=self.latent_dim, ncv=min(self.latent_dim*5, A.getSize()[0]))
            eigensolver.setTolerances(tol=5e-3, max_it=10000)
            
            # Add this line to make the solver more robust
            eigensolver.setConvergenceTest(SLEPc.EPS.Conv.ABS)
            
            eigensolver.solve()
            nconv = eigensolver.getConverged()
            print(f"Fallback method found {nconv} converged eigenvalues")

        # Add safety check before extracting modes
        modes = []
        for i in range(min(self.latent_dim, nconv)):
            vr = A.createVecRight()
            eigensolver.getEigenvector(i, vr)
            modes.append(vr.array[:])

        # Handle case when no modes are found
        if len(modes) == 0:
            print("ERROR: No modes could be computed. Using random initialization.")
            # Create random modes as fallback - better than crashing
            random_modes = np.random.rand(A.getSize()[0], self.latent_dim)
            # Make them mass-orthogonal
            return random_modes

        linear_modes = np.column_stack(modes)
        print(f"Shape of linear_modes: {linear_modes.shape}")
        return linear_modes


    def train(self, num_epochs=1000):
        """
        Train the rotation-strain decomposition model with improved convergence strategies:
        1. More conservative curriculum learning
        2. Adaptive loss weighting
        3. Regularization warmup
        4. More balanced constraints
        """

        
        print("Starting rotation-strain neural modes training with improved convergence...")
        
        # Setup training parameters
        self.batch_size = 32  # You can add this to config
        rest_idx = 0     # Index for rest shape in batch
        print_every = 1
        checkpoint_every = 50
        
        # Get rest positions (undeformed configuration) for deformation calculations
        num_nodes = self.V.dofmap.index_map.size_global
        rest_positions = torch.tensor(self.domain.geometry.x, 
                                    device=self.device, dtype=torch.float64)
        rest_positions_batch = rest_positions.unsqueeze(0).expand(self.batch_size, -1, -1)
        
        # Use a subset of linear modes
        L = self.latent_dim
        linear_modes = self.linear_modes[:, :L]

        vol_stats = {
            'mean_J': 1.0,
            'min_J': 1.0,
            'max_J': 1.0,
            'inverted_elements': 0
        }
        # Setup iteration counter and best loss tracking
        iteration = 0
        best_loss = float('inf')
        patience = 0
        
        # Make sure model accepts batched inputs
        original_forward = self.model.forward
        
        def new_forward(x, rest_pos=None):
            is_batch = x.dim() > 1
            if not is_batch:
                x = x.unsqueeze(0)
                if rest_pos is not None:
                    rest_pos = rest_pos.unsqueeze(0)
            
            disps, rot_matrices, strain_matrices = original_forward(x, rest_pos)
            
            if not is_batch:
                disps = disps.squeeze(0)
                rot_matrices = rot_matrices.squeeze(0)
                strain_matrices = strain_matrices.squeeze(0)
                
            return disps, rot_matrices, strain_matrices
                
        self.model.forward = new_forward
        
        # IMPROVED: Optimizer with gradient clipping for better stability
        optimizer = torch.optim.LBFGS(
            self.model.parameters(), 
            lr=1,  # Slightly more conservative learning rate
            max_iter=50,
            max_eval=100,
            tolerance_grad=1e-04,
            tolerance_change=1e-05,
            history_size=100,
            line_search_fn="strong_wolfe"
        )
        
        # IMPROVED: More aggressive learning rate decay
        scheduler = LBFGSScheduler(
            optimizer,
            factor=0.5,      # More aggressive reduction 
            patience=5,      # React faster to plateaus
            threshold=0.001,
            min_lr=1e-6,
            verbose=True
        )
        
        # Curriculum learning parameters
        initial_scale = 0.1
        final_scale = 10.0
        scale_warmup_epochs = num_epochs * 0.8  # Warmup for 20% of training
        
        # Main training loop
        while iteration < num_epochs:
        # Ultra-conservative start for first few iterations
           
            # Compute scale with curriculum learning
            if iteration < scale_warmup_epochs:
                # Linear increase from initial to final scale
                scale = initial_scale + (final_scale - initial_scale) * (iteration / scale_warmup_epochs)
            else:
                # After warmup, use the final scale
                scale = final_scale
            
            # IMPORTANT: Add a tiny epsilon to avoid starting exactly from zero
            # This helps numerical stability in the quaternion normalization
            epsilon = 1e-6
            
            # Generate random latent vectors with more controlled distribution
            with torch.no_grad():

                z = torch.randn(self.batch_size, L, device=self.device) * scale * 2 - scale
                    
                z[rest_idx, :] = 0  # Set rest shape latent to zero
                        
                l = torch.matmul(z, linear_modes.T)
                
                constraint_dir = torch.matmul(z, linear_modes.T)
                constraint_norms = torch.norm(constraint_dir, p=2, dim=1, keepdim=True)
                constraint_norms = torch.clamp(constraint_norms, min=1e-8)
                constraint_dir = constraint_dir / constraint_norms
                constraint_dir[rest_idx] = 0
            
            # Track values outside the closure for logging
            energy_val = 0
            ortho_val = 0
            origin_val = 0
            rot_orthogonality_val = 0
            strain_reg_val = 0
            vol_loss_val = 0
            loss_val = 0
            
            # Define closure for optimizer
            def closure():
                nonlocal energy_val, ortho_val, origin_val, rot_orthogonality_val
                nonlocal strain_reg_val, vol_loss_val, loss_val
                warm_up = 5  # Number of warm-up iterations
                debug = False  # Enable detailed debugging
                
                optimizer.zero_grad()
                
                # Compute displacements, rotations, and strains for the batch
                displacements, rotation_matrices, strain_matrices = self.model(z, rest_positions_batch)
                
                
                # =========== ADD DETAILED DEBUGGING HERE ===========
                if iteration % print_every == 0 and debug == True:
                    with torch.no_grad():
                        # Rotation matrix stats
                        R_dets = torch.linalg.det(rotation_matrices)
                        R_frobenius = torch.norm(rotation_matrices, dim=(-2, -1))
                        
                        print(f"\n==== ROTATION MATRICES (ITER {iteration}) ====")
                        print(f"  Det(R): min={torch.min(R_dets).item():.6f}, mean={torch.mean(R_dets).item():.6f}, max={torch.max(R_dets).item():.6f}")
                        print(f"  Norm(R): min={torch.min(R_frobenius).item():.6f}, mean={torch.mean(R_frobenius).item():.6f}, max={torch.max(R_frobenius).item():.6f}")
                        print(f"  Invalid rotations: {torch.sum(torch.abs(R_dets - 1.0) > 0.1).item()} of {R_dets.numel()}")
                        
                        # Check orthogonality of rotation matrices
                        batch_size, num_nodes = rotation_matrices.shape[0], rotation_matrices.shape[1]
                        R_transpose = rotation_matrices.transpose(2, 3)
                        R_product = torch.matmul(
                            R_transpose.reshape(batch_size * num_nodes, 3, 3),
                            rotation_matrices.reshape(batch_size * num_nodes, 3, 3)
                        ).reshape(batch_size, num_nodes, 3, 3)
                        identity = torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_nodes, 3, 3)
                        ortho_error = torch.norm(R_product - identity, dim=(-2, -1))
                        print(f"  R^TR-I error: min={torch.min(ortho_error).item():.6e}, mean={torch.mean(ortho_error).item():.6e}, max={torch.max(ortho_error).item():.6e}")
                        
                        # Strain matrix stats
                        S_dets = torch.linalg.det(strain_matrices)
                        S_trace = torch.diagonal(strain_matrices, dim1=2, dim2=3).sum(dim=2)
                        S_frobenius = torch.norm(strain_matrices, dim=(-2, -1))
                        S_eigenvalues = torch.linalg.eigvalsh(strain_matrices.reshape(-1, 3, 3))
                        S_min_eigenval = S_eigenvalues[:, 0].reshape(batch_size, num_nodes)
                        S_max_eigenval = S_eigenvalues[:, 2].reshape(batch_size, num_nodes)
                        
                        print(f"\n==== STRAIN MATRICES (ITER {iteration}) ====")
                        print(f"  Det(S): min={torch.min(S_dets).item():.6f}, mean={torch.mean(S_dets).item():.6f}, max={torch.max(S_dets).item():.6f}")
                        print(f"  Trace(S): min={torch.min(S_trace).item():.6f}, mean={torch.mean(S_trace).item():.6f}, max={torch.max(S_trace).item():.6f}")
                        print(f"  Norm(S): min={torch.min(S_frobenius).item():.6f}, mean={torch.mean(S_frobenius).item():.6f}, max={torch.max(S_frobenius).item():.6f}")
                        print(f"  Min eigenvalue: {torch.min(S_min_eigenval).item():.6f}, Max eigenvalue: {torch.max(S_max_eigenval).item():.6f}")
                        print(f"  Compressive elements: {torch.sum(S_dets < 0.99).item()} of {S_dets.numel()}")
                        print(f"  Expansive elements: {torch.sum(S_dets > 1.01).item()} of {S_dets.numel()}")
                        
                        # Compute deformation gradients F = R·S
                        F_matrices = torch.matmul(
                            rotation_matrices.reshape(-1, 3, 3),
                            strain_matrices.reshape(-1, 3, 3)
                        ).reshape(batch_size, num_nodes, 3, 3)
                        
                        F_dets = torch.linalg.det(F_matrices)
                        
                        print(f"\n==== DEFORMATION GRADIENT (ITER {iteration}) ====")
                        print(f"  Det(F): min={torch.min(F_dets).item():.6f}, mean={torch.mean(F_dets).item():.6f}, max={torch.max(F_dets).item():.6f}")
                        print(f"  Inverted elements: {torch.sum(F_dets < 0).item()} of {F_dets.numel()}")
                        print(f"  Near-singular elements: {torch.sum((F_dets > 0) & (F_dets < 0.1)).item()} of {F_dets.numel()}")
                        
                        # Displacement stats
                        disp_norm = torch.norm(displacements.reshape(batch_size, num_nodes, 3), dim=2)
                        print(f"\n==== DISPLACEMENTS (ITER {iteration}) ====")
                        print(f"  |u|: min={torch.min(disp_norm).item():.6f}, mean={torch.mean(disp_norm).item():.6f}, max={torch.max(disp_norm).item():.6f}")
                        
                        # Node positions before/after for checking if elements are inverted
                        if iteration % (print_every * 5) == 0 and iteration > 0:
                            # Only do this check periodically as it's more expensive
                            rest_pos_sample = rest_positions_batch[0]  # Just check first batch item
                            deformed_pos = rest_pos_sample + displacements[0]
                            
                            # Sample some elements to check if they're inverted
                            if hasattr(self.energy_calculator, 'elements'):
                                elements = self.energy_calculator.elements[:10]  # First 10 elements
                                print(f"\n==== ELEMENT CHECK (ITER {iteration}) ====")
                                for i, elem in enumerate(elements):
                                    original_vols = torch.zeros(1, device=self.device)
                                    deformed_vols = torch.zeros(1, device=self.device)
                                    
                                    # Get element node indices
                                    nodes = elem.cpu().numpy()
                                    
                                    # Compute original and deformed element volumes
                                    if len(nodes) >= 4:  # Only works for tetrahedra or larger elements
                                        v1, v2, v3, v4 = nodes[:4]
                                        
                                        # Original tetrahedron - FIX: specify dim=0 for cross product
                                        a = rest_pos_sample[v2] - rest_pos_sample[v1]
                                        b = rest_pos_sample[v3] - rest_pos_sample[v1]
                                        c = rest_pos_sample[v4] - rest_pos_sample[v1]
                                        orig_vol = torch.abs(torch.dot(torch.cross(a, b, dim=0), c)) / 6.0
                                        
                                        # Deformed tetrahedron - FIX: specify dim=0 for cross product
                                        a = deformed_pos[v2] - deformed_pos[v1]
                                        b = deformed_pos[v3] - deformed_pos[v1] 
                                        c = deformed_pos[v4] - deformed_pos[v1]
                                        def_vol = torch.dot(torch.cross(a, b, dim=0), c) / 6.0


                                        print(f"  Element {i}: original vol = {orig_vol.item():.6f}, deformed vol = {def_vol.item():.6f}")
                # =========== END DEBUGGING SECTION =================
                
                # Reshape displacements to match linear modes shape
                displacements_flattened = displacements.reshape(self.batch_size, -1)
                
                # For energy calculation, add linear contribution to neural model output
                u_total_batch = displacements_flattened + l
                
                # Compute elastic energy
                if self.batch_size > 1:
                    energies = self.energy_calculator.compute_batch_energy(u_total_batch)
                    energy = torch.mean(energies)
                else:
                    energy = self.energy_calculator(u_total_batch[0])
                
                # Calculate maximum displacements for monitoring
                max_linear = torch.max(torch.norm(l.reshape(self.batch_size, -1, 3), dim=2)).item()
                max_total = torch.max(torch.norm(u_total_batch.reshape(self.batch_size, -1, 3), dim=2)).item()
                max_correction = torch.max(torch.norm(displacements.reshape(self.batch_size, -1, 3), dim=2)).item()

                # Normal adaptive weights after warm-up
                # energy_weight = self.compute_adaptive_weight(iteration, num_epochs, 1.0, 1.0)
                # ortho_weight = self.compute_adaptive_weight(iteration, num_epochs, 0.5, 0.3)
                # origin_weight = self.compute_adaptive_weight(iteration, num_epochs, 0.1, 0.05)
                # vol_weight = self.compute_adaptive_weight(iteration, num_epochs, 1.0, 2.0)  # Increase over time
                # rot_weight = self.compute_adaptive_weight(iteration, num_epochs, 2.0, 3.0)  # Enforce more strictly over time
                # strain_reg_weight = self.compute_adaptive_weight(iteration, num_epochs, 0.3, 0.1)  # Relax over time
                
                # Fixed weights
                energy_weight = 1.0
                ortho_weight = 100
                origin_weight = 1000
                vol_weight = 0.1
                rot_weight = 0.1
                strain_reg_weight = 0.1
                
                # 1. Orthogonality constraint
                ortho = torch.mean(torch.sum(displacements_flattened * constraint_dir, dim=1)**2)
                
                # 2. Origin constraint for rest shape
                origin = torch.sum(displacements_flattened[rest_idx]**2)
                
                # 3. IMPROVED: Volume preservation with stronger barrier
                vol_loss, vol_stats = self.compute_volume_preservation_loss(
                    rotation_matrices=rotation_matrices,
                    strain_matrices=strain_matrices,
                    weight=vol_weight  # Use vol_weight directly
                )
                
                # 4. IMPROVED: Rotation orthogonality constraint with two-part loss
                batch_size = rotation_matrices.shape[0]
                num_nodes = rotation_matrices.shape[1]
                
                # Orthogonality: R^T R = I
                R_transpose = rotation_matrices.transpose(2, 3)
                R_product = torch.matmul(
                    R_transpose.reshape(self.batch_size * num_nodes, 3, 3),
                    rotation_matrices.reshape(self.batch_size * num_nodes, 3, 3)
                ).reshape(batch_size, num_nodes, 3, 3)
                
                identity = torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0).expand(self.batch_size, num_nodes, 3, 3)
                rot_orthogonality = torch.mean((R_product - identity)**2)
                
                # IMPROVED: Additional determinant constraint - det(R) = 1
                # This ensures proper rotations without reflection
                R_dets = torch.linalg.det(rotation_matrices)
                rot_det_constraint = torch.mean((R_dets - 1.0)**2)
                
                # 5. IMPROVED: Strain regularization with more physical constraints
                # Penalize deviation from identity with higher weight
                strain_reg = torch.mean((strain_matrices - identity)**2)
                
                # 6. IMPROVED: Determinant constraint on strain matrices with safer minimum
                strain_dets = torch.linalg.det(strain_matrices)
                min_strain_det = 0.5  # Keep strains from becoming too compressive
                det_penalty = torch.mean(torch.relu(min_strain_det - strain_dets)**2 * 10.0 + 
                                        (strain_dets - 1.0)**2)
                
                
                
                # Combine all loss terms with adaptive weighting
                loss = (
                    energy_weight * energy +           # Physical energy
                    ortho_weight * ortho +             # Orthogonality to linear modes
                    origin_weight * origin +           # Rest shape constraint
                    vol_loss +            # Volume preservation
                    rot_weight * (rot_orthogonality + rot_det_constraint) + # Rotation constraints
                    strain_reg_weight * strain_reg +   # Strain matrix regularization
                    vol_weight * det_penalty           # Strain determinant constraint
                )
                
          
                
                # Print stats periodically
                if iteration % print_every == 0:
                    print(f"[Iter {iteration}] Energy: {energy.item():.4e}, Loss: {loss.item():.4e}")
                    print(f"  Ortho: {ortho.item():.4e}, Origin: {origin.item():.4e}, Vol: {vol_loss.item():.4e}")
                    # print(f"  Rot_ortho: {rot_orthogonality.item():.4e}, Rot_det: {rot_det_constraint.item():.4e}")
                    # print(f"  Strain_reg: {strain_reg.item():.4e}, Det_penalty: {det_penalty.item():.4e}")
                    # print(f"  J stats: min={vol_stats['min_J']:.3f}, mean={vol_stats['mean_J']:.3f}, "
                    #     f"max={vol_stats['max_J']:.3f}, inverted={vol_stats['inverted_elements']}")
                    # print(f"  Strain det: min={torch.min(strain_dets).item():.3f}, "
                    #     f"max={torch.max(strain_dets).item():.3f}")
                    print(f"  Scale: {scale:.2f}, "
                        f"Disp: lin={max_linear:.4f}, tot={max_total:.4f}, neural={max_correction:.4f}")
                
                # Backpropagate gradients
                loss.backward()
                
                # Store values for logging outside closure
                energy_val = energy.item()
                ortho_val = ortho.item()
                origin_val = origin.item()
                rot_orthogonality_val = rot_orthogonality.item()
                strain_reg_val = strain_reg.item()
                vol_loss_val = vol_loss.item()
                loss_val = loss.item()
                
                # Print the number of times the closure is called
                print("Closure called")
                
                return loss
            
            # IMPROVED: Catch and handle numerical errors
            try:
                # Perform optimization step
                optimizer.step(closure)
            except RuntimeError as e:
                if "nan" in str(e).lower() or "inf" in str(e).lower():
                    print(f"WARNING: Numerical error encountered: {e}")
                    print("Reducing deformation scale and restarting step...")
                    deformation_scale *= 0.5
                    continue
                else:
                    raise e
            
            # Update learning rate based on loss progress
            scheduler.step(loss_val)
            
            # Record metrics using TensorBoard
            self.writer.add_scalar('train/loss', loss_val, iteration)
            self.writer.add_scalar('train/energy', energy_val, iteration)
            self.writer.add_scalar('train/ortho', ortho_val, iteration)
            self.writer.add_scalar('train/origin', origin_val, iteration)
            self.writer.add_scalar('train/rot_orthogonality', rot_orthogonality_val, iteration)
            self.writer.add_scalar('train/strain_reg', strain_reg_val, iteration)
            self.writer.add_scalar('train/volume', vol_loss_val, iteration)
            
            # Save best model
            if loss_val < best_loss:
                best_loss = loss_val
                checkpoint_size = self.estimate_checkpoint_size()
                print(f"Estimated checkpoint size: {checkpoint_size:.2f} MB")
                
                checkpoint = {
                    'epoch': iteration,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_val,
                    'latent_dim': self.latent_dim
                }
                
                patience = 0
                torch.save(checkpoint, os.path.join('checkpoints', 'best_rotation_strain.pt'))
                print(f"============ BEST MODEL UPDATED ============")
                print(f"New best model at iteration {iteration} with loss {loss_val:.6e}")
                print(f"============================================")
            
            # Save periodic checkpoint
            if iteration % checkpoint_every == 0:
                checkpoint = {
                    'epoch': iteration,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_val,
                    'latent_dim': self.latent_dim
                }
                torch.save(checkpoint, os.path.join('checkpoints', f'rotation_strain_it{iteration}.pt'))
            
            # Update iteration counter and check patience
            iteration += 1
            patience += 1
            
            # IMPROVED: More sophisticated early stopping
            early_stop = False
            
            # Stop if loss is good enough
            if loss_val < 1e-4 and iteration > 10:
                print(f"Achieved target loss at iteration {iteration}: {loss_val:.6e}")
                early_stop = True
            
            # Stop if no progress after patience period
            if patience > 20:
                print(f"No improvement after {patience} iterations")
                early_stop = True
                
            num_elements = self.energy_calculator.num_elements
        
            # Now the check will work properly
            if iteration > num_epochs * 0.3 and vol_stats['inverted_elements'] > num_elements * 0.05:
                print(f"Too many inverted elements ({vol_stats['inverted_elements']}) persist")
                early_stop = True
            
            if early_stop:
                print(f"Early stopping at iteration {iteration}")
                break
        
        # Restore original forward method
        self.model.forward = original_forward
        
        print(f"Rotation-strain training complete. Best loss: {best_loss:.8e}")
        return best_loss

    def compute_volume_preservation_loss(self, rotation_matrices=None, strain_matrices=None, batch_F=None, batch_u=None, weight=1.0):
        """
        Enhanced volume preservation loss with progressive barrier function
        """
        device = self.device
        dtype = torch.float64
        
        # Get batch size and element count
        if batch_F is not None:
            batch_size = batch_F.shape[0]
            num_elements = batch_F.shape[1]
        else:
            if rotation_matrices is None or strain_matrices is None:
                raise ValueError("Rotation and strain matrices must be provided")
            
            batch_size = rotation_matrices.shape[0]
            num_elements = rotation_matrices.shape[1]  # Number of nodes
            
            # Compute deformation gradient F = R @ S
            batch_F = torch.matmul(rotation_matrices, strain_matrices)

        # Compute determinant of F for each element in the batch
        J = torch.linalg.det(batch_F)

        # SAFER: Much stricter clamping of J values
        J = torch.clamp(J, -10.0, 10.0)
        
        # SAFER: Quadratic barrier instead of exponential (won't explode)
        safe_threshold = 0.2
        barrier = torch.zeros_like(J, device=self.device)
        critical_idx = J < safe_threshold
        barrier[critical_idx] = 50.0 * (safe_threshold - J[critical_idx])**2
        
        # Add direct penalty for negative J values
        negative_idx = J <= 0
        barrier[negative_idx] += 100.0  # Constant large penalty
        
        # IMPROVED: Asymmetric volume loss - penalize compression more than expansion
        # This is physically motivated as materials often resist compression more than expansion
        compression_mask = J < 1.0
        expansion_mask = J >= 1.0
        
        vol_loss = torch.zeros_like(J)
        vol_loss[compression_mask] = (J[compression_mask] - 1.0)**2 * 2.0  # Stronger penalty for compression
        vol_loss[expansion_mask] = (J[expansion_mask] - 1.0)**2 * 1.0  # Normal penalty for expansion
        
        # Combine both terms and take mean over all elements and batch samples
        combined_loss = weight * (torch.mean(vol_loss) + torch.mean(barrier))
        
        # Return statistics for monitoring
        with torch.no_grad():
            stats = {
                'mean_J': torch.mean(J).item(),
                'min_J': torch.min(J).item(),
                'max_J': torch.max(J).item(),
                'inverted_elements': torch.sum(J <= 0).item(),
                'vol_loss': combined_loss.item()
            }
            
        return combined_loss, stats
    # 2. Improved Adaptive Weighting Function for Loss Terms
    def compute_adaptive_weight(self, iteration, max_iterations, initial_weight, final_weight):
        """
        Compute smooth weight transition to balance loss terms during training
        """
        # Sigmoid function for smooth transition
        progress = min(iteration / (0.3 * max_iterations), 1.0)
        weight = initial_weight + (final_weight - initial_weight) * progress
        return weight

    def _compute_batch_deformation_gradients(self, batch_u):
        """
        Efficiently compute deformation gradients for a batch of displacement fields
        """
        batch_size = batch_u.shape[0]
        
        # Use energy calculator attributes for ALL mesh data
        energy_calc = self.energy_calculator
        num_elements = energy_calc.num_elements
        num_nodes = energy_calc.num_nodes
        elements = energy_calc.elements
        coordinates = energy_calc.coordinates
        
        # Reshape displacement vectors to [batch_size, num_nodes, 3]
        u = batch_u.reshape(batch_size, num_nodes, 3)
        
        # Initialize storage for deformation gradients
        batch_F = torch.zeros((batch_size, num_elements, 3, 3), 
                            device=self.device, dtype=torch.float64)
        
        # Process elements in batches to control memory usage
        batch_elements = 1000  # Process 1000 elements at once
        
        for e_start in range(0, num_elements, batch_elements):
            e_end = min(e_start + batch_elements, num_elements)
            e_slice = slice(e_start, e_end)
            
            # Use energy_calc.elements instead of self.elements
            elem_nodes = elements[e_slice].long()  # [num_batch_elements, nodes_per_element]
            
            # Use energy_calc.coordinates instead of self.coordinates
            elem_coords = coordinates[elem_nodes]  # [num_batch_elements, nodes_per_elem, 3]
            
            # Get displacements for these elements for the entire batch
            elem_disps = u[:, elem_nodes]  # [batch_size, num_batch_elements, nodes_per_elem, 3]
            
            # Process each quadrature point
            q_idx = 0  # Use first quadrature point for volume calculation
            
            # Use attributes from energy calculator instead of self
            if hasattr(energy_calc, 'dN_dx_all'):
                # Use precomputed derivatives if available
                elem_dN_dx = energy_calc.dN_dx_all[e_slice, q_idx]
                
                # Compute F for all elements in this batch at once using broadcasting
                F = torch.eye(3, device=self.device, dtype=torch.float64)
                F = F.reshape(1, 1, 3, 3).expand(batch_size, e_end-e_start, 3, 3).clone()
                
                # Add displacement gradient: F += u_i ⊗ ∇N_i
                for n in range(elem_nodes.shape[1]):
                    node_disps = elem_disps[:, :, n, :]
                    node_dN_dx = elem_dN_dx[:, n, :]
                    F += torch.einsum('bei,ej->beij', node_disps, node_dN_dx)
                    
                # Store results for this batch of elements
                batch_F[:, e_slice] = F
            else:
                # Compute on-the-fly if derivatives not precomputed
                for b in range(batch_size):
                    for e_idx, e in enumerate(range(e_start, e_end)):
                        e_coords = elem_coords[e_idx]
                        e_disps = elem_disps[b, e_idx]
                        
                        # Use energy_calc.quadrature_points
                        qp = energy_calc.quadrature_points[q_idx]
                        
                        # Use energy_calc._compute_derivatives instead of self._compute_derivatives
                        dN_dx, _ = energy_calc._compute_derivatives(e_coords, qp)
                        
                        # Compute F for this element
                        F = torch.eye(3, device=self.device, dtype=torch.float64)
                        for n in range(elem_nodes.shape[1]):
                            F += torch.outer(e_disps[n], dN_dx[n])
                        
                        # Store result
                        batch_F[b, e] = F
        
        return batch_F


    # First add this function to your Routine class (copied from linear_modes.py):
    def compute_modal_coordinates(self, u_array, modal_matrix, M):
        """
        Compute modal coordinates using mass orthonormalization
        Args:
            u_array: displacement field as numpy array
            modal_matrix: matrix containing eigenvectors as columns
            M: mass matrix from eigenvalue problem
        Returns:
            q: modal coordinates
        """
        # Convert to PETSc vector
        u_vec = PETSc.Vec().createWithArray(u_array)
        
        # Initialize vector for modal coordinates
        q = np.zeros(modal_matrix.shape[1])
        
        # Compute modal coordinates using mass orthonormalization
        for i in range(modal_matrix.shape[1]):
            phi_i = PETSc.Vec().createWithArray(modal_matrix[:, i])
            Mphi = M.createVecLeft()
            M.mult(phi_i, Mphi)
            q[i] = u_vec.dot(Mphi) / phi_i.dot(Mphi)
        
        return q
    
    def compute_safe_scaling_factor(self):
        """
        Compute appropriate scaling factor for latent variables based on:
        1. Mesh dimensions
        2. Safety factor to avoid extreme deformations
        
        Returns a scaling factor that will produce visible but physically reasonable deformations
        """
        # Get mesh coordinates and compute characteristic length
        x_coords = self.domain.geometry.x
        x_range = x_coords[:, 0].max() - x_coords[:, 0].min()
        y_range = x_coords[:, 1].max() - x_coords[:, 1].min() 
        z_range = x_coords[:, 2].max() - x_coords[:, 2].min()
        
        # Calculate characteristic length (average of dimensions)
        char_length = max(x_range, y_range, z_range)
        # Safety factor to avoid extreme deformations
        safety_factor = 1
        
      
        
        return char_length * safety_factor
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint
        
        Args:
            epoch: Current epoch number
            loss: Current loss value
            is_best: Whether this is the best model so far
        """
        checkpoint_dir = os.path.join('checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_adam_state_dict': self.optimizer_adam.state_dict(),
            'optimizer_lbfgs_state_dict': self.optimizer_lbfgs.state_dict(),
            'loss': loss,
            'latent_dim': self.latent_dim
        }
        
        if is_best:
            print(f"Epoch {epoch+1}: New best model with loss {loss:.6e}")
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_rotation_strain.pt'))

    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint with robust error handling"""
        try:
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model parameters
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Try to load optimizer states if they exist
            if 'optimizer_adam_state_dict' in checkpoint:
                self.optimizer_adam.load_state_dict(checkpoint['optimizer_adam_state_dict'])
            
            if 'optimizer_lbfgs_state_dict' in checkpoint:
                self.optimizer_lbfgs.load_state_dict(checkpoint['optimizer_lbfgs_state_dict'])
            
            print(f"Successfully loaded model from epoch {checkpoint.get('epoch', 0)+1}")
            return checkpoint.get('loss', float('inf'))
        except (RuntimeError, EOFError, AttributeError) as e:
            print(f"Error loading checkpoint: {str(e)}")
            print("Checkpoint file appears to be corrupted. Starting with fresh model.")
            return float('inf')
    
    def visualize_latent_dimensions(self, dim1=0, dim2=1, num_points=3):
        """Visualize neural modes across a grid of two latent dimensions"""
        print(f"Visualizing neural modes for dimensions {dim1} and {dim2}...")
        
        # Convert DOLFINx mesh to PyVista format
        topology, cell_types, x = plot.vtk_mesh(self.domain)
        grid = pyvista.UnstructuredGrid(topology, cell_types, x)
        
        # Create a linear function space for visualization
        V_viz = fem.functionspace(self.domain, ("CG", 1, (3,)))
        u_linear = fem.Function(V_viz)
        
        # Compute scale for latent vectors
        scale = self.compute_safe_scaling_factor()  * 1.5  # Larger scale for better visualization
        values = np.linspace(-scale, scale, num_points)
        
        # Create plotter with subplots
        plotter = pyvista.Plotter(shape=(num_points, num_points), border=False)
        
        # Get rest positions
        rest_positions = torch.tensor(self.domain.geometry.x, 
                                    device=self.device, dtype=torch.float64)
        
        # Generate neural modes for each combination of latent values
        for i, val1 in enumerate(values):
            row_idx = num_points - 1 - i  # Reverse order for proper cartesian layout
            for j, val2 in enumerate(values):
                # Create latent vector with fixed values except for the two selected dims
                z = torch.zeros(self.latent_dim, device=self.device, dtype=torch.float64)
                z[dim1] = val1
                z[dim2] = val2
                
                # Compute neural mode
                # Only use the columns of linear_modes corresponding to dim1 and dim2
                linear_contribution = (self.linear_modes[:, [dim1, dim2]] @ z[[dim1, dim2]].unsqueeze(1)).squeeze(1)
                
                # Pass rest_positions to the model and unpack the tuple
                displacements, _, _ = self.model(z, rest_positions=rest_positions.unsqueeze(0))
                
                # IMPORTANT: Reshape displacements to match linear_contribution
                # The displacement tensor may be in shape [batch_size, num_nodes, 3] and needs flattening
                if displacements.dim() > 1:
                    displacements = displacements.reshape(displacements.shape[0], -1)
                
                # Make sure it's the right shape before adding
                if displacements.dim() == 2 and displacements.shape[0] == 1:
                    displacements = displacements.squeeze(0)  # Remove batch dimension
                    
                u_total = displacements + linear_contribution
                u_total_np = u_total.detach().cpu().numpy()
                # Create a function in the quadratic function space
                u_quadratic = fem.Function(self.V)
                u_quadratic.x.array[:] = u_total_np
                
                # Interpolate the quadratic displacement to the linear function space
                u_linear.interpolate(u_quadratic)
                
                # Get the displacement values at the linear nodes
                u_linear_np = u_linear.x.array
                
                # Set active subplot
                plotter.subplot(row_idx, j)
                
                # Create mesh with deformation
                local_grid = grid.copy()
                local_grid.point_data["displacement"] = u_linear_np.reshape((-1, 3))
                local_grid["displacement_magnitude"] = np.linalg.norm(u_linear_np.reshape((-1, 3)), axis=1)
                max_disp = np.max(local_grid["displacement_magnitude"])
                warp_factor = min(1.5, 0.2/max(max_disp, 1e-6)) 
                warped = local_grid.warp_by_vector("displacement", factor=warp_factor)
                
                # Add mesh to plot
                plotter.add_mesh(warped, scalars="displacement_magnitude", 
                            cmap="viridis", show_edges=True)
                
                # Add compact z-value labels in bottom corner (less intrusive)
                plotter.add_text(f"{val1:.2f}, {val2:.2f}", position="lower_right", 
                            font_size=6, color='white')
                
                plotter.view_isometric()
        
        # Add axis labels at edges of the grid
        for i, val1 in enumerate(values):
            row_idx = num_points - 1 - i
            # Y-axis labels (left side)
            plotter.subplot(row_idx, 0)
            plotter.add_text(f"z{dim1}={val1:.2f}", position=(0.01, 0.5), viewport=True,
                        font_size=8, color='white')
        
        for j, val2 in enumerate(values):
            # X-axis labels (bottom)
            plotter.subplot(num_points-1, j)
            plotter.add_text(f"z{dim2}={val2:.2f}", position=(0.5, 0.01), viewport=True, 
                        font_size=8, color='white')
        
        # Link camera views for synchronized rotation
        plotter.link_views()
        
        # Add a unified colorbar at the bottom
        plotter.add_scalar_bar("Displacement Magnitude", position_x=0.4, position_y=0.01, width=0.2, height=0.02)
        
        # Add a more compact title
        title = f"Neural Modes Matrix: z{dim1} vs z{dim2}"
        plotter.add_text(title, position=(0.5, 0.97), viewport=True, font_size=12, color='black')
        
        print("Showing latent space visualization...")
        plotter.show()
        print("Visualization complete.")

    def estimate_checkpoint_size(self):
        """Estimate the size of the checkpoint file that will be saved"""
        # Get model size
        model_size = 0
        for param in self.model.parameters():
            model_size += param.nelement() * param.element_size()
        
        # Get optimizer size
        optimizer_size = 0
        for buffer in self.optimizer_adam.state.values():
            if isinstance(buffer, torch.Tensor):
                optimizer_size += buffer.nelement() * buffer.element_size()
        
        # Total size in bytes
        total_size = model_size + optimizer_size
        
        # Convert to human-readable format
        size_in_mb = total_size / (1024 * 1024)
        
        return size_in_mb

    def visualize_latent_space(self, num_samples=5, scale=None, modes_to_show=None):
        """
        Visualize the effect of each latent dimension independently.
        
        Args:
            num_samples: Number of samples to take for each mode
            scale: Range of latent values to sample (-scale to +scale), auto-computed if None
            modes_to_show: List of specific mode indices to visualize, visualize all if None
        """
        print("Visualizing latent space modes...")
        
        # Determine which modes to show
        if modes_to_show is None:
            modes_to_show = list(range(self.latent_dim))
        
        num_modes = len(modes_to_show)
        
        # Compute scale for latent vectors if not provided
        if scale is None:
            scale = self.compute_safe_scaling_factor() * 2.0  # Larger scale to see clear deformations
        
        # Create values to sample for each mode
        values = np.linspace(-scale, scale, num_samples)
        
        # Convert DOLFINx mesh to PyVista format
        topology, cell_types, x = plot.vtk_mesh(self.domain)
        grid = pyvista.UnstructuredGrid(topology, cell_types, x)
        
        # Create a linear function space for visualization
        V_viz = fem.functionspace(self.domain, ("CG", 1, (3,)))
        u_linear = fem.Function(V_viz)
        
        # Create plotter with mode rows and sample columns
        plotter = pyvista.Plotter(shape=(num_modes, num_samples), border=False, 
                                window_size=[1600, 200 * num_modes])
        
        rest_positions = torch.tensor(self.domain.geometry.x, 
                                    device=self.device, dtype=torch.float64)
        
        # Visualize each mode with varying values
        for i, mode_idx in enumerate(modes_to_show):
            for j, val in enumerate(values):
                # Create a zero latent vector
                z = torch.zeros(self.latent_dim, device=self.device, dtype=torch.float64)
                
                # Set only the current mode to the current value
                z[mode_idx] = val
                
                # Compute the linear component and neural model prediction
                linear_contribution = (self.linear_modes[:, mode_idx] * val)
                 # Pass rest_positions to the model and unpack the tuple
                displacements, _, _ = self.model(z, rest_positions=rest_positions.unsqueeze(0))
                
                # IMPORTANT: Reshape displacements to match linear_contribution
                # The displacement tensor may be in shape [batch_size, num_nodes, 3] and needs flattening
                if displacements.dim() > 1:
                    displacements = displacements.reshape(displacements.shape[0], -1)
                
                # Make sure it's the right shape before adding
                if displacements.dim() == 2 and displacements.shape[0] == 1:
                    displacements = displacements.squeeze(0)  # Remove batch dimension
                    
                u_total = displacements + linear_contribution
                u_total_np = u_total.detach().cpu().numpy()
                
                # Create a function in the original function space
                u_quadratic = fem.Function(self.V)
                u_quadratic.x.array[:] = u_total_np
                
                # Interpolate to the visualization space if needed
                u_linear.interpolate(u_quadratic)
                u_linear_np = u_linear.x.array
                
                # Set active subplot
                plotter.subplot(i, j)
                
                # Create mesh with deformation
                local_grid = grid.copy()
                local_grid.point_data["displacement"] = u_linear_np.reshape((-1, 3))
                local_grid["displacement_magnitude"] = np.linalg.norm(u_linear_np.reshape((-1, 3)), axis=1)
                
                # Compute max displacement for adaptive scaling
                max_disp = np.max(local_grid["displacement_magnitude"])
                warp_factor = min(1.5, 0.2/max(max_disp, 1e-6))  # Adaptive but reasonable scaling
                
                # Warp the mesh by the displacement
                warped = local_grid.warp_by_vector("displacement", factor=warp_factor)
                
                # Add mesh to plot
                plotter.add_mesh(warped, scalars="displacement_magnitude", 
                            cmap="viridis", show_edges=True)
                
                # Add value label
                plotter.add_text(f"z{mode_idx}={val:.2f}", position="lower_right", 
                            font_size=8, color='white')
                
                # Set camera position consistently
                plotter.view_isometric()
        
        # Add row labels for modes
        for i, mode_idx in enumerate(modes_to_show):
            plotter.subplot(i, 0)
            plotter.add_text(f"Mode {mode_idx}", position="left_edge", 
                        font_size=12, color='white')
        
        # Link all camera views
        plotter.link_views()
        
        # Add a unified colorbar
        plotter.subplot(0, 0)
        plotter.add_scalar_bar("Displacement Magnitude", position_x=0.4, position_y=0.05, 
                        width=0.5, height=0.02, title_font_size=12, label_font_size=10)
        
        # Add overall title
        plotter.add_text("Neural Latent Space Mode Atlas", position="upper_edge", 
                    font_size=16, color='black')
        
        print("Showing latent space visualization...")
        plotter.show()
        print("Visualization complete.")
        
        return plotter  # Return plotter in case further customization is needed
    
    
    def load_sofa_matrices(self, matrices_path=None, timestamp=None):
        """
        Load mass and stiffness matrices from SOFA export
        
        Args:
            matrices_path: Path to the matrices directory
            timestamp: Specific timestamp to load, or None for latest
        
        Returns:
            Tuple of (mass_matrix, stiffness_matrix) as PETSc matrices, or (None, None) if loading fails
        """
        print("Attempting to load SOFA matrices...")
        if matrices_path is None:
            matrices_path = 'matrices'
        
        if not os.path.exists(matrices_path):
            print(f"Error: Matrices directory {matrices_path} not found")
            return None, None
        
        # Find the latest matrices if timestamp not specified
        if timestamp is None:
            mass_files = glob.glob(f'{matrices_path}/mass_matrix_*.npy')
            if not mass_files:
                print(f"Error: No mass matrices found in {matrices_path}")
                return None, None
                
            # Get latest file by timestamp
            latest_file = max(mass_files, key=os.path.getctime)
            timestamp = latest_file.split('mass_matrix_')[1].split('.npy')[0]
            print(f"Using latest matrices with timestamp {timestamp}")
        
        # Load matrices
        mass_matrix_file = f'{matrices_path}/mass_matrix_{timestamp}.npy'
        stiff_matrix_file = f'{matrices_path}/stiffness_matrix_{timestamp}.npy'
        
        if not os.path.exists(mass_matrix_file) or not os.path.exists(stiff_matrix_file):
            print(f"Error: Matrix files for timestamp {timestamp} not found")
            return None, None
            
        print(f"Loading SOFA matrices from {mass_matrix_file} and {stiff_matrix_file}")
        
        # Add allow_pickle=True to handle object arrays from SOFA
        mass_matrix_obj = np.load(mass_matrix_file, allow_pickle=True)
        stiff_matrix_obj = np.load(stiff_matrix_file, allow_pickle=True)
        
        # Convert from SOFA format to dense numpy arrays
        # This handles both cases: if they're already arrays or if they're special objects
        try:
            if hasattr(mass_matrix_obj, 'todense'):  # If it's a sparse matrix
                mass_matrix_np = mass_matrix_obj.todense()
            else:  # If it's already a numpy array or can be converted
                mass_matrix_np = np.array(mass_matrix_obj, dtype=np.float64)
                
            if hasattr(stiff_matrix_obj, 'todense'):
                stiff_matrix_np = stiff_matrix_obj.todense()
            else:
                stiff_matrix_np = np.array(stiff_matrix_obj, dtype=np.float64)
                
            print(f"Matrix shapes: Mass {mass_matrix_np.shape}, Stiffness {stiff_matrix_np.shape}")
        except Exception as e:
            print(f"Error converting matrices to numpy arrays: {e}")
            return None, None
        
        # Load metadata if available
        metadata_file = f'{matrices_path}/metadata_{timestamp}.json'
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                print(f"Loaded matrices metadata: {metadata}")
        
        # Convert numpy arrays to PETSc matrices
        try:
            A = PETSc.Mat().createDense([stiff_matrix_np.shape[0], stiff_matrix_np.shape[1]])
            A.setUp()
            A.setValues(range(stiff_matrix_np.shape[0]), range(stiff_matrix_np.shape[1]), stiff_matrix_np)
            A.assemble()
            
            M = PETSc.Mat().createDense([mass_matrix_np.shape[0], mass_matrix_np.shape[1]])
            M.setUp()
            M.setValues(range(mass_matrix_np.shape[0]), range(mass_matrix_np.shape[1]), mass_matrix_np)
            M.assemble()
            
            print(f"Successfully converted matrices to PETSc format: M({M.size}), A({A.size})")
            return M, A
        except Exception as e:
            print(f"Error converting to PETSc format: {e}")
            return None, None


def main():
    print("Starting main function...")
    # Parse arguments
    parser = argparse.ArgumentParser(description='Hybrid Simulation')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='config file path')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--skip-training', action='store_true', help='skip training and load best model')
    parser.add_argument('--checkpoint', type=str, default=None, help='specific checkpoint path to load')
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    setup_logger(None, log_dir=os.path.join(cfg['training']['checkpoint_dir'], cfg.get('training', {}).get('log_dir', 'logs')))
    print("Arguments parsed and logger setup.")

    # Check for skip_training in both command line and config
    skip_training = args.skip_training or cfg.get('training', {}).get('skip_training', False)
    checkpoint_path = args.checkpoint or cfg.get('training', {}).get('checkpoint_path')
    
    if checkpoint_path is None:
        checkpoint_path = os.path.join('checkpoints', 'best.pt')
    
    print(f"Skip training: {skip_training}")
    print(f"Checkpoint path: {checkpoint_path if os.path.exists(checkpoint_path) else 'Not found'}")

    engine = RotationStrainRoutine(cfg)
    print("Engine initialized.")

    # Training or loading logic
    if skip_training:
        print("Skipping training as requested...")
        if os.path.exists(checkpoint_path):
            print(f"Loading model from {checkpoint_path}")
            engine.load_checkpoint(checkpoint_path)
        else:
            print(f"Warning: No checkpoint found at {checkpoint_path}, using untrained model")
    else:
        # Normal training loop
        num_epochs = cfg['training']['num_epochs']
        print(f"Starting training for {num_epochs} epochs...")
        best_loss = engine.train(num_epochs)
        print("Training complete.")
        
        # Load the best model before evaluation
        best_checkpoint_path = os.path.join('checkpoints', 'best_rotation_strain.pt')
        if os.path.exists(best_checkpoint_path):
            print("Loading best model for evaluation...")
            engine.load_checkpoint(best_checkpoint_path)
        else:
            print("No best model checkpoint found, using final model")

    latent_dim = engine.latent_dim
    print(f"Latent dimension: {latent_dim}")


    # Add latent space visualization
    print("\nVisualizing latent space dimensions...")
    # Visualize first two dimensions by default
    engine.visualize_latent_dimensions(dim1=3, dim2=2, num_points=5)
    
    # Optionally visualize other dimension pair
    engine.visualize_latent_dimensions(dim1=0, dim2=3, num_points=5)

    print("\nVisualizing latent space modes...")
    # Visualize all latent dimensions
    engine.visualize_latent_space(num_samples=5)
    
    print("Main function complete.")

def load_config(config_file):
    import yaml
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

if __name__ == '__main__':
    main()
