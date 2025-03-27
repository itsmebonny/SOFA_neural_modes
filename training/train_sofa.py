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

import traceback
from scipy import sparse

from tests.solver import EnergyModel, NeoHookeanEnergyModel, ModularNeoHookeanEnergy



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

class Net(torch.nn.Module):
    def __init__(self, latent_dim, output_dim, hid_layers=2, hid_dim=64):
        """
        Neural network with configurable architecture
        
        Args:
            latent_dim: Dimension of input latent space
            output_dim: Dimension of output space
            hid_layers: Number of hidden layers
            hid_dim: Width of hidden layers
        """
        super(Net, self).__init__()
        
        # Input layer
        layers = [torch.nn.Linear(latent_dim, hid_dim)]
        
        # Hidden layers
        for _ in range(hid_layers-1):
            layers.append(torch.nn.Linear(hid_dim, hid_dim))
            
        # Output layer
        layers.append(torch.nn.Linear(hid_dim, output_dim))
        
        self.layers = torch.nn.ModuleList(layers)

        # self.layers[-1].weight.data.fill_(0.0)
        # self.layers[-1].bias.data.fill_(0.0)
        
    def forward(self, x):
        # Handle both single vectors and batches
        is_batched = x.dim() > 1
        if not is_batched:
            x = x.unsqueeze(0)  # Add batch dimension if not present
        
        # Apply GELU activation to all but the last layer
        for i in range(len(self.layers)-1):
            x = torch.nn.functional.gelu(self.layers[i](x))
        
        # No activation on output layer
        x = self.layers[-1](x)
        
        # Remove batch dimension if input wasn't batched
        if not is_batched:
            x = x.squeeze(0)
        return x
        







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
    
    def _compute_F_batch_multi(self, batch_coords, batch_disps, batch_dN_dx):
        """
        Compute deformation gradient F = I + ∇u for a batch of elements across multiple samples
        
        Args:
            batch_coords: Element coordinates [chunk_size, batch_size, nodes_per_element, 3]
            batch_disps: Element displacements [chunk_size, batch_size, nodes_per_element, 3]
            batch_dN_dx: Shape function derivatives [chunk_size, batch_size, nodes_per_element, 3]
            
        Returns:
            Batch of deformation gradients [chunk_size, batch_size, 3, 3]
        """
        # Initialize deformation gradient as identity for each element in batch
        chunk_size, local_batch_size = batch_disps.shape[0], batch_disps.shape[1]
        batch_F = torch.eye(3, dtype=self.dtype, device=self.device).expand(
            chunk_size, local_batch_size, 3, 3).clone()
        
        # Use einsum for efficient computation across all dimensions
        batch_F += torch.einsum('cbij,cbik->cbjk', batch_disps, batch_dN_dx)
        
        return batch_F
    
    def _compute_PK1_batch_multi(self, batch_F):
        """
        Compute the First Piola-Kirchhoff stress tensor for multiple batches of deformation gradients
        
        Args:
            batch_F: Tensor of shape [chunk_size, batch_size, 3, 3] containing deformation gradients
            
        Returns:
            Tensor of shape [chunk_size, batch_size, 3, 3] containing PK1 tensors
        """
        # Compute J = det(F)
        batch_J = torch.linalg.det(batch_F)
        
        # Compute inverse of F
        batch_inv_F = torch.linalg.inv(batch_F)
        
        # Compute F^-T
        batch_inv_F_T = batch_inv_F.transpose(2, 3)
        
        # First Piola-Kirchhoff stress tensor
        term1 = self.mu * batch_F
        term2 = -self.mu * batch_inv_F_T
        term3 = 0.5 * self.lmbda * ((batch_J * batch_J - 1.0).unsqueeze(-1).unsqueeze(-1)) * batch_inv_F_T
        
        batch_P = term1 + term2 + term3
        
        return batch_P
        
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
    
    def compute_div_p(self, u_batch):
        """
        Fully vectorized implementation to compute the raw divergence of Piola-Kirchoff tensor.
        Returns the raw div(P) values without any transformation for post-processing in the training loop.
        
        Args:
            u_batch: Batch of displacement fields [batch_size, num_nodes * 3]
            
        Returns:
            Raw div(P) values [batch_size, num_nodes, 3]
        """
        batch_size = u_batch.shape[0]
        
        # Process the entire batch at once if memory permits, otherwise use chunking
        max_chunk_size = 4  # Adjust based on GPU memory
        div_p_all = []
        
        for chunk_start in range(0, batch_size, max_chunk_size):
            chunk_end = min(chunk_start + max_chunk_size, batch_size)
            chunk_u_batch = u_batch[chunk_start:chunk_end]
            chunk_size = chunk_end - chunk_start
            
            # Reshape all displacements in the chunk
            u_reshaped = chunk_u_batch.reshape(chunk_size, self.num_nodes, self.dim)
            
            # Initialize divergence storage for all samples in chunk
            div_p = torch.zeros((chunk_size, self.num_nodes, 3), 
                            dtype=self.dtype, device=self.device)
            node_counts = torch.zeros((chunk_size, self.num_nodes), 
                                    dtype=self.dtype, device=self.device)
            
            # Process elements in batches for memory efficiency
            for batch_start in range(0, self.num_elements, self.batch_size):
                batch_end = min(batch_start + self.batch_size, self.num_elements)
                batch_elements = self.elements[batch_start:batch_end]
                local_batch_size = batch_end - batch_start
                
                # Get element coordinates and displacements
                batch_coords = self.coordinates[batch_elements].unsqueeze(0).expand(
                    chunk_size, local_batch_size, self.nodes_per_element, 3)
                    
                # Gather displacements for all elements in batch for all samples in chunk
                batch_disps = torch.stack([u_reshaped[b, batch_elements] for b in range(chunk_size)])
                
                # Initialize divergence contribution
                batch_div_p = torch.zeros((chunk_size, local_batch_size, self.nodes_per_element, 3), 
                                        dtype=self.dtype, device=self.device)
                
                # Compute for all quadrature points
                for q_idx in range(len(self.quadrature_points)):
                    if self.precomputed:
                        # Use precomputed derivatives
                        batch_dN_dx = self.dN_dx_all[batch_start:batch_end, q_idx].unsqueeze(0).expand(
                            chunk_size, -1, -1, -1)
                        batch_detJ = self.detJ_all[batch_start:batch_end, q_idx].unsqueeze(0).expand(
                            chunk_size, -1)
                    else:
                        # Compute derivatives vectorized
                        batch_dN_dx, batch_detJ = self._compute_derivatives_batch(batch_coords, 
                                                self.quadrature_points[q_idx])
                    
                    # Compute F tensors for all elements in batch for all samples in chunk
                    batch_F = self._compute_F_batch_multi(batch_coords, batch_disps, batch_dN_dx)
                    
                    # Compute PK1 stress for all elements
                    batch_P = self._compute_PK1_batch_multi(batch_F)
                    
                    # Vectorized divergence computation using einsum
                    qp_div_p = torch.einsum('cbij,cbnj->cbni', batch_P, batch_dN_dx)

                    # Weight by quadrature weight and jacobian determinant (no tanh)
                    weighted_div_p = qp_div_p * self.quadrature_weights[q_idx] * batch_detJ.unsqueeze(-1).unsqueeze(-1)
                    
                    # Accumulate divergence contribution
                    batch_div_p += weighted_div_p
                
                # Assemble element contributions to nodes
                for e in range(local_batch_size):
                    element_nodes = batch_elements[e].long()
                    div_p[:, element_nodes] += batch_div_p[:, e]
                    node_counts[:, element_nodes] += 1
            
            # Average contributions at nodes with multiple element memberships
            mask = node_counts > 0
            div_p[mask] = div_p[mask] / node_counts[mask].unsqueeze(-1)
            
            # Save this chunk's results
            div_p_all.append(div_p)
        
        # Combine all chunks
        return torch.cat(div_p_all, dim=0)





class SOFANeoHookeanEnergy(torch.nn.Module):
    """
    NeoHookean energy implementation following SOFA's formulation, fully vectorized
    for batch processing with PyTorch.
    
    This implementation follows the formulation from SOFA's neohookean.cpp:
    W = 0.5 * mu * (IC - 3) - mu * log(J) + 0.5 * lambda * log(J)^2
    """
    def __init__(self, domain, degree, E, nu, precompute_matrices=True, device=None,
                 dtype=torch.float64, batch_size=100):
        super(SOFANeoHookeanEnergy, self).__init__()
        
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
        
        # Generate quadrature points
        self.quadrature_points, self.quadrature_weights = self._generate_quadrature()
        print(f"Using {len(self.quadrature_weights)} quadrature points per element")
        
        # Memory-efficient precomputation strategy
        self.precomputed = False
        if precompute_matrices:
            mem_estimate = self._estimate_precompute_memory()
            print(f"Estimated memory for precomputation: {mem_estimate:.2f} GB")
            if mem_estimate > 2.0:  # Arbitrary threshold, adjust based on your hardware
                print("Warning: High memory usage for precomputation. Proceeding with caution.")
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
    
    def _compute_sofa_neohookean_energy_batch(self, batch_F):
        """
        Compute NeoHookean strain energy density following SOFA's formulation
        for a batch of deformation gradients with numerical stability improvements
        
        Args:
            batch_F: Tensor of shape [batch_size, 3, 3] containing deformation gradients
        
        Returns:
            Tensor of shape [batch_size] containing energy densities
        """
        # Compute J = det(F) for all elements in batch
        batch_J = torch.linalg.det(batch_F)
        
        # Compute right Cauchy-Green tensor C = F^T·F
        batch_C = torch.bmm(batch_F.transpose(1, 2), batch_F)
        
        # Compute first invariant of C (trace of C)
        batch_I1 = torch.diagonal(batch_C, dim1=1, dim2=2).sum(dim=1)
        
        # Create storage for energy
        batch_size = batch_F.shape[0]
        batch_W = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
        
        # Handle different J regimes for numerical stability
        
        # Valid J values (J > threshold)
        valid_mask = batch_J > 1e-6
        if torch.any(valid_mask):
            valid_J = batch_J[valid_mask]
            valid_I1 = batch_I1[valid_mask]
            valid_log_J = torch.log(valid_J)
            
            # SOFA formulation: W = 0.5 * mu * (IC - 3) - mu * log(J) + 0.5 * lambda * log(J)^2
            term1 = 0.5 * self.mu * (valid_I1 - 3.0)
            term2 = -self.mu * valid_log_J
            term3 = 0.5 * self.lmbda * torch.pow(valid_log_J, 2)
            
            batch_W[valid_mask] = term1 + term2 + term3
        
        # Small positive J values (need Taylor expansion for log)
        small_mask = (batch_J > 0) & (batch_J <= 1e-6)
        if torch.any(small_mask):
            small_J = batch_J[small_mask]
            small_I1 = batch_I1[small_mask]
            
            # Taylor series for log(J) near 0: log(J) ≈ (J-1) - (J-1)^2/2 + ...
            approx_log_J = small_J - 1.0 - (small_J - 1.0)**2 / 2.0
            
            # Apply SOFA's formula with approximated log(J)
            term1 = 0.5 * self.mu * (small_I1 - 3.0)
            term2 = -self.mu * approx_log_J
            term3 = 0.5 * self.lmbda * torch.pow(approx_log_J, 2)
            
            batch_W[small_mask] = term1 + term2 + term3
            
            # Add barrier term to prevent J → 0
            barrier = 1e2 * torch.pow(1e-6 - small_J, 2)
            batch_W[small_mask] += barrier
        
        # Negative J values (inverted elements)
        invalid_mask = batch_J <= 0
        if torch.any(invalid_mask):
            # Apply strong penalty for inverted elements
            batch_W[invalid_mask] = 1e6 * torch.abs(batch_J[invalid_mask])
        
        return batch_W
    
    def _compute_PK2_batch(self, batch_F):
        """
        Compute the Second Piola-Kirchhoff stress tensor following SOFA's formulation
        for a batch of deformation gradients
        
        Args:
            batch_F: Tensor of shape [batch_size, 3, 3] containing deformation gradients
            
        Returns:
            Tensor of shape [batch_size, 3, 3] containing PK2 tensors
        """
        # Create identity matrix (repeated for the batch)
        batch_size = batch_F.shape[0]
        batch_I = torch.eye(3, dtype=self.dtype, device=self.device).expand(batch_size, 3, 3)
        
        # Compute J = det(F)
        batch_J = torch.linalg.det(batch_F)
        
        # Compute right Cauchy-Green tensor C = F^T·F
        batch_C = torch.bmm(batch_F.transpose(1, 2), batch_F)
        
        # Compute inverse of C (batched)
        batch_inv_C = torch.linalg.inv(batch_C)
        
        # Compute log(J) with stability
        # For valid J, use direct formula
        batch_log_J = torch.zeros_like(batch_J)
        valid_mask = batch_J > 1e-6
        batch_log_J[valid_mask] = torch.log(batch_J[valid_mask])
        
        # For small J, use Taylor expansion
        small_mask = (batch_J > 0) & (batch_J <= 1e-6)
        small_J = batch_J[small_mask]
        batch_log_J[small_mask] = small_J - 1.0 - (small_J - 1.0)**2 / 2.0
        
        # SOFA's formula: SPK = mu * I + (lambda * log(J) - mu) * C^-1
        term1 = self.mu * batch_I
        term2 = (self.lmbda * batch_log_J.unsqueeze(-1).unsqueeze(-1) - self.mu) * batch_inv_C
        
        batch_PK2 = term1 + term2
        
        # Handle inverted elements separately
        invalid_mask = batch_J <= 0
        if torch.any(invalid_mask):
            # For inverted elements, apply a stabilizing force to uninvert
            inv_indices = torch.where(invalid_mask)[0]
            for idx in inv_indices:
                # Create a strong restorative stress that pushes back toward identity
                F_inv = batch_F[idx]
                batch_PK2[idx] = 1e4 * (batch_I[0] - F_inv)
        
        return batch_PK2
    
    def forward(self, u_tensor):
        """
        Compute total elastic energy for the displacement field using SOFA's NeoHookean formula
        with optimized batch operations
        
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
                
                # Compute energy densities for all elements at once using SOFA's formula
                batch_energy_density = self._compute_sofa_neohookean_energy_batch(batch_F)
                
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

class Routine:
    def __init__(self, cfg):
        print("Initializing Routine...")
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda') if use_cuda else torch.device('cpu')
        self.device = device
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
        self.energy_calculator = ModularNeoHookeanEnergy(
                        self.domain, self.fem_degree, self.E, self.nu,
                        precompute_matrices=True, device=self.device
                    ).to(self.device)
        self.scale = self.compute_safe_scaling_factor()
        print(f"Scaling factor: {self.scale}")

        # Load neural network
        print("Loading neural network...")
        self.latent_dim = cfg['model']['latent_dim']
        self.num_modes = self.latent_dim  - 6
        output_dim = self.V.dofmap.index_map.size_global * self.domain.geometry.dim
        hid_layers = cfg['model'].get('hid_layers', 2)
        hid_dim = cfg['model'].get('hid_dim', 64)
        print(f"Output dimension: {output_dim}")
        print(f"Network architecture: {hid_layers} hidden layers with {hid_dim} neurons each")
        self.model = Net(self.num_modes, output_dim, hid_layers, hid_dim).to(device).double()

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
        print(f"Linear modes shape: {self.linear_modes.shape}")
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
        M, A, eigenvalues, eigenvectors = self.load_sofa_matrices()
        
        print("Using SOFA-generated matrices and modes for linear modes computation")
        # Store matrices in self for later use
        self.M = M
        self.A = A
        self.eigenvalues = eigenvalues
        linear_modes = eigenvectors
        return linear_modes
        
    def load_sofa_matrices(self, matrices_path=None, timestamp=None):
        """
        Load mass and stiffness matrices from SOFA export with robust format handling
        """
        print("Attempting to load SOFA matrices and modes...")
        if matrices_path is None:
            matrices_path = 'matrices'
        
        if not os.path.exists(matrices_path):
            print(f"Matrices path does not exist: {matrices_path}")
            return None, None, None, None
        
        # Find the latest matrices if timestamp not specified
        if timestamp is None:
            mass_files = glob.glob(f'{matrices_path}/mass_matrix_*.npy')
            if not mass_files:
                print(f"No mass matrices found in {matrices_path}")
                return None, None, None, None
            
            # Get latest file by timestamp
            latest_file = max(mass_files, key=os.path.getctime)
            timestamp = latest_file.split('mass_matrix_')[1].split('.npy')[0]
            print(f"Using latest matrices with timestamp {timestamp}")
        
        # Load matrices
        mass_matrix_file = f'{matrices_path}/mass_matrix_{timestamp}.npy'
        stiff_matrix_file = f'{matrices_path}/stiffness_matrix_{timestamp}.npy'
        eigenvalues_file =  f'{matrices_path}/{timestamp}/eigenvalues_{timestamp}.npy'
        eigenvectors_file = f'{matrices_path}/{timestamp}/eigenvectors_{timestamp}.npy'
        
        if not os.path.exists(mass_matrix_file) or not os.path.exists(stiff_matrix_file) or not os.path.exists(eigenvalues_file) or not os.path.exists(eigenvectors_file):
            print(f"Matrix or mode files for timestamp {timestamp} not found")
            return None, None, None, None
        
        print(f"Loading matrices from {mass_matrix_file} and {stiff_matrix_file}")
        print(f"Loading modes from {eigenvalues_file} and {eigenvectors_file}")
        
        try:
            mass_matrix = np.load(mass_matrix_file, allow_pickle=True)
            stiff_matrix = np.load(stiff_matrix_file, allow_pickle=True)
            eigenvalues = np.load(eigenvalues_file, allow_pickle=True)
            eigenvectors = np.load(eigenvectors_file, allow_pickle=True)
            
            # Handle 0-dimensional array (scalar container)
            if isinstance(mass_matrix, np.ndarray) and mass_matrix.ndim == 0:
                print("Detected 0-dimensional array. Extracting contained object...")
                mass_matrix = mass_matrix.item()
                stiff_matrix = stiff_matrix.item()
            
            # Ensure matrices are in CSR format for efficiency
            if not isinstance(mass_matrix, sparse.csr_matrix):
                if hasattr(mass_matrix, 'tocsr'):
                    mass_matrix = mass_matrix.tocsr()
                    stiff_matrix = stiff_matrix.tocsr()
                    print("Converted matrices to CSR format")
                else:
                    raise TypeError(f"Cannot convert {type(mass_matrix)} to CSR format")
            
            print(f"Matrix shapes: Mass {mass_matrix.shape}, Stiffness {stiff_matrix.shape}")
            print(f"Mode shapes: Eigenvalues {eigenvalues.shape}, Eigenvectors {eigenvectors.shape}")
            
            return mass_matrix, stiff_matrix, eigenvalues, eigenvectors
        
        except Exception as e:
            print(f"Error loading SOFA matrices: {e}")
            traceback.print_exc()
            return None, None, None, None

    def compute_eigenvalue_based_scale(self, mode_index=None):
        """
        Compute scaling factor for latent variables based on eigenvalues
        
        Args:
            mode_index: Specific mode index to get scaling for, or None for all modes
            
        Returns:
            Scaling factor or array of scaling factors
        """
        if not hasattr(self, 'eigenvalues') or self.eigenvalues is None:
            print("Warning: No eigenvalues available, using default scaling")
            return self.compute_safe_scaling_factor()
        
        # Check if we have all eigenvalues needed
        if mode_index is not None and mode_index >= len(self.eigenvalues):
            print(f"Warning: Requested mode {mode_index} exceeds available eigenvalues")
            return self.compute_safe_scaling_factor()
        
        # For neo-Hookean materials, scale is inversely proportional to sqrt(eigenvalue)
        # This is because energy is proportional to eigenvalue * displacement^2
        if mode_index is not None:
            # Return scale for specific mode
            return 1.0 / np.sqrt(max(1e-8, self.eigenvalues[mode_index]))
        else:
            # Return array of scales for all modes
            return 1.0 / np.sqrt(np.maximum(1e-8, self.eigenvalues))
        

    def train(self, num_epochs=1000):
        """
        Train the model using batched processing with strong orthogonality constraints.
        Similar to the reference implementation with St. Venant-Kirchhoff energy.
        """
        print("Starting training...")
        
        # Setup training parameters
        batch_size = 32  # You can add this to config
        rest_idx = 0    # Index for rest shape in batch
        print_every = 1
        checkpoint_every = 50
        
        # Get rest shape (undeformed configuration)
        X = torch.zeros((self.V.dofmap.index_map.size_global * self.domain.geometry.dim), 
                    device=self.device, dtype=torch.float64)
        X = X.view(1, -1).expand(batch_size, -1)
        
        # Use a subset of linear modes (you might need to adjust indices)
        L = self.num_modes  # Use at most 3 linear modes
        linear_modes = self.linear_modes[:, :L]  # Use the first L modes
        
        # Setup iteration counter and best loss tracking
        iteration = 0
        best_loss = float('inf')
        
        # Make sure model accepts batched inputs
        # Modify Net forward method to handle batched inputs
        original_forward = self.model.forward
        
        def new_forward(x):
            is_batch = x.dim() > 1
            if not is_batch:
                x = x.unsqueeze(0)  # Add batch dimension
            
            # Process through network
            result = original_forward(x)
            
            if not is_batch:
                result = result.squeeze(0)  # Remove batch dimension if input wasn't batched
            return result
            
        self.model.forward = new_forward
        
        # Use LBFGS optimizer
        optimizer = torch.optim.LBFGS(self.model.parameters(), 
                                    lr=1,
                                    max_iter=25,
                                    max_eval=35,
                                    tolerance_grad=1e-05,
                                    tolerance_change=1e-07,
                                    history_size=100,
                                    line_search_fn="strong_wolfe")
        
        # Add scheduler for learning rate reduction
        scheduler = LBFGSScheduler(
            optimizer,
            factor=0.5,  # Reduce LR by factor of 5 when plateau is detected
            patience=100,  # More patience for batch training
            threshold=0.01,  # Consider 1% improvement as significant
            min_lr=1e-6,  # Don't go below this LR
            verbose=True   # Print LR changes
        )
        patience = 0
        # Main training loop

        # Add this near the beginning of the train method, after initializing variables:
        if not hasattr(self, 'viz_plotter'):
            self.create_live_visualizer()

        

       

        while iteration < num_epochs:  # Set a maximum iteration count or use other stopping criteria
            # Generate random latent vectors and linear displacements

            lbfgs_iter = 0
            with torch.no_grad():
                
                # Generate latent vectors 
                deformation_scale_init = 0.5
                deformation_scale_final = 3
                #current_scale = deformation_scale_init * (deformation_scale_final/deformation_scale_init)**(iteration/num_epochs) #expoential scaling
                current_scale = deformation_scale_init + (deformation_scale_final - deformation_scale_init) # * (iteration/num_epochs) #linear scaling

                print(f"Current scale: {current_scale}")
                mode_scales = torch.tensor(self.compute_eigenvalue_based_scale(), device=self.device, dtype=torch.float64)
                mode_scales = mode_scales * current_scale

                # Generate samples with current scale
                z = torch.rand(batch_size, L, device=self.device) * mode_scales * 2 - mode_scales
                # Generate mixed batch with varying scales

                # max_scale = 10.0
                # z = torch.zeros(batch_size, L, device=self.device, dtype=torch.float64)
                # scales = torch.linspace(0.5, max_scale, batch_size)
                # for i in range(batch_size):
                # z[i] = torch.rand(L, device=self.device) * scales[i] * 2 - scales[i]


                # z = torch.rand(batch_size, L, device=self.device) * mode_scales * 2 - mode_scales
                z[rest_idx, :] = 0  # Set rest shape latent to zero
                #concatenate the generated samples with the rest shape
                
                # Compute linear displacements
                l = torch.matmul(z, linear_modes.T)
                
                # Create normalized constraint directions
                constraint_dir = torch.matmul(z, linear_modes.T)
                constraint_norms = torch.norm(constraint_dir, p=2, dim=1, keepdim=True)
                # Avoid division by zero
                constraint_norms = torch.clamp(constraint_norms, min=1e-8)
                constraint_dir = constraint_dir / constraint_norms
                constraint_dir[rest_idx] = 0  # Zero out rest shape constraints
            
                # Track these values outside the closure
                energy_val = 0
                ortho_val = 0
                origin_val = 0
                loss_val = 0
                
                # Define closure for optimizer
                def closure():
                    nonlocal energy_val, ortho_val, origin_val, loss_val, lbfgs_iter, iteration, best_loss
                    nonlocal z, l, rest_idx, constraint_dir

                    lbfgs_iter += 1
                    
                    optimizer.zero_grad()
                    
                    # Compute nonlinear correction
                    y = self.model(z)
                    
                    # Compute energy (use your energy calculator)
                    u_total_batch = l + y
                    
                    
                    
                    # After (if processing a batch):
                    batch_size = u_total_batch.shape[0]
                    if batch_size > 1:
                        # Use batch processing for multiple samples
                        energies = self.energy_calculator(u_total_batch)
                        energy = torch.mean(energies)  # Average energy across batch
                    else:
                        # Use single-sample processing
                        energy = self.energy_calculator(u_total_batch[0])

                    # Calculate maximum displacements
                    mean_linear = torch.mean(torch.norm(l.reshape(batch_size, -1, 3), dim=2)).item()
                    mean_total = torch.mean(torch.norm(u_total_batch.reshape(batch_size, -1, 3), dim=2)).item()
                    mean_correction = torch.mean(torch.norm(y.reshape(batch_size, -1, 3), dim=2)).item()

                    nonlinear_ratio = mean_correction / mean_total
                    
                    # Compute orthogonality constraint (using the same approach as reference)
                    ortho = torch.mean(torch.sum(y * constraint_dir, dim=1)**2)
                    
                    # Compute origin constraint for rest shape
                    origin = torch.sum(y[rest_idx]**2)


                    displacement_magnitude = torch.norm(l, dim=1, keepdim=True)
                    displacement_magnitude_mean = torch.mean(displacement_magnitude)

                    # 1. Scale-normalized energy: E/|u|²
                    # This measures energy efficiency rather than raw energy
                    # For linear elasticity, this should be constant (scale-invariant)
                    scale_factor = torch.clamp(displacement_magnitude_mean, min=1e-6)**2
                    normalized_energy = energy / scale_factor

                
                    # 2. Energy with physically expected scaling
                    # For Neo-Hookean materials, energy scales approximately with |u|²

                    energy_scaling = torch.log10(torch.square(normalized_energy) + 1)

                    # Add incentive for beneficial nonlinearity (energy improvement term)
                    u_linear_only = l.detach()  # Detach to avoid affecting linear gradients
                    energy_linear = self.energy_calculator(u_linear_only).mean()
                    energy_improvement = (energy_linear - energy)
                    nonlinear_weight = torch.clamp(normalized_energy, min=0.1, max=10.0)
                    nonlinear_reward = nonlinear_weight * torch.clamp(energy_improvement, min=0) 



                    # Get the raw div(P) tensor
                    raw_div_p = self.energy_calculator.compute_div_p(u_total_batch)

                    raw_div_p_L2_mean = torch.mean(torch.norm(raw_div_p, dim=2))

                    # Apply tanh transformation
                    div_p_magnitude = torch.norm(raw_div_p, dim=2, keepdim=True)
                    div_p_direction = raw_div_p / (div_p_magnitude + 1e-8)

                    # Apply log scaling to magnitude only
                    log_div_p_magnitude = torch.log10(div_p_magnitude + 1)

                    # Recombine with original direction
                    log_scaled_div_p_tensor = log_div_p_magnitude * div_p_direction

                    # log_Scaled_div_p is a tensor of shape [batch_size, num_nodes, 3]
                    #let's do a mean over the nodes and the batch and resh

                    log_scaled_div_p = torch.mean(torch.norm(log_scaled_div_p_tensor, dim=2))


                    #ADD a plotter with only linear modes prediction alongside the neural network prediction

                    div_p_weight = 2.0  # Weight for divergence loss


                    # Modified loss
                    loss =  energy + ortho + origin  

                    loss.backward()

                    # Choose a random latent vector from the batch
                    random_idx = np.random.randint(1, batch_size)
                    random_z = z[random_idx].detach().clone()
                    self.visualize_latent_vector(random_z, iteration, loss_val)

                    # Print stats periodically
                    if iteration % print_every == 0:
                        # Create a clean, organized training progress display
                        progress = f"{iteration}/{num_epochs}"
                        progress_pct = f"({100 * iteration / num_epochs:.1f}%)"
                        lbfgs_progress = f"LBFGS iteration: {lbfgs_iter}"
                        
                        # Create separator line and header
                        sep_line = "=" * 80
                        print(f"\n{sep_line}")
                        print(f"TRAINING ITERATION {progress} {progress_pct} - Best Loss: {best_loss:.6e}")
                        print(f"{sep_line}")
                        
                        # Energy metrics section
                        print(f"│ ENERGY METRICS:")
                        print(f"│ {'Raw Energy:':<20} {energy.item():<12.6f} │ {'Normalized Energy:':<20} {normalized_energy.item():<12.6f}")
                        print(f"│ {'Energy Improvement:':<20} {energy_improvement.item():<12.6f} │ {'Energy Loss:':<20} {energy_scaling.item():<12.6f}")
                        
                        # Constraint metrics section
                        print(f"│ CONSTRAINT METRICS:")
                        print(f"│ {'Orthogonality:':<20} {ortho.item():<12.6f} │ {'Origin Constraint:':<20} {origin.item():<12.6f}")
                        
                        # Displacement metrics section
                        print(f"│ DISPLACEMENT METRICS:")
                        print(f"│ {'Mean Linear:':<20} {mean_linear:<12.6f} │ {'Mean Total:':<20} {mean_total:<12.6f}")
                        print(f"│ {'Mean Correction:':<20} {mean_correction:<12.6f} │ {'Nonlinear Ratio:':<20} {nonlinear_ratio*100:.2f}%")
                        
                        # Divergence metrics section
                        div_p_means = torch.mean(raw_div_p, dim=0).mean(dim=0)
                        print(f"│ DIVERGENCE METRICS:")
                        print(f"│ {'Direction:':<20} {'X':<17} {'Y':<17} {'Z':<17}")
                        print(f"│ {'Div(P):':<12} {div_p_means[0].item():15.6e} {div_p_means[1].item():15.6e} {div_p_means[2].item():15.6e}")
                        print(f"│ {'Div(P) Loss:':<20} {log_scaled_div_p.item():<12.6f} │ {'Raw Div(P) L2:':<20} {raw_div_p_L2_mean.item():<12.6e}")
                        
                        # Final loss value
                        print(f"{sep_line}")
                        print(f"TOTAL LOSS: {loss.item():.6e} - {lbfgs_progress}")
                        print(f"{sep_line}\n")
                    

                                    # Add these lines before return loss
                    energy_val = energy.item()  # Convert tensor to Python scalar
                    ortho_val = ortho.item()
                    origin_val = origin.item()
                    loss_val = loss.item()

                    return loss
                
                # Perform optimization step
                optimizer.step(closure)

                scheduler.step(loss_val)  # Use the loss value to determine if we should reduce LR

            if iteration % 1 == 0:  # Update visualization every 5 iterations
                pass
                # Update visualization

            
            # Record metrics using values captured from closure
            self.writer.add_scalar('train/loss', loss_val, iteration)
            self.writer.add_scalar('train/energy', energy_val, iteration)
            self.writer.add_scalar('train/ortho', ortho_val, iteration)
            self.writer.add_scalar('train/origin', origin_val, iteration)
            
            # Save checkpoint if this is the best model so far
            if loss_val < best_loss or loss_val < 10:
                best_loss = loss_val
                # Estimate checkpoint size
                checkpoint_size = self.estimate_checkpoint_size()
                print(f"Estimated checkpoint size: {checkpoint_size:.2f} MB")
                checkpoint = {
                    'epoch': iteration,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_val,
                }
                patience = 0
                torch.save(checkpoint, os.path.join('checkpoints', 'best.pt'))
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
                }
                torch.save(checkpoint, os.path.join('checkpoints', f'model_it{iteration}.pt'))
            
            iteration += 1
            patience += 1
            
            # Early stopping criterion (optional)
            if loss_val < 1e-8 or patience > 30:
                print(f"Converged to target loss at iteration {iteration}")
                break
        
        # Restore original forward method
        self.model.forward = original_forward
        
        print(f"Training complete. Best loss: {best_loss:.8e}")
        return best_loss
    
    def create_live_visualizer(self):
        """Create and return a persistent PyVista plotter for visualization during training"""
        # Convert DOLFINx mesh to PyVista format
        topology, cell_types, x = plot.vtk_mesh(self.domain)
        grid = pyvista.UnstructuredGrid(topology, cell_types, x)
        
        # Create a linear function space for visualization
        self.V_viz = fem.functionspace(self.domain, ("CG", 1, (3,)))
        self.u_linear_viz = fem.Function(self.V_viz)
        
        # Create plotter with two viewports for side-by-side comparison
        plotter = pyvista.Plotter(shape=(1, 2), title="Neural Modes Training Visualization", 
                            window_size=[1600, 720], off_screen=False)
        
        # Store grid and visualization components
        self.viz_grid = grid
        self.viz_plotter = plotter
        self.mesh_actor_left = None
        self.mesh_actor_right = None
        self.info_actor = None
        
        # Initialize the render window
        plotter.show(interactive=False, auto_close=False)
        
        # Set camera position for both viewports (same as validate_twist.py)
        for i in range(2):
            plotter.subplot(0, i)
            plotter.camera_position = [(20.0, 3.0, 2.0), (0.0, -2.0, 0.0), (0.0, 0.0, 2.0)]
            plotter.camera.zoom(1.5)
        
        # Link camera views so they move together
        plotter.link_views()
        
        return plotter

    def visualize_latent_vector(self, z, iteration=None, loss=None):
        """Update visualization with current training state showing both linear and neural predictions"""
        try:
            # Ensure z is properly formatted
            if not isinstance(z, torch.Tensor):
                z = torch.tensor(z, device=self.device, dtype=torch.float64)
            
            if z.dim() > 1:
                z = z.squeeze()
            
            # Compute displacements for both visualizations
            with torch.no_grad():
                # 1. Linear component only
                linear_contribution = torch.matmul(z, self.linear_modes.T)
                linear_only_np = linear_contribution.detach().cpu().numpy()
                
                # 2. Full neural network prediction (linear + nonlinear correction)
                neural_correction = self.model(z)
                u_total = linear_contribution + neural_correction
                u_total_np = u_total.detach().cpu().numpy()
            
            # Create functions in the original function space
            u_quadratic_linear = fem.Function(self.V)
            u_quadratic_linear.x.array[:] = linear_only_np
            
            u_quadratic_total = fem.Function(self.V)
            u_quadratic_total.x.array[:] = u_total_np
            
            # Interpolate to the visualization space
            u_linear_viz_linear = fem.Function(self.V_viz)
            u_linear_viz_linear.interpolate(u_quadratic_linear)
            linear_np = u_linear_viz_linear.x.array
            
            u_linear_viz_total = fem.Function(self.V_viz)
            u_linear_viz_total.interpolate(u_quadratic_total)
            total_np = u_linear_viz_total.x.array
            
            # Compute displacement magnitudes for both
            linear_mag = np.linalg.norm(linear_np.reshape((-1, 3)), axis=1)
            total_mag = np.linalg.norm(total_np.reshape((-1, 3)), axis=1)
            
            # Find global color range for consistent comparison
            max_mag = max(np.max(linear_mag), np.max(total_mag))
            min_mag = min(np.min(linear_mag), np.min(total_mag))
            color_range = [min_mag, max_mag]
            
            # Left subplot - Linear modes only
            self.viz_plotter.subplot(0, 0)
            
            # Remove previous mesh actor if it exists
            if self.mesh_actor_left is not None:
                self.viz_plotter.remove_actor(self.mesh_actor_left)
            
            # Create mesh with linear deformation only
            linear_grid = self.viz_grid.copy()
            linear_grid.point_data["displacement"] = linear_np.reshape((-1, 3))
            linear_grid["displacement_magnitude"] = linear_mag
            
            # Warp the mesh by the displacement
            linear_warped = linear_grid.warp_by_vector("displacement", factor=1)
            
            # Add mesh to left subplot
            self.mesh_actor_left = self.viz_plotter.add_mesh(
                linear_warped, 
                scalars="displacement_magnitude",
                cmap="viridis", 
                show_edges=False,
                clim=color_range,
                reset_camera=False
            )
            
            # Add title to left subplot
            self.viz_plotter.add_text("Linear Modes Only", position="upper_edge", font_size=12, color='black')
            
            # Right subplot - Full neural network prediction
            self.viz_plotter.subplot(0, 1)
            
            # Remove previous mesh actor if it exists
            if self.mesh_actor_right is not None:
                self.viz_plotter.remove_actor(self.mesh_actor_right)
            
            # Create mesh with full deformation
            total_grid = self.viz_grid.copy()
            total_grid.point_data["displacement"] = total_np.reshape((-1, 3))
            total_grid["displacement_magnitude"] = total_mag
            
            # Warp the mesh by the displacement
            total_warped = total_grid.warp_by_vector("displacement", factor=1)
            
            # Add mesh to right subplot
            self.mesh_actor_right = self.viz_plotter.add_mesh(
                total_warped, 
                scalars="displacement_magnitude",
                cmap="viridis", 
                show_edges=False,
                clim=color_range,
                reset_camera=False
            )
            
            # Add title to right subplot
            self.viz_plotter.add_text("Neural Network Prediction", position="upper_edge", font_size=12, color='black')
            
            # Update the training info text (on the main plotter)
            self.viz_plotter.subplot(0, 0)  # Add info to left plot for balance
            if iteration is not None and loss is not None:
                if self.info_actor is not None:
                    self.viz_plotter.remove_actor(self.info_actor)
                
                # Compute nonlinear contribution percentage
                nonlinear_mag = np.linalg.norm(neural_correction.detach().cpu().numpy())
                total_mag = np.linalg.norm(u_total_np)
                nonlinear_percent = (nonlinear_mag / total_mag) * 100 if total_mag > 0 else 0
                
                info_text = f"Iteration: {iteration}\nLoss: {loss:.6e}\nNonlinear Contribution: {nonlinear_percent:.2f}%"
                self.info_actor = self.viz_plotter.add_text(
                    info_text, 
                    position=(10, 10),
                    font_size=10,
                    color='black'
                )
            
            # Add colorbar (to the right subplot for space efficiency)
            self.viz_plotter.subplot(0, 1)
            self.viz_plotter.add_scalar_bar(title="Displacement Magnitude", n_labels=5, position_x=0.05)
            
            # Update the visualization without blocking training
            self.viz_plotter.update()
            self.viz_plotter.render()
            
        except Exception as e:
            # Don't let visualization errors halt the training process
            print(f"Visualization error (continuing training): {str(e)}")
            import traceback
            print(traceback.format_exc())
    

    
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
        safety_factor = 5
        
      
        
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
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best.pt'))

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
        scale = self.compute_safe_scaling_factor()    # Larger scale for better visualization
        values = np.linspace(-scale, scale, num_points)
        
        # Create plotter with subplots
        plotter = pyvista.Plotter(shape=(num_points, num_points), border=False)
        
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
                y = self.model(z)
                u_total =  y + linear_contribution
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
        
        # Visualize each mode with varying values
        for i, mode_idx in enumerate(modes_to_show):
            for j, val in enumerate(values):
                # Create a zero latent vector
                z = torch.zeros(self.latent_dim, device=self.device, dtype=torch.float64)
                
                # Set only the current mode to the current value
                z[mode_idx] = val
                
                # Compute the linear component and neural model prediction
                linear_contribution = (self.linear_modes[:, mode_idx] * val)
                y = self.model(z)
                u_total = y +  linear_contribution
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

    engine = Routine(cfg)
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
        best_checkpoint_path = os.path.join('checkpoints', 'best.pt')
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
    engine.visualize_latent_dimensions(dim1=1, dim2=2, num_points=5)
    
    # Optionally visualize other dimension pair
    engine.visualize_latent_dimensions(dim1=0, dim2=2, num_points=5)

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
