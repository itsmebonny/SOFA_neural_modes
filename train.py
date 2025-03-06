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
        





class TorchElasticEnergy(torch.nn.Module):
    """
    Optimized PyTorch implementation of elastic energy calculation for FEM.
    Takes the best approaches from both implementations.
    """
    def __init__(self, domain, degree, E, nu, precompute_matrices=True, device=None):
        """Initialize with simplified parameters and better defaults"""
        super(TorchElasticEnergy, self).__init__()
        
        # Set device once and stick with it for all tensors
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Material properties
        self.E = E
        self.nu = nu
        self.mu = E / (2 * (1 + nu))
        self.lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
        
        # Extract mesh geometry
        self.coordinates = torch.tensor(domain.geometry.x, dtype=torch.float64, device=self.device)
        self.num_nodes = self.coordinates.shape[0]
        self.dim = self.coordinates.shape[1]
        
        # Get mesh connectivity with simplified approach
        elements_list = []
        tdim = domain.topology.dim
        for cell in range(domain.topology.index_map(tdim).size_local):
            elements_list.append(domain.topology.connectivity(tdim, 0).links(cell))
        
        # Convert directly to device tensor in one step
        self.elements = torch.tensor(np.array(elements_list), dtype=torch.long, device=self.device)
        self.num_elements = len(self.elements)
        self.nodes_per_element = self.elements.shape[1]
        
        # Determine element type for optimal quadrature
        if self.nodes_per_element == 4:
            self.element_type = "tetrahedron"
        elif self.nodes_per_element == 8:
            self.element_type = "hexahedron"
        else:
            raise ValueError(f"Unsupported element type with {self.nodes_per_element} nodes")
        
        print(f"Mesh: {self.num_elements} {self.element_type} elements, {self.num_nodes} nodes")
        
        # Generate optimal but simple quadrature
        self.quadrature_points, self.quadrature_weights = self._generate_simple_quadrature()
        print(f"Using {len(self.quadrature_weights)} quadrature points per element")
        
        # Precompute constitutive matrix (constant for all elements)
        self.D = self._compute_D_matrix()
        
        # Precompute B matrices - use a more efficient approach
        if precompute_matrices:
            print(f"Precomputing B matrices for {self.num_elements} elements...")
            # Store as a single batched tensor rather than lists
            self.B_matrices, self.detJs = self._precompute_B_matrices_efficiently()
            self.precomputed = True
        else:
            print(f"Computing B matrices on-the-fly")
            self.precomputed = False
    
    def _generate_simple_quadrature(self):
        """Generate simple but optimal quadrature based on element type"""
        if self.element_type == "tetrahedron":
            # 1-point quadrature for linear tetrahedron - very efficient
            points = torch.tensor([[0.25, 0.25, 0.25]], dtype=torch.float64, device=self.device)
            weights = torch.tensor([1.0/6.0], dtype=torch.float64, device=self.device)
        else:  # hexahedron
            # 2×2×2 Gaussian quadrature for hexahedrons
            gp = 1.0 / math.sqrt(3)
            points = []
            weights = []
            for i in [-gp, gp]:
                for j in [-gp, gp]:
                    for k in [-gp, gp]:
                        points.append([i, j, k])
                        weights.append(1.0)
            points = torch.tensor(points, dtype=torch.float64, device=self.device)
            weights = torch.tensor(weights, dtype=torch.float64, device=self.device)
        
        return points, weights
    
    def _precompute_B_matrices_efficiently(self):
        """Precompute B matrices more efficiently using batched operations where possible"""
        num_qp = len(self.quadrature_points)
        
        # Create storage tensors with appropriate shape for batch access
        # Instead of list of lists, use a 3D tensor directly
        B_shape = (6, self.nodes_per_element * self.dim)
        all_B = torch.zeros((self.num_elements, num_qp, *B_shape), 
                           dtype=torch.float64, device=self.device)
        all_detJ = torch.zeros((self.num_elements, num_qp), 
                              dtype=torch.float64, device=self.device)
        
        # Process elements in batches if there are many
        batch_size = min(100, self.num_elements)  # Process in smaller batches
        
        for batch_start in range(0, self.num_elements, batch_size):
            batch_end = min(batch_start + batch_size, self.num_elements)
            batch_elements = self.elements[batch_start:batch_end]
            
            for e_idx, element_nodes in enumerate(batch_elements):
                element_coords = self.coordinates[element_nodes]
                
                for q_idx, qp in enumerate(self.quadrature_points):
                    B, detJ = self._compute_B_matrix(element_coords, qp)
                    all_B[batch_start + e_idx, q_idx] = B
                    all_detJ[batch_start + e_idx, q_idx] = detJ
        
        return all_B, all_detJ
            
    def _compute_B_matrix(self, element_coords, qp):
        """Compute strain-displacement matrix - optimized by element type"""
        num_nodes = element_coords.shape[0]
        
        if num_nodes == 4:  # tetrahedron
            # Shape function derivatives for tetrahedron (constant)
            dN_dxi = torch.tensor([
                [-1.0, -1.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=torch.float64, device=element_coords.device)
            
            # Jacobian calculation
            J = torch.matmul(element_coords.T, dN_dxi)
            detJ = torch.det(J)
            invJ = torch.linalg.inv(J)
            
            # Shape function derivatives w.r.t. physical coordinates
            dN_dx = torch.matmul(dN_dxi, invJ)
            
            # Vectorized B matrix assembly for tetrahedral element (6x12)
            B = torch.zeros((6, 12), dtype=torch.float64, device=element_coords.device)
            
            # Fast vectorized assignment using slicing
            B[0, 0::3] = dN_dx[:, 0]  # du/dx
            B[1, 1::3] = dN_dx[:, 1]  # dv/dy
            B[2, 2::3] = dN_dx[:, 2]  # dw/dz
            B[3, 0::3] = dN_dx[:, 1]  # du/dy
            B[3, 1::3] = dN_dx[:, 0]  # dv/dx
            B[4, 1::3] = dN_dx[:, 2]  # dv/dz
            B[4, 2::3] = dN_dx[:, 1]  # dw/dy
            B[5, 0::3] = dN_dx[:, 2]  # du/dz
            B[5, 2::3] = dN_dx[:, 0]  # dw/dx
            
        else:  # hexahedron with 8 nodes
            xi, eta, zeta = qp
            
            # Precompute terms for efficiency
            xim = 1.0 - xi
            xip = 1.0 + xi
            etam = 1.0 - eta
            etap = 1.0 + eta
            zetam = 1.0 - zeta
            zetap = 1.0 + zeta
            
            # Shape function derivatives (optimized)
            dN_dxi = torch.tensor([
                [-0.125 * etam * zetam, -0.125 * xim * zetam, -0.125 * xim * etam],
                [0.125 * etam * zetam, -0.125 * xip * zetam, -0.125 * xip * etam],
                [0.125 * etap * zetam, 0.125 * xip * zetam, -0.125 * xip * etap],
                [-0.125 * etap * zetam, 0.125 * xim * zetam, -0.125 * xim * etap],
                [-0.125 * etam * zetap, -0.125 * xim * zetap, 0.125 * xim * etam],
                [0.125 * etam * zetap, -0.125 * xip * zetap, 0.125 * xip * etam],
                [0.125 * etap * zetap, 0.125 * xip * zetap, 0.125 * xip * etap],
                [-0.125 * etap * zetap, 0.125 * xim * zetap, 0.125 * xim * etap]
            ], dtype=torch.float64, device=element_coords.device)
            
            # Jacobian calculation
            J = torch.matmul(element_coords.T, dN_dxi)
            detJ = torch.det(J)
            invJ = torch.linalg.inv(J)
            
            # Shape function derivatives w.r.t. physical coordinates
            dN_dx = torch.matmul(dN_dxi, invJ)
            
            # Vectorized B matrix assembly for hexahedral element (6x24)
            B = torch.zeros((6, 24), dtype=torch.float64, device=element_coords.device)
            
            # Fast vectorized assignment using slicing
            B[0, 0::3] = dN_dx[:, 0]  # du/dx
            B[1, 1::3] = dN_dx[:, 1]  # dv/dy
            B[2, 2::3] = dN_dx[:, 2]  # dw/dz
            B[3, 0::3] = dN_dx[:, 1]  # du/dy
            B[3, 1::3] = dN_dx[:, 0]  # dv/dx
            B[4, 1::3] = dN_dx[:, 2]  # dv/dz
            B[4, 2::3] = dN_dx[:, 1]  # dw/dy
            B[5, 0::3] = dN_dx[:, 2]  # du/dz
            B[5, 2::3] = dN_dx[:, 0]  # dw/dx
            
        return B, detJ
    
    def _compute_D_matrix(self):
        """Compute the constitutive matrix D"""
        D = torch.zeros((6, 6), dtype=torch.float64, device=self.device)
        
        # Fill the matrix (using engineering notation for shear strains)
        factor = self.E / ((1 + self.nu) * (1 - 2 * self.nu))
        
        # Normal stress components
        D[0, 0] = D[1, 1] = D[2, 2] = factor * (1 - self.nu)
        D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = factor * self.nu
        
        # Shear stress components
        D[3, 3] = D[4, 4] = D[5, 5] = factor * (1 - 2 * self.nu) / 2
        
        return D
    
    def forward(self, u_tensor):
        """Compute elastic energy with optimized precomputed approach"""
        # Move u_tensor to device if needed
        if u_tensor.device != self.device:
            u_tensor = u_tensor.to(self.device)
        
        # Reshape displacement vector
        u = u_tensor.reshape(self.num_nodes, self.dim)
        
        # Get element displacements - single efficient operation
        element_displacements = u[self.elements].reshape(self.num_elements, -1)
        
        # Initialize total energy
        total_energy = 0.0
        
        if self.precomputed:
            # Highly optimized version using batched operations
            # Calculate strains and stresses for all elements and quadrature points at once
            # This avoids expensive Python loops and uses efficient BLAS operations
            
            # Loop only over quadrature points (usually very few)
            for q in range(len(self.quadrature_points)):
                # Get B matrices for this quadrature point for all elements
                B_q = self.B_matrices[:, q]  # [num_elements, 6, nodes*dim]
                
                # Calculate strains for all elements at once
                # B_q: [num_elements, 6, nodes*dim], element_displacements: [num_elements, nodes*dim]
                strains = torch.bmm(B_q, element_displacements.unsqueeze(2)).squeeze(2)  # [num_elements, 6]
                
                # Calculate stresses for all elements at once
                stresses = torch.matmul(strains, self.D.T)  # [num_elements, 6]
                
                # Calculate energy contributions
                element_energies = 0.5 * torch.sum(strains * stresses, dim=1) * self.detJs[:, q] * self.quadrature_weights[q]
                
                # Accumulate total energy
                total_energy += torch.sum(element_energies)
        else:
            # Fallback implementation with on-the-fly calculation
            for e in range(self.num_elements):
                e_coords = self.coordinates[self.elements[e]]
                e_displ = element_displacements[e]
                
                for q, qp in enumerate(self.quadrature_points):
                    B, detJ = self._compute_B_matrix(e_coords, qp)
                    
                    # Compute strain and stress
                    strain = torch.matmul(B, e_displ)
                    stress = torch.matmul(self.D, strain)
                    
                    # Energy increment
                    total_energy += 0.5 * torch.dot(strain, stress) * detJ * self.quadrature_weights[q]
        
        return total_energy
    
    def compute_batch_energy(self, batch_u):
        """Efficiently compute energy for multiple displacement fields with less memory overhead"""
        batch_size = batch_u.shape[0]
        energies = torch.zeros(batch_size, dtype=torch.float64, device=self.device)
        
        # For small batches, it's faster to process directly
        if batch_size <= 4:
            for i in range(batch_size):
                energies[i] = self.forward(batch_u[i])
            return energies
        
        # For larger batches, we need to be more careful with memory
        # Process in chunks to avoid OOM
        chunk_size = 4
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            sub_batch = batch_u[i:end_idx]
            
            # Reshape each displacement field in the sub-batch
            u_reshaped = sub_batch.reshape(-1, self.num_nodes, self.dim)
            sub_batch_size = end_idx - i
            
            # Optimized version for precomputed B matrices
            if self.precomputed:
                # Gather element displacements for each batch item in one operation
                for b in range(sub_batch_size):
                    # Get element displacements - single efficient operation
                    element_disps = u_reshaped[b][self.elements].reshape(self.num_elements, -1)
                    
                    # Process quadrature points
                    energy = 0.0
                    for q in range(len(self.quadrature_points)):
                        # Get B matrices for this quadrature point for all elements
                        B_q = self.B_matrices[:, q]
                        
                        # Calculate strains for all elements at once
                        strains = torch.bmm(B_q, element_disps.unsqueeze(2)).squeeze(2)
                        
                        # Calculate stresses for all elements at once
                        stresses = torch.matmul(strains, self.D.T)
                        
                        # Calculate energy contributions
                        element_energies = 0.5 * torch.sum(strains * stresses, dim=1) * self.detJs[:, q] * self.quadrature_weights[q]
                        energy += torch.sum(element_energies)
                    
                    energies[i + b] = energy
            else:
                # Fallback path for non-precomputed B matrices
                for b in range(sub_batch_size):
                    energies[i + b] = self.forward(sub_batch[b])
        
        return energies
    
    
class NeoHookeanElasticEnergy(torch.nn.Module):
    """
    Memory-efficient implementation of Neo-Hookean elastic energy calculation,
    inspired by the formulation from pablo_loss.py
    """
    def __init__(self, domain, degree, E, nu, precompute_matrices=True, device=None, 
                 dtype=torch.float64):
        super(NeoHookeanElasticEnergy, self).__init__()
        
        # Set device and precision (keeping float64 for numerical stability)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        
        # Material properties
        self.E = E
        self.nu = nu
        self.mu = E / (2 * (1 + nu))  # Shear modulus
        self.lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))  # First Lamé parameter
        
        # Extract mesh information
        self.coordinates = torch.tensor(domain.geometry.x, dtype=self.dtype, device=self.device)
        self.num_nodes = self.coordinates.shape[0]
        self.dim = self.coordinates.shape[1]
        
        # Create elements tensor with int32 for memory efficiency
        elements_list = []
        tdim = domain.topology.dim
        for cell in range(domain.topology.index_map(tdim).size_local):
            elements_list.append(domain.topology.connectivity(tdim, 0).links(cell))
        
        self.elements = torch.tensor(np.array(elements_list), dtype=torch.int32, device=self.device)
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
        if precompute_matrices and self.num_elements < 100000:
            est_memory = self._estimate_precompute_memory()
            if est_memory < 1.0:  # Less than 1GB
                print(f"Precomputing shape derivatives (est. memory: {est_memory:.2f} GB)")
                self.dN_dx_list = []
                self.detJ_list = []
                
                # Process in chunks to manage memory
                chunk_size = 1000
                for chunk_start in range(0, self.num_elements, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, self.num_elements)
                    for e in range(chunk_start, chunk_end):
                        element_nodes = self.elements[e].long()
                        element_coords = self.coordinates[element_nodes]
                        
                        element_dN_dx = []
                        element_detJ = []
                        for q in range(len(self.quadrature_points)):
                            dN_dx, detJ = self._compute_derivatives(element_coords, self.quadrature_points[q])
                            element_dN_dx.append(dN_dx)
                            element_detJ.append(detJ)
                        
                        self.dN_dx_list.append(element_dN_dx)
                        self.detJ_list.append(element_detJ)
                    
                    # Clear GPU cache periodically
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                self.precomputed = True
            else:
                print(f"On-the-fly computation (est. precompute memory: {est_memory:.2f} GB)")
        else:
            print("Computing derivatives on-the-fly to save memory")

    def _estimate_precompute_memory(self):
        """Estimate memory needed for precomputation in GB"""
        num_qp = len(self.quadrature_points)
        bytes_per_element = (
            # dN_dx: (nodes_per_element × dim) values per quadrature point
            num_qp * self.nodes_per_element * self.dim * self.dtype.itemsize +
            # detJ: 1 value per quadrature point
            num_qp * self.dtype.itemsize
        )
        return (bytes_per_element * self.num_elements) / (1024**3)  # Convert to GB

    def _generate_quadrature(self):
        """Generate appropriate quadrature rules based on element type"""
        if self.element_type == "tetrahedron":
            # Use 4-point quadrature for tetrahedron (higher order for nonlinear materials)
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
    
    def _compute_derivatives(self, element_coords, qp):
        """Compute shape function derivatives for an element at a quadrature point"""
        num_nodes = element_coords.shape[0]
        
        if num_nodes == 4:  # tetrahedron
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
            
            # Shape function derivatives (optimized)
            dN_dxi = torch.zeros((8, 3), dtype=self.dtype, device=element_coords.device)
            
            # Fill dN_dxi directly for memory efficiency
            dN_dxi[0, 0] = -0.125 * etam * zetam
            dN_dxi[0, 1] = -0.125 * xim * zetam
            dN_dxi[0, 2] = -0.125 * xim * etam
            
            dN_dxi[1, 0] = 0.125 * etam * zetam
            dN_dxi[1, 1] = -0.125 * xip * zetam
            dN_dxi[1, 2] = -0.125 * xip * etam
            
            dN_dxi[2, 0] = 0.125 * etap * zetam
            dN_dxi[2, 1] = 0.125 * xip * zetam
            dN_dxi[2, 2] = -0.125 * xip * etap
            
            dN_dxi[3, 0] = -0.125 * etap * zetam
            dN_dxi[3, 1] = 0.125 * xim * zetam
            dN_dxi[3, 2] = -0.125 * xim * etap
            
            dN_dxi[4, 0] = -0.125 * etam * zetap
            dN_dxi[4, 1] = -0.125 * xim * zetap
            dN_dxi[4, 2] = 0.125 * xim * etam
            
            dN_dxi[5, 0] = 0.125 * etam * zetap
            dN_dxi[5, 1] = -0.125 * xip * zetap
            dN_dxi[5, 2] = 0.125 * xip * etam
            
            dN_dxi[6, 0] = 0.125 * etap * zetap
            dN_dxi[6, 1] = 0.125 * xip * zetap
            dN_dxi[6, 2] = 0.125 * xip * etap
            
            dN_dxi[7, 0] = -0.125 * etap * zetap
            dN_dxi[7, 1] = 0.125 * xim * zetap
            dN_dxi[7, 2] = 0.125 * xim * etap
            
            # Jacobian calculation
            J = torch.matmul(element_coords.T, dN_dxi)
            detJ = torch.det(J)
            invJ = torch.linalg.inv(J)
            
            # Shape function derivatives w.r.t. physical coordinates
            dN_dx = torch.matmul(dN_dxi, invJ)
            
        return dN_dx, detJ
    
    def _compute_F(self, element_coords, element_disps, dN_dx):
        """Compute deformation gradient F = I + ∇u"""
        # Initialize deformation gradient as identity
        F = torch.eye(3, dtype=self.dtype, device=self.device)
        
        # Add displacement gradient
        for i in range(self.nodes_per_element):
            for j in range(3):  # u, v, w components
                for k in range(3):  # x, y, z derivatives
                    F[j, k] += element_disps[i, j] * dN_dx[i, k]
                    
        return F
    
    def _compute_energy_density(self, F):
        """
        Compute Neo-Hookean strain energy density using the formulation from pablo_loss.py
        
        W = 0.5 * mu * (I₁ - 3 - 2*ln(J)) + 0.25 * lambda * (J² - 1 - 2*ln(J))
        
        Where:
        - I₁ = trace(F^T F) = trace(C) is first invariant of right Cauchy-Green tensor
        - J = det(F) is the determinant of deformation gradient
        """
        # Compute J = det(F)
        J = torch.det(F)
        
        # First invariant of right Cauchy-Green tensor C = F^T F
        I1 = torch.einsum('ji,ij->', F, F)  # = trace(F^T F)
        
        # Handle numerical issues for extreme compression
        if J < 1e-10:
            # Apply severe penalty
            return torch.tensor(1e10, device=self.device, dtype=self.dtype)
            
        # Neo-Hookean strain energy density with volumetric regularization
        log_J = torch.log(J)
        isochoric_term = 0.5 * self.mu * (I1 - 3.0 - 2.0 * log_J)
        volumetric_term = 0.25 * self.lambda_ * ((J*J) - 1.0 - 2.0 * log_J)
        
        energy = isochoric_term + volumetric_term
        return energy
    
    def forward(self, u_tensor):
        """Compute total elastic energy for the displacement field"""
        # Ensure tensor is on the correct device and type
        if u_tensor.device != self.device or u_tensor.dtype != self.dtype:
            u_tensor = u_tensor.to(device=self.device, dtype=self.dtype)
        
        # Reshape displacement vector
        u = u_tensor.reshape(self.num_nodes, self.dim)
        
        # Initialize total energy
        total_energy = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        
        # Process elements in chunks for memory efficiency
        batch_size = 50
        for batch_start in range(0, self.num_elements, batch_size):
            batch_end = min(batch_start + batch_size, self.num_elements)
            batch_energy = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            
            for e in range(batch_start, batch_end):
                # Get element nodes and displacements
                element_nodes = self.elements[e].long()
                element_coords = self.coordinates[element_nodes]
                element_disps = u[element_nodes]
                
                # Element energy from all quadrature points
                element_energy = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                
                # Sum over quadrature points
                for q_idx in range(len(self.quadrature_points)):
                    # Get or compute derivatives
                    if self.precomputed:
                        dN_dx = self.dN_dx_list[e][q_idx]
                        detJ = self.detJ_list[e][q_idx]
                    else:
                        qp = self.quadrature_points[q_idx]
                        dN_dx, detJ = self._compute_derivatives(element_coords, qp)
                    
                    # Compute deformation gradient
                    F = self._compute_F(element_coords, element_disps, dN_dx)
                    
                    # Compute strain energy density
                    energy_density = self._compute_energy_density(F)
                    
                    # Numerical integration: energy_density * det(J) * weight
                    element_energy += energy_density * detJ * self.quadrature_weights[q_idx]
                
                batch_energy += element_energy
            
            # Add batch energy to total
            total_energy += batch_energy
            
            # Clean up to save memory
            if torch.cuda.is_available() and batch_start % 200 == 0:
                torch.cuda.empty_cache()
        
        return total_energy
    
    def compute_batch_energy(self, batch_u):
        """Process batch of displacement fields with memory optimization"""
        batch_size = batch_u.shape[0]
        energies = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
        
        # For small batches, process directly
        if batch_size <= 2:
            for i in range(batch_size):
                energies[i] = self.forward(batch_u[i])
            return energies
        
        # For larger batches, process one at a time to save memory
        for i in range(batch_size):
            energies[i] = self.forward(batch_u[i])
            
            # Free memory periodically
            if torch.cuda.is_available() and i % 2 == 1:
                torch.cuda.empty_cache()
        
        return energies
    


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

class Routine:
    def __init__(self, cfg):
        print("Initializing Routine...")
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda') if use_cuda else torch.device('cpu')
        self.device = device
        print(f"Using device: {self.device}")

        # Load mesh from file
        filename = cfg['data']['mesh_file']
        print(f"Loading mesh from file: {filename}")
        self.domain, self.cell_tags, self.facet_tags = gmshio.read_from_msh(filename, MPI.COMM_WORLD, gdim=3)
        print("Mesh loaded successfully.")

        # Define function space
        print("Defining function space...")
        self.fem_degree = cfg['data'].get('fem_degree', 1)  # Default to 1 if not specified
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

        # Gravity and scaling
        self.g = cfg['physics']['gravity']
        print(f"g = {self.g}")

 

        # self.energy_calculator = TorchElasticEnergy(
        # self.domain, 
        # self.fem_degree, 
        # self.E, 
        # self.nu,
        # precompute_matrices=True,
        # device=self.device
        # ).to(self.device)

        self.energy_calculator = PabloNeoHookeanEnergy(
        self.domain, 
        self.fem_degree, 
        self.E, 
        self.nu,
        precompute_matrices=True,
        device=self.device
    ).to(self.device)

        self.scale = self.compute_safe_scaling_factor()
        print(f"Scaling factor: {self.scale}")


        # In __init__ method of Routine class:
        print("Loading neural network...")
        self.latent_dim = cfg['model']['latent_dim']
        self.num_modes = self.latent_dim  # Make them the same
        output_dim = self.V.dofmap.index_map.size_global * self.domain.geometry.dim  # Full DOF space size
        hid_layers = cfg['model'].get('hid_layers', 2)  # Default to 2 hidden layers if not specified
        hid_dim = cfg['model'].get('hid_dim', 64)      # Default to 64 neurons per layer if not specified
        print(f"Output dimension: {output_dim}")
        print(f"Network architecture: {hid_layers} hidden layers with {hid_dim} neurons each")
        self.model = Net(self.latent_dim, output_dim, hid_layers, hid_dim).to(device).double()



        print(f"Neural network loaded. Latent dim: {self.latent_dim}, Num Modes: {self.num_modes}")

        # Load linear eigenmodes (replace with your actual loading)
        print("Loading linear eigenmodes...")
        self.linear_modes = self.compute_linear_modes()
        self.linear_modes = torch.tensor(self.linear_modes, device=self.device).double()
        print("Linear eigenmodes loaded.")

        # Tensorboard
        self.writer = SummaryWriter(os.path.join(cfg['training']['checkpoint_dir'], cfg['training']['tensorboard_dir']))

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
                
            elif isinstance(self.energy_calculator, NeoHookeanElasticEnergy) or \
                isinstance(self.energy_calculator, PabloNeoHookeanEnergy):
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
        L = self.latent_dim  # Use at most 3 linear modes
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
                                    max_iter=20,
                                    max_eval=30,
                                    tolerance_grad=1e-05,
                                    tolerance_change=1e-07,
                                    history_size=100,
                                    line_search_fn="strong_wolfe")
        
        # Add scheduler for learning rate reduction
        scheduler = LBFGSScheduler(
            optimizer,
            factor=0.5,  # Reduce LR by factor of 5 when plateau is detected
            patience=5,  # More patience for batch training
            threshold=0.01,  # Consider 1% improvement as significant
            min_lr=1e-6,  # Don't go below this LR
            verbose=True   # Print LR changes
        )
        patience = 0
        # Main training loop
        while iteration < num_epochs:  # Set a maximum iteration count or use other stopping criteria
            # Generate random latent vectors and linear displacements
            with torch.no_grad():
                # Generate latent vectors with scaling (-0.625 to 0.625)
                z = torch.rand(batch_size, L, device=self.device) * self.scale * 2 - self.scale
                z[rest_idx, :] = 0  # Set rest shape latent to zero
                
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
                nonlocal energy_val, ortho_val, origin_val, loss_val
                
                optimizer.zero_grad()
                
                # Compute nonlinear correction
                y = self.model(z)
                
                # Compute energy (use your energy calculator)
                u_total_batch = l + y
                
                
                
                # After (if processing a batch):
                batch_size = u_total_batch.shape[0]
                if batch_size > 1:
                    # Use batch processing for multiple samples
                    energies = self.energy_calculator.compute_batch_energy(u_total_batch)
                    energy = torch.mean(energies)  # Average energy across batch
                else:
                    # Use single-sample processing
                    energy = self.energy_calculator(u_total_batch[0])

                # Calculate maximum displacements
                max_linear = torch.max(torch.norm(l.reshape(batch_size, -1, 3), dim=2)).item()
                max_total = torch.max(torch.norm(u_total_batch.reshape(batch_size, -1, 3), dim=2)).item()
                max_correction = torch.max(torch.norm(y.reshape(batch_size, -1, 3), dim=2)).item()
                
                # Compute orthogonality constraint (using the same approach as reference)
                ortho = torch.mean(torch.sum(y * constraint_dir, dim=1)**2)
                
                # Compute origin constraint for rest shape
                origin = torch.sum(y[rest_idx]**2)
                
                # Total loss with much stronger weights for constraints
                loss = energy + 0.5 * ortho + 0.1 * origin
                
                # Store values for use outside closure
                energy_val = energy.item()
                ortho_val = ortho.item()
                origin_val = origin.item()
                loss_val = loss.item()
                
                # Print components periodically
                if iteration % print_every == 0:
                    print(f"[Iter {iteration}] Energy: {energy_val:.6f}, "
                        f"Ortho: {ortho_val:.6e}, Origin: {origin_val:.6e}")
                    print(f"===> Max linear: {max_linear:.4f}, "
                        f"Max correction: {max_correction:.4f}, Max total: {max_total:.4f}")
                
                loss.backward()
                return loss
            
            # Perform optimization step
            optimizer.step(closure)

            scheduler.step(loss_val)  # Use the loss value to determine if we should reduce LR

            
            # Record metrics using values captured from closure
            self.writer.add_scalar('train/loss', loss_val, iteration)
            self.writer.add_scalar('train/energy', energy_val, iteration)
            self.writer.add_scalar('train/ortho', ortho_val, iteration)
            self.writer.add_scalar('train/origin', origin_val, iteration)
            
            # Save checkpoint if this is the best model so far
            if loss_val < best_loss:
                best_loss = loss_val
                checkpoint = {
                    'epoch': iteration,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_val,
                }
                patience = 0
                torch.save(checkpoint, os.path.join('checkpoints', 'best.pt'))
                print(f"New best model at iteration {iteration} with loss {loss_val:.6e}")
            
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
            if loss_val < 1e-3 or patience > 30:
                print(f"Converged to target loss at iteration {iteration}")
                break
        
        # Restore original forward method
        self.model.forward = original_forward
        
        print(f"Training complete. Best loss: {best_loss:.8e}")
        return best_loss
    
    def visualize_stresses(self, latent_vector):
        """Visualize von Mises stresses for a given latent vector"""
        z = torch.tensor(latent_vector, device=self.device, dtype=torch.float64)
        
        # Generate displacement field
        with torch.no_grad():
            l = (self.linear_modes @ z.unsqueeze(1)).squeeze(1)
            y = self.model(z)
            u_total = l + y
        
        # Compute strains and stresses
        result = self.energy_calculator.compute_strains_stresses(u_total)
        stresses = result['stresses']
        
        # Convert from Voigt notation to tensor form and compute von Mises stress
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
                
                # Compute von Mises stress: sqrt(3/2 * s_dev : s_dev)
                # where s_dev = s - (trace(s)/3) * I
                trace_s = torch.trace(stress_tensor)
                s_dev = stress_tensor - (trace_s/3) * torch.eye(3, device=stresses.device)
                von_mises_val = torch.sqrt(1.5 * torch.sum(s_dev * s_dev))
                von_mises.append(von_mises_val.item())
        


  


    def evaluate_generalization(self, num_samples=10):
        """Evaluate the generalization capability by sampling different latent variables"""
        print("\nEvaluating generalization capability...")
        
        # Compute ground truth solutions for different force configurations
        force_configs = []
        ground_truths = []
        
        for i in range(num_samples):
            # Create different force configurations by scaling the base forces
            scale = 0.5 + i * 0.5  # 0.5, 1.0, 1.5, ...
            force_magnitudes = [scale * 1e4, -scale * 4e4, -scale * 3e4, scale * 1e4]
            
            # Solve for this configuration
            print(f"\nSolving for force configuration {i+1}/{num_samples} (scale={scale})...")
            gt = self.solve_static(force_magnitudes=force_magnitudes)
            
            force_configs.append(force_magnitudes)
            ground_truths.append(gt)
        
        # Evaluate neural network predictions for each configuration
        rmse_values = []
        energy_diff_values = []
        
        for i, (force_config, gt) in enumerate(zip(force_configs, ground_truths)):
            print(f"\nEvaluating configuration {i+1}/{num_samples}...")
            
            # Find latent vector that best approximates this configuration
            z = self.compute_modal_coordinates(
                gt.x.array,
                self.linear_modes,
                self.M
            )

            # Convert z to a PyTorch tensor
            z = torch.tensor(z, device=self.device, dtype=torch.float64)
            
            # Compute prediction using this latent vector
            l = (self.linear_modes @ z.unsqueeze(1)).squeeze(1)
            y = self.model(z)
            u_total = l + y
            u_total_np = u_total.detach().cpu().numpy()
            
            # Calculate metrics
            gt_array = gt.x.array
            error = u_total_np - gt_array
            rmse = np.sqrt(np.mean(error**2))
            
            u_fenics = fem.Function(self.V)
            u_fenics.x.array[:] = u_total_np
            
            gt_energy = fem.assemble_scalar(fem.form(0.5 * ufl.inner(self.sigma(gt), 
                                                                self.epsilon(gt)) * ufl.dx))
            pred_energy = fem.assemble_scalar(fem.form(0.5 * ufl.inner(self.sigma(u_fenics), 
                                                                self.epsilon(u_fenics)) * ufl.dx))
            energy_diff = abs(gt_energy - pred_energy) / gt_energy
            
            rmse_values.append(rmse)
            energy_diff_values.append(energy_diff)
            
            print(f"Configuration {i+1}: RMSE = {rmse:.6e}, Rel. Energy Diff = {energy_diff:.6f}")
        
        # Print summary statistics
        print("\n----- Generalization Evaluation Summary -----")
        print(f"Average RMSE: {np.mean(rmse_values):.6e}")
        print(f"Average Relative Energy Difference: {np.mean(energy_diff_values):.6f}")
        print("-------------------------------------------\n")
        
        # Create visualization of results
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, num_samples+1), rmse_values, 'o-')
        plt.title('RMSE for Different Configurations')
        plt.xlabel('Configuration')
        plt.ylabel('RMSE')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, num_samples+1), energy_diff_values, 'o-')
        plt.title('Relative Energy Difference')
        plt.xlabel('Configuration')
        plt.ylabel('Relative Energy Diff')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def optimize_latent_for_target(self, target, num_iters=100):
        """Find the latent vector that best approximates a target solution"""
        z = torch.zeros(self.latent_dim, requires_grad=True, device=self.device)
        optimizer = torch.optim.LBFGS([z], lr=0.1, max_iter=20)
        
        target_array = torch.tensor(target.x.array, device=self.device)
        
        for i in range(num_iters):
            def closure():
                optimizer.zero_grad()
                l = (self.linear_modes @ z.unsqueeze(1)).squeeze(1)
                y = self.model(z)
                u_total = l + y
                loss = torch.nn.functional.mse_loss(u_total, target_array)
                loss.backward()
                return loss
            
            optimizer.step(closure)
            
        return z
    
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
        safety_factor = 0.75
        
      
        
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
        """Load model from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} not found")
            return
        
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model parameters
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded model from epoch {checkpoint['epoch']+1} with loss {checkpoint['loss']:.6e}")
        return checkpoint['loss']
    
    def visualize_latent_dimensions(self, dim1=0, dim2=1, num_points=3, fixed_value=0.0):
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
        
        # Generate neural modes for each combination of latent values
        for i, val1 in enumerate(values):
            row_idx = num_points - 1 - i  # Reverse order for proper cartesian layout
            for j, val2 in enumerate(values):
                # Create latent vector with fixed values except for the two selected dims
                z = torch.full((self.latent_dim,), fixed_value, device=self.device)
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
                z = torch.rand(self.latent_dim, device=self.device, dtype=torch.float64)
                
                # Set only the current mode to the current value
                z[mode_idx] = val
                
                # Compute the linear component and neural model prediction
                linear_contribution = (self.linear_modes[:, mode_idx] * val)
                y = self.model(z)
                u_total = y + linear_contribution
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
                            cmap="coolwarm", show_edges=True)
                
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
    engine.visualize_latent_dimensions(dim1=0, dim2=2, num_points=5, fixed_value=3.0)
    
    # Optionally visualize other dimension pair
    engine.visualize_latent_dimensions(dim1=2, dim2=1, num_points=5, fixed_value=-3.0)

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
