import argparse
import os, sys, logging
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from torch.utils.tensorboard import SummaryWriter
import math
import glob
import json
import matplotlib.pyplot as plt
import pyvista
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmshio

import ufl
import gmsh
from mpi4py import MPI
import ufl
from scipy.linalg import eig
from slepc4py import SLEPc
from petsc4py import PETSc
import yaml

logger = logging.getLogger('train_shell')

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

class StVenantKirchhoffShellEnergy(torch.nn.Module):
    """
    St. Venant-Kirchhoff shell energy with both membrane and bending terms.
    """
    def __init__(self, domain, V, thickness, E, nu, device=None, dtype=torch.float64):
        super(StVenantKirchhoffShellEnergy, self).__init__()

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype

        self.E = E
        self.nu = nu
        self.mu = E / (2 * (1 + nu))  # Shear modulus
        self.lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))  # Lamé's first parameter
        self.thickness = thickness
        
        # Calculate bending stiffness
        self.D = E * thickness**3 / (12 * (1 - nu**2))  # Bending stiffness
        self.kappa = 5.0/6.0  # Shear correction factor
        
        self.coordinates = torch.tensor(domain.geometry.x, dtype=self.dtype, device=self.device)
        self.num_nodes = self.coordinates.shape[0]
        
        # Element connectivity
        elements_list = []
        tdim = domain.topology.dim
        for cell in range(domain.topology.index_map(tdim).size_local):
            elements_list.append(domain.topology.connectivity(tdim, 0).links(cell))
        self.elements = torch.tensor(np.array(elements_list), dtype=torch.long, device=self.device)
        self.num_elements = len(self.elements)
        self.nodes_per_element = self.elements.shape[1]

        # Quadrature points (need more points for accurate bending)
        self.quadrature_points, self.quadrature_weights = self._generate_quadrature()
        print(f"Using {len(self.quadrature_weights)} quadrature points per element")

    def _generate_quadrature(self):
        """Generate Gauss quadrature points for triangles with 3-point rule"""
        # 3-point quadrature for triangles (better for bending)
        points = torch.tensor([
            [1/6, 1/6],
            [2/3, 1/6],
            [1/6, 2/3]
        ], dtype=self.dtype, device=self.device)
        weights = torch.tensor([1/6, 1/6, 1/6], dtype=self.dtype, device=self.device)
        return points, weights

    def forward(self, u_tensor):
        """Compute total elastic energy using vectorized operations"""
        # Handle both single vectors and batches
        batch_size = u_tensor.shape[0] if len(u_tensor.shape) > 1 else 1
        if batch_size == 1 and len(u_tensor.shape) == 1:
            u_tensor = u_tensor.unsqueeze(0)  # Add batch dimension
        
        # Reshape to [batch, nodes, 4]
        u_reshaped = u_tensor.reshape(batch_size, -1, 4)
        
        # Split displacement and rotation components
        u_disp = u_reshaped[:, :, :2]  # [batch, nodes, 2]
        u_rot = u_reshaped[:, :, 2:4]  # [batch, nodes, 2]
        
        # Get element coordinates (not batch-dependent)
        element_coords = self.coordinates[self.elements]  # [num_elements, 3, 2]
        
        # Process elements for each batch item (more reliable than gather for this case)
        element_energies = 0
        for b in range(batch_size):
            # Extract displacements and rotations for this batch item
            batch_disps = u_disp[b]  # [num_nodes, 2]
            batch_rots = u_rot[b]    # [num_nodes, 2]
            
            # Get displacement and rotation values for each element's nodes
            element_disps_b = batch_disps[self.elements]  # [num_elements, 3, 2]
            element_rots_b = batch_rots[self.elements]    # [num_elements, 3, 2]
            
            # Compute energy for this batch item
            membrane = self._compute_membrane_energy_single_batch(element_coords, element_disps_b)
            bending = self._compute_bending_energy_single_batch(element_coords, element_rots_b)
            stab = self._compute_stabilization_energy_single_batch(element_rots_b)
            
            # Sum energies for this batch item
            element_energies_b = membrane + bending + stab
            element_energies += element_energies_b.sum()
        
        return element_energies / batch_size  # Return average energy across batch
    
    def _compute_membrane_energy_single_batch(self, element_coords, element_disps):
        """Compute membrane energy for a single batch"""
        # element_coords: [num_elements, 3, 2]
        # element_disps: [num_elements, 3, 2]
        
        num_elements = element_coords.shape[0]
        
        # Compute derivatives once for all elements
        dN_dx, detJ = self._compute_derivatives_vectorized(element_coords)
        
        # Initialize deformation gradient for all elements: [num_elements, 2, 2]
        F = torch.eye(2, dtype=self.dtype, device=self.device)
        F = F.unsqueeze(0).expand(num_elements, -1, -1).clone()  # Add clone() here!
        
        # Add displacement contribution
        for i in range(self.nodes_per_element):
            for j in range(2):  # coordinate dimension
                for k in range(2):  # derivative dimension
                    # Use temporary values to avoid in-place overlapping operations
                    update = element_disps[:, i, j] * dN_dx[:, i, k]
                    F[:, j, k] = F[:, j, k] + update  # Replace += with explicit assignment
        
        # Compute Green-Lagrange strain tensor
        C = torch.bmm(F.transpose(1, 2), F)  # [num_elements, 2, 2]
        I = torch.eye(2, dtype=self.dtype, device=self.device).unsqueeze(0).expand(num_elements, -1, -1)
        E = 0.5 * (C - I)  # [num_elements, 2, 2]
        
        # Compute St. Venant-Kirchhoff energy
        trE = torch.diagonal(E, dim1=1, dim2=2).sum(dim=1)  # [num_elements]
        trEE = torch.bmm(E, E).diagonal(dim1=1, dim2=2).sum(dim=1)  # [num_elements]
        
        # Calculate energy density
        energy_density = 0.5 * self.lmbda * trE**2 + self.mu * trEE
        
        # Integrate over element
        return energy_density * detJ * self.thickness
    
    def _compute_bending_energy_single_batch(self, element_coords, element_rots):
        """Compute bending energy for a single batch"""
        # element_coords: [num_elements, 3, 2]
        # element_rots: [num_elements, 3, 2]
        
        num_elements = element_coords.shape[0]
        
        # Compute derivatives once for all elements
        dN_dx, detJ = self._compute_derivatives_vectorized(element_coords)
        
        # Initialize curvature tensor for all elements: [num_elements, 2, 2]
        kappa = torch.zeros((num_elements, 2, 2), dtype=self.dtype, device=self.device)
        
        # Add rotation contribution to curvature
        for i in range(self.nodes_per_element):
            for j in range(2):  # rotation component
                for k in range(2):  # derivative dimension
                    kappa[:, j, k] += element_rots[:, i, j] * dN_dx[:, i, k]
        
        # Extract curvature components
        kappa_xx = kappa[:, 0, 0]
        kappa_yy = kappa[:, 1, 1]
        kappa_xy = 0.5 * (kappa[:, 0, 1] + kappa[:, 1, 0])
        
        # Calculate bending energy density
        bending_energy_density = self.D * (
            kappa_xx**2 + kappa_yy**2 + 
            2*self.nu*kappa_xx*kappa_yy + 
            2*(1-self.nu)*kappa_xy**2
        )
        
        # Integrate over element
        return bending_energy_density * detJ

    def _compute_stabilization_energy_single_batch(self, element_rots):
        """Compute stabilization energy for a single batch"""
        # element_rots: [num_elements, 3, 2]
        
        # Simple drill stiffness for stabilization
        drill_factor = 1e-4 * self.E * self.thickness**3
        
        # Sum drill energy contributions from all nodes
        drill_energy = torch.sum(element_rots[:, :, 0]**2, dim=1)  # [num_elements]
        
        return drill_factor * drill_energy


    def _compute_derivatives_vectorized(self, element_coords):
        """Compute shape function derivatives for all elements at once"""
        # element_coords shape: [num_elements, 3, 2]
        num_elements = element_coords.shape[0]
        
        # Constant derivatives for linear triangles
        dN_dxi = torch.tensor([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]],
                            dtype=self.dtype, device=element_coords.device)
        
        # Expand for broadcasting: [1, 3, 2] -> [num_elements, 3, 2]
        dN_dxi = dN_dxi.unsqueeze(0).expand(num_elements, -1, -1)
        
        # Initialize Jacobian tensor for all elements: [num_elements, 2, 2]
        J = torch.zeros((num_elements, 2, 2), dtype=self.dtype, device=element_coords.device)
        
        # Vectorized Jacobian calculation
        for i in range(3):  # Loop over the 3 nodes (still needed but reduced loops)
            # element_coords[:, i, :] shape: [num_elements, 2]
            # dN_dxi[:, i, :] shape: [num_elements, 2]
            
            # Outer product for each element
            # For each element, add contribution from node i
            for j in range(2):  # coordinate dimension
                for k in range(2):  # derivative dimension
                    J[:, j, k] += element_coords[:, i, j] * dN_dxi[:, i, k]
        
        # Compute determinant and inverse for all elements at once
        detJ = torch.linalg.det(J)  # [num_elements]
        invJ = torch.linalg.inv(J)  # [num_elements, 2, 2]
        
        # Initialize derivative matrix for all elements: [num_elements, 3, 2]
        dN_dx = torch.zeros_like(dN_dxi)
        
        # Vectorized computation of derivatives
        for i in range(3):  # Shape functions
            for j in range(2):  # x,y derivatives
                for k in range(2):  # xi,eta derivatives
                    dN_dx[:, i, j] += dN_dxi[:, i, k] * invJ[:, k, j]
        
        return dN_dx, detJ
    
    
    

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
                            print(f"Epoch {self.last_epoch}: Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}")
                        self.wait_epochs = 0  # Reset patience counter

class Routine:
    def __init__(self, cfg):
        """
        Initializes the training routine for shell meshes.

        Args:
            cfg (dict): Configuration dictionary containing simulation parameters.
        """
        print("Initializing Routine...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float64
        self.cfg = cfg
        
        # Add this line to define fem_degree from config
        self.fem_degree = cfg['data']['fem_degree']
        print(f"Using FEM degree: {self.fem_degree}")

        # Mesh loading
        self.mesh_file = cfg['data']['mesh_file']
        print(f"Loading mesh: {self.mesh_file}")
        self.domain, self.cell_tags, self.facet_tags = gmshio.read_from_msh(self.mesh_file, MPI.COMM_WORLD, gdim=2)

        # Create function space directly with the dimension specified in the element tuple
        # This is the proper way in modern FEniCSx
        self.V = fem.functionspace(self.domain, 
                                ("Lagrange", self.fem_degree, (4,)))  # 4-vector field (2 disps + 2 rots)
        
        print("Function space defined for shell simulation.")
        print(f"Function space dimension: {self.V.dofmap.index_map.size_global}")


        # Material properties
        print("Defining material properties...")
        self.E = float(cfg['material']['youngs_modulus'])
        self.nu = float(cfg['material']['poissons_ratio'])
        self.rho = float(cfg['material']['density'])
        self.thickness = 0.05
        print(f"E = {self.E}, nu = {self.nu}, rho = {self.rho}, thickness = {self.thickness}")

        # Energy calculator
        print("Initializing energy calculator...")
        self.energy_calculator = StVenantKirchhoffShellEnergy(
        self.domain, self.V, self.thickness, self.E, self.nu, device=self.device, dtype=self.dtype
    ).to(self.device)

        # Model setup
        print("Loading neural network...")
        self.latent_dim = cfg['model']['latent_dim']
        num_nodes = self.V.dofmap.index_map.size_global
        components_per_node = 4  # 2 displacement + 2 rotation components per node
        self.output_dim = num_nodes * components_per_node

        print(f"Output dimension (total DOFs): {self.output_dim}")

       

        # Linear modes
        print("Computing linear modes...")
        self.linear_modes = self.compute_linear_modes()
        print(f"Shape of linear_modes: {self.linear_modes.shape}")
        print("Linear eigenmodes loaded.")
        self.latent_dim = self.linear_modes.shape[1]

         # Make sure we're using the correct output dimension when initializing the model
        self.model = Net(self.latent_dim, self.output_dim, 
                        cfg['model']['hid_layers'], 
                        cfg['model']['hid_dim']).to(self.device).double()
        print(f"Model initialized with output dimension: {self.model.layers[-1].out_features}")

        # Scaling factor
        self.scale = self.compute_safe_scaling_factor()
        print(f"Scaling factor: {self.scale}")

        # Optimizer and scheduler
        self.optimizer_adam = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.optimizer_lbfgs = torch.optim.LBFGS(self.model.parameters(),
                                                   lr=1,
                                                   max_iter=20,
                                                   max_eval=30,
                                                   tolerance_grad=1e-05,
                                                   tolerance_change=1e-07,
                                                   history_size=100,
                                                   line_search_fn="strong_wolfe")
        self.scheduler = LBFGSScheduler(self.optimizer_lbfgs,
                                         factor=0.5, patience=5, threshold=0.01,
                                         min_lr=1e-6, verbose=True)

        # Tensorboard writer
        self.writer = SummaryWriter(os.path.join('checkpoints', 'tensorboard'))

        print("Routine initialized.")
    

    def compute_linear_modes(self):
        """Compute linear modes using specialized MITC shell elements"""
        def eps_m_voigt(u):
            """Return membrane strain directly in Voigt notation"""
            e = ufl.sym(ufl.grad(u))
            return ufl.as_vector([e[0, 0], e[1, 1], 2*e[0, 1]])

        def eps_b_voigt(r):
            """Return bending strain directly in Voigt notation"""
            e = ufl.sym(ufl.grad(r))
            return ufl.as_vector([e[0, 0], e[1, 1], 2*e[0, 1]])
        # Define trial and test functions
        u_trial = ufl.TrialFunction(self.V)
        v_test = ufl.TestFunction(self.V)
        
        # Extract components with proper vector structure
        u_disp = ufl.as_vector([u_trial[0], u_trial[1]])  # displacements
        u_rot = ufl.as_vector([u_trial[2], u_trial[3]])   # rotations
        
        v_disp = ufl.as_vector([v_test[0], v_test[1]])
        v_rot = ufl.as_vector([v_test[2], v_test[3]])
        
        # Material parameters
        E = self.E
        nu = self.nu
        t = self.thickness
        D = E * t**3 / (12 * (1 - nu**2))  # Bending stiffness
        
        # MITC element formulation - improves thin shell performance
        # 1. Define transverse shear modulus with correction factor
        G = self.energy_calculator.mu
        kappa = 5.0/6.0  # Shear correction factor
        G_corr = kappa * G
        
        # 2. Membrane strain (standard)
        def eps_m(u):
            return ufl.sym(ufl.grad(u))
        
        # 3. Bending strain (curvature)
        def eps_b(r):
            return ufl.sym(ufl.grad(r))
        
        # 4. Improved transverse shear strain - key for MITC formulation
        def eps_s(u, r):
            """Transverse shear strain for MITC shell elements"""
            grad_u = ufl.grad(u)
            gamma_x = r[0] + grad_u[1, 0]  # rotation_x + ∂u_y/∂x
            gamma_y = r[1] + grad_u[0, 1]  # rotation_y + ∂u_x/∂y
            return ufl.as_vector([gamma_x, gamma_y])
        
        # 5. Define material matrices in Voigt notation
        # Membrane stiffness
        Dm = E*t/(1-nu**2) * ufl.as_matrix([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1-nu)/2]
        ])
        
        # Bending stiffness
        Db = D * ufl.as_matrix([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1-nu)/2]
        ])
        
        # Shear stiffness
        Ds = G_corr * t * ufl.Identity(2)
        
        # Add drilling stiffness to prevent zero-energy modes (critical for thin shells)
        drill_factor = 1e-4 * E * t**3  # Small but nonzero drilling stiffness
        
        # Membrane energy
        a_m = ufl.inner(ufl.dot(Dm, eps_m_voigt(u_disp)), eps_m_voigt(v_disp)) * ufl.dx
        
        # Bending energy term
        a_b = ufl.inner(ufl.dot(Db, eps_b_voigt(u_rot)), eps_b_voigt(v_rot)) * ufl.dx
            
        # Shear energy (reduced for thin shells to avoid locking)
        a_s = G_corr * t * ufl.inner(eps_s(u_disp, u_rot), eps_s(v_disp, v_rot)) * ufl.dx        
        
        # Drilling stiffness (stabilization term)
        a_d = drill_factor * ufl.inner(u_rot[0], v_rot[0]) * ufl.dx
        
        # Combined stiffness
        a = a_m + a_b + a_s + a_d
        
        # Mass matrix with proper rotational inertia
        rot_inertia_factor = t**2/12.0  # Physically correct factor
        M_form = self.rho * t * (ufl.inner(u_disp, v_disp) * ufl.dx + 
                            rot_inertia_factor * ufl.inner(u_rot, v_rot) * ufl.dx)
        
        # Create Dirichlet boundary condition for x_min edge
        # First locate the x_min nodes
        x_coordinates = self.domain.geometry.x
        x_min = np.min(x_coordinates[:, 0])
        x_min_tol = 1e-10  # Tolerance for identifying boundary nodes
        
        # Create boundary condition function
        def x_min_boundary(x):
            return np.isclose(x[0], x_min, atol=x_min_tol)
        
        # Create boundary condition
        boundary_dofs = fem.locate_dofs_geometrical(self.V, x_min_boundary)
        fixed_value = np.zeros(4)  # Fix all 4 components (displacements and rotations)
        bc = fem.dirichletbc(fixed_value, boundary_dofs, self.V)
        
        # Assemble matrices
        print("Assembling A matrix")
        A = fem.petsc.assemble_matrix(fem.form(a), bcs=[bc])
        A.assemble()
        print("Assembling M matrix")
        M = fem.petsc.assemble_matrix(fem.form(M_form))
        M.assemble()
        print("Matrices assembled")
        
        # Setup eigensolver
        print("Setting up eigensolver with robust settings for Neo-Hookean materials...")
        eigensolver = SLEPc.EPS().create(self.domain.comm)
        eigensolver.setOperators(A, M)
    
    # The rest of your code remains the same...

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
        #exclude the first 2 modes
        return torch.tensor(linear_modes, dtype=self.dtype, device=self.device)

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
        # Calculate characteristic length (average of dimensions)
        char_length = max(x_range, y_range)
        # Safety factor to avoid extreme deformations
        safety_factor = 0.2
        
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

    def train(self, num_epochs=1000):
        """
        Train the neural network to approximate the nonlinear solution manifold.

        Args:
            num_epochs (int): Number of training epochs.
        """
        print("Starting training...")
        self.model.train()  # Set the model to training mode
        
        # Define optimizer (Adam)
        self.optimizer_adam = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        # Define optimizer (LBFGS)
        self.optimizer_lbfgs = torch.optim.LBFGS(self.model.parameters(), 
                                            lr=1,
                                            max_iter=20,
                                            max_eval=30,
                                            tolerance_grad=1e-05,
                                            tolerance_change=1e-07,
                                            history_size=100,
                                            line_search_fn="strong_wolfe")
        
        # Add scheduler for learning rate reduction
        self.scheduler = LBFGSScheduler(
            self.optimizer_lbfgs,
            factor=0.5,  # Reduce LR by factor of 5 when plateau is detected
            patience=5,  # More patience for batch training
            threshold=0.01,  # Consider 1% improvement as significant
            min_lr=1e-6,  # Don't go below this LR
            verbose=True   # Print LR changes
        )
        
        best_loss = float('inf')
        iteration = 0
        print_every = 1
        batch_size = 32
        L = self.latent_dim
        
        # Find index of rest shape latent vector
        rest_idx = 0  # Assuming rest shape is always the first one
        
        # Main training loop
        while iteration < num_epochs:  # Set a maximum iteration count or use other stopping criteria
            # Generate random latent vectors and linear displacements
            with torch.no_grad():
                # Generate latent vectors with scaling (-0.625 to 0.625)
                z = torch.rand(batch_size, L, device=self.device) * self.scale * 2 - self.scale
                z[rest_idx, :] = 0  # Set rest shape latent to zero
                
                # Compute linear displacements
                l = torch.matmul(z, self.linear_modes.T)
                
                # Create normalized constraint directions
                constraint_dir = torch.matmul(z, self.linear_modes.T)
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
            # def closure():
            #     nonlocal energy_val, ortho_val, origin_val, loss_val
                
            #     self.optimizer_lbfgs.zero_grad()
                
            #     # Compute nonlinear correction
            #     y = self.model(z)
                
            #     # Compute energy (use your energy calculator)
            #     u_total_batch = l + y
                
               
                
            #     # After (if processing a batch):
            #     batch_size = u_total_batch.shape[0]
            #     if batch_size > 1:
            #         # Use batch processing for multiple samples
            #         energies = self.energy_calculator(u_total_batch)
            #         energy = torch.mean(energies)  # Average energy across batch
            #     else:
            #         # Use single-sample processing
            #         energy = self.energy_calculator(u_total_batch[0])

            #      # Calculate maximum displacements
            #     max_linear = torch.max(torch.norm(l.reshape(batch_size, -1, 2), dim=2)).item()
            #     max_total = torch.max(torch.norm(u_total_batch.reshape(batch_size, -1, 2), dim=2)).item()
            #     max_correction = torch.max(torch.norm(y.reshape(batch_size, -1, 2), dim=2)).item()
                
            #     # Compute orthogonality constraint (using the same approach as reference)
            #     ortho = torch.mean(torch.sum(y * constraint_dir, dim=1)**2)
                
            #     # Compute origin constraint for rest shape
            #     origin = torch.sum(y[rest_idx]**2)
                
            #     # Total loss with much stronger weights for constraints
            #     loss = energy + 1e2 * ortho + 1e2 * origin
                
            #     # Store values for use outside closure
            #     energy_val = energy.item()
            #     ortho_val = ortho.item()
            #     origin_val = origin.item()
            #     loss_val = loss.item()
                
            #     # Print components periodically
            #     if iteration % print_every == 0:
            #         print(f"[Iter {iteration}] Energy: {energy_val:.6f}, "
            #               f"Ortho: {ortho_val:.6e}, Origin: {origin_val:.6e}")
            #         print(f"===> Max linear: {max_linear:.4f}, "
            #               f"Max correction: {max_correction:.4f}, Max total: {max_total:.4f}")
                
            #     loss.backward()
            #     return loss
            def closure():
                nonlocal energy_val, ortho_val, origin_val, loss_val
                
                self.optimizer_lbfgs.zero_grad()
                
                # Compute nonlinear correction
                y = self.model(z)
                
                # Compute total displacements
                u_total_batch = l + y
                
                # Calculate linear and total energies based on batch size
                batch_size = u_total_batch.shape[0]
                
                # Calculate linear energy (only from linear components)
                with torch.no_grad():
                    if batch_size > 1:
                        linear_energies = self.energy_calculator(l)
                        linear_energy = torch.mean(linear_energies)
                    else:
                        linear_energy = self.energy_calculator(l[0])
                
                # Calculate total energy (linear + nonlinear contributions)
                if batch_size > 1:
                    total_energies = self.energy_calculator(u_total_batch)
                    total_energy = torch.mean(total_energies)
                else:
                    total_energy = self.energy_calculator(u_total_batch[0])
                
                # Compute nonlinearity measure (difference between total and linear energy)
                nonlinearity_measure = torch.abs(total_energy - linear_energy)
                
                # Apply nonlinearity reward (subtract to minimize energy while maximizing nonlinearity)
                nonlinearity_weight = 0.99 
                energy = total_energy - nonlinearity_weight * nonlinearity_measure
                
                # Calculate maximum displacements
                max_linear = torch.max(torch.norm(l.reshape(batch_size, -1, 2), dim=2)).item()
                max_total = torch.max(torch.norm(u_total_batch.reshape(batch_size, -1, 2), dim=2)).item()
                max_correction = torch.max(torch.norm(y.reshape(batch_size, -1, 2), dim=2)).item()
                
                # Compute orthogonality constraint (using the same approach as reference)
                ortho = torch.mean(torch.sum(y * constraint_dir, dim=1)**2)
                
                # Compute origin constraint for rest shape
                origin = torch.sum(y[rest_idx]**2)
                
                # Total loss with constraints
                loss = energy + 1e2 * ortho + 1e2 * origin
                
                # Store values for use outside closure
                energy_val = energy.item()
                ortho_val = ortho.item()
                origin_val = origin.item()
                loss_val = loss.item()
                
                # Print components periodically
                if iteration % print_every == 0:
                    print(f"[Iter {iteration}] Energy: {energy_val:.6f}, "
                        f"Ortho: {ortho_val:.6e}, Origin: {origin_val:.6e}")
                    print(f"Linear energy: {linear_energy.item():.6e}, Total energy: {total_energy.item():.6e}")
                    print(f"Nonlinearity: {nonlinearity_measure.item():.6e} (weight: {nonlinearity_weight:.2f})")
                    print(f"===> Max linear: {max_linear:.4f}, "
                        f"Max correction: {max_correction:.4f}, Max total: {max_total:.4f}")
                
                loss.backward()
                return loss
            self.optimizer_lbfgs.step(closure)
            
            # Update scheduler
            self.scheduler.step(loss_val)
            
            # Save best model
            if loss_val < best_loss:
                best_loss = loss_val
                self.save_checkpoint(iteration, best_loss, is_best=True)
                print(f"New best model at iteration {iteration} with loss {best_loss:.6e}")
            
            # Save checkpoint periodically
            if iteration % 50 == 0:
                self.save_checkpoint(iteration, loss_val, is_best=False)
            
            iteration += 1


    def visualize_latent_dimensions(self, dim1=0, dim2=1, num_points=3, fixed_value=0.0):
        """Visualize neural modes across a grid of two latent dimensions"""
        print(f"Visualizing neural modes for dimensions {dim1} and {dim2}...")
        
        # Convert DOLFINx mesh to PyVista format
        topology, cell_types, x = plot.vtk_mesh(self.domain)
        grid = pyvista.UnstructuredGrid(topology, cell_types, x)
        
        # Create a linear function space for visualization (3D vector field)
        V_viz = fem.functionspace(self.domain, ("CG", 1, (3,)))
        u_linear = fem.Function(V_viz)
        
        # Create an intermediate function space for extraction
        V_disp = fem.functionspace(self.domain, ("CG", 1, (2,)))
        u_disp = fem.Function(V_disp)
        
        # Compute scale for latent vectors
        scale = self.compute_safe_scaling_factor() * 1.5  # Larger scale for better visualization
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
                linear_contribution = (self.linear_modes[:, [dim1, dim2]] @ z[[dim1, dim2]].unsqueeze(1)).squeeze(1)
                y = self.model(z)
                u_total = y + linear_contribution
                u_total_np = u_total.detach().cpu().numpy()

                # Create a function in the quadratic function space
                u_quadratic = fem.Function(self.V)
                u_quadratic.x.array[:] = u_total_np
                
                # Extract only displacement components (first 2 components per node)
                # Create array for holding just displacements
                displacements = np.zeros((self.domain.geometry.x.shape[0], 3))
                
                # Extract the first 2 components per node (displacements) and pad with zeros for z
                num_nodes = self.domain.geometry.x.shape[0]
                for n in range(num_nodes):
                    # In-plane displacements (directly from solution)
                    displacements[n, 0] = u_quadratic.x.array[n*4]     # x displacement
                    displacements[n, 1] = u_quadratic.x.array[n*4+1]   # y displacement
                    
                    # Compute out-of-plane (Z) displacement from rotations
                    # For Reissner-Mindlin shells, Z displacement is related to rotations
                    # Negative rotation around Y axis causes positive Z displacement in X direction
                    # Positive rotation around X axis causes positive Z displacement in Y direction
                    theta_x = u_quadratic.x.array[n*4+2]  # Rotation around X
                    theta_y = u_quadratic.x.array[n*4+3]  # Rotation around Y
                    
                    # Compute Z displacement (simplified relationship for visualization)
                    # This approximation assumes small rotations
                    displacements[n, 2] = -10 * theta_y  # Scale factor to make bending visible
                
                # Set the 3D displacement values directly
                u_linear.x.array[:] = displacements.flatten()
                
                # Set active subplot
                plotter.subplot(row_idx, j)
                
                # Create mesh with deformation
                local_grid = grid.copy()
                local_grid.point_data["displacement"] = displacements
                local_grid["displacement_magnitude"] = np.linalg.norm(displacements, axis=1)
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
    
    # Rest of your visualization code remains unchanged...
        
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
                z = torch.zeros(self.latent_dim, device=self.device, dtype=torch.float64)
                
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
                
                # Extract only displacement components (first 2 components per node)
                displacements = np.zeros((self.domain.geometry.x.shape[0], 3))
                
                # Extract the first 2 components per node (displacements) and pad with zeros for z
                num_nodes = self.domain.geometry.x.shape[0]
                for n in range(num_nodes):
                    # In-plane displacements (directly from solution)
                    displacements[n, 0] = u_quadratic.x.array[n*4]     # x displacement
                    displacements[n, 1] = u_quadratic.x.array[n*4+1]   # y displacement
                    
                    # Compute out-of-plane (Z) displacement from rotations
                    # For Reissner-Mindlin shells, Z displacement is related to rotations
                    # Negative rotation around Y axis causes positive Z displacement in X direction
                    # Positive rotation around X axis causes positive Z displacement in Y direction
                    theta_x = u_quadratic.x.array[n*4+2]  # Rotation around X
                    theta_y = u_quadratic.x.array[n*4+3]  # Rotation around Y
                    
                    # Compute Z displacement (simplified relationship for visualization)
                    # This approximation assumes small rotations
                    displacements[n, 2] = -10 * theta_y  # Scale factor to make bending visible
                
                # Set the 3D displacement values directly
                u_linear.x.array[:] = displacements.flatten()
                
                # Set active subplot
                plotter.subplot(i, j)
                
                # Create mesh with deformation
                local_grid = grid.copy()
                local_grid.point_data["displacement"] = displacements
                local_grid["displacement_magnitude"] = np.linalg.norm(displacements, axis=1)
                
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

def load_config(config_file):
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

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
    engine.visualize_latent_dimensions(dim1=1, dim2=0, num_points=5, fixed_value=3.0)
    
    # Optionally visualize other dimension pair
    engine.visualize_latent_dimensions(dim1=0, dim2=1, num_points=5, fixed_value=-3.0)

    print("\nVisualizing latent space modes...")
    # Visualize all latent dimensions
    engine.visualize_latent_space(num_samples=5)
    
    print("Main function complete.")



if __name__ == '__main__':
    main()