import argparse
import os, sys, logging
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from torch.utils.tensorboard import SummaryWriter
import math
from torch.func import vmap, jvp


from dolfinx.fem.petsc import assemble_matrix as assemble_matrix_petsc
from dolfinx.fem import form 

from ufl import TrialFunction, TestFunction, inner, dx, grad, sym, Identity, div

import traceback
from scipy import sparse as sp








import matplotlib.pyplot as plt

import pyvista
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmshio
import gmsh
from mpi4py import MPI
import ufl
from scipy.linalg import eig
from tests.solver import UFLNeoHookeanModel

# Add after imports
from slepc4py import SLEPc
from petsc4py import PETSc

logger = logging.getLogger('train')

def setup_logger(name, log_dir="logs", level=logging.INFO, use_colors=True):
    """Set up logger with console and file handlers"""
    # Create logs directory if not exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear any existing handlers
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[94m',     # Blue
        'INFO': '\033[97m',      # White
        'WARNING': '\033[93m',   # Yellow
        'ERROR': '\033[91m',     # Red
        'CRITICAL': '\033[91m\033[1m',  # Bright Red
        'RESET': '\033[0m'       # Reset color
    }
    
    # Format for console (with colors)
    if use_colors:
        class ColoredFormatter(logging.Formatter):
            def format(self, record):
                levelname = record.levelname
                if levelname in COLORS:
                    record.levelname = f"{COLORS[levelname]}{levelname}{COLORS['RESET']}"
                    record.msg = f"{COLORS[levelname]}{record.msg}{COLORS['RESET']}"
                return super().format(record)
        
        console_formatter = ColoredFormatter('[%(asctime)s] %(name)s %(levelname)s: %(message)s',
                                           datefmt='%m/%d %H:%M:%S')
    else:
        console_formatter = logging.Formatter('[%(asctime)s] %(name)s %(levelname)s: %(message)s',
                                           datefmt='%m/%d %H:%M:%S')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (no colors in file)
    file_formatter = logging.Formatter('[%(asctime)s] %(name)s %(levelname)s: %(message)s',
                                     datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
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

       
        # Initialize the last layer with smaller weights to avoid large initial outputs
        torch.nn.init.zeros_(self.layers[-1].weight)
        
    def forward(self, x):
        # Handle both single vectors and batches
        is_batched = x.dim() > 1
        if not is_batched:
            x = x.unsqueeze(0)  # Add batch dimension if not present
        
        # Apply GELU activation to all but the last layer
        for i in range(len(self.layers)-1):
            x = torch.nn.functional.leaky_relu(self.layers[i](x))
        
        # No activation on output layer
        x = self.layers[-1](x)
        
        # Remove batch dimension if input wasn't batched
        if not is_batched:
            x = x.squeeze(0)
        return x
    

class ResidualNet(torch.nn.Module):
    """
    Residual network architecture that ensures model(0) = 0 by design.
    Uses skip connections and ensures zero output for zero input.
    """
    def __init__(self, latent_dim, output_dim, hid_layers=3, hid_dim=128, 
                 activation=torch.nn.functional.leaky_relu, zero_init_output=True):
        """
        Neural network with residual connections and guaranteed zero-mapping property.
        
        Args:
            latent_dim: Dimension of input latent space
            output_dim: Dimension of output space
            hid_layers: Number of hidden layers
            hid_dim: Width of hidden layers
            activation: Activation function to use (default: leaky_relu)
            zero_init_output: Whether to initialize the output layer to near-zero weights
        """
        super(ResidualNet, self).__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.activation = activation

        # Input projection layer (latent → hidden) - No bias
        self.input_proj = torch.nn.Linear(latent_dim, hid_dim, bias=False)

        # Residual blocks - No bias in linear layers
        self.res_blocks = torch.nn.ModuleList()
        for _ in range(hid_layers):
            block = torch.nn.Sequential(
            torch.nn.Linear(hid_dim, hid_dim, bias=False),
            torch.nn.LayerNorm(hid_dim, elementwise_affine=False), # Keep affine=False to avoid learnable shift (bias)
            torch.nn.Linear(hid_dim, hid_dim, bias=False),
            torch.nn.LayerNorm(hid_dim, elementwise_affine=False) # Keep affine=False
            )
            self.res_blocks.append(block)

        # Output projection (hidden → output) - No bias
        self.output_proj = torch.nn.Linear(hid_dim, output_dim, bias=False)

        # Initialize weights
        # self._init_weights(zero_init_output)
        #zero last layer weights
        torch.nn.init.zeros_(self.output_proj.weight)
        # torch.nn.init.normal_(self.output_proj.bias, mean=0.0, std=0.01)


    def _init_weights(self, zero_init_output):
        """Initialize network weights properly for stable training and zero mapping."""
        # Initialize input projection with Kaiming
        torch.nn.init.kaiming_normal_(self.input_proj.weight, nonlinearity='leaky_relu')
        
        # Initialize residual blocks
        for block in self.res_blocks:
            for i, layer in enumerate(block):
                if isinstance(layer, torch.nn.Linear):
                    # For linear layers, use Kaiming initialization
                    torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
                    # Scale down the second layer in each block for stability
                    if i == 2:  # Second linear layer in the block
                        layer.weight.data.mul_(0.1)
        
        # Initialize output projection to small values
        if zero_init_output:
            torch.nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.01)
        else:
            torch.nn.init.kaiming_normal_(self.output_proj.weight, nonlinearity='linear')
            self.output_proj.weight.data.mul_(0.01)  # Scale down regardless
    
    def forward(self, x):
        """
        Forward pass with residual connections.
        Ensures model(0) = 0 by design due to absence of bias terms.
        
        Args:
            x: Input latent vector(s)
        """
        # Handle both single vectors and batches
        is_batched = x.dim() > 1
        if not is_batched:
            x = x.unsqueeze(0)  # Add batch dimension
            
        # Input projection & activation
        h = self.activation(self.input_proj(x))
        
        # Process through residual blocks
        for block in self.res_blocks:
            # Compute block output
            block_output = block[0](h)  # First linear
            block_output = self.activation(block_output)
            block_output = block[1](block_output)  # First normalization
            block_output = block[2](block_output)  # Second linear
            block_output = self.activation(block_output)
            block_output = block[3](block_output)  # Second normalization
            
            # Add residual connection
            h = h + block_output
            
        # Output projection
        y = self.output_proj(h)
        
        # Remove batch dimension if input wasn't batched
        if not is_batched:
            y = y.squeeze(0)
            
        return y
    
    def verify_zero_property(self, epsilon=1e-6):
        """Verify that the network maps zero to zero by testing f(0) ≈ 0"""
        with torch.no_grad():
            zero_input = torch.zeros(self.latent_dim, device=next(self.parameters()).device)
            zero_output = self.forward(zero_input)
            zero_norm = torch.norm(zero_output).item()
            
            print(f"Zero-mapping test: ||f(0)|| = {zero_norm:.3e}")
            passed = zero_norm < epsilon * self.output_dim**0.5
            print(f"Zero-mapping test {'PASSED' if passed else 'FAILED'}")
            
            return passed, zero_norm

class Routine:
    def __init__(self, cfg):
        print("Initializing Routine...")
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda') if use_cuda else torch.device('cpu')
        self.device = device
        # print(f"Using device: {self.device}")

        # Load mesh from file
        # Modified to match default.yaml structure
        filename = cfg.get('mesh', {}).get('filename', cfg.get('data', {}).get('mesh_file', 'mesh/beam_732.msh'))
        # print(f"Loading mesh from file: {filename}")
        self.domain, self.cell_tags, self.facet_tags = gmshio.read_from_msh(filename, MPI.COMM_WORLD, gdim=3)
        # print("Mesh loaded successfully.")

        # Define function space - handle both key structures
        # print("Defining function space...")
        self.fem_degree = cfg.get('mesh', {}).get('fem_degree', 
                        cfg.get('data', {}).get('fem_degree', 1))  # Default to 1 if not specified
        # print(f"Using FEM degree: {self.fem_degree}")
        self.V = fem.functionspace(self.domain, ("Lagrange", self.fem_degree, (self.domain.geometry.dim, )))
        # print("Function space defined.")
        # print(f"Function space dimension: {self.V.dofmap.index_map.size_global * 3}")

        # Define material properties
        # print("Defining material properties...")
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
        # print(f"g = {self.g}")

        # Choose energy calculator 
        self.energy_calculator = UFLNeoHookeanModel(
                self.domain, self.fem_degree, self.E, self.nu,
                precompute_matrices=True, device=self.device
            ).to(self.device)
        self.scale = self.compute_safe_scaling_factor()
        # print(f"Scaling factor: {self.scale}")

        # Load neural network
        
        self.latent_dim = cfg['model']['latent_dim']
       

        # Load linear modes
        # print("Loading linear eigenmodes...")
        self.linear_modes = self.compute_linear_modes()
        self.linear_modes = torch.tensor(self.linear_modes, device=self.device).double()
        #remove the first 6 modes
        # print(f"Initial linear modes shape: {self.linear_modes.shape}")
        # self.linear_modes = self.linear_modes[:, 6:]
        print(f"Linear eigenmodes loaded. Shape: {self.linear_modes.shape}")

        # Convert boundary DOFs to tensor for use in training
        if hasattr(self, 'boundary_dofs') and self.boundary_dofs is not None:
            # Ensure boundary_dofs are unique and sorted if necessary
            unique_boundary_dofs = np.unique(self.boundary_dofs)
            self.boundary_dofs_th = torch.tensor(unique_boundary_dofs, device=self.device, dtype=torch.long)
            print(f"Stored {len(self.boundary_dofs_th)} unique boundary DOFs for penalty.")
        else:
            self.boundary_dofs_th = None
            print("Warning: Boundary DOFs not found or stored. BC penalty will not be applied.")


        # print("Loading neural network...")
        self.num_modes = self.latent_dim  # Make them the same
        output_dim = self.V.dofmap.index_map.size_global * self.domain.geometry.dim
        hid_layers = cfg['model'].get('hid_layers', 2)
        hid_dim = cfg['model'].get('hid_dim', 64)
        print(f"Output dimension: {output_dim}")


        # self.model = Net(
        #     self.num_modes, 
        #     output_dim, 
        #     hid_layers=hid_layers, 
        #     hid_dim=hid_dim
        # ).to(device).double()
        # print(f"Network architecture: {hid_layers} hidden layers with {hid_dim} neurons each")


        self.model = ResidualNet(
            self.num_modes, 
            output_dim, 
            hid_layers=hid_layers, 
            hid_dim=hid_dim,
            activation=torch.nn.functional.leaky_relu,
            zero_init_output=True
        ).to(device).double()

        print(f"Neural network loaded. Latent dim: {self.latent_dim}, Num Modes: {self.num_modes}")


        # Tensorboard setup
        checkpoint_dir = cfg.get('training', {}).get('checkpoint_dir', 'checkpoints')
        tensorboard_dir = cfg.get('training', {}).get('tensorboard_dir', 'tensorboard')
        self.writer = SummaryWriter(os.path.join(checkpoint_dir, tensorboard_dir))


    
    

    def compute_linear_modes(self):
        """Compute linear eigenmodes with improved method from linear_modes.py"""
        print("Computing linear modes with improved solver settings...")
        
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

        # Set up forms based on energy calculator type
        if hasattr(self, 'energy_calculator') and hasattr(self.energy_calculator, 'type'):
            print(f"Using {self.energy_calculator.type} forms for linear modes")
        else:
            print("Using standard linear elasticity forms for linear modes")
        
        # Define strain tensor
        def epsilon(u):
            return sym(grad(u))

        # Linear elasticity stress tensor
        def sigma(u):
            return self.lambda_ * div(u) * Identity(3) + 2 * self.mu * epsilon(u)
            
        # Define stiffness and mass forms
        a_form = inner(sigma(u_tr), epsilon(u_test)) * dx
        m_form = self.rho * inner(u_tr, u_test) * dx

        # Define boundary condition at fixed end (x_min)
        x_min_tol = 1e-10
        def x_min_boundary(x):
            return np.isclose(x[0], x_min, atol=x_min_tol)
        
        # Create a function for the fixed values
        u_fixed = fem.Function(self.V)
        u_fixed.x.array[:] = 0.0  # Set all values to zero
        
        # Create boundary condition using the function
        boundary_dofs = fem.locate_dofs_geometrical(self.V, x_min_boundary)
        bc = fem.dirichletbc(u_fixed, boundary_dofs)
        self.boundary_dofs = boundary_dofs # Store the boundary DOFs
        self.bc = bc


        # ---- Assemble matrices WITHOUT boundary conditions (new approach) ----
        print("Assembling A matrix WITHOUT boundary conditions")
        A = assemble_matrix_petsc(form(a_form)) # No bcs here
        A.assemble()
        
        print("Assembling M matrix WITHOUT boundary conditions")
        M = assemble_matrix_petsc(form(m_form)) # No bcs here
        M.assemble()
        
        # ---- Apply boundary conditions AFTER assembly to preserve symmetry ----
        print("Applying boundary conditions symmetrically with zeroRowsColumns")
        constrained_dofs = bc.dof_indices()[0]
        
        if constrained_dofs.size > 0:
            A.zeroRowsColumns(constrained_dofs, diag=1.0)
            M.zeroRowsColumns(constrained_dofs, diag=0.0)
            print(f"Applied zeroRowsColumns to {constrained_dofs.size} DOFs")
        else:
            print("Warning: No constrained DOFs found, check boundary condition setup")
        
        # ---- Check symmetry ----
        is_A_sym = A.isSymmetric(tol=1e-9)
        is_M_sym = M.isSymmetric(tol=1e-9)
        print(f"Stiffness matrix A is symmetric: {is_A_sym}")
        print(f"Mass matrix M is symmetric: {is_M_sym}")
        if not is_A_sym or not is_M_sym:
            print("WARNING: Matrices are not symmetric! Eigenvalue solver may fail or give incorrect results.")
        
        # ---- Optional: Create hybrid mass matrix (improved approach) ----
        use_hybrid_mass = True  # Set to False to use consistent mass matrix
        lumping_ratio = 0.4     # Ratio of lumped mass (0=consistent, 1=fully lumped)
        
        if use_hybrid_mass:
            print(f"Creating hybrid mass matrix with lumping ratio {lumping_ratio}")
            # Convert M to SciPy CSR for lumping calculations
            ai, aj, av = M.getValuesCSR()
            M_scipy_consistent = sp.csr_matrix((av, aj, ai), shape=M.getSize())

            # Create a fully lumped mass matrix (diagonal only)
            lumped_diag = np.array(M_scipy_consistent.sum(axis=1)).flatten()
            M_scipy_lumped = sp.diags(lumped_diag, format='csr')

            # Blend the matrices
            M_scipy_hybrid = M_scipy_lumped * lumping_ratio + M_scipy_consistent * (1 - lumping_ratio)

            # Preserve total mass
            total_mass_consistent = M_scipy_consistent.sum()
            total_mass_hybrid = M_scipy_hybrid.sum()
            if total_mass_hybrid > 1e-15:
                mass_scaling = total_mass_consistent / total_mass_hybrid
                M_scipy_hybrid = M_scipy_hybrid * mass_scaling
                print(f"Hybrid mass matrix scaled by {mass_scaling:.4f} to preserve total mass")
            else:
                print("Warning: Hybrid mass matrix has near-zero total mass, skipping scaling")

            # Convert back to PETSc
            M_hybrid_petsc = PETSc.Mat().createAIJ(size=M_scipy_hybrid.shape,
                                    csr=(M_scipy_hybrid.indptr, M_scipy_hybrid.indices, M_scipy_hybrid.data))
            M_hybrid_petsc.assemble()
            M_final = M_hybrid_petsc
            print("Hybrid mass matrix created and converted to PETSc")
        else:
            M_final = M
            print("Using consistent mass matrix")
        
        # Store for later use
        self.A = A
        self.M = M_final
        
        # ---- Setup eigensolver with improved settings ----
        print("Creating eigensolver with improved settings")
        eigensolver = SLEPc.EPS().create(self.domain.comm)
        eigensolver.setOperators(A, M_final)
        eigensolver.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)

        # Use target magnitude with a small positive target
        eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)  
        eigensolver.setTarget(0.001)  # Small positive target

        # Use spectral transformation for better convergence
        st = eigensolver.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        st.setShift(0.0)

        # Set dimensions with some extra padding
        requested_modes = self.latent_dim * 2  # Request more to ensure we get enough
        eigensolver.setDimensions(requested_modes, PETSc.DECIDE)

        # Increase solver tolerance and max iterations
        eigensolver.setTolerances(tol=1e-6, max_it=1000)

        print(f"Solving eigenvalue problem for {requested_modes} requested modes...")
        eigensolver.solve()
        
        # ---- Extract eigenvectors and eigenvalues with improved filtering ----
        nconv = eigensolver.getConverged()
        print(f"Number of converged eigenvalues: {nconv}")
        
        # Initialize arrays to store valid modes
        modes = []
        eigenvalues = []
        actual_modes_found = 0
        
        # Prototype vector for extracting eigenvectors
        vr = A.createVecRight()
        
        for i in range(nconv):
            # Extract eigenvalue
            eigenvalue = eigensolver.getEigenvalue(i)
            lambda_real = eigenvalue.real
            
            # Filter out negative or near-zero eigenvalues (likely rigid body modes)
            if lambda_real < 1e-9:
                print(f"  Skipping eigenvalue {i+1}: Near-zero or negative ({lambda_real:.4e})")
                continue
            
            # Extract eigenvector for valid eigenvalue
            eigensolver.getEigenvector(i, vr)
            
            # Store valid eigenvalue and eigenvector
            eigenvalues.append(lambda_real)
            modes.append(vr.array.copy())
            
            # Display frequency information
            frequency = np.sqrt(lambda_real) / (2 * np.pi)  # Convert to Hz
            print(f"  Mode {actual_modes_found+1}: λ={lambda_real:.6e}, Frequency: {frequency:.4f} Hz")
            
            actual_modes_found += 1
            
            # Stop once we have enough modes
            if actual_modes_found >= self.latent_dim:
                break
        
        print(f"Successfully extracted {actual_modes_found} valid eigenmodes")
        
        # Handle insufficient modes case
        if actual_modes_found < self.latent_dim:
            print(f"WARNING: Requested {self.latent_dim} modes, but only found {actual_modes_found}")
            print("Reducing latent dimension to match number of modes found")
            self.latent_dim = actual_modes_found
        
        # Handle case when no modes are found
        if len(modes) == 0:
            print("ERROR: No valid modes could be computed. Using random initialization.")
            random_modes = np.random.rand(A.getSize()[0], self.latent_dim)
            self.eigenvalues = np.ones(self.latent_dim)  # Default eigenvalues
            return random_modes
        
        # Store eigenvalues for later use in scaling
        self.eigenvalues = np.array(eigenvalues)
        print(f"Eigenvalues: {self.eigenvalues}")
        
        # Return modal matrix
        linear_modes = np.column_stack(modes)
        print(f"Shape of linear_modes: {linear_modes.shape}")
        return linear_modes
    


    
    def compute_eigenvalue_based_scale(self, mode_index=None):
        """
        Compute scaling factor for latent variables based on eigenvalues
        
        Args:
            mode_index: Specific mode index to get scaling for, or None for all modes
            
        Returns:
            Scaling factor or array of scaling factors
        """
        if not hasattr(self, 'eigenvalues') or self.eigenvalues is None:
            # print("Warning: No eigenvalues available, using default scaling")
            return self.compute_safe_scaling_factor()
        
        # Check if we have all eigenvalues needed
        if mode_index is not None and mode_index >= len(self.eigenvalues):
            # print(f"Warning: Requested mode {mode_index} exceeds available eigenvalues")
            return self.compute_safe_scaling_factor()
        
        # For neo-Hookean materials, scale is inversely proportional to sqrt(eigenvalue)
        # This is because energy is proportional to eigenvalue * displacement^2
        if mode_index is not None:
            # Return scale for specific mode
            return 1.0 / np.sqrt(max(1e-8, self.eigenvalues[mode_index]))
        else:
            # Return array of scales for all modes
            return 1.0 / np.sqrt(np.maximum(1e-8, self.eigenvalues))
        

    def compute_volume_comparison(self, z):
        """Use energy calculator to compute volume comparison for a given latent vector"""
        # Ensure proper tensor format
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z, device=self.device, dtype=torch.float64)
        
        if z.dim() > 1:
            z = z.squeeze()
        
        # Compute displacements
        with torch.no_grad():
            # Linear component only
            linear_contribution = torch.matmul(z, self.linear_modes.T)
            
            # Full neural network prediction
            neural_correction = self.model(z)
            u_total = linear_contribution + neural_correction
        
        # Use energy calculator to compute volume comparison
        result = self.energy_calculator.compute_volume_comparison(
            linear_contribution.detach(), u_total.detach())
        
        # Print detailed results
        print("\n====== VOLUME PRESERVATION ANALYSIS ======")
        print(f"Original volume:         {result['original_volume']:.6f}")
        print(f"Linear modes volume:     {result['linear_volume']:.6f} (ratio: {result['linear_volume_ratio']:.6f})")
        print(f"Neural network volume:   {result['neural_volume']:.6f} (ratio: {result['neural_volume_ratio']:.6f})")
        print(f"Volume change - Linear:  {(result['linear_volume_ratio']-1)*100:.4f}%")
        print(f"Volume change - Neural:  {(result['neural_volume_ratio']-1)*100:.4f}%")
 
        return result
        

    def train(self, num_epochs=1000):
        """
        Train the model using batched processing with strong orthogonality constraints.
        Similar to the reference implementation with St. Venant-Kirchhoff energy.
        """
        print("Starting training...")
        
        # Setup training parameters
        batch_size = 64  # You can add this to config
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
                                    lr=5,
                                    max_iter=20,
                                    tolerance_grad=1e-05,
                                    tolerance_change=1e-07,
                                    history_size=100,
                                    line_search_fn="strong_wolfe")
        

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
                deformation_scale_init = 2
                deformation_scale_final = 30
                #current_scale = deformation_scale_init * (deformation_scale_final/deformation_scale_init)**(iteration/num_epochs) #expoential scaling
                current_scale = deformation_scale_init + (deformation_scale_final - deformation_scale_init)# * (iteration/num_epochs) #linear scaling

                print(f"Current scale: {current_scale}")
                mode_scales = torch.tensor(self.compute_eigenvalue_based_scale(), device=self.device, dtype=torch.float64)[:L]
                mode_scales = mode_scales * current_scale

                # Generate samples with current scale
                z = torch.rand(batch_size, L, device=self.device) * current_scale * 2 - current_scale

                #scale each sample to randomly between 0.001 and 1 to cover smaller and larger scales
                z = z * torch.rand(batch_size, 1, device=self.device) * 0.999 + 0.001


                # z = torch.rand(batch_size, L, device=self.device) * mode_scales * 2 - mode_scales
                # z[rest_idx, :] = 0  # Set rest shape latent to zero
                #concatenate the generated samples with the rest shape
                
                # Compute linear displacements
                l = torch.matmul(z, linear_modes.T)
                
                # Create normalized constraint directions
                constraint_dir = torch.matmul(z, linear_modes.T)
                constraint_norms = torch.norm(constraint_dir, p=2, dim=1, keepdim=True)
                # Avoid division by zero
                constraint_norms = torch.clamp(constraint_norms, min=1e-8)
                constraint_dir = constraint_dir / constraint_norms
                # constraint_dir[rest_idx] = 0  # Zero out rest shape constraints
            
                # Track th0ese values outside the closure
                energy_val = 0
                ortho_val = 0
                origin_val = 0
                bc_penalty_val = 0 # Add variable for BC penalty
                loss_val = 0

                # Define closure for optimizer
                def closure():
                    nonlocal energy_val, ortho_val, origin_val, bc_penalty_val, loss_val, lbfgs_iter, iteration, best_loss # Add bc_penalty_val
                    nonlocal z, l, rest_idx, constraint_dir


                    lbfgs_iter += 1
                    
                    optimizer.zero_grad()
                    
                    # Compute nonlinear correction
                    y = self.model(z)
                    
                    # Compute energy (use your energy calculator)
                    u_total_batch = l + y
                    
                    
                    
                    # After (if processing a batch):
                    batch_size = u_total_batch.shape[0]

                    energies = self.energy_calculator(u_total_batch)
                    energy = torch.mean(energies)  # Average energy across batch
             

                    volume_sample_indices = [0, min(5, batch_size-1), rest_idx]  # Rest shape + a couple samples
                    volume_results = []
                    for idx in volume_sample_indices:
                        vol_result = self.energy_calculator.compute_volume_comparison(
                            l[idx:idx+1], u_total_batch[idx:idx+1])
                        volume_results.append(vol_result)
                    
                    # Calculate average volume metrics across the samples
                    avg_linear_ratio = sum(r['linear_volume_ratio'] for r in volume_results) / len(volume_results)
                    avg_neural_ratio = sum(r['neural_volume_ratio'] for r in volume_results) / len(volume_results)
                    
                    # Compute volume preservation penalty (squared deviation from 1.0)
                    vol_penalty = 1000.0 * torch.mean((torch.tensor(
                        [r['neural_volume_ratio'] for r in volume_results], 
                        device=self.device, dtype=torch.float64) - 1.0)**2)
                    
                    # # We need gradients w.r.t. z for JVP, and w.r.t. u_total_batch for energy gradient
                    # # Recompute l inside closure if needed, or ensure z requires grad if l depends on it outside
                    # # Assuming z is generated outside and doesn't require grad initially for JVP purposes.
                    # # Let's make a version of z that requires grad for the energy gradient part.
                    # z_for_grad = z.detach().clone().requires_grad_(True)
                    # l_for_grad = torch.matmul(z_for_grad, linear_modes.T) # Recompute l with grad tracking

                    # y = self.model(z_for_grad) # Pass grad-tracking z
                    # u_total_batch = l_for_grad + y
                    # # Ensure energy calculation tracks gradients w.r.t. u_total_batch
                    # energies = self.energy_calculator(u_total_batch) # Assumes energy_calculator is differentiable
                    # energy = torch.mean(energies)
                    # # --- End Energy Calculation ---

                    # # --- Standard Loss Components ---
                    # # Use y computed with grad-tracking z here as well
                    # ortho = torch.mean(torch.sum(y * constraint_dir, dim=1)**2) # constraint_dir is from non-grad z
                    # origin = torch.sum(y[rest_idx]**2)
                    # bc_penalty = torch.tensor(0.0, device=self.device, dtype=torch.float64)
                    # if self.boundary_dofs_th is not None and len(self.boundary_dofs_th) > 0:
                    #     fixed_displacements = u_total_batch[:, self.boundary_dofs_th]
                    #     log_cosh_penalty = torch.log(torch.cosh(fixed_displacements))
                    #     bc_penalty = torch.mean(log_cosh_penalty)

                    # # --- Energy-based Monotonicity Penalty ---
                    # monotonicity_penalty = torch.tensor(0.0, device=self.device, dtype=torch.float64)
                    # mono_weight = 1e2 # TUNABLE WEIGHT

                    # # Exclude the rest shape (z=0)
                    # valid_indices = torch.arange(batch_size, device=self.device) #!= rest_idx
                    # z_valid_no_grad = z[valid_indices] # Use original z for JVP
                    # l_valid_no_grad = l[valid_indices] # Use original l

                    # if z_valid_no_grad.shape[0] > 0:
                    #     # 1. Compute JVP: J(z) * z
                    #     def single_model_forward(latent_vec):
                    #         # Ensure model uses non-grad-tracking input for JVP
                    #         return self.model(latent_vec.detach())
                    #     compute_jvp_batch = vmap(lambda zi: jvp(single_model_forward, (zi,), (zi,))[1])
                    #     jvp_results = compute_jvp_batch(z_valid_no_grad) # Shape: (batch_size-1, output_dim)

                    #     # 2. Compute v = l + J(z)*z
                    #     v = l_valid_no_grad + jvp_results

                    #     # 3. Compute Energy Gradient: ∇_u E(u(z))
                    #     # We need the gradient of the *sum* of energies w.r.t u_total_batch
                    #     # to get per-sample gradients correctly scaled.
                    #     # Use retain_graph=True if energies or u_total_batch are needed later in loss
                    #     grad_E_all = torch.autograd.grad(torch.sum(energies), u_total_batch, retain_graph=True, create_graph=True)[0]
                    #     grad_E_valid = grad_E_all[valid_indices] # Select gradients for valid indices

                    #     # 4. Compute Dot Product: ∇_u E ⋅ v
                    #     dot_products = torch.sum(grad_E_valid * v, dim=1) # Shape: (batch_size-1,)

                    #     # 5. Apply Penalty: relu(-dot_product)
                    #     penalty_values = torch.relu(-dot_products)
                    #     monotonicity_penalty = torch.mean(penalty_values)


                    # Calculate maximum displacements
                    mean_linear = torch.mean(torch.norm(l.reshape(batch_size, -1, 3), dim=2)).item()
                    mean_total = torch.mean(torch.norm(u_total_batch.reshape(batch_size, -1, 3), dim=2)).item()
                    mean_correction = torch.mean(torch.norm(y.reshape(batch_size, -1, 3), dim=2)).item()

                    nonlinear_ratio = mean_correction / mean_total
                    
                    # Compute orthogonality constraint (using the same approach as reference)
                    ortho = torch.mean(torch.sum(y * constraint_dir, dim=1)**2)
                    
                    # # Compute origin constraint for rest shape
                    origin = torch.sum(y[rest_idx]**2)

                    # origin = energies[rest_idx]  # Use energy for rest shape as origin constraint

                    bc_penalty = torch.tensor(0.0, device=self.device, dtype=torch.float64) # Initialize as tensor
                    if self.boundary_dofs_th is not None and len(self.boundary_dofs_th) > 0:
                        fixed_displacements = u_total_batch[:, self.boundary_dofs_th] # Shape: (batch_size, num_fixed_dofs)

                        # --- Option 1: Log-Cosh Loss ---
                        # Calculate log(cosh(error)) for each element, then average
                        log_cosh_penalty = torch.log(torch.cosh(fixed_displacements))
                        bc_penalty = torch.mean(log_cosh_penalty) # Average over all fixed DOFs and batch
                        # --- End Option 1 ---

                        # --- Option 2: L2 Loss (Current Method) ---
                        # bc_penalty = torch.mean(torch.sum(fixed_displacements**2, dim=1))
                        # --- End Option 2 ---

              


                 

                    # Scale energy by maximum linear displacement to get comparable units
                    max_linear_disp = torch.max(torch.norm(l.reshape(batch_size, -1, 3), dim=2))
                    energies_scaling = energies / (max_linear_disp + 1e-8)  # Add epsilon to avoid division by zero

                    energy_scaling = torch.mean(energies_scaling)  # Average energy across batch

                    # Add incentive for beneficial nonlinearity (energy improvement term)
                    u_linear_only = l.detach()  # Detach to avoid affecting linear gradients
                    energy_linear = self.energy_calculator(u_linear_only).mean()
                    energy_improvement = (torch.relu(energy_linear - energy))/(energy_linear + 1e-8)  
                    improvement_loss = (energy_linear - energy) / (energy_linear + 1e-8)  

                                
                    # Get the raw div(P) tensor
                    raw_div_p = self.energy_calculator.compute_div_p(u_total_batch)






                    # Modified loss
                    loss = -improvement_loss+1 + 1e2 * ortho +  1e1 * bc_penalty  #+ 1e4 *  origin

                    # --- NaN Safeguard ---
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"\n !!! WARNING: Loss became {loss.item()} at iteration {iteration}, LBFGS step {lbfgs_iter} !!!")
                        print(f"    Components: Energy={energy.item():.4e}, Ortho={ortho.item():.4e}, Origin={origin.item():.4e}")
                        # Option 1: Raise error to stop LBFGS step (recommended)
                        # raise ValueError("Loss is NaN or Inf, stopping optimization step.")
                        # Option 2: Return a large value (might mess up line search)
                        return torch.tensor(float('inf'), device=self.device, dtype=torch.float64) 


                    loss.backward()

                    

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
                        print(f"│ {'Raw Energy:':<20} {energy.item():<12.6f} │ {'Linear Energy:':<20} {energy_linear.item():<12.6f}")
                        print(f"│ {'Energy Improvement:':<20} {energy_improvement.item()*100:.2f}% │ {'Monotonicity:':<20} {energy_scaling.item():<12.6f}")
                        
                        # Constraint metrics section
                        print(f"│ CONSTRAINT METRICS:")
                        print(f"│ {'Orthogonality:':<20} {ortho.item():<12.6f} │ {'Origin Constraint:':<20} {origin.item():<12.6f} | {'BC Penalty:':<20} {bc_penalty.item():<12.6f}")
                        
                        # # Displacement metrics section
                        # print(f"│ DISPLACEMENT METRICS:")
                        # print(f"│ {'Mean Linear:':<20} {mean_linear:<12.6f} │ {'Mean Total:':<20} {mean_total:<12.6f}")
                        # print(f"│ {'Mean Correction:':<20} {mean_correction:<12.6f} │ {'Nonlinear Ratio:':<20} {nonlinear_ratio*100:.2f}%")
                        
                        # Divergence metrics section
                        div_p_means = torch.mean(raw_div_p, dim=0).mean(dim=0)
                        # print(f"│ DIVERGENCE METRICS:")
                        # print(f"│ {'Direction:':<20} {'X':<17} {'Y':<17} {'Z':<17}")
                        # print(f"│ {'Div(P):':<12} {div_p_means[0].item():15.6e} {div_p_means[1].item():15.6e} {div_p_means[2].item():15.6e}")
                        # print(f"│ {'Div(P) Loss:':<20} {log_scaled_div_p.item():<12.6f} │ {'Raw Div(P) L2:':<20} {raw_div_p_L2_mean.item():<12.6e}")

                        # # Add volume metrics section
                        # print(f"│ VOLUME PRESERVATION:")
                        # print(f"│ {'Linear Volume Ratio:':<20} {avg_linear_ratio:<12.6f} │ {'Neural Volume Ratio:':<20} {avg_neural_ratio:<12.6f}")
                        # print(f"│ {'Linear Volume Change:':<20} {(avg_linear_ratio-1)*100:<12.4f}% │ {'Neural Volume Change:':<20} {(avg_neural_ratio-1)*100:<12.4f}%")
                        # print(f"│ {'Volume Penalty:':<20} {vol_penalty.item():<12.6f}")
        
                        
                        # Final loss value
                        print(f"{sep_line}")
                        print(f"TOTAL LOSS: {loss.item():.6e} - {lbfgs_progress}")
                        print(f"{sep_line}\n")
                    

                    energy_val = energy.item()  # Convert tensor to Python scalar
                    ortho_val = ortho.item()
                    origin_val = origin.item()
                    loss_val = loss.item()

                    return loss
                
                # Perform optimization step
                optimizer.step(closure)


                # Choose a random latent vector from the batch
                random_idx = np.random.randint(1, batch_size)
                random_z = z[random_idx].detach().clone()
                self.visualize_latent_vector(random_z, iteration, loss_val)
                self.compute_volume_comparison(random_z)

            if iteration % 1 == 0:  # Update visualization every 5 iterations
                pass
                # Update visualization

            
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
            if patience > 200: #loss_val < 1e-8 or 
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
        plotter.open_gif("training_visualization.gif")

        
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
                z = torch.zeros(self.num_modes, device=self.device, dtype=torch.float64)
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
                warped = local_grid.warp_by_vector("displacement", factor=1)
                
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
            modes_to_show = list(range(self.num_modes))
        
        num_modes = len(modes_to_show)
        
        # Compute scale for latent vectors if not provided
        if scale is None:
            scale = self.compute_safe_scaling_factor() * 2  # Larger scale to see clear deformations
        
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
                z = torch.zeros(self.num_modes, device=self.device, dtype=torch.float64)
                
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
                warped = local_grid.warp_by_vector("displacement", factor=1)
                
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
    
    
    def plot_training_metrics(self, save_path=None):
        """
        Plot key training metrics (energy and constraints) over training iterations.
        
        Args:
            save_path: Optional path to save the plot image
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Fetch metrics from tensorboard logs
        try:
            # Get events from tensorboard file
            from tensorboard.backend.event_processing import event_accumulator
            
            # Find the tensorboard log path
            import os
            import glob
            
            # Look for tensorboard files in writer's log_dir
            logdir = self.writer.log_dir
            tb_files = glob.glob(os.path.join(logdir, "events.out.tfevents.*"))
            
            if not tb_files:
                print("No tensorboard log files found. Using dummy data.")
                # Use dummy data for the plot in case logs aren't found
                iterations = np.arange(100)
                energy_values = np.random.exponential(1.0, size=100)[::-1]  # Decreasing values
                ortho_values = np.random.exponential(0.1, size=100)[::-1]
                origin_values = np.random.exponential(0.01, size=100)[::-1]
            else:
                print(f"Loading training metrics from {tb_files[0]}")
                ea = event_accumulator.EventAccumulator(tb_files[0])
                ea.Reload()  # Load all data
                
                # Extract scalar values
                energy_events = ea.Scalars('train/energy')
                ortho_events = ea.Scalars('train/ortho')
                origin_events = ea.Scalars('train/origin')
                
                # Extract iteration and value pairs
                iterations = [e.step for e in energy_events]
                energy_values = [e.value for e in energy_events]
                ortho_values = [e.value for e in ortho_events]
                origin_values = [e.value for e in origin_events]
        
        except Exception as e:
            print(f"Error loading tensorboard logs: {str(e)}")
            # Fall back to dummy data
            iterations = np.arange(100)
            energy_values = np.random.exponential(1.0, size=100)[::-1]
            ortho_values = np.random.exponential(0.1, size=100)[::-1]
            origin_values = np.random.exponential(0.01, size=100)[::-1]
        
        # Create figure with two subplots (energy and constraints)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle('Neural Modes Training Metrics', fontsize=16)
        
        # Plot energy on the first subplot
        ax1.plot(iterations, energy_values, 'b-', linewidth=2, label='Energy')
        ax1.set_yscale('log')  # Log scale often better for energy values
        ax1.set_ylabel('Energy (log scale)')
        ax1.set_title('Energy Loss Over Training')
        ax1.grid(True, which='both', linestyle='--', alpha=0.6)
        ax1.legend()
        
        # Plot constraint values on the second subplot
        ax2.plot(iterations, ortho_values, 'r-', linewidth=2, label='Orthogonality Constraint')
        ax2.plot(iterations, origin_values, 'g-', linewidth=2, label='Origin Constraint')
        ax2.set_yscale('log')  # Log scale for constraints too
        ax2.set_xlabel('Training Iteration')
        ax2.set_ylabel('Constraint Value (log scale)')
        ax2.set_title('Constraint Losses Over Training')
        ax2.grid(True, which='both', linestyle='--', alpha=0.6)
        ax2.legend()
        
        # Add horizontal grid lines
        for ax in [ax1, ax2]:
            ax.yaxis.grid(True, linestyle='-', which='major', color='gray', alpha=0.5)
            ax.yaxis.grid(True, linestyle=':', which='minor', color='gray', alpha=0.3)
            if iterations:  # Only format if we have data
                ax.set_xlim([min(iterations), max(iterations)])
        
        # Annotate best values
        if energy_values:
            min_energy_idx = np.argmin(energy_values)
            min_energy = energy_values[min_energy_idx]
            min_energy_iter = iterations[min_energy_idx]
            
        
        # Add overall improvement rate
        if len(energy_values) > 1:
            first_energy = energy_values[0]
            last_energy = energy_values[-1]
            improvement = (first_energy - last_energy) / first_energy * 100 if first_energy != 0 else 0
            fig.text(0.5, 0.01, f'Total Energy Improvement: {improvement:.2f}%', 
                    ha='center', fontsize=12, bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1) # Make space for the text
        
        # Save plot if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training metrics plot saved to {save_path}")
        
        # Show the plot
        plt.show()
        
        return fig
    
    def analyze_latent_correlations(self, delta=0.01, save_path=None):
        """
        Analyze correlations between latent dimensions based on the gradient of the 
        TOTAL displacement (linear + neural) at the origin.
        
        Args:
            delta: Small perturbation for finite difference approximation
            save_path: Path to save visualization
        """
        print("Analyzing latent space correlations (based on total displacement)...")
        
        # Get latent dimension and linear modes
        latent_dim = self.latent_dim
        # Ensure linear_modes are available and correctly sliced if needed
        linear_modes = self.linear_modes[:, :latent_dim] 

        # Create base zero latent vector (origin)
        z0 = torch.zeros(latent_dim, device=self.device, dtype=torch.float64)
        
        # Get base TOTAL output at origin
        with torch.no_grad():
            linear_output0 = torch.matmul(z0, linear_modes.T) # Should be zero
            neural_output0 = self.model(z0).flatten()
            total_output0 = linear_output0 + neural_output0 # Effectively neural_output0
        
        # Initialize correlation matrix and norm vectors
        corr_matrix = torch.zeros((latent_dim, latent_dim), device=self.device, dtype=torch.float64)
        norm_squared = torch.zeros(latent_dim, device=self.device, dtype=torch.float64)
        
        # Store perturbed TOTAL outputs
        perturbed_total_outputs = []
        for i in range(latent_dim):
            # Perturb the i-th dimension
            zi = z0.clone()
            zi[i] += delta
            
            # Compute TOTAL output with perturbation
            with torch.no_grad():
                linear_output_i = torch.matmul(zi, linear_modes.T)
                neural_output_i = self.model(zi).flatten()
                total_output_i = linear_output_i + neural_output_i
            
            # Store perturbed total output
            perturbed_total_outputs.append(total_output_i)
        
        # --- Calculate direction vectors based on TOTAL displacement ---
        direction_vectors = []
        for i in range(latent_dim):
            # Direction vector for dimension i (gradient approximation of total displacement)
            dir_i = (perturbed_total_outputs[i] - total_output0) / delta
            direction_vectors.append(dir_i)
            # Update squared norm
            norm_squared[i] = torch.dot(dir_i, dir_i)

        # Compute E = [e1|e2|...|en] matrix where each ei is a gradient direction of total displacement
        # And compute correlation matrix ET E incrementally
        unnormalized_corr_matrix = torch.zeros_like(corr_matrix) # Store unnormalized version
        for i in range(latent_dim):
            dir_i = direction_vectors[i]
            for j in range(latent_dim):
                dir_j = direction_vectors[j]
                dot_product = torch.dot(dir_i, dir_j)
                unnormalized_corr_matrix[i, j] = dot_product
        # --- End of modification ---

        # --- Add eigenvalue calculation and print statement here ---
        try:
            # Calculate eigenvalues of the unnormalized correlation matrix (E^T E)
            # Use eigvalsh since the matrix is symmetric
            eigenvalues_unnormalized = torch.linalg.eigvalsh(unnormalized_corr_matrix)
            print(f"Eigenvalues of unnormalized correlation matrix (E^T E based on total displacement): {eigenvalues_unnormalized.cpu().numpy()}")
        except Exception as e:
            print(f"Could not compute eigenvalues of unnormalized correlation matrix: {e}")
        
        # Normalize to get correlations (-1 to 1 scale)
        corr_matrix = unnormalized_corr_matrix.clone() # Start with unnormalized values
        for i in range(latent_dim):
            for j in range(latent_dim):
                denominator = torch.sqrt(norm_squared[i] * norm_squared[j])
                if denominator > 1e-10:
                    corr_matrix[i, j] /= denominator
                else:
                    # Handle case where norm is zero
                    if i != j:
                        corr_matrix[i, j] = 0.0
                    else:
                        corr_matrix[i, j] = 1.0 if norm_squared[i] > 1e-10 else 0.0

        # --- Add eigenvalue calculation for normalized matrix ---
        try:
            # Calculate eigenvalues of the normalized correlation matrix
            # Use eigvalsh since the matrix is symmetric
            eigenvalues_normalized = torch.linalg.eigvalsh(corr_matrix)
            print(f"Eigenvalues of normalized correlation matrix (based on total displacement): {eigenvalues_normalized.cpu().numpy()}")
        except Exception as e:
            print(f"Could not compute eigenvalues of normalized correlation matrix: {e}")

        # Visualize the correlation matrix
        fig = self.visualize_correlation_matrix(corr_matrix, save_path, title_suffix="(Total Displacement)")
        
        print("Latent correlation analysis (total displacement) complete.")
        return corr_matrix, fig

    def analyze_linear_mode_correlations(self, delta=0.01, save_path=None):
        """
        Analyze correlations between latent dimensions based ONLY on the linear modes.
        This should ideally result in an identity matrix if linear modes are orthogonal.
        
        Args:
            delta: Small perturbation (value doesn't strictly matter here but kept for consistency)
            save_path: Path to save visualization
        """
        print("Analyzing latent space correlations (based on linear modes ONLY)...")
        
        # Get latent dimension and linear modes
        latent_dim = self.latent_dim
        linear_modes = self.linear_modes[:, :latent_dim] 

        # Create base zero latent vector (origin)
        z0 = torch.zeros(latent_dim, device=self.device, dtype=torch.float64)
        
        # Get base LINEAR output at origin (should be zero)
        with torch.no_grad():
            linear_output0 = torch.matmul(z0, linear_modes.T) 
        
        # Initialize correlation matrix and norm vectors
        corr_matrix = torch.zeros((latent_dim, latent_dim), device=self.device, dtype=torch.float64)
        norm_squared = torch.zeros(latent_dim, device=self.device, dtype=torch.float64)
        
        # Store perturbed LINEAR outputs
        perturbed_linear_outputs = []
        for i in range(latent_dim):
            # Perturb the i-th dimension
            zi = z0.clone()
            zi[i] += delta # Use delta=1 for direct mode extraction
            
            # Compute LINEAR output with perturbation
            with torch.no_grad():
                linear_output_i = torch.matmul(zi, linear_modes.T)
            
            # Store perturbed linear output
            perturbed_linear_outputs.append(linear_output_i)
        
        # --- Calculate direction vectors based on LINEAR displacement ---
        # Note: With delta=1, dir_i is essentially the i-th linear mode vector
        direction_vectors = []
        for i in range(latent_dim):
            # Direction vector for dimension i (gradient approximation of linear displacement)
            # Equivalent to (linear_modes[:, i] * delta) / delta = linear_modes[:, i]
            dir_i = (perturbed_linear_outputs[i] - linear_output0) / delta 
            direction_vectors.append(dir_i)
            # Update squared norm
            norm_squared[i] = torch.dot(dir_i, dir_i)

        # Compute E = [e1|e2|...|en] matrix where each ei is a linear mode vector
        # And compute correlation matrix ET E incrementally
        unnormalized_corr_matrix = torch.zeros_like(corr_matrix) # Store unnormalized version
        for i in range(latent_dim):
            dir_i = direction_vectors[i]
            for j in range(latent_dim):
                dir_j = direction_vectors[j]
                dot_product = torch.dot(dir_i, dir_j)
                unnormalized_corr_matrix[i, j] = dot_product
        # --- End of modification ---

        # --- Add eigenvalue calculation and print statement here ---
        try:
            # Calculate eigenvalues of the unnormalized correlation matrix (E^T E)
            eigenvalues_unnormalized = torch.linalg.eigvalsh(unnormalized_corr_matrix)
            print(f"Eigenvalues of unnormalized correlation matrix (E^T E based on linear modes): {eigenvalues_unnormalized.cpu().numpy()}")
        except Exception as e:
            print(f"Could not compute eigenvalues of unnormalized linear correlation matrix: {e}")
        
        # Normalize to get correlations (-1 to 1 scale)
        corr_matrix = unnormalized_corr_matrix.clone() # Start with unnormalized values
        for i in range(latent_dim):
            for j in range(latent_dim):
                denominator = torch.sqrt(norm_squared[i] * norm_squared[j])
                if denominator > 1e-10:
                    corr_matrix[i, j] /= denominator
                else:
                    # Handle case where norm is zero
                    if i != j:
                        corr_matrix[i, j] = 0.0
                    else:
                        corr_matrix[i, j] = 1.0 if norm_squared[i] > 1e-10 else 0.0

        # --- Add eigenvalue calculation for normalized matrix ---
        try:
            # Calculate eigenvalues of the normalized correlation matrix
            eigenvalues_normalized = torch.linalg.eigvalsh(corr_matrix)
            print(f"Eigenvalues of normalized correlation matrix (based on linear modes): {eigenvalues_normalized.cpu().numpy()}")
        except Exception as e:
            print(f"Could not compute eigenvalues of normalized linear correlation matrix: {e}")

        # Visualize the correlation matrix
        fig = self.visualize_correlation_matrix(corr_matrix, save_path, title_suffix="(Linear Modes Only)")
        
        print("Linear mode correlation analysis complete.")
        return corr_matrix, fig

    def visualize_correlation_matrix(self, corr_matrix, save_path=None, title_suffix=""):
        """Visualize the correlation matrix between latent dimensions."""
        # ... (rest of visualize_correlation_matrix method - maybe add title_suffix to the title) ...
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        
        # Convert to numpy for visualization
        corr_np = corr_matrix.cpu().numpy()
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Use seaborn for nicer heatmap
        mask = np.zeros_like(corr_np, dtype=bool)
        #mask[np.triu_indices_from(mask)] = True  # Optional: show only lower triangle
        
        # Plot correlation matrix
        sns.heatmap(corr_np, mask=mask, cmap='RdBu_r', vmin=-1, vmax=1,
                    square=True, linewidths=.5, annot=True, fmt='.2f',
                    cbar_kws={"shrink": .8, "label": "Correlation"})
        
        # Set labels
        latent_dim = corr_np.shape[0]
        plt.xticks(np.arange(latent_dim) + 0.5, [f'z{i}' for i in range(latent_dim)])
        plt.yticks(np.arange(latent_dim) + 0.5, [f'z{i}' for i in range(latent_dim)], rotation=0)
        
        # Add title
        plt.title(f'Latent Space Correlation Matrix {title_suffix}', fontsize=14) # Added suffix
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            # Modify save path slightly if needed to avoid overwriting
            base, ext = os.path.splitext(save_path)
            if "Linear" in title_suffix:
                save_path_mod = f"{base}_linear{ext}"
            else:
                save_path_mod = save_path # Keep original for total displacement
            plt.savefig(save_path_mod, dpi=300, bbox_inches='tight')
            print(f"Correlation matrix visualization saved to {save_path_mod}")
        
        plt.show()
        return plt.gcf()
    



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
    engine.visualize_latent_dimensions(dim1=1, dim2=0, num_points=5)
    
    # Optionally visualize other dimension pair
    engine.visualize_latent_dimensions(dim1=3, dim2=4, num_points=5)

    print("\nVisualizing latent space modes...")
    # Visualize all latent dimensions
    engine.visualize_latent_space(num_samples=5)



    print("\nAnalyzing latent space correlations (TOTAL displacement)...")
    correlation_matrix_path = os.path.join('checkpoints', 'latent_correlations_total.png')
    corr_matrix_total, _ = engine.analyze_latent_correlations(delta=0.0001, save_path=correlation_matrix_path)

    print("\nAnalyzing latent space correlations (LINEAR modes only)...")
    linear_correlation_matrix_path = os.path.join('checkpoints', 'latent_correlations_linear.png')
    corr_matrix_linear, _ = engine.analyze_linear_mode_correlations(delta=1.0, save_path=linear_correlation_matrix_path) # delta=1 is fine here


    print("\nPlotting training metrics...")
    metrics_plot_path = os.path.join('checkpoints', 'training_metrics.png')
    engine.plot_training_metrics(save_path=metrics_plot_path)
    
    print("Main function complete.")
    
    print("Main function complete.")

def load_config(config_file):
    import yaml
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

if __name__ == '__main__':
    main()
