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








import matplotlib.pyplot as plt

import pyvista
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmshio
import gmsh
from mpi4py import MPI
import ufl
from scipy.linalg import eig
from tests.solver import EnergyModel, NeoHookeanEnergyModel, ModularNeoHookeanEnergy  

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

        #self.layers[-1].weight.data.fill_(0.0)
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
        self.energy_calculator = ModularNeoHookeanEnergy(
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
        # print("Linear eigenmodes loaded.")

        # print("Loading neural network...")
        self.num_modes = self.latent_dim  # Make them the same
        output_dim = self.V.dofmap.index_map.size_global * self.domain.geometry.dim
        hid_layers = cfg['model'].get('hid_layers', 2)
        hid_dim = cfg['model'].get('hid_dim', 64)
        # print(f"Output dimension: {output_dim}")
        # print(f"Network architecture: {hid_layers} hidden layers with {hid_dim} neurons each")
        self.model = Net(self.latent_dim, output_dim, hid_layers, hid_dim).to(device).double()

        # print(f"Neural network loaded. Latent dim: {self.latent_dim}, Num Modes: {self.num_modes}")


        # Tensorboard setup
        checkpoint_dir = cfg.get('training', {}).get('checkpoint_dir', 'checkpoints')
        tensorboard_dir = cfg.get('training', {}).get('tensorboard_dir', 'tensorboard')
        self.writer = SummaryWriter(os.path.join(checkpoint_dir, tensorboard_dir))


    
    

    def compute_linear_modes(self):
        # print("Computing linear modes with FEniCS...")
        # Get domain extents
        x_coords = self.domain.geometry.x
        x_min = x_coords[:, 0].min()
        x_max = x_coords[:, 0].max()
        y_min = x_coords[:, 1].min()
        y_max = x_coords[:, 1].max()
        z_min = x_coords[:, 2].min()
        z_max = x_coords[:, 2].max()
        # print(f"Domain extents: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}], z=[{z_min}, {z_max}]")

        # Use the same function space as the main model
        V = self.V
        u_tr = TrialFunction(V)
        u_test = TestFunction(V)

        
        if isinstance(self.energy_calculator, ModularNeoHookeanEnergy):
            # print("Using Neo-Hookean forms for linear modes")
            
            # Define strain tensor
            def epsilon(u):
                return sym(grad(u))

            # Linearized Neo-Hookean stress tensor at undeformed state
            def sigma(u):
                return self.lambda_ * div(u) * Identity(3) + self.mu * (grad(u) + grad(u).T)
                
            a_form = inner(sigma(u_tr), epsilon(u_test)) * dx
            m_form = self.rho * inner(u_tr, u_test) * dx
            
        else:
            # print("Using standard linear elasticity forms for linear modes")
            
            # Default to linear elasticity forms
            def epsilon(u):
                return sym(grad(u))

            def sigma(u):
                return self.lambda_ * div(u) * Identity(3) + 2 * self.mu * epsilon(u)
                
            a_form = inner(sigma(u_tr), epsilon(u_test)) * dx
            m_form = self.rho * inner(u_tr, u_test) * dx

        x_coordinates = self.domain.geometry.x
        x_min = np.min(x_coordinates[:, 0])
        y_min = np.min(x_coordinates[:, 1])
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
        
        # print("Assembling A matrix")
        A = assemble_matrix_petsc(form(a_form), bcs=[bc])
        A.assemble()
        self.A = A
        
        # print("Assembling M matrix")
        M = assemble_matrix_petsc(form(m_form))
        M.assemble()
        self.M = M

        # print("Matrices assembled")

        # Setup eigensolver - USE LINEAR_MODES.PY CONFIGURATION
        # print("Creating main eigensolver")
        eigensolver = SLEPc.EPS().create(self.domain.comm)
        eigensolver.setOperators(A, M)
        eigensolver.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)

        # Change back to TARGET_MAGNITUDE with a small non-zero target
        eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)  
        eigensolver.setTarget(0.001)  # Small positive target value

        st = eigensolver.getST()
        st.setType(SLEPc.ST.Type.SINVERT)  
        st.setShift(0.0)

        # Set explicit dimensions to increase chances of convergence
        eigensolver.setDimensions(self.latent_dim*2, PETSc.DECIDE)  # Request more to ensure we get enough

        # Increase max iterations and tolerance
        eigensolver.setTolerances(tol=1e-6, max_it=1000)

        # print("Solving eigenvalue problem...")
        eigensolver.solve()
        # print("Eigenvalue problem solved")

        # Extract eigenvectors and eigenvalues
        nconv = eigensolver.getConverged()
        # # print(f"Number of converged eigenvalues: {nconv}")
        
        # Initialize arrays to store both
        modes = []
        eigenvalues = []
        if self.latent_dim > nconv:
            # print(f"WARNING: Requested {self.latent_dim} modes, but only {nconv} could be computed")
            self.latent_dim = nconv
        
        for i in range(min(self.latent_dim, nconv)):
            # Extract eigenvector
            vr = A.createVecRight()
            eigensolver.getEigenvector(i, vr)
            modes.append(vr.array[:])
            
            # Extract eigenvalue (store for scaling)
            eigenvalue = eigensolver.getEigenvalue(i)
            eigenvalues.append(eigenvalue.real)  # Keep only real part
            
            # Display frequency information
            frequency = np.sqrt(np.abs(eigenvalue.real)) / (2 * np.pi)  # Convert to Hz
            # print(f"Eigenvalue {i+1}: {eigenvalue.real:.6e}, Frequency: {frequency:.4f} Hz")
        
        # Store eigenvalues for later use in scaling
        self.eigenvalues = np.array(eigenvalues)
        # print(f"Eigenvalues: {self.eigenvalues}")
        
        # Handle case when no modes are found
        if len(modes) == 0:
            # print("ERROR: No modes could be computed. Using random initialization.")
            # Create random modes and eigenvalues as fallback
            random_modes = np.random.rand(A.getSize()[0], self.latent_dim)
            self.eigenvalues = np.ones(self.latent_dim)  # Default eigenvalues
            return random_modes
        
        linear_modes = np.column_stack(modes)
        # print(f"Shape of linear_modes: {linear_modes.shape}")
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
        
        if abs(result['linear_volume_ratio'] - 1.0) > 1e-10:
            print(f"Improvement factor:      {result['volume_preservation_improvement']:.2f}x")
        print("==========================================\n")
        
        return result
        

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
                deformation_scale_final = 10
                #current_scale = deformation_scale_init * (deformation_scale_final/deformation_scale_init)**(iteration/num_epochs) #expoential scaling
                current_scale = deformation_scale_init + (deformation_scale_final - deformation_scale_init) # * (iteration/num_epochs) #linear scaling

                print(f"Current scale: {current_scale}")
                mode_scales = torch.tensor(self.compute_eigenvalue_based_scale(), device=self.device, dtype=torch.float64)
                mode_scales = mode_scales * current_scale
                mode_scales[-1] *= 2  # Increase scale for last mode

                # Generate samples with current scale
                z = torch.rand(batch_size, L, device=self.device) * mode_scales * 2 - mode_scales


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


                    # Calculate maximum displacements
                    mean_linear = torch.mean(torch.norm(l.reshape(batch_size, -1, 3), dim=2)).item()
                    mean_total = torch.mean(torch.norm(u_total_batch.reshape(batch_size, -1, 3), dim=2)).item()
                    mean_correction = torch.mean(torch.norm(y.reshape(batch_size, -1, 3), dim=2)).item()

                    nonlinear_ratio = mean_correction / mean_total
                    
                    # Compute orthogonality constraint (using the same approach as reference)
                    ortho = torch.mean(torch.sum(y * constraint_dir, dim=1)**2)
                    
                    # Compute origin constraint for rest shape
                    origin = torch.sum(y[rest_idx]**2)


                 

                    energy_scaling = torch.log10(energy + 1)

                    # Add incentive for beneficial nonlinearity (energy improvement term)
                    u_linear_only = l.detach()  # Detach to avoid affecting linear gradients
                    energy_linear = self.energy_calculator(u_linear_only).mean()
                    energy_improvement = (energy_linear - energy)
                   
                    # Get the raw div(P) tensor
                    raw_div_p = self.energy_calculator.compute_div_p(u_total_batch)

                    raw_div_p_L2_mean = torch.mean(torch.norm(raw_div_p, dim=2))

                    div_p_magnitude = torch.norm(raw_div_p, dim=2, keepdim=True)
                    div_p_direction = raw_div_p / (div_p_magnitude + 1e-8)

                    # Apply log scaling to magnitude only
                    log_div_p_magnitude = torch.log10(div_p_magnitude + 1)

                    # Recombine with original direction
                    log_scaled_div_p_tensor = log_div_p_magnitude * div_p_direction

                    # log_Scaled_div_p is a tensor of shape [batch_size, num_nodes, 3]

                    log_scaled_div_p = torch.mean(torch.norm(log_scaled_div_p_tensor, dim=2))



                    div_p_weight = 2.0  # Weight for divergence loss


                    # Modified loss
                    loss = energy + 1e6* ortho + 1e5 * origin  


                    loss.backward()

                    # Choose a random latent vector from the batch
                    random_idx = np.random.randint(1, batch_size)
                    random_z = z[random_idx].detach().clone()
                    self.visualize_latent_vector(random_z, iteration, loss_val)
                    self.compute_volume_comparison(random_z)

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
                        print(f"│ {'Energy Improvement:':<20} {energy_improvement.item():<12.6f} │ {'Energy Loss:':<20} {energy_scaling.item():<12.6f}")
                        
                        # Constraint metrics section
                        print(f"│ CONSTRAINT METRICS:")
                        print(f"│ {'Orthogonality:':<20} {ortho.item():<12.6f} │ {'Origin Constraint:':<20} {origin.item():<12.6f}")
                        
                        # # Displacement metrics section
                        # print(f"│ DISPLACEMENT METRICS:")
                        # print(f"│ {'Mean Linear:':<20} {mean_linear:<12.6f} │ {'Mean Total:':<20} {mean_total:<12.6f}")
                        # print(f"│ {'Mean Correction:':<20} {mean_correction:<12.6f} │ {'Nonlinear Ratio:':<20} {nonlinear_ratio*100:.2f}%")
                        
                        # Divergence metrics section
                        div_p_means = torch.mean(raw_div_p, dim=0).mean(dim=0)
                        print(f"│ DIVERGENCE METRICS:")
                        print(f"│ {'Direction:':<20} {'X':<17} {'Y':<17} {'Z':<17}")
                        print(f"│ {'Div(P):':<12} {div_p_means[0].item():15.6e} {div_p_means[1].item():15.6e} {div_p_means[2].item():15.6e}")
                        print(f"│ {'Div(P) Loss:':<20} {log_scaled_div_p.item():<12.6f} │ {'Raw Div(P) L2:':<20} {raw_div_p_L2_mean.item():<12.6e}")

                        # Add volume metrics section
                        print(f"│ VOLUME PRESERVATION:")
                        print(f"│ {'Linear Volume Ratio:':<20} {avg_linear_ratio:<12.6f} │ {'Neural Volume Ratio:':<20} {avg_neural_ratio:<12.6f}")
                        print(f"│ {'Linear Volume Change:':<20} {(avg_linear_ratio-1)*100:<12.4f}% │ {'Neural Volume Change:':<20} {(avg_neural_ratio-1)*100:<12.4f}%")
                        print(f"│ {'Volume Penalty:':<20} {vol_penalty.item():<12.6f}")
        
                        
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

                scheduler.step(loss_val)  

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
            modes_to_show = list(range(self.latent_dim))
        
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
    
    print("Main function complete.")

def load_config(config_file):
    import yaml
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

if __name__ == '__main__':
    main()
