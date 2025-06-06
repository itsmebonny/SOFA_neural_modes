




import argparse
import os, sys, logging
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from torch.utils.tensorboard import SummaryWriter
import math


import traceback
from scipy import sparse

# --- Import the renamed solver class ---

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tests.solver import SOFANeoHookeanModel, SOFAStVenantKirchhoffModel, SOFAStVenantKirchhoffModelModified

# In train.py - add these imports
import glob
import json
import datetime
import matplotlib.pyplot as plt
import pyvista # Keep pyvista for visualization
import seaborn as sns          # Add seaborn



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
        
        # Input projection layer (latent → hidden), no bias
        self.input_proj = torch.nn.Linear(latent_dim, hid_dim, bias=False)

        # Residual blocks
        self.res_blocks = torch.nn.ModuleList()
        for _ in range(hid_layers):
            # Linear layers within blocks also have no bias
            # LayerNorm layers also have no learnable affine parameters (weight and bias)
            block = torch.nn.Sequential(
            torch.nn.Linear(hid_dim, hid_dim, bias=False),
            torch.nn.LayerNorm(hid_dim, elementwise_affine=False), # No learnable scale/shift
            torch.nn.Linear(hid_dim, hid_dim, bias=False),
            torch.nn.LayerNorm(hid_dim, elementwise_affine=False)  # No learnable scale/shift
            )
            self.res_blocks.append(block)

        # Output projection (hidden → output), no bias
        self.output_proj = torch.nn.Linear(hid_dim, output_dim, bias=False)

        # Initialize weights
        # self._init_weights(zero_init_output) # Keep this commented if you want zero init below
        # Zero last layer weights (as in original selection)
        torch.nn.init.zeros_(self.output_proj.weight)
        # Note: Biases are handled by bias=False in Linear and elementwise_affine=False in LayerNorm.

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

        # --- Load Mesh Data and Matrices from SOFA export ---
        print("Loading SOFA matrices and mesh data...")
        matrices_cfg = cfg.get('matrices', {})
        matrices_path = matrices_cfg.get('matrices_path', 'matrices')
        timestamp = matrices_cfg.get('timestamp', None) # Allow specific timestamp override

        # Load matrices, modes, and mesh data
        M, A, eigenvalues, eigenvectors, coordinates_np, elements_np, fixed_dofs_np, metadata = self.load_sofa_data(matrices_path, timestamp) # Add fixed_dofs_np

        if M is None or A is None or eigenvalues is None or eigenvectors is None or coordinates_np is None or elements_np is None:
             raise RuntimeError("Failed to load required data from SOFA export. Check paths and files.")

        # Store loaded data
        self.M = M
        self.A = A
        self.eigenvalues = eigenvalues
        self.linear_modes = torch.tensor(eigenvectors, device=self.device, dtype=torch.float64)
        self.coordinates_np = coordinates_np # Keep numpy version for PyVista
        self.elements_np = elements_np
        self.fixed_dofs_np = fixed_dofs_np # Store fixed DOFs (numpy array)
        if self.fixed_dofs_np is not None:
            self.fixed_dofs_th = torch.tensor(fixed_dofs_np, device=self.device, dtype=torch.long) # Convert to tensor for indexing
            print(f"Loaded {len(self.fixed_dofs_np)} fixed DOFs.")
        else:
            self.fixed_dofs_th = None
            print("Warning: Fixed DOFs not loaded.")


        # Extract necessary info from loaded data
        self.num_nodes = coordinates_np.shape[0]
        self.dim = coordinates_np.shape[1]
        self.output_dim = self.num_nodes * self.dim
        self.nodes_per_element = elements_np.shape[1]
        # --- End Data Loading ---


        # Define material properties (same as before)
        print("Defining material properties...")
        self.E = float(cfg['material']['youngs_modulus'])
        self.nu = float(cfg['material']['poissons_ratio'])
        self.rho = float(cfg['material']['density'])
        print(f"E = {self.E}, nu = {self.nu}, rho = {self.rho}")
        self.mu = self.E / (2 * (1 + self.nu))
        self.lambda_ = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        print(f"mu = {self.mu}, lambda = {self.lambda_}")

        # --- Instantiate Energy Calculator using loaded data ---
        print("Instantiating energy calculator...")
        # Use the degree from the config file, needed for quadrature
        self.fem_degree = cfg.get('mesh', {}).get('fem_degree', 1)
        print(f"Using FEM degree for quadrature: {self.fem_degree}")

        self.energy_calculator = SOFANeoHookeanModel(
                        coordinates_np, elements_np, # Pass loaded mesh data
                        self.fem_degree, self.E, self.nu,
                        precompute_matrices=True, device=self.device, dtype=torch.float64
                    ).to(self.device)
        print("Energy calculator instantiated.")


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

        self.scale = self.compute_safe_scaling_factor()
        print(f"Scaling factor: {self.scale}")

 # Load neural network (same as before)
        print("Loading neural network...")
        self.latent_dim = cfg['model']['latent_dim']
        self.num_modes = self.latent_dim
        # output_dim is now derived from loaded coordinates
        hid_layers = cfg['model'].get('hid_layers', 2)
        hid_dim = cfg['model'].get('hid_dim', 64)
        print(f"Output dimension: {self.output_dim}")        
        self.model = Net(self.num_modes, self.output_dim, hid_layers, hid_dim).to(device).double()

        print(f"Neural network loaded. Latent dim: {self.latent_dim}, Num Modes: {self.num_modes}")

        # self.model = ResidualNet(
        #     self.num_modes, 
        #     self.output_dim, 
        #     hid_layers=hid_layers, 
        #     hid_dim=hid_dim,
        #     activation=torch.nn.functional.leaky_relu,
        #     zero_init_output=True
        # ).to(device).double()

        # # # After model creation, verify zero property
        # # # self.model.verify_zero_property()

        # Load linear modes
        print(f"Linear modes shape: {self.linear_modes.shape}")

        print("Linear eigenmodes loaded.")

        # Tensorboard setup
        checkpoint_dir = cfg.get('training', {}).get('checkpoint_dir', 'checkpoints')
        base_tensorboard_dir = cfg.get('training', {}).get('tensorboard_dir', 'tensorboard') # e.g., "tensorboard"
        
        # Create a unique directory for this run
        import datetime
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        run_specific_log_dir = os.path.join(checkpoint_dir, base_tensorboard_dir, current_time)
        # Alternatively, you could use a run name from cfg if you add one
        # run_name = cfg.get('experiment_name', current_time) 
        # run_specific_log_dir = os.path.join(checkpoint_dir, base_tensorboard_dir, run_name)

        self.writer = SummaryWriter(log_dir=run_specific_log_dir)
        print(f"TensorBoard logs will be saved to: {run_specific_log_dir}")

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

        self.load_z_dataset() #         
        print("Routine initialized.")


    # Replace the energy calculation in train_step with this implementation:

    
    

    
        
    def load_sofa_data(self, matrices_path=None, timestamp=None):
        """
        Load matrices, modes, mesh data, and metadata from SOFA export.
        More robustly handles matrix file formats (.npy vs .npz).
        """
        print("Attempting to load SOFA matrices, modes, and mesh data...")
        if matrices_path is None:
            matrices_path = 'matrices' # Default relative path

        # --- Convert to absolute path if not already ---
        if not os.path.isabs(matrices_path):
            # project_root is defined at the module level
            matrices_path = os.path.join(project_root, matrices_path)
            print(f"Converted matrices_path to absolute: {matrices_path}")
        # --- End absolute path conversion ---

        if not os.path.exists(matrices_path):
            print(f"Matrices path does not exist: {matrices_path}")
            return None, None, None, None, None, None, None, None # Added one None for fixed_dofs_np

            # Find the latest subdirectory if timestamp not specified
        if timestamp is None:
            subdirs = [d for d in os.listdir(matrices_path) if os.path.isdir(os.path.join(matrices_path, d))]
            if not subdirs:
                 print(f"No timestamped subdirectories found in {matrices_path}")
                 return None, None, None, None, None, None, None
            # Assuming directory names are timestamps like YYYYMMDD_HHMMSS
            try:
                 # Filter out non-timestamp directories if any
                 valid_subdirs = []
                 for d in subdirs:
                     try:
                         datetime.datetime.strptime(d, '%Y%m%d_%H%M%S')
                         valid_subdirs.append(d)
                     except ValueError:
                         pass # Ignore directories not matching the format
                 if not valid_subdirs:
                      print(f"No valid timestamped subdirectories found in {matrices_path}")
                      return None, None, None, None, None, None, None

                 latest_subdir = max(valid_subdirs, key=lambda d: datetime.datetime.strptime(d, '%Y%m%d_%H%M%S'))
                 timestamp = latest_subdir
                 print(f"Using latest data subdirectory: {timestamp}")
            except ValueError:
                 print(f"Could not determine latest subdirectory based on timestamp format. Please specify timestamp.")
                 return None, None, None, None, None, None, None

        data_subdir = os.path.join(matrices_path, timestamp)
        if not os.path.exists(data_subdir):
             print(f"Data subdirectory not found: {data_subdir}")
             return None, None, None, None, None, None, None

        # --- Load Metadata ---
        metadata_file = os.path.join(data_subdir, f'metadata_{timestamp}.json')
        if not os.path.exists(metadata_file):
             print(f"Metadata file not found: {metadata_file}")
             return None, None, None, None, None, None, None
        try:
             with open(metadata_file, 'r') as f:
                 metadata = json.load(f)
             print("Metadata loaded.")
        except Exception as e:
             print(f"Error loading metadata: {e}")
             return None, None, None, None, None, None, None

        # --- Load Matrices (Robustly) ---
        mass_matrix = None
        stiff_matrix = None
        # Prefer 'npz' as default if not specified, as it's the intended primary format now
        expected_format = metadata.get('matrix_format', 'npz')
        alt_format = 'npy' if expected_format == 'npz' else 'npz'

        mass_base_name = f'mass_matrix_{timestamp}'
        stiff_base_name = f'stiffness_matrix_{timestamp}'

        # Try loading with the expected format first
        try:
            mass_file = os.path.join(data_subdir, f'{mass_base_name}.{expected_format}')
            stiff_file = os.path.join(data_subdir, f'{stiff_base_name}.{expected_format}')
            if os.path.exists(mass_file) and os.path.exists(stiff_file):
                if expected_format == 'npz':
                    mass_matrix = sparse.load_npz(mass_file)
                    stiff_matrix = sparse.load_npz(stiff_file)
                    print(f"Loaded sparse matrices (.{expected_format}).")
                else: # expected 'npy'
                    mass_matrix = np.load(mass_file, allow_pickle=True)
                    stiff_matrix = np.load(stiff_file, allow_pickle=True)
                    print(f"Loaded numpy matrices (.{expected_format}).")
            else:
                 print(f"Matrix files not found with expected format '.{expected_format}'.")
                 raise FileNotFoundError # Trigger fallback

        except (FileNotFoundError, Exception) as e1:
            print(f"Failed loading with format '.{expected_format}': {e1}. Trying alternative format '.{alt_format}'...")
            # Try loading with the alternative format
            try:
                mass_file = os.path.join(data_subdir, f'{mass_base_name}.{alt_format}')
                stiff_file = os.path.join(data_subdir, f'{stiff_base_name}.{alt_format}')
                if os.path.exists(mass_file) and os.path.exists(stiff_file):
                    if alt_format == 'npz':
                        mass_matrix = sparse.load_npz(mass_file)
                        stiff_matrix = sparse.load_npz(stiff_file)
                        print(f"Loaded sparse matrices (.{alt_format}).")
                    else: # alt 'npy'
                        mass_matrix = np.load(mass_file, allow_pickle=True)
                        stiff_matrix = np.load(stiff_file, allow_pickle=True)
                        print(f"Loaded numpy matrices (.{alt_format}).")
                else:
                    print(f"Matrix files also not found with alternative format '.{alt_format}'.")
                    return None, None, None, None, None, None, metadata # Give up if neither format found

            except Exception as e2:
                print(f"Error loading matrices with alternative format '.{alt_format}': {e2}")
                traceback.print_exc()
                return None, None, None, None, None, None, metadata # Give up on error

        # Ensure matrices are sparse CSR format if loaded as numpy
        try:
            if mass_matrix is not None and not isinstance(mass_matrix, sparse.spmatrix):
                if mass_matrix.ndim == 0: mass_matrix = mass_matrix.item() # Handle 0-dim array
                mass_matrix = sparse.csr_matrix(mass_matrix)
                print("Converted loaded mass matrix to CSR.")
            if stiff_matrix is not None and not isinstance(stiff_matrix, sparse.spmatrix):
                if stiff_matrix.ndim == 0: stiff_matrix = stiff_matrix.item() # Handle 0-dim array
                stiff_matrix = sparse.csr_matrix(stiff_matrix)
                print("Converted loaded stiffness matrix to CSR.")
        except Exception as e_conv:
             print(f"Error converting loaded matrices to CSR: {e_conv}")
             return None, None, None, None, None, None, metadata


        if mass_matrix is None or stiff_matrix is None:
             print("Failed to load matrices in any format.")
             return None, None, None, None, None, None, metadata

        print(f"Matrix shapes: Mass {mass_matrix.shape}, Stiffness {stiff_matrix.shape}")


        # --- Load Eigenmodes ---
        eigenvalues_file = os.path.join(data_subdir, metadata.get('eigenvalues_file', ''))
        eigenvectors_file = os.path.join(data_subdir, metadata.get('eigenvectors_file', ''))

        if not metadata.get('eigenvalues_file') or not metadata.get('eigenvectors_file') or \
           not os.path.exists(eigenvalues_file) or not os.path.exists(eigenvectors_file):
            print(f"Eigenmode files not found or not specified in metadata ({data_subdir})")
            # Allow continuing without modes if matrices loaded? Maybe not for this script.
            return mass_matrix, stiff_matrix, None, None, None, None, metadata

        try:
            eigenvalues = np.load(eigenvalues_file, allow_pickle=True)
            eigenvectors = np.load(eigenvectors_file, allow_pickle=True)
            print(f"Mode shapes: Eigenvalues {eigenvalues.shape}, Eigenvectors {eigenvectors.shape}")
        except Exception as e:
            print(f"Error loading eigenmodes: {e}")
            traceback.print_exc()
            return mass_matrix, stiff_matrix, None, None, None, None, metadata

        # --- Load Mesh Data ---
        coords_file = os.path.join(data_subdir, metadata.get('coordinates_file', ''))
        elements_file = os.path.join(data_subdir, metadata.get('elements_file', ''))

        if not metadata.get('coordinates_file') or not metadata.get('elements_file') or \
           not os.path.exists(coords_file) or not os.path.exists(elements_file):
            print(f"Mesh data files not found or not specified in metadata ({data_subdir})")
            return mass_matrix, stiff_matrix, eigenvalues, eigenvectors, None, None, metadata

        try:
            coordinates_np = np.load(coords_file, allow_pickle=True)
            elements_np = np.load(elements_file, allow_pickle=True)
            print(f"Mesh data loaded: Coords {coordinates_np.shape}, Elements {elements_np.shape}")
        except Exception as e:
            print(f"Error loading mesh data: {e}")
            traceback.print_exc()
            return mass_matrix, stiff_matrix, eigenvalues, eigenvectors, None, None, metadata

        # --- Load Fixed DOFs ---
        fixed_dofs_np = None
        fixed_dofs_file_key = 'fixed_dofs_file'
        if fixed_dofs_file_key in metadata and metadata[fixed_dofs_file_key]:
            fixed_dofs_file = os.path.join(data_subdir, metadata[fixed_dofs_file_key])
            if os.path.exists(fixed_dofs_file):
                try:
                    fixed_dofs_np = np.load(fixed_dofs_file, allow_pickle=True)
                    print(f"Fixed DOFs loaded: Shape {fixed_dofs_np.shape}")
                except Exception as e:
                    print(f"Error loading fixed DOFs: {e}")
                    traceback.print_exc()
            else:
                print(f"Fixed DOFs file not found: {fixed_dofs_file}")
        else:
            print("Fixed DOFs file not specified in metadata or is None.")


        print("Successfully loaded all required SOFA data.")
        return mass_matrix, stiff_matrix, eigenvalues, eigenvectors, coordinates_np, elements_np, fixed_dofs_np, metadata # Added fixed_dofs_np



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


    def load_z_dataset(self, dataset_base_path="z_dataset"):
        """
        Load z coordinates, ground truth energies, and ground truth displacements
        from the z_dataset folder. Skips files containing NaN values.
        """
        print(f"Loading z_dataset from base path: {dataset_base_path} for {self.num_modes} modes...")
        
        # self.num_modes is equivalent to L (latent_dim), which should match the subfolder name
        mode_subfolder_name = f"{self.num_modes}_modes"
        mode_subfolder_path = os.path.join(dataset_base_path, mode_subfolder_name)

        if not os.path.isabs(mode_subfolder_path):
            # project_root is defined at the module level
            # Ensure project_root is accessible here or pass it if it's not a global/module var
            # For this example, assuming project_root is available as defined at the top of the script
            mode_subfolder_path = os.path.join(project_root, mode_subfolder_path)
            print(f"Converted mode_subfolder_path to absolute: {mode_subfolder_path}")

        if not os.path.exists(mode_subfolder_path):
            print(f"Error: z_dataset subfolder not found: {mode_subfolder_path}")
            print("Please ensure you have run the 'sofa_scripts/z_dataset_builder.py' script to generate the dataset.")
            self.loaded_z_coords = None
            self.loaded_energies_gt = None
            self.loaded_displacements_gt = None
            return

        data_files = sorted(glob.glob(os.path.join(mode_subfolder_path, "data_*.npz")))
        displacement_files = sorted(glob.glob(os.path.join(mode_subfolder_path, "displacement_*.npz")))

        if not data_files:
            print(f"No 'data_*.npz' files found in {mode_subfolder_path}")
            print("Please ensure you have run the 'sofa_scripts/z_dataset_builder.py' script to generate the dataset.")
            self.loaded_z_coords = None
            self.loaded_energies_gt = None
            self.loaded_displacements_gt = None
            return
        
        loaded_z_list = []
        loaded_energies_list = []
        loaded_displacements_list = []
        
        # Track statistics
        skipped_files = 0
        total_files = len(data_files)

        # Determine how many pairs to load
        # If displacement files are missing for some data files, we'll only load z and energy for those.
        
        processed_data_indices = set()

        for data_file_path in data_files:
            try:
                file_index_str = os.path.basename(data_file_path).replace('data_', '').replace('.npz', '')
                
                with np.load(data_file_path) as data:
                    z_sample = data['z']
                    energy_sample = data['energy']
                
                # Check for NaN values in z and energy
                if np.isnan(z_sample).any():
                    print(f"Warning: Skipping {data_file_path} - z contains NaN values")
                    skipped_files += 1
                    continue
                    
                if np.isnan(energy_sample).any():
                    print(f"Warning: Skipping {data_file_path} - energy contains NaN values")
                    skipped_files += 1
                    continue
                    
                # Check for infinite values as well
                if np.isinf(z_sample).any():
                    print(f"Warning: Skipping {data_file_path} - z contains infinite values")
                    skipped_files += 1
                    continue
                    
                if np.isinf(energy_sample).any():
                    print(f"Warning: Skipping {data_file_path} - energy contains infinite values")
                    skipped_files += 1
                    continue
                
                loaded_z_list.append(z_sample)
                loaded_energies_list.append(energy_sample)
                processed_data_indices.add(file_index_str)

                # Try to find corresponding displacement file
                disp_file_path = os.path.join(mode_subfolder_path, f"displacement_{file_index_str}.npz")
                if os.path.exists(disp_file_path):
                    with np.load(disp_file_path) as disp_data:
                        displacement_sample = disp_data['u_flat']
                        
                        # Check for NaN values in displacement
                        if np.isnan(displacement_sample).any():
                            print(f"Warning: Displacement file {disp_file_path} contains NaN values - using zero placeholder")
                            loaded_displacements_list.append(np.zeros(self.output_dim, dtype=np.float64))
                        elif np.isinf(displacement_sample).any():
                            print(f"Warning: Displacement file {disp_file_path} contains infinite values - using zero placeholder")
                            loaded_displacements_list.append(np.zeros(self.output_dim, dtype=np.float64))
                        else:
                            loaded_displacements_list.append(displacement_sample)
                else:
                    # If no displacement file, append None or a zero array of expected shape
                    # For simplicity, let's append None and handle it later if necessary
                    # Or, ensure your logic can handle cases where displacements might be missing for some z
                    print(f"Warning: Displacement file not found for {data_file_path}. Displacement will be missing for this sample.")
                    loaded_displacements_list.append(np.zeros(self.output_dim, dtype=np.float64)) # Placeholder

            except Exception as e:
                print(f"Error loading data from {data_file_path}: {e}")
                skipped_files += 1
                continue
        
        # Print statistics
        valid_files = len(loaded_z_list)
        print(f"Dataset loading statistics:")
        print(f"  Total files found: {total_files}")
        print(f"  Valid files loaded: {valid_files}")
        print(f"  Files skipped (NaN/Inf/Error): {skipped_files}")
        print(f"  Success rate: {100 * valid_files / total_files:.1f}%")
        
        if not loaded_z_list:
            print("No valid data successfully loaded from z_dataset after NaN/Inf filtering.")
            print("Please check your dataset for data quality issues.")
            self.loaded_z_coords = None
            self.loaded_energies_gt = None
            self.loaded_displacements_gt = None
            return

        self.loaded_z_coords = torch.tensor(np.stack(loaded_z_list), device=self.device, dtype=torch.float64)
        self.loaded_energies_gt = torch.tensor(np.stack(loaded_energies_list), device=self.device, dtype=torch.float64)
        
        if loaded_displacements_list and len(loaded_displacements_list) == len(loaded_z_list):
            try:
                # Filter out any None placeholders if they were used and stacking fails
                valid_displacements = [d for d in loaded_displacements_list if d is not None]
                if len(valid_displacements) == len(loaded_z_list): # All had displacements
                    self.loaded_displacements_gt = torch.tensor(np.stack(valid_displacements), device=self.device, dtype=torch.float64)
                elif len(valid_displacements) > 0:
                    print(f"Warning: Some displacement data was missing. Loaded {len(valid_displacements)} displacement samples.")
                    # This case needs careful handling if you strictly need paired data.
                    # For now, this might lead to shape mismatches if not all z have a displacement.
                    # A safer approach if strict pairing is needed is to only keep z/energy for which displacement exists.
                    # However, the request was to load all three. The placeholder zeros will allow stacking.
                    self.loaded_displacements_gt = torch.tensor(np.stack(loaded_displacements_list), device=self.device, dtype=torch.float64)
                else:
                    self.loaded_displacements_gt = None
            except Exception as e:
                print(f"Error stacking displacement data (possibly due to missing files or shape mismatches): {e}")
                self.loaded_displacements_gt = None
        else:
            self.loaded_displacements_gt = None

        print(f"Successfully loaded {len(self.loaded_z_coords)} z samples and energies from {mode_subfolder_path}.")
        if self.loaded_displacements_gt is not None:
            print(f"  Loaded {len(self.loaded_displacements_gt)} displacement samples.")
            print(f"  Loaded z shape: {self.loaded_z_coords.shape}")
            print(f"  Loaded energies shape: {self.loaded_energies_gt.shape}")
            print(f"  Loaded displacements shape: {self.loaded_displacements_gt.shape}")
        else:
            print("  Displacement data not fully loaded or missing.")
            print(f"  Loaded z shape: {self.loaded_z_coords.shape}")
            print(f"  Loaded energies shape: {self.loaded_energies_gt.shape}")



    def train(self, num_epochs=1000):
        """
        Train the model using batched processing with strong orthogonality constraints.
        Similar to the reference implementation with St. Venant-Kirchhoff energy.
        """
        print("Starting training...")
        
        # Setup training parameters
        batch_size = 128  # You can add this to config
        rest_idx = 0    # Index for rest shape in batch
        print_every = 1
        checkpoint_every = 50
        
        # Get rest shape (undeformed configuration)
        coordinates_th = torch.tensor(self.coordinates_np, device=self.device, dtype=torch.float64)
        X = torch.zeros_like(coordinates_th, device=self.device, dtype=torch.float64)
        # X = X.view(1, -1).expand(batch_size, -1) # This X is not directly used later, can be removed if not needed
        
        # Use a subset of linear modes (you might need to adjust indices)
        L = self.num_modes  # Use at most L linear modes
        linear_modes = self.linear_modes[:, :L].detach()  # Use the first L modes, detach as it's fixed
        
        # Setup iteration counter and best loss tracking
        iteration = 0
        best_loss = float('inf')
        
        # Make sure model accepts batched inputs
        # Modify Net forward method to handle batched inputs
        original_forward = self.model.forward # Assuming self.model is already instantiated
        
        # This new_forward wrapper might not be necessary if your model's forward already handles batches correctly.
        # If ResidualNet's forward already handles batches (it seems to), you can remove this wrapper.
        def new_forward(x):
            is_batch = x.dim() > 1
            if not is_batch:
                x = x.unsqueeze(0)  # Add batch dimension
            
            # Process through network
            result = original_forward(x) # Call the original model's forward
            
            if not is_batch:
                result = result.squeeze(0)  # Remove batch dimension if input wasn't batched
            return result
            
        self.model.forward = new_forward # Apply the wrapper
        
        # Use LBFGS optimizer
        optimizer = torch.optim.LBFGS(self.model.parameters(), 
                                    lr=1, # LBFGS often uses lr=1
                                    max_iter=100, # Max LBFGS iterations per optimizer.step()
                                    max_eval=200, # Max function evaluations per optimizer.step()
                                    tolerance_grad=1e-05,
                                    tolerance_change=1e-07,
                                    history_size=100,
                                    line_search_fn="strong_wolfe")
        
        # Add scheduler for learning rate reduction
        scheduler = LBFGSScheduler(
            optimizer,
            factor=0.5,
            patience=100, 
            threshold=0.01,
            min_lr=1e-6,
            verbose=True
        )
        # patience = 0 # This local patience seems unused, scheduler has its own
        
        example_eigenvalue_scale = self.compute_eigenvalue_based_scale() # This might be for info only now
        print(f"Example eigenvalue scale (for reference): {example_eigenvalue_scale[:min(L, len(example_eigenvalue_scale))]}")

        if not hasattr(self, 'viz_plotter'):
            self.create_live_visualizer()

        while iteration < num_epochs:
            lbfgs_iter = 0 # Reset LBFGS iteration count for each main training iteration

            # --- Load z, (optional) energies_gt, (optional) displacements_gt from dataset ---
            with torch.no_grad():
                if self.loaded_z_coords is None or len(self.loaded_z_coords) == 0:
                    raise RuntimeError("z_dataset not loaded or empty. Training cannot continue.")
                
                current_dataset_size = len(self.loaded_z_coords)
                if current_dataset_size < batch_size:
                    print(f"Warning: Requested batch_size {batch_size} is larger than loaded dataset size {current_dataset_size}. Using all {current_dataset_size} loaded samples.")
                    indices = torch.arange(current_dataset_size, device=self.device)
                else:
                    indices = torch.randperm(current_dataset_size, device=self.device)[:batch_size]

                z = self.loaded_z_coords[indices]
                
                # Ensure the rest shape (if used) has zero latent activation
                #z = z * torch.rand(batch_size, 1, device=self.device) * 0.999 + 0.001


                # z = torch.rand(batch_size, L, device=self.device) * mode_scales * 2 - mode_scales
                # z[rest_idx, :] = 0  # Set rest shape latent to zero
                #concatenate the generated samples with the rest shape
                
                # Compute linear displacements
                # l = torch.matmul(z, linear_modes.T)
                
                # Create normalized constraint directions
                # constraint_dir = torch.matmul(z, linear_modes.T)
                # constraint_norms = torch.norm(constraint_dir, p=2, dim=1, keepdim=True)
                # # Avoid division by zero
                # constraint_norms = torch.clamp(constraint_norms, min=1e-8)
                # constraint_dir = constraint_dir / constraint_norms
                # constraint_dir[rest_idx] = 0  # Zero out rest shape constraints
            
                # Track these values outside the closure
                energy_val = 0
                ortho_val = 0
                origin_val = 0
                loss_val = 0
                
                # Extract ground truth data outside closure for proper scope
                ground_truth_energy = self.loaded_energies_gt[indices] if self.loaded_energies_gt is not None else None
                ground_truth_displacement = self.loaded_displacements_gt[indices] if self.loaded_displacements_gt is not None else None
                
                # Define closure for optimizer
                def closure():
                    nonlocal energy_val, ortho_val, origin_val, loss_val, lbfgs_iter, iteration, best_loss
                    nonlocal z, ground_truth_energy, ground_truth_displacement

                    lbfgs_iter += 1
                    
                    optimizer.zero_grad()
                    
                    # Compute nonlinear correction
                    y = self.model(z)
                    
                    linear_displacement = torch.matmul(z, linear_modes.T)  # Linear displacement
                    
                    # Compute energy (use your energy calculator)
                    u_total_batch = linear_displacement + y  # Total displacement   
                                    
                    
                    
                    # After (if processing a batch):
                    batch_size = u_total_batch.shape[0]

                    energies = self.energy_calculator(u_total_batch)
                    energy = torch.mean(energies)  # Average energy across batch
             

                    # volume_sample_indices = [0, min(10, batch_size-1), rest_idx]  # Rest shape + a couple samples
                    # volume_results = []
                    # for idx in volume_sample_indices:
                    #     vol_result = self.energy_calculator.compute_volume_comparison(
                    #         l[idx:idx+1], u_total_batch[idx:idx+1])
                    #     volume_results.append(vol_result)
                    
                    # # Calculate average volume metrics across the samples
                    # avg_linear_ratio = sum(r['linear_volume_ratio'] for r in volume_results) / len(volume_results)
                    # avg_neural_ratio = sum(r['neural_volume_ratio'] for r in volume_results) / len(volume_results)
                    
                    # # Compute volume preservation penalty (squared deviation from 1.0)
                    # vol_penalty = 1000.0 * torch.mean((torch.tensor(
                    #     [r['neural_volume_ratio'] for r in volume_results], 
                    #     device=self.device, dtype=torch.float64) - 1.0)**2)


                    # Calculate maximum displacements
                    # mean_linear = torch.mean(torch.norm(l.reshape(batch_size, -1, 3), dim=2)).item()
                    # mean_total = torch.mean(torch.norm(u_total_batch.reshape(batch_size, -1, 3), dim=2)).item()
                    # mean_correction = torch.mean(torch.norm(y.reshape(batch_size, -1, 3), dim=2)).item()

                    # nonlinear_ratio = mean_correction / mean_total
                    
                    # Compute orthogonality constraint (using the same approach as reference)
                    # ortho = torch.mean(torch.sum(y * constraint_dir, dim=1)**2)
                    
                    # Compute origin constraint for rest shape
                    # origin = torch.sum(y[rest_idx]**2)

                    bc_penalty = 0.0
                    if self.fixed_dofs_th is not None and len(self.fixed_dofs_th) > 0:
                        # Get the total displacement at the fixed DOFs for the entire batch
                        # u_total_batch shape: (batch_size, num_dofs)
                        # self.fixed_dofs_th shape: (num_fixed_dofs,)
                        fixed_displacements = u_total_batch[:, self.fixed_dofs_th] # Shape: (batch_size, num_fixed_dofs)
                        # Calculate the squared L2 norm of displacements at fixed DOFs, averaged over batch
                        bc_penalty = torch.mean(torch.sum(fixed_displacements**2, dim=1))



                 

                    # Scale energy by maximum linear displacement to get comparable units
                    # max_linear_disp = torch.max(torch.norm(l.reshape(batch_size, -1, 3), dim=2))
                    # energies_scaling = energies / max_linear_disp**2

                    # energy_scaling = torch.log10(torch.mean(energies_scaling))  # Average energy across batch

                    # Add incentive for beneficial nonlinearity (energy improvement term)
                    # u_linear_only = l.detach()  # Detach to avoid affecting linear gradients
                    # energy_linear = self.energy_calculator(u_linear_only).mean()
                    # energy_improvement = (torch.relu(energy_linear - energy))/(energy_linear + 1e-8)  
                    # improvement_loss = (energy_linear - energy) / (energy_linear + 1e-8)  
                    mse_energy_val_for_print = float('nan')
                    if ground_truth_energy is not None:
                        # 'energy' is the mean predicted energy for the batch (scalar tensor)
                        # 'ground_truth_energy' is a tensor of GT energies for the batch
                        mean_gt_energy_for_batch = torch.mean(ground_truth_energy.squeeze()) # scalar tensor

                        # Using Absolute Error (AE) to compare the mean predicted energy
                        # with the mean ground truth energy for the current batch.
                        # AE = |mean_predicted_batch_energy - mean_gt_batch_energy|
                        # This directly addresses comparing two "single scalar numbers".
                        absolute_error_calc = torch.abs(energy - mean_gt_energy_for_batch)
                        mse_energy_val_for_print = absolute_error_calc.item()
                        # Note: The variable 'mse_energy_val_for_print' now stores the
                        # Absolute Error between the batch's mean predicted energy and
                        # the batch's mean ground truth energy. The print label for this
                        # value (outside this code block) might still refer to "MSE".
                        mse_displacement_val_for_print = float('nan')
                    if ground_truth_displacement is not None:
                        # Ensure u_total_batch and ground_truth_displacement are compatible
                        # u_total_batch shape: (batch_size, num_dofs)
                        # ground_truth_displacement shape: (batch_size, num_dofs)
                        
                        # Calculate MSE per sample in the batch, then average
                        # mse_per_sample = torch.mean((u_total_batch - ground_truth_displacement)**2, dim=1) # MSE for each sample
                        # mse_displacement_calc = torch.mean(mse_per_sample) # Average MSE over the batch
                        
                        # RMSE: sqrt of the mean of squared errors
                        squared_errors = (u_total_batch - ground_truth_displacement)**2
                        mean_squared_errors = torch.mean(squared_errors) # MSE across the entire batch
                        rmse_displacement_calc = torch.sqrt(mean_squared_errors)
                        mse_displacement_val_for_print = rmse_displacement_calc.item() # Storing RMSE here



                    scale_value = 0.00000000001
                    energy_tanh = torch.tanh(scale_value * energy)

                                
                    # Get the raw div(P) tensor
                    # raw_div_p = self.energy_calculator.compute_div_p(u_total_batch)






                    # Initialize MSE loss terms
                    mse_displacement_loss_term = torch.tensor(0.0, device=self.device, dtype=torch.float64)
                    mse_energy_loss_term = torch.tensor(0.0, device=self.device, dtype=torch.float64)

                    # Calculate MSE for displacement if ground truth is available
                    if ground_truth_displacement is not None and ground_truth_displacement.numel() > 0:
                        # u_total_batch shape: (batch_size, num_dofs)
                        # ground_truth_displacement shape: (batch_size, num_dofs)
                        if u_total_batch.shape == ground_truth_displacement.shape:
                            mse_displacement_loss_term = torch.mean((u_total_batch - ground_truth_displacement)**2)
                        else:
                            # This case should ideally not happen if data loading is consistent
                            logger.warning(f"Shape mismatch for displacement MSE. Pred: {u_total_batch.shape}, GT: {ground_truth_displacement.shape}. Skipping displacement MSE term for this batch.")

                    # Calculate MSE for energy if ground truth is available
                    if ground_truth_energy is not None and ground_truth_energy.numel() > 0:
                        # 'energy' is the mean predicted energy for the batch (scalar tensor)
                        # 'ground_truth_energy' is a tensor of GT energies for the batch
                        # Squeeze ground_truth_energy in case it's (batch_size, 1)
                        mean_gt_energy_for_batch = torch.mean(ground_truth_energy.squeeze()) # scalar tensor
                        
                        # Compare log10 of energies to make gradients easier
                        log_energy_pred = torch.log10(torch.clamp(energy, min=1e-10))
                        log_energy_gt = torch.log10(torch.clamp(mean_gt_energy_for_batch, min=1e-10))
                        mse_energy_loss_term = (log_energy_pred - log_energy_gt)**2     
                    # Define weights for the new loss terms. 
                    # These weights might need tuning.
                    # Using a large weight similar to ortho and bc_penalty.
                    weight_mse_displacement = 1e6
                    weight_mse_energy = 1e2

                    # Modified loss: includes original energy term, ortho, bc, and new MSE terms
                    loss = 1e10 * bc_penalty + \
                           weight_mse_displacement * mse_displacement_loss_term + \
                           weight_mse_energy * mse_energy_loss_term
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
                        print(f"│ {'Raw Energy:':<20} {energy.item():<12.6f}")
                        # Inside the closure, before the print block:
                        
                        # Ground Truth Comparison Metrics
                        print(f"│ GROUND TRUTH COMPARISON:")
                        rmse_linear_vs_gt_val_for_print = float('nan')

                        print(f"│ {'AE Energy (Pred vs GT):':<25} {mse_energy_val_for_print:<12.6e} │ {'RMSE Disp (Total vs GT):':<25} {mse_displacement_val_for_print:<12.6e}")
                        print(f"│ {'RMSE Disp (Linear vs GT):':<25} {rmse_linear_vs_gt_val_for_print:<12.6e} │")
                        # Constraint metrics section
                        print(f"│ CONSTRAINT METRICS:")
                                        
                                            # # Displacement metrics section
                        # print(f"│ DISPLACEMENT METRICS:")
                        # print(f"│ {'Mean Linear:':<20} {mean_linear:<12.6f} │ {'Mean Total:':<20} {mean_total:<12.6f}")
                        # print(f"│ {'Mean Correction:':<20} {mean_correction:<12.6f} │ {'Nonlinear Ratio:':<20} {nonlinear_ratio*100:.2f}%")
                        
                        # Divergence metrics section
                        # div_p_means = torch.mean(raw_div_p, dim=0).mean(dim=0)
                        # print(f"│ DIVERGENCE METRICS:")
                        # print(f"│ {'Direction:':<20} {'X':<17} {'Y':<17} {'Z':<17}")
                        # print(f"│ {'Div(P):':<12} {div_p_means[0].item():15.6e} {div_p_means[1].item():15.6e} {div_p_means[2].item():15.6e}")
                        # print(f"│ {'Div(P) Loss:':<20} {log_scaled_div_p.item():<12.6f} │ {'Raw Div(P) L2:':<20} {raw_div_p_L2_mean.item():<12.6e}")

                        # # Add volume metrics section
                        # print(f"│ VOLUME PRESERVATION:")
                        # print(f"│ {'Linear Volume Ratio:':<20} {avg_linear_ratio:<12.6f} │ {'Neural Volume Ratio:':<20} {avg_neural_ratio:<12.6f}")
                        # print(f"│ {'Linear Volume Change:':<20} {(avg_linear_ratio-1)*100:<12.4f}% │ {'Neural Volume Change:':<20} {(avg_neural_ratio-1)*100:<12.4f}%")
                        # print(f"│ {'Volume Penalty:':<20} {vol_penalty.item():<12.6f}")
        
                        # Print z vector (first 10 modes of the first batch sample)
                        z_to_print = z[0, :min(10, z.shape[1])].detach().cpu().numpy()
                        z_str = ", ".join([f"{val:.3f}" for val in z_to_print])
                        print(f"│ {'z (first sample, up to 10 modes):':<30} [{z_str}]")
                        # Final loss value
                        print(f"{sep_line}")
                        print(f"TOTAL LOSS: {loss.item():.6e} - {lbfgs_progress}")
                        print(f"{sep_line}\n")
                    

                    energy_val = energy.item()  # Convert tensor to Python scalar

                    loss_val = loss.item()

                    return loss
                
                # Perform optimization step
                optimizer.step(closure)

                scheduler.step(loss_val)  

                # Choose a random latent vector from the batch
                random_idx = np.random.randint(0, batch_size)
                random_z = z[random_idx].detach().clone()

                # Compute predicted displacement for visualization
                with torch.no_grad():
                    neural_correction = self.model(random_z)
                    predicted_displacement = torch.matmul(random_z, linear_modes.T) + neural_correction     # Get ground truth displacement if available
                gt_displacement = None
                if ground_truth_displacement is not None:
                    gt_displacement = ground_truth_displacement[random_idx]

                # Update visualization with computed displacements
                self.visualize_latent_vector(random_z, iteration=iteration, loss=loss_val)
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
                torch.save(checkpoint, os.path.join('checkpoints', 'best_sofa_dataset.pt'))
                print(f"============ BEST MODEL UPDATED ============")
                print(f"New best model at iteration {iteration} with loss {loss_val:.6e}")
                print(f"============================================")
            
            # # Save periodic checkpoint
            # if iteration % checkpoint_every == 0:
            #     checkpoint = {
            #         'epoch': iteration,
            #         'model_state_dict': self.model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'loss': loss_val,
            #     }
            #     torch.save(checkpoint, os.path.join('checkpoints', f'model_it{iteration}.pt'))
            
            iteration += 1
            patience += 1
            
            # # Early stopping criterion (optional)
            # if patience > 30: #loss_val < 1e-8 or 
            #     print(f"Converged to target loss at iteration {iteration}")
            #     break
        
        # Restore original forward method
        self.model.forward = original_forward # This was outside the loop, ensure it's handled correctly
        print(f"Training complete. Best loss: {best_loss:.8e}")
        if hasattr(self, 'writer') and self.writer:
            self.writer.close()
            print("TensorBoard writer closed.")


        return best_loss    


    def create_live_visualizer(self):
        """Create and return a persistent PyVista plotter using loaded mesh data"""
        print("Creating live visualizer using loaded mesh data...")
        # Create PyVista grid directly from loaded NumPy arrays
        # Need element type information. Assuming tetrahedra if nodes_per_element is 4 or 10.
        # Assuming hexahedra if nodes_per_element is 8 or 27.
        # PyVista expects cell connectivity in a specific format: [n_points, idx1, idx2, ..., n_points, idx1, ...]
        num_elements = self.elements_np.shape[0]
        nodes_per_elem = self.elements_np.shape[1]

        if nodes_per_elem == 4: # Linear Tetrahedron
            cell_type = pyvista.CellType.TETRA
            # Format: [4, node0, node1, node2, node3, 4, node0, ...]
            cells = np.hstack((np.full((num_elements, 1), 4), self.elements_np)).flatten()
        elif nodes_per_elem == 8: # Linear Hexahedron
            cell_type = pyvista.CellType.HEXAHEDRON
            cells = np.hstack((np.full((num_elements, 1), 8), self.elements_np)).flatten()
        elif nodes_per_elem == 10: # Quadratic Tetrahedron
             cell_type = pyvista.CellType.QUADRATIC_TETRA
             cells = np.hstack((np.full((num_elements, 1), 10), self.elements_np)).flatten()
        # Add other element types if needed (e.g., quadratic hex)
        else:
             print(f"Warning: Unsupported element type ({nodes_per_elem} nodes) for PyVista visualization. Using points.")
             # Fallback: visualize as points
             grid = pyvista.PolyData(self.coordinates_np)
             cell_type = None # Indicate no cells
             cells = None
             # Or raise error: raise ValueError(f"Unsupported element type for visualization: {nodes_per_elem} nodes")

        if cell_type is not None:
             grid = pyvista.UnstructuredGrid(cells, [cell_type] * num_elements, self.coordinates_np)
        else: # Fallback to PolyData (points)
             grid = pyvista.PolyData(self.coordinates_np)


        # Create plotter (same as before)
        plotter = pyvista.Plotter(shape=(1, 2), title="Neural Modes Training Visualization",
                            window_size=[1600, 720], off_screen=False)

        # Store grid and visualization components
        self.viz_grid = grid # Store the PyVista grid
        self.viz_plotter = plotter
        self.mesh_actor_left = None
        self.mesh_actor_right = None
        self.info_actor = None

        # Initialize the render window (same as before)
        plotter.show(interactive=False, auto_close=False)
        for i in range(2):
            plotter.subplot(0, i)
            plotter.camera_position = [(20.0, 3.0, 2.0), (0.0, -2.0, 0.0), (0.0, 0.0, 2.0)]
            plotter.camera.zoom(0.5)
        plotter.link_views()

        print("Visualizer created.")
        return plotter

    def visualize_latent_vector(self, z, iteration=None, loss=None):
        """Update visualization using loaded mesh data"""
        if not hasattr(self, 'viz_plotter') or self.viz_plotter is None:
             print("Visualizer not initialized. Skipping visualization.")
             return
        try:
            # Ensure z is properly formatted (same as before)
            # ...
            if not isinstance(z, torch.Tensor): z = torch.tensor(z, device=self.device, dtype=torch.float64)
            if z.dim() > 1: z = z.squeeze()


            # Compute displacements (same as before)
            with torch.no_grad():
                linear_contribution = torch.matmul(z, self.linear_modes.T)
                neural_correction = self.model(z)
                u_total = linear_contribution + neural_correction
                linear_only_np = linear_contribution.detach().cpu().numpy()
                u_total_np = u_total.detach().cpu().numpy()

            # --- No dolfinx interpolation needed ---
            # Displacements are already defined on the nodes
            linear_np = linear_only_np.reshape((-1, 3))
            total_np = u_total_np.reshape((-1, 3))

            # Compute magnitudes (same as before)
            linear_mag = np.linalg.norm(linear_np, axis=1)
            total_mag = np.linalg.norm(total_np, axis=1)
            max_mag = max(np.max(linear_mag), np.max(total_mag), 1e-9) # Avoid zero max
            min_mag = min(np.min(linear_mag), np.min(total_mag))
            color_range = [min_mag, max_mag]

            # --- Update PyVista Plotter ---
            # Left subplot - Linear
            self.viz_plotter.subplot(0, 0)
            if self.mesh_actor_left is not None: self.viz_plotter.remove_actor(self.mesh_actor_left)
            linear_grid = self.viz_grid.copy()
            linear_grid.points = self.coordinates_np + linear_np # Apply displacement directly
            linear_grid["displacement_magnitude"] = linear_mag
            self.mesh_actor_left = self.viz_plotter.add_mesh(
                linear_grid, scalars="displacement_magnitude", cmap="viridis",
                show_edges=False, clim=color_range, reset_camera=False
            )
            self.viz_plotter.add_text("Linear Modes Only", position="upper_edge", font_size=12, color='black')

            # Right subplot - Neural
            self.viz_plotter.subplot(0, 1)
            if self.mesh_actor_right is not None: self.viz_plotter.remove_actor(self.mesh_actor_right)
            total_grid = self.viz_grid.copy()
            total_grid.points = self.coordinates_np + total_np # Apply displacement directly
            total_grid["displacement_magnitude"] = total_mag
            self.mesh_actor_right = self.viz_plotter.add_mesh(
                total_grid, scalars="displacement_magnitude", cmap="viridis",
                show_edges=False, clim=color_range, reset_camera=False
            )
            self.viz_plotter.add_text("Neural Network Prediction", position="upper_edge", font_size=12, color='black')

            # Update info text (same as before)
            self.viz_plotter.subplot(0, 0)
            if iteration is not None and loss is not None:
                if self.info_actor is not None: self.viz_plotter.remove_actor(self.info_actor)
                nonlinear_mag = np.linalg.norm(neural_correction.detach().cpu().numpy())
                total_mag_val = np.linalg.norm(u_total_np)
                nonlinear_percent = (nonlinear_mag / total_mag_val) * 100 if total_mag_val > 1e-9 else 0
                info_text = f"Iteration: {iteration}\nLoss: {loss:.6e}\nNonlinear Contribution: {nonlinear_percent:.2f}%"
                self.info_actor = self.viz_plotter.add_text(info_text, position=(10, 10), font_size=10, color='black')

            # Update render window (same as before)
            self.viz_plotter.update()
            self.viz_plotter.render()

        except Exception as e:
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
        x_coords = self.coordinates_np
        x_range = x_coords[:, 0].max() - x_coords[:, 0].min()
        y_range = x_coords[:, 1].max() - x_coords[:, 1].min() 
        z_range = x_coords[:, 2].max() - x_coords[:, 2].min()
        
        # Calculate characteristic length (average of dimensions)
        char_length = max(x_range, y_range, z_range)
        # Safety factor to avoid extreme deformations
        safety_factor = 0.7
        
      
        
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
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_sofa.pt'))

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
        """Visualize neural modes across a grid of two latent dimensions using loaded mesh data"""
        print(f"Visualizing neural modes for dimensions {dim1} and {dim2}...")

        # Create base PyVista grid directly from loaded NumPy arrays
        num_elements = self.elements_np.shape[0]
        nodes_per_elem = self.elements_np.shape[1]

        if nodes_per_elem == 4: # Linear Tetrahedron
            cell_type = pyvista.CellType.TETRA
            cells = np.hstack((np.full((num_elements, 1), 4), self.elements_np)).flatten()
        elif nodes_per_elem == 8: # Linear Hexahedron
            cell_type = pyvista.CellType.HEXAHEDRON
            cells = np.hstack((np.full((num_elements, 1), 8), self.elements_np)).flatten()
        elif nodes_per_elem == 10: # Quadratic Tetrahedron
             cell_type = pyvista.CellType.QUADRATIC_TETRA
             cells = np.hstack((np.full((num_elements, 1), 10), self.elements_np)).flatten()
        # Add other element types if needed (e.g., quadratic hex)
        else:
             print(f"Warning: Unsupported element type ({nodes_per_elem} nodes) for PyVista visualization. Cannot visualize latent dimensions.")
             return

        try:
            grid = pyvista.UnstructuredGrid(cells, [cell_type] * num_elements, self.coordinates_np)
        except Exception as e:
            print(f"Error creating PyVista grid: {e}. Cannot visualize.")
            return

        # Compute scale for latent vectors
        scale = self.compute_safe_scaling_factor() * 2.0 # Larger scale for better visualization
        values = np.linspace(-scale, scale, num_points)

        # Create plotter with subplots
        plotter = pyvista.Plotter(shape=(num_points, num_points), border=False)

        # Generate modes and plot
        max_overall_disp = 0.0 # Track max displacement for consistent color range
        grids_to_plot = [] # Store grids to determine color range first

        for i, val1 in enumerate(values):
            row_idx = num_points - 1 - i # Reverse order for proper cartesian layout
            for j, val2 in enumerate(values):
                # Create latent vector
                z = torch.zeros(self.latent_dim, device=self.device, dtype=torch.float64)
                z[dim1] = val1
                z[dim2] = val2

                # Compute total displacement
                with torch.no_grad():
                    # Consider only the relevant linear modes for this visualization
                    # This assumes linear_modes columns correspond to latent dimensions
                    linear_contribution = torch.zeros_like(self.linear_modes[:, 0]) # Zero vector of correct size
                    if dim1 < self.linear_modes.shape[1]:
                        linear_contribution += self.linear_modes[:, dim1] * z[dim1]
                    if dim2 < self.linear_modes.shape[1]:
                         linear_contribution += self.linear_modes[:, dim2] * z[dim2]

                    y = self.model(z)
                    u_total = y + linear_contribution
                    u_total_np = u_total.detach().cpu().numpy().reshape((-1, 3))

                # Create deformed grid
                local_grid = grid.copy()
                local_grid.points = self.coordinates_np + u_total_np # Apply displacement directly
                local_grid["displacement_magnitude"] = np.linalg.norm(u_total_np, axis=1)
                max_overall_disp = max(max_overall_disp, np.max(local_grid["displacement_magnitude"]))
                grids_to_plot.append({'grid': local_grid, 'row': row_idx, 'col': j})

        # Now plot with consistent color range
        color_range = [0, max(max_overall_disp, 1e-9)] # Avoid zero range
        for item in grids_to_plot:
            plotter.subplot(item['row'], item['col'])
            # Use show_edges=False for potentially cleaner visualization with many elements
            plotter.add_mesh(item['grid'], scalars="displacement_magnitude",
                             cmap="viridis", show_edges=False, clim=color_range, reset_camera=False)
            # Add compact z-value labels
            val1 = values[num_points - 1 - item['row']]
            val2 = values[item['col']]
            plotter.add_text(f"{val1:.2f}, {val2:.2f}", position="lower_right", font_size=6, color='white')
            plotter.view_isometric() # Set consistent view

        # Add axis labels at edges of the grid
        for i, val1 in enumerate(values):
            row_idx = num_points - 1 - i
            plotter.subplot(row_idx, 0) # Left edge
            plotter.add_text(f"z{dim1}={val1:.2f}", position=(0.01, 0.5), viewport=True, font_size=8, color='white')
        for j, val2 in enumerate(values):
            plotter.subplot(num_points-1, j) # Bottom edge
            plotter.add_text(f"z{dim2}={val2:.2f}", position=(0.5, 0.01), viewport=True, font_size=8, color='white')

        # Link camera views
        plotter.link_views()

        # Add a unified colorbar at the bottom
        plotter.subplot(num_points-1, 0) # Place relative to a subplot
        plotter.add_scalar_bar("Displacement Magnitude", position_x=0.4, position_y=0.01, width=0.2, height=0.02)

        # Add title
        title = f"Neural Modes Matrix: z{dim1} vs z{dim2}"
        plotter.add_text(title, position=(0.5, 0.97), viewport=True, font_size=12, color='black')

        print("Showing latent space visualization...")
        plotter.show()
        print("Visualization complete.")


    def visualize_latent_space(self, num_samples=5, scale=None, modes_to_show=None):
        """
        Visualize the effect of each latent dimension independently using loaded mesh data.

        Args:
            num_samples: Number of samples to take for each mode (-scale to +scale).
            scale: Range of latent values to sample, auto-computed if None.
            modes_to_show: List of specific mode indices to visualize, visualize all if None.
        """
        print("Visualizing latent space modes...")

        # Determine which modes to show
        if modes_to_show is None:
            modes_to_show = list(range(self.latent_dim))
        num_modes = len(modes_to_show)
        if num_modes == 0:
            print("No modes selected to visualize.")
            return

        # Compute scale for latent vectors if not provided
        if scale is None:
            scale = self.compute_safe_scaling_factor() * 2.0 # Larger scale to see clear deformations

        # Create values to sample for each mode
        values = np.linspace(-scale, scale, num_samples)

        # Create base PyVista grid directly from loaded NumPy arrays
        num_elements = self.elements_np.shape[0]
        nodes_per_elem = self.elements_np.shape[1]

        if nodes_per_elem == 4: # Linear Tetrahedron
            cell_type = pyvista.CellType.TETRA
            cells = np.hstack((np.full((num_elements, 1), 4), self.elements_np)).flatten()
        elif nodes_per_elem == 8: # Linear Hexahedron
            cell_type = pyvista.CellType.HEXAHEDRON
            cells = np.hstack((np.full((num_elements, 1), 8), self.elements_np)).flatten()
        elif nodes_per_elem == 10: # Quadratic Tetrahedron
             cell_type = pyvista.CellType.QUADRATIC_TETRA
             cells = np.hstack((np.full((num_elements, 1), 10), self.elements_np)).flatten()
        # Add other element types if needed
        else:
             print(f"Warning: Unsupported element type ({nodes_per_elem} nodes) for PyVista visualization. Cannot visualize latent space.")
             return

        try:
            grid = pyvista.UnstructuredGrid(cells, [cell_type] * num_elements, self.coordinates_np)
        except Exception as e:
            print(f"Error creating PyVista grid: {e}. Cannot visualize.")
            return

        # Create plotter with mode rows and sample columns
        plotter = pyvista.Plotter(shape=(num_modes, num_samples), border=False,
                                window_size=[1600, 200 * num_modes])

        # Visualize each mode with varying values
        max_overall_disp = 0.0
        grids_to_plot = []

        for i, mode_idx in enumerate(modes_to_show):
            if mode_idx >= self.latent_dim:
                print(f"Warning: Skipping mode index {mode_idx} as it exceeds latent dimension {self.latent_dim}.")
                continue
            if mode_idx >= self.linear_modes.shape[1]:
                 print(f"Warning: Skipping mode index {mode_idx} as it exceeds available linear modes {self.linear_modes.shape[1]}.")
                 continue

            for j, val in enumerate(values):
                # Create a zero latent vector
                z = torch.zeros(self.latent_dim, device=self.device, dtype=torch.float64)
                # Set only the current mode to the current value
                z[mode_idx] = val

                # Compute total displacement
                with torch.no_grad():
                    linear_contribution = self.linear_modes[:, mode_idx] * val # Assumes linear_modes columns match latent dims
                    y = self.model(z)
                    u_total = y + linear_contribution
                    u_total_np = u_total.detach().cpu().numpy().reshape((-1, 3))

                # Create deformed grid
                local_grid = grid.copy()
                local_grid.points = self.coordinates_np + u_total_np # Apply displacement directly
                local_grid["displacement_magnitude"] = np.linalg.norm(u_total_np, axis=1)
                max_overall_disp = max(max_overall_disp, np.max(local_grid["displacement_magnitude"]))
                grids_to_plot.append({'grid': local_grid, 'row': i, 'col': j})

        # Plot with consistent color range
        color_range = [0, max(max_overall_disp, 1e-9)] # Avoid zero range
        for item in grids_to_plot:
            plotter.subplot(item['row'], item['col'])
            # Use show_edges=False for potentially cleaner visualization
            plotter.add_mesh(item['grid'], scalars="displacement_magnitude",
                             cmap="viridis", show_edges=False, clim=color_range, reset_camera=False)
            # Add value label
            mode_idx = modes_to_show[item['row']]
            val = values[item['col']]
            plotter.add_text(f"z{mode_idx}={val:.2f}", position="lower_right", font_size=8, color='white')
            plotter.view_isometric() # Set consistent view

        # Add row labels for modes
        for i, mode_idx in enumerate(modes_to_show):
            plotter.subplot(i, 0) # Left edge
            plotter.add_text(f"Mode {mode_idx}", position="left_edge", font_size=12, color='white')

        # Link all camera views
        plotter.link_views()

        # Add a unified colorbar
        plotter.subplot(0, 0) # Place relative to a subplot
        plotter.add_scalar_bar("Displacement Magnitude", position_x=0.4, position_y=0.05,
                        width=0.5, height=0.02, title_font_size=12, label_font_size=10)

        # Add overall title
        plotter.add_text("Neural Latent Space Mode Atlas", position="upper_edge",
                    font_size=16, color='black')

        print("Showing latent space visualization...")
        plotter.show()
        print("Visualization complete.")
        return plotter # Return plotter in case further customization is needed

    def analyze_latent_correlations(self, delta=1e-5, save_path=None):
        """
        Analyze correlations between latent dimensions based on the gradient of the
        TOTAL displacement (linear + neural) at the origin.

        Args:
            delta: Small perturbation for finite difference approximation
            save_path: Path to save visualization
        """
        print("Analyzing latent space correlations (based on total displacement)...")
        self.model.eval() # Ensure model is in evaluation mode

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
                unnormalized_corr_matrix[i, j] = torch.dot(dir_i, dir_j)
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
                norm_i = torch.sqrt(norm_squared[i])
                norm_j = torch.sqrt(norm_squared[j])
                if norm_i > 1e-9 and norm_j > 1e-9:
                    corr_matrix[i, j] /= (norm_i * norm_j)
                else:
                    corr_matrix[i, j] = 0.0 # Avoid division by zero

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

    def analyze_linear_mode_correlations(self, delta=1e-5, save_path=None):
        """
        Analyze correlations between latent dimensions based ONLY on the linear modes.
        This should ideally result in an identity matrix if linear modes are orthogonal.

        Args:
            delta: Perturbation value (set to 1.0 to directly use modes).
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
                unnormalized_corr_matrix[i, j] = torch.dot(dir_i, dir_j)
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
                norm_i = torch.sqrt(norm_squared[i])
                norm_j = torch.sqrt(norm_squared[j])
                if norm_i > 1e-9 and norm_j > 1e-9:
                    corr_matrix[i, j] /= (norm_i * norm_j)
                else:
                    corr_matrix[i, j] = 0.0 # Avoid division by zero

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
        # Convert to numpy for visualization
        corr_np = corr_matrix.detach().cpu().numpy() # Use detach()

        # Create figure
        plt.figure(figsize=(10, 8))

        # Use seaborn for nicer heatmap
        mask = np.zeros_like(corr_np, dtype=bool)

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
            save_path_mod = save_path # Default
            if "Linear" in title_suffix:
                save_path_mod = f"{base}_linear{ext}"
            else:
                save_path_mod = f"{base}_total{ext}"

            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path_mod), exist_ok=True)

            plt.savefig(save_path_mod, dpi=300, bbox_inches='tight')
            print(f"Correlation matrix visualization saved to {save_path_mod}")

        plt.show(block=False) # Use block=False to avoid stopping execution if run non-interactively
        plt.pause(1) # Pause briefly to allow plot to render
        return plt.gcf()

    
    
    

def main():
    print("Starting main function...")
    # Parse arguments
    parser = argparse.ArgumentParser(description='Hybrid Simulation SOFA')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='config file path') # Default to sofa config
    parser.add_argument('--resume', action='store_true', help='resume from best checkpoint')
    parser.add_argument('--skip-training', action='store_true', help='skip training and load best model')
    parser.add_argument('--checkpoint', type=str, default=None, help='specific checkpoint path to load')
    parser.add_argument('--analyze', action='store_true', help='perform analysis (visualization, correlations) after training/loading')
    parser.add_argument('--init-checkpoint', type=str, default=None, help='path to checkpoint to initialize model weights before training (does not load optimizer)')
    args = parser.parse_args()
    config_file_path = args.config
    if not os.path.isabs(config_file_path):
        # project_root is defined at the module level
        config_file_path = os.path.join(project_root, config_file_path)


    cfg = load_config(config_file_path)
    # Use checkpoint_dir from config for logs
    log_dir_base = cfg.get('training', {}).get('checkpoint_dir', 'checkpoints')
    log_dir_specific = cfg.get('training', {}).get('log_dir', 'logs')
    setup_logger('train', log_dir=os.path.join(log_dir_base, log_dir_specific)) # Pass logger name
    print("Arguments parsed and logger setup.")

    # Check for skip_training in both command line and config
    skip_training = args.skip_training or cfg.get('training', {}).get('skip_training', False)

    # Determine checkpoint path
    checkpoint_dir = cfg.get('training', {}).get('checkpoint_dir', 'checkpoints')
    best_checkpoint_filename = 'best_sofa.pt' # Specific name for SOFA training
    default_checkpoint_path = os.path.join(checkpoint_dir, best_checkpoint_filename)

    # Use specific checkpoint if provided, otherwise default best, handle resume flag
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        print(f"Using specified checkpoint: {checkpoint_path}")
    elif args.resume:
        checkpoint_path = default_checkpoint_path
        print(f"Attempting to resume from default best checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = default_checkpoint_path # Default to best even if not resuming explicitly
        print(f"Default checkpoint path set to: {checkpoint_path}")


    print(f"Skip training: {skip_training}")
    print(f"Checkpoint path to load (if exists): {checkpoint_path}")

    engine = Routine(cfg)
    print("Engine initialized.")

    # --- Handle Initial Checkpoint Loading (Model Weights Only) ---
    initial_model_loaded = False
    if args.init_checkpoint and not args.resume: # Only if --init-checkpoint is given AND we are NOT resuming
        init_checkpoint_path = args.init_checkpoint
        if os.path.exists(init_checkpoint_path):
            try:
                print(f"Initializing model weights from: {init_checkpoint_path}")
                checkpoint = torch.load(init_checkpoint_path, map_location=engine.device)
                if 'model_state_dict' in checkpoint:
                    engine.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Successfully loaded model weights from epoch {checkpoint.get('epoch', 'N/A')}.")
                    initial_model_loaded = True
                else:
                    print(f"Warning: Checkpoint {init_checkpoint_path} does not contain 'model_state_dict'.")
            except Exception as e:
                print(f"Error loading initial checkpoint {init_checkpoint_path}: {e}. Starting with fresh model.")
        else:
            print(f"Warning: Initial checkpoint path not found: {init_checkpoint_path}. Starting with fresh model.")
    # --- End Initial Checkpoint Loading ---


    # Training or loading logic
    if skip_training:
        print("Skipping training as requested...")
        if os.path.exists(checkpoint_path):
            print(f"Loading model from {checkpoint_path}")
            engine.load_checkpoint(checkpoint_path)
        else:
            print(f"Warning: No checkpoint found at {checkpoint_path}, using untrained model")
    else:
        # Resume or start fresh training
        start_epoch = 0 # Default start epoch
        if args.resume: # Resume loads model AND optimizer state
             if initial_model_loaded:
                 print("Warning: --init-checkpoint provided but --resume is active. --resume takes precedence for loading.")
             if os.path.exists(checkpoint_path):
                 print(f"Resuming training from {checkpoint_path} (loading model and optimizer)")
                 # load_checkpoint should handle loading both model and optimizer
                 engine.load_checkpoint(checkpoint_path)
                 # Potentially extract start_epoch if load_checkpoint doesn't handle it
                 # start_epoch = checkpoint['epoch'] + 1
             else:
                 print(f"Warning: --resume requested but checkpoint {checkpoint_path} not found. Starting fresh.")
                 if initial_model_loaded:
                      print("Using model weights initialized from --init-checkpoint.")
                 # else: starting completely fresh

        elif not initial_model_loaded: # Neither resume nor init-checkpoint used
             print("Starting training from scratch.")
        # If initial_model_loaded is True and not resuming, we just proceed with the loaded weights

        num_epochs = cfg['training']['num_epochs']
        print(f"Starting training for {num_epochs} epochs (from epoch {start_epoch})...")
        # Modify train call if you implement epoch tracking
        best_loss = engine.train(num_epochs=num_epochs) # Pass start_epoch if needed
        print("Training complete.")

        # Load the best model after training finishes before analysis
        if os.path.exists(default_checkpoint_path):
            print("Loading best model for analysis...")
            engine.load_checkpoint(default_checkpoint_path)
        else:
            print("No best model checkpoint found after training, using final model state.")

    # --- Perform Analysis if requested ---
    if args.analyze:
        print("\n--- Starting Post-Training Analysis ---")
        latent_dim = engine.latent_dim
        print(f"Latent dimension: {latent_dim}")

        # Ensure checkpoint directory exists for saving plots
        analysis_save_dir = cfg.get('training', {}).get('checkpoint_dir', 'checkpoints')
        os.makedirs(analysis_save_dir, exist_ok=True)

        # Add latent space visualization (optional, can be time-consuming)
        # print("\nVisualizing latent space dimensions...")
        # engine.visualize_latent_dimensions(dim1=1, dim2=0, num_points=3)
        # engine.visualize_latent_dimensions(dim1=3, dim2=4, num_points=3)
        print("\nVisualizing latent space modes...")
        engine.visualize_latent_space(num_samples=5)

        print("\nAnalyzing latent space correlations (TOTAL displacement)...")
        correlation_matrix_path_total = os.path.join(analysis_save_dir, 'latent_correlations_total.png')
        corr_matrix_total, _ = engine.analyze_latent_correlations(delta=1e-10, save_path=correlation_matrix_path_total) # Smaller delta might be needed

        print("\nAnalyzing latent space correlations (LINEAR modes only)...")
        correlation_matrix_path_linear = os.path.join(analysis_save_dir, 'latent_correlations_linear.png')
        corr_matrix_linear, _ = engine.analyze_linear_mode_correlations(delta=1e-10, save_path=correlation_matrix_path_linear)

        # Plotting training metrics might require loading tensorboard data, skipping for now
        # print("\nPlotting training metrics...")
        # metrics_plot_path = os.path.join(analysis_save_dir, 'training_metrics.png')
        # engine.plot_training_metrics(save_path=metrics_plot_path)

        print("--- Analysis Complete ---")
    # --- End Analysis Block ---

    print("Main function complete.")

def load_config(config_file):
    import yaml
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

if __name__ == '__main__':
    main()