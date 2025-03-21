import Sofa
import SofaRuntime
import numpy as np 
import os
from Sofa import SofaDeformable
from time import process_time, time
import datetime
from sklearn.preprocessing import MinMaxScaler
# add network path to the python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../network'))
# Add this import at the top of the file
from scipy.sparse.linalg import eigsh
import scipy.sparse as sparse

import json

import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import json
import datetime
import numpy as np

import traceback

import torch
import os
import glob
import json

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




class AnimationStepController(Sofa.Core.Controller):
    def __init__(self, node, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.mass = kwargs.get('mass')
        self.modal_mass = kwargs.get('modal_mass')
        self.fem = kwargs.get('fem')
        self.linear_solver = kwargs.get('linear_solver')
        self.surface_topo = kwargs.get('surface_topo')
        self.MO1 = kwargs.get('MO1')  # Exact solution MO
        self.MO2 = kwargs.get('MO2')  # Modal model MO
        self.MO3 = kwargs.get('MO3')
        self.fixed_box = kwargs.get('fixed_box')
        
        # Add force-related objects from createScene
        self.force_box = kwargs.get('force_box')
        self.force_field = kwargs.get('force_field')
        
        # Store visual models
        self.exact_visual = kwargs.get('exact_visual')
        self.modal_visual = kwargs.get('modal_visual')
        self.neural_visual = kwargs.get('neural_visual')
        
        self.key = kwargs.get('key')
        self.iteration = kwargs.get("sample")
        self.start_time = 0
        self.root = node
        self.save = False
        self.l2_error, self.MSE_error = [], []
        self.l2_deformation, self.MSE_deformation = [], []
        self.RMSE_error, self.RMSE_deformation = [], []
        self.directory = kwargs.get('directory')
        
        # Add material properties to controller
        self.young_modulus = kwargs.get('young_modulus', 5000)
        self.poisson_ratio = kwargs.get('poisson_ratio', 0.25)
        self.density = kwargs.get('density', 10)
        self.mesh_filename = kwargs.get('mesh_filename', 'unknown')
        
        # For modal analysis
        self.mass_matrix = None
        self.stiffness_matrix = None
        self.eigenvectors = None
        self.eigenvalues = None
        self.num_modes = kwargs.get('num_modes', 5)  # Number of modes to use
        
        # Neural network attributes
        self.neural_model = None
        self.neural_model_loaded = False  # Track if we've tried loading
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.checkpoint_dir = kwargs.get('checkpoint_dir', 'checkpoints')

        self.previous_displacements = None  # u_{n-1}
        self.current_displacements = None   # u_n
        self.time_step = kwargs.get('dt', 0.01)
        self.iteration_count = 0
        self.warmup_iterations = 2  # Number of iterations to run numerical solver only
        self.optimizer = None

        # Add these instance variables to store modal and neural displacements
        self.U_modal = None
        self.U_neural = None
        
        # Add reference to store energy calculator
        self.energy_calculator = None

        self.force_cycle_counter = 0
        self.force_cycle_period = 100  # Apply force every 100 iterations
        self.force_active = False
        self.force_magnitude_range = (5, 10)  # Force magnitude range


        
        # Store references to parent contexts
        self.exactSolution = self.MO1.getContext()
        
        print(f"Using directory: {self.directory}")
        print(f"Material properties: E={self.young_modulus}, nu={self.poisson_ratio}, rho={self.density}")




    def onSimulationInitDoneEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the scene graph initialisation.
        """
        self.inputs = []
        self.outputs = []       
        if self.save:
            if not os.path.exists('modal_data'):
                os.mkdir('modal_data')
            # get current time from computer format yyyy-mm-dd-hh-mm-ss and create a folder with that name
            if not os.path.exists(f'modal_data/{self.directory}'):
                os.makedirs(f'modal_data/{self.directory}')
            print(f"Saving data to modal_data/{self.directory}")
        self.sampled = False

        surface = self.surface_topo

        self.idx_surface = surface.triangles.value.reshape(-1)

        self.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        






    def onAnimateBeginEvent(self, event):
        """Apply random force to the simulation in a cyclic pattern"""
        self.bad_sample = False

        # Try loading neural model if not already loaded
        if self.neural_model is None and not self.neural_model_loaded:
            self._load_neural_model()

        # Get mesh extents for reference
        positions = np.array(self.MO1.rest_position.value)
        x_min, y_min, z_min = np.min(positions, axis=0)
        x_max, y_max, z_max = np.max(positions, axis=0)
        
        # Increment force cycle counter
        self.force_cycle_counter += 1
        
        # Check if we need to apply a new force (every 100 iterations)
        if self.force_cycle_counter % self.force_cycle_period == 0:
            # Generate random direction for force
            self.z = np.random.uniform(-1, 1)
            self.phi = np.random.uniform(0, 2*np.pi)
            self.versor = np.array([np.sqrt(1 - self.z**2) * np.cos(self.phi), 
                                np.sqrt(1 - self.z**2) * np.sin(self.phi), 
                                self.z])
                                
            # Scale force based on material stiffness
            self.magnitude = np.random.uniform(*self.force_magnitude_range) * self.young_modulus
            self.externalForce = self.magnitude * self.versor
            
            print(f"Applying force {self.externalForce} in direction {self.versor}")
            
            # Get indices from cff_box
            indices_forces = list(self.force_box.indices.value)
            
            # Skip if no nodes are affected
            if len(indices_forces) == 0:
                print("No nodes in force application box - trying a different approach")
                fallback_indices = np.where(positions[:, 2] > (z_min + z_max) / 2)[0]
                
                if len(fallback_indices) == 0:
                    print("Still no nodes found - skipping this sample")
                    self.bad_sample = True
                    return
                    
                indices_forces = fallback_indices.tolist()
            
            print(f"Applying force to {len(indices_forces)} nodes")
            
            # Remove old force field
            self.remove_force_field()
            
            # Create a new force field with object reference
            try:
                self.force_field = self.exactSolution.addObject('ConstantForceField', 
                                                        name='ExternalForce', 
                                                        indices=indices_forces,
                                                        totalForce=self.externalForce,
                                                        showArrowSize=0.1,
                                                        showColor="0.2 0.2 0.8 1")
                self.force_field.init()
                self.force_active = True
                print("Created new force field")
            except Exception as e:
                print(f"Error creating force field: {str(e)}")
                self.force_active = False
        
        # If force was active in the previous step, remove it now (so it's only active for 1 iteration)
        elif self.force_active:
            print("Removing force after one iteration")
            self.remove_force_field()
            self.force_active = False

        # Track iteration for dynamics
        self.iteration_count += 1
        
        # Initialize optimizer for latent variable optimization if we're beyond warmup
        if self.iteration_count >= self.warmup_iterations and self.optimizer is None and self.neural_model is not None:
            self.setup_latent_optimizer()
        
        self.start_time = process_time()

    def remove_force_field(self):
        """Safely remove the force field if it exists"""
        try:
            if self.force_field is not None:
                self.exactSolution.removeObject(self.force_field)
                print("Removed force field")
                self.force_field = None
        except Exception as e:
            print(f"Error removing force field: {str(e)}")


    def onAnimateEndEvent(self, event):
        
        
        # Get matrices from SOFA
        self.mass_matrix = self.mass.assembleMMatrix()
        self.stiffness_matrix = self.fem.assembleKMatrix()
        
        
            
        # Solve eigenvalue problem if not already done
        if self.eigenvectors is None or self.eigenvalues is None:
            success = self.compute_eigenmodes()
            if not success:
                print("ERROR: Failed to compute eigenmodes - skipping modal projection")
                return  # Exit early if we couldn't compute eigenmodes
        
        # Get current displacement from exact solution
        U_exact = self.compute_displacement(self.MO1)
        
        # For the first two iterations, just store displacements and use standard approach
        if self.iteration_count <= self.warmup_iterations:
            # Store displacement history
            if self.previous_displacements is None:
                # First iteration - just store current as previous
                self.previous_displacements = U_exact.copy()
            else:
                # Second iteration - update history
                self.previous_displacements, self.current_displacements = self.previous_displacements.copy(), U_exact.copy()
                
            # Use standard modal projection for visualization
            modal_error = self.standard_modal_projection(U_exact)
            
            print(f"Warmup iteration {self.iteration_count}/{self.warmup_iterations}, storing displacement history")
            return
            
        # Beyond warmup: use neural prediction
        neural_error = self.neural_dynamic_prediction()
        
        # Update displacement history
        self.previous_displacements, self.current_displacements = self.current_displacements.copy(), U_exact.copy()
        
        # Print performance metrics
        print(f"Time step {self.iteration_count}, Neural prediction error: {neural_error:.6f}")

        # Compute errors between exact and approximated solutions - use stored U_modal
        if self.U_modal is not None:
            modal_error = np.linalg.norm(U_exact - self.U_modal) / np.linalg.norm(U_exact)
            print(f"Modal reconstruction error: {modal_error:.6f}")
        
        if self.U_neural is not None:
            # Neural model is a correction to modal model
            full_neural_solution = self.U_modal + self.U_neural
            neural_error = np.linalg.norm(U_exact - full_neural_solution) / np.linalg.norm(U_exact)
            print(f"Neural model error: {neural_error:.6f}")
        
        # Save matrices and metadata as before
        matrices_dir = 'matrices'
        os.makedirs(matrices_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'mesh_file': self.mesh_filename,
            'young_modulus': self.young_modulus,
            'poisson_ratio': self.poisson_ratio,
            'density': self.density,
            'size': self.mass_matrix.shape[0] if hasattr(self, 'mass_matrix') and self.mass_matrix is not None else 0,
            'num_modes': self.num_modes,
            'modal_error': float(modal_error) if 'modal_error' in locals() else None,
            'neural_error': float(neural_error)
        }
        
        with open(f'{matrices_dir}/metadata_{timestamp}.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Matrices and data saved to {matrices_dir} with timestamp {timestamp}")
        self.end_time = process_time()

        print(f"Time taken: {self.end_time - self.start_time:.2f} seconds")




    def standard_modal_projection(self, U_exact):
        """Use standard modal projection for comparison and visualization"""
        # Get matrices from SOFA if needed
        if self.mass_matrix is None or self.stiffness_matrix is None:
            self.mass_matrix = self.mass.assembleMMatrix()
            self.stiffness_matrix = self.fem.assembleKMatrix()
            
        # Solve eigenvalue problem if not already done
        if self.eigenvectors is None or self.eigenvalues is None:
            success = self.compute_eigenmodes()
            if not success:
                print("ERROR: Failed to compute eigenmodes - skipping modal projection")
                return float('inf')
                
        # Project to modal coordinates
        q = self.project_to_modal_coords(U_exact)
        
        # Store for later use
        self.current_modal_coords = q
        
        # Reconstruct displacement using direct modal coordinates
        U_modal = self.reconstruct_from_modal_coords(q)
        
        # Store for later use (FIX: Store as an instance variable)
        self.U_modal = U_modal
        
        # Update mechanical objects
        self.update_modal_model(U_modal)
        
        # Compute errors between exact and approximated solutions
        modal_error = np.linalg.norm(U_exact - U_modal) / np.linalg.norm(U_exact)
        print(f"Modal reconstruction error: {modal_error:.6f}")
        
        return modal_error

    def neural_dynamic_prediction(self):
        """Solve the dynamic optimization problem for the next time step"""
        # Initialize energy calculator if needed
        if self.energy_calculator is None:
            success = self.initialize_energy_calculator()
            
        if self.neural_model is None:
            print("Neural model not available, falling back to standard modal projection")
            return self.standard_modal_projection(self.current_displacements)
            
        if self.optimizer is None:
            self.setup_latent_optimizer()
        
        # Convert numpy arrays to PyTorch tensors
        current_u_tensor = torch.tensor(self.current_displacements.flatten(), 
                                        dtype=torch.float64, device=self.device)
        previous_u_tensor = torch.tensor(self.previous_displacements.flatten(), 
                                        dtype=torch.float64, device=self.device)
        self.z_opt = self.project_to_modal_coords(self.current_displacements)
        # Convert to tensor
        self.z_opt = torch.tensor(self.z_opt, dtype=torch.float64, device=self.device)
        
        # Mass matrix to PyTorch (sparse or dense as needed)
        M = torch.tensor(self.mass_matrix.toarray(), dtype=torch.float64, device=self.device)
        
        # Target acceleration from Verlet: (u_{n+1} - 2u_n + u_{n-1})
        # In this formulation, we're predicting u_{n+1}
        target = -2 * current_u_tensor + previous_u_tensor


        def closure():
            self.optimizer.zero_grad()
            
            # Forward pass through neural model to get predicted displacements
            pred_displacements = self.neural_model(self.z_opt)
            
            # Compute squared M-norm of acceleration difference: |n(z) - 2u_n + u_{n-1}|_M
            accel_diff = pred_displacements - target
            
            # Use torch.matmul for matrix multiplication
            M_norm = torch.matmul(accel_diff.unsqueeze(0), 
                                torch.matmul(M, accel_diff.unsqueeze(1))).squeeze()
            
            # Add elastic energy E(n(z)) from the model
            # This computes the elastic energy of the predicted configuration
            elastic_energy = 0.0
            try:
                if self.energy_calculator is not None:
                    # Try using the energy calculator
                    elastic_energy = self.energy_calculator(pred_displacements).item()
                    elastic_energy_tensor = torch.tensor(elastic_energy, dtype=torch.float64, device=self.device)
                    
                    # Total loss: acceleration matching + regularization from elastic energy
                    loss = M_norm + 0.1 * elastic_energy_tensor  # Weight elastic energy less if causing issues
                else:
                    # If energy calculator not available, use only acceleration matching
                    loss = M_norm
                    print("Warning: Energy calculation skipped - using only acceleration matching")
            except Exception as e:
                # If energy calculation fails, use only acceleration matching
                loss = M_norm
                print(f"Warning: Energy calculation failed: {e} - using only acceleration matching")
                elastic_energy = 0.0
            
            # Compute gradients
            loss.backward()
            
            # Print optimization progress
            print(f"Optimization loss: {loss.item():.6f}, Accel match: {M_norm.item():.6f}, Energy: {elastic_energy:.6f}")
            
            return loss
        
        # Run the optimization
        try:
            self.optimizer.step(closure)
        except Exception as e:
            print(f"Optimization error: {str(e)}")
            print(traceback.format_exc())
            
        # Get the optimized prediction
        with torch.no_grad():
            optimized_u = self.neural_model(self.z_opt).cpu().numpy()
        
        # Reshape to (nodes, 3) format
        optimized_u = optimized_u.reshape(-1, 3)
        
        # Store neural prediction for later use
        self.U_neural = optimized_u
        
        # Update the neural model visualization
        self.update_neural_model(optimized_u)
        
        # Compute error against the current exact solution
        error = np.linalg.norm(optimized_u - self.current_displacements) / np.linalg.norm(self.current_displacements)
        
        return error
    


    def setup_latent_optimizer(self):
        """Set up the optimizer for the latent variable optimization problem"""
        # Create latent variable with gradient tracking AND MATCHING PRECISION
        self.z_opt = torch.zeros(self.num_modes, 
                                requires_grad=True, 
                                device=self.device,
                                dtype=torch.float64)
                                
        # Create LBFGS optimizer with better parameters
        self.optimizer = torch.optim.LBFGS(
            [self.z_opt],
            lr=0.5,  # More conservative learning rate
            max_iter=20,  # More iterations
            max_eval=25,  # More function evaluations
            tolerance_grad=1e-5,  # Less strict tolerance
            tolerance_change=1e-5,  # Less strict tolerance
            history_size=50,
            line_search_fn="strong_wolfe"
        )
        print("Optimizer for latent variables initialized")
        
    def compute_elastic_energy(self, displacements):
        """Compute the elastic energy for given displacements using PabloNeoHookeanEnergy if available"""
        # Initialize energy calculator if needed
        if self.energy_calculator is None:
            self.initialize_energy_calculator()
            
        # If energy calculator initialization failed, fall back to a simpler approach
        if self.energy_calculator is None:
            # Reshape displacements if needed
            if displacements.ndim == 1:
                displacements = displacements.reshape(-1, 3)
                
            # Get rest positions
            rest_pos = np.array(self.MO1.rest_position.value)
            
            # Compute positions with displacements
            positions = rest_pos + displacements
            
            # Create a temporary array to hold positions for SOFA
            temp_positions = self.MO1.position.value.copy()
            
            # Temporarily set positions to compute energy
            self.MO1.position.value = positions.tolist()
            
            # Compute the potential energy (only the elastic component)
            energy = self.fem.getPotentialEnergy()
            
            # Restore original positions
            self.MO1.position.value = temp_positions
            
            return energy
        else:
            # Use the PabloNeoHookeanEnergy to compute the energy
            # Convert to tensor if not already
            if not torch.is_tensor(displacements):
                displacements_tensor = torch.tensor(displacements.flatten(), 
                                                device=self.energy_calculator.device, 
                                                dtype=self.energy_calculator.dtype)
            else:
                displacements_tensor = displacements
                
            # Compute energy using the calculator
            energy = self.energy_calculator(displacements_tensor).item()
            return energy

    # Add method to update neural model visualization
    def update_neural_model(self, U_neural):
        """Update the neural model's positions using neural network prediction"""
        # Get rest positions
        rest_pos = np.array(self.MO3.rest_position.value)
        
        # Get modal displacement
        U_modal = self.reconstruct_from_modal_coords(self.current_modal_coords)
        
        # Apply BOTH modal displacement AND neural correction to get new positions
        # Formula: X + l + y (rest_position + modal_displacement + neural_correction)
        new_pos = rest_pos + U_modal + U_neural
        
        # Update the mechanical object
        self.MO3.position.value = new_pos.tolist()
        
        # Update the visual model directly if available
        if hasattr(self, 'neural_visual') and self.neural_visual is not None:
            visual_obj = self.neural_visual.getObject('OglModel')
            if visual_obj:
                visual_obj.position.value = new_pos.tolist()
        
    def compute_eigenmodes(self):
        """
        Compute eigenmodes from mass and stiffness matrices
        Returns:
            bool: True if computation succeeded, False otherwise
        """
        print("Computing eigenmodes...")
        
        try:
            # Get matrices from SOFA - these are sparse matrices
            K = self.stiffness_matrix
            M = self.mass_matrix
            
            # Check if matrices are valid
            if K is None or M is None:
                print("ERROR: Empty matrices - cannot compute eigenmodes")
                return False
            
            # Convert to proper scipy CSR sparse format if needed
            if not isinstance(K, sparse.csr_matrix):
                print("Converting K matrix to CSR format")
                K = sparse.csr_matrix(K)
            if not isinstance(M, sparse.csr_matrix):
                print("Converting M matrix to CSR format")
                M = sparse.csr_matrix(M)
                
            print(f"Matrix sizes: {K.shape}")
            
            # Make sure matrices are symmetrical (average with transpose)
            K = (K + K.transpose()) / 2
            M = (M + M.transpose()) / 2
            
            # For large systems, we need to use a sparse eigensolver
            print("Computing eigenmodes using sparse solver...")
            sigma = 0.0  # Solve for modes near the origin (rigid + lowest frequency modes)
            
            # Use shift-invert mode for better numerical stability
            eigenvalues, eigenvectors = eigsh(K, k=self.num_modes + 6,  # +6 for rigid body modes
                                            M=M, sigma=sigma, which='LM')
            
            # Sort by eigenvalues (should be sorted already but just to be sure)
            idx = eigenvalues.argsort()
            self.eigenvalues = eigenvalues[idx]
            self.eigenvectors = eigenvectors[:, idx]
            
            # Skip the first 6 modes (rigid body modes)
            start_idx = 6
            end_idx = min(len(self.eigenvalues), start_idx + self.num_modes)
            
            # Keep only the desired modes
            self.eigenvalues = self.eigenvalues[start_idx:end_idx]
            self.eigenvectors = self.eigenvectors[:, start_idx:end_idx]
            self.num_modes = len(self.eigenvalues)  # Update in case we got fewer modes
            
            print(f"Computed {self.num_modes} eigenmodes successfully")
            print(f"Eigenvalues range: [{self.eigenvalues[0]:.2f}, {self.eigenvalues[-1]:.2f}]")
            return True
            
        except Exception as e:
            print(f"ERROR computing eigenmodes: {str(e)}")
            print("Traceback:", traceback.format_exc())
            return False
        
    def project_to_modal_coords(self, U):
        """Project displacement to modal coordinates"""
        # Reshape U to match eigenvectors format
        U_flat = U.flatten()
        
        # Initialize modal coordinates vector
        q = np.zeros(self.num_modes)
        
        # Project using mass orthonormality
        # Keep the mass matrix in sparse format
        M = self.mass_matrix  # Don't convert to numpy array
        
        for i in range(self.num_modes):
            phi_i = self.eigenvectors[:, i]
            
            # Use sparse matrix operations
            M_phi = M.dot(phi_i)  # This returns a numpy array from the sparse dot product
            
            # For mass orthonormality: q_i = (u·M·φ_i)/(φ_i·M·φ_i)
            numerator = np.dot(U_flat, M_phi)
            denominator = np.dot(phi_i, M_phi)
            q[i] = numerator / denominator
        
        return q
        
    
    def reconstruct_from_modal_coords(self, q):
        """Reconstruct displacement from modal coordinates"""
        # Initialize displacement vector
        U_modal = np.zeros(self.eigenvectors.shape[0])
        
        # Sum contributions from each mode
        for i in range(self.num_modes):
            U_modal += q[i] * self.eigenvectors[:, i]
        
        # Reshape back to 3D displacement format (n_nodes, 3)
        U_modal = U_modal.reshape(-1, 3)
        
        return U_modal
    
    def update_modal_model(self, U_modal):
        """Update the modal model's positions using computed displacements"""
        # Get rest positions
        rest_pos = np.array(self.MO2.rest_position.value)
        
        # Apply displacements to get new positions
        new_pos = rest_pos + U_modal
        
        # Update the mechanical object
        self.MO2.position.value = new_pos.tolist()
        
        # Update the visual model directly if available
        if hasattr(self, 'modal_visual') and self.modal_visual is not None:
            visual_obj = self.modal_visual.getObject('OglModel')
            if visual_obj:
                visual_obj.position.value = new_pos.tolist()
        

        

    def _load_neural_model(self):
        """Load the trained neural network model"""
        print(f"Looking for neural model in: {os.path.abspath(self.checkpoint_dir)}")
        print(f"Current working directory: {os.getcwd()}")
        
        # If we've already tried loading and failed, don't try again
        if hasattr(self, 'neural_model_loaded') and self.neural_model_loaded:
            return self.neural_model is not None
        
        try:
            # Check if we have the mass matrix
            if self.mass_matrix is None:
                print("Mass matrix not available yet, deferring neural model loading")
                self.neural_model_loaded = False
                return False
                
            # Look for the best checkpoint file
            best_checkpoint = os.path.join(self.checkpoint_dir, 'best.pt')
            if not os.path.exists(best_checkpoint):
                # Try to find any checkpoint
                checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, '*.pt'))
                if not checkpoint_files:
                    print("No neural model checkpoints found, neural prediction disabled")
                    return False
                best_checkpoint = checkpoint_files[0]
            
            # Load the checkpoint
            print(f"Loading neural model from {best_checkpoint}")
            checkpoint = torch.load(best_checkpoint, map_location=self.device)
            
            # Print checkpoint keys for debugging
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            # Extract model architecture parameters
            # Note: The saved model might have these as parameters or hardcoded
            latent_dim = checkpoint.get('latent_dim', self.num_modes)
            
            # Try to determine output dimension from the last layer weight
            if 'model_state_dict' in checkpoint:
                # Find the last layer's weight
                last_layer_keys = [k for k in checkpoint['model_state_dict'].keys() 
                                if 'weight' in k and 'layers' in k]
                last_layer_keys.sort()
                if last_layer_keys:
                    last_layer_key = last_layer_keys[-1]
                    print(f"Last layer key: {last_layer_key}")
                    last_layer_shape = checkpoint['model_state_dict'][last_layer_key].shape
                    output_dim = last_layer_shape[0]
                    print(f"Detected output dimension: {output_dim}")
                else:
                    # Fallback to matrix size
                    output_dim = self.mass_matrix.shape[0]
                    print(f"Using mass matrix shape for output dim: {output_dim}")
            else:
                # Fallback to matrix size
                output_dim = self.mass_matrix.shape[0]
                print(f"No model_state_dict found, using mass matrix shape: {output_dim}")
            
            # Extract the number of hidden layers by checking the state_dict
            if 'model_state_dict' in checkpoint:
                layer_weights = [k for k in checkpoint['model_state_dict'].keys() 
                            if 'weight' in k and 'layers' in k]
                num_layers = len(layer_weights)
                # Account for input and output layers
                hid_layers = num_layers - 1  # Subtract 1 because we count layers not transitions
                print(f"Detected hidden layers: {hid_layers}")
            else:
                # Use the saved value or a reasonable default
                hid_layers = checkpoint.get('hid_layers', 16)
                print(f"Using saved/default hidden layers: {hid_layers}")
            
            # Get hidden dimension
            if 'model_state_dict' in checkpoint and layer_weights:
                # Use the first hidden layer's shape to determine hidden dimension
                first_hidden_shape = checkpoint['model_state_dict'][layer_weights[0]].shape
                hid_dim = first_hidden_shape[0]
                print(f"Detected hidden dimension: {hid_dim}")
            else:
                # Use the saved value or a reasonable default
                hid_dim = checkpoint.get('hid_dim', 64)
                print(f"Using saved/default hidden dimension: {hid_dim}")
            
            print(f"Creating neural model with: latent_dim={latent_dim}, output_dim={output_dim}, "
                f"hid_layers={hid_layers}, hid_dim={hid_dim}")
            
            # Create the model with the exact same architecture
            self.neural_model = Net(
                latent_dim=latent_dim,
                output_dim=output_dim,
                hid_layers=hid_layers,
                hid_dim=hid_dim
            ).to(self.device).double()
            
            # Load model weights
            self.neural_model.load_state_dict(checkpoint['model_state_dict'])
            self.neural_model.eval()  # Set to evaluation mode
            
            print(f"Neural model loaded successfully: {latent_dim} → {output_dim}")
            self.neural_model_loaded = True
            return True
        
        except Exception as e:
            print(f"Error loading neural model: {str(e)}")
            print(traceback.format_exc())
            self.neural_model = None
            self.neural_model_loaded = True  # Mark as tried so we don't try again
            return False

    def predict_with_neural_model(self, modal_coords):
        """Use the neural model to predict displacements from modal coordinates"""
        if self.neural_model is None:
            if not self._load_neural_model():
                print("Neural model not available, cannot predict")
                return None
        
        # Convert modal coordinates to torch tensor
        q_tensor = torch.tensor(modal_coords, dtype=torch.float64, device=self.device)
        
        # Run prediction
        with torch.no_grad():
            try:
                u_predicted = self.neural_model(q_tensor).cpu().numpy()
                return u_predicted.reshape(-1, 3)  # Reshape to (num_nodes, 3)
            except Exception as e:
                print(f"Error in neural model prediction: {str(e)}")
                print(traceback.format_exc())
                return None
    def compute_displacement(self, mechanical_object):
        # Compute the displacement between the high and low resolution solutions
        U = mechanical_object.position.value.copy() - mechanical_object.rest_position.value.copy()
        return U
    
    def compute_velocity(self, mechanical_object):
        # Compute the velocity of the high resolution solution
        return mechanical_object.velocity.value.copy()
    

    def compute_rest_position(self, mechanical_object):
        # Compute the position of the high resolution solution
        return mechanical_object.rest_position.value.copy()
    
    def compute_position(self, mechanical_object):
        # Compute the position of the high resolution solution
        return mechanical_object.position.value.copy()
    
    def initialize_energy_calculator(self):
        """Initialize the energy calculator using PabloNeoHookeanEnergy from train.py"""
        if self.energy_calculator is not None:
            return  # Already initialized
            
        try:
            # Import the energy calculator class from train.py
            from training.train import PabloNeoHookeanEnergy
            
            # Create a dummy domain structure with the necessary attributes
            class DummyDomain:
                def __init__(self, coordinates, elements):
                    self.geometry = type('obj', (object,), {'x': coordinates})
                    self.topology = type('obj', (object,), {
                        'dim': 3,
                        'index_map': lambda dim: type('obj', (object,), {'size_local': len(elements)}),
                        'connectivity': lambda dim1, dim2: type('obj', (object,), {'links': lambda cell: elements[cell]})
                    })
                    
            # Get coordinates and elements
            coordinates = np.array(self.MO1.rest_position.value)
            
            # Try multiple approaches to get valid tetrahedral elements
            elements = []
            valid_elements = False
            
            # Approach 1: Get elements directly from SOFA's topology container
            try:
                if hasattr(self.surface_topo, 'tetrahedra'):
                    # Direct access to tetrahedra
                    tetrahedra = self.surface_topo.tetrahedra.value
                    if len(tetrahedra) > 0:
                        elements = np.array(tetrahedra)
                        print(f"Extracted {len(elements)} tetrahedra from topology")
                        valid_elements = True
                elif hasattr(self.surface_topo, 'indices'):
                    # For TetrahedronSetTopologyContainer
                    if hasattr(self.surface_topo, 'indices'):
                        topo_elements = np.array(self.surface_topo.indices.value)
                        if topo_elements.size > 0:
                            try:
                                elements = topo_elements.reshape(-1, 4)
                                print(f"Extracted {len(elements)} tetrahedra from indices")
                                valid_elements = True
                            except ValueError:
                                print("Topology indices couldn't be reshaped to tetrahedra")
            except Exception as e:
                print(f"Error extracting tetrahedra from topology: {e}")
            
            # Approach 2: If we still don't have valid elements, try to build tetrahedra from triangles
            if not valid_elements and hasattr(self.surface_topo, 'triangles'):
                try:
                    triangles = np.array(self.surface_topo.triangles.value)
                    if len(triangles) > 0:
                        # Create properly formed tetrahedra by adding a 4th point
                        # This is a simplification but better than dummy elements
                        elements = []
                        for tri in triangles:
                            # Find a point that's not in the triangle to form a tetrahedron
                            p1, p2, p3 = coordinates[tri]
                            center = (p1 + p2 + p3) / 3
                            
                            # Create a point slightly above the triangle
                            normal = np.cross(p2-p1, p3-p1)
                            normal = normal / (np.linalg.norm(normal) + 1e-10)  # Normalize
                            
                            # Find closest point not in triangle
                            distances = np.linalg.norm(coordinates - center, axis=1)
                            sorted_indices = np.argsort(distances)
                            for idx in sorted_indices:
                                if idx not in tri:
                                    elements.append([tri[0], tri[1], tri[2], idx])
                                    break
                        
                        if elements:
                            elements = np.array(elements, dtype=int)
                            print(f"Built {len(elements)} tetrahedra from triangles")
                            valid_elements = True
                except Exception as e:
                    print(f"Error building tetrahedra from triangles: {e}")
            
            # Approach 3: If all else fails, try to create a valid tetrahedral mesh
            if not valid_elements:
                try:
                    # Use a simple Delaunay tetrahedralization
                    from scipy.spatial import Delaunay
                    points = coordinates
                    
                    # Add some jitter to avoid coplanar points
                    jitter = np.random.randn(*points.shape) * 1e-6
                    points_with_jitter = points + jitter
                    
                    # Generate tetrahedral mesh
                    delaunay = Delaunay(points_with_jitter)
                    elements = delaunay.simplices
                    print(f"Created {len(elements)} tetrahedra using Delaunay triangulation")
                    valid_elements = True
                except Exception as e:
                    print(f"Error creating Delaunay tetrahedralization: {e}")
            
            # Final check - if we still don't have valid elements, create an error
            if not valid_elements or len(elements) == 0:
                raise ValueError("Could not create valid tetrahedral elements from the mesh")
                
            print(f"Mesh: {len(elements)} tetrahedron elements, {len(coordinates)} nodes")
            
            # Create dummy domain
            dummy_domain = DummyDomain(coordinates, elements)
            
            # Create energy calculator with lower batch size and more error checking
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            self.energy_calculator = PabloNeoHookeanEnergy(
                domain=dummy_domain,
                degree=1,
                E=self.young_modulus,
                nu=self.poisson_ratio,
                precompute_matrices=False,  # Disable precomputation for safety
                device=device,
                batch_size=1  # Use very small batch size for safety
            )
            print("Energy calculator initialized using PabloNeoHookeanEnergy")
            return True
        except Exception as e:
            print(f"Failed to initialize energy calculator: {str(e)}")
            print(traceback.format_exc())
            return False
    
   
    
    def close(self):
        print("Closing simulation")


def createScene(rootNode, config=None, directory=None, sample=0, key=(0, 0, 0), *args, **kwargs):
    """
    Create SOFA scene with parameters from a YAML config file
    
    Args:
        rootNode: SOFA root node
        config: Dict with configuration parameters from YAML file (or None to use defaults)
        directory: Output directory
        sample: Sample index
        key: Key tuple (x, y, r) for the simulation
    """
    # Handle default config if not provided
    if config is None:
        config = {
            'physics': {'gravity': [0, 0, 0], 'dt': 0.01},
            'material': {'youngs_modulus': 5000, 'poissons_ratio': 0.25, 'density': 10},
            'mesh': {'filename': 'mesh/beam_615.msh'},
            'constraints': {'fixed_box': [-0.01, -0.01, -0.02, 1.01, 0.01, 0.02]}
        }
    
    # Set basic simulation parameters
    rootNode.dt = config['physics'].get('dt', 0.01)
    rootNode.gravity = config['physics'].get('gravity', [0, 0, 0])
    rootNode.name = 'root'

    # Add required plugins
    required_plugins = [
        'MultiThreading',
        'Sofa.Component.Constraint.Projective',
        'Sofa.Component.Engine.Select',
        'Sofa.Component.LinearSolver.Iterative',
        'Sofa.Component.LinearSolver.Direct',
        'Sofa.Component.Mass',
        'Sofa.Component.Mapping.Linear', 
        'Sofa.Component.MechanicalLoad',
        'Sofa.Component.ODESolver.Backward',
        'Sofa.Component.SolidMechanics.FEM.Elastic',
        'Sofa.Component.StateContainer',
        'Sofa.Component.Topology.Container.Dynamic',
        'Sofa.Component.Topology.Container.Grid',
        'Sofa.Component.Visual',
        'SofaMatrix'
    ]
    
    for plugin in required_plugins:
        rootNode.addObject('RequiredPlugin', name=plugin)

    # Add basic scene components
    rootNode.addObject('DefaultAnimationLoop')
    rootNode.addObject('DefaultVisualManagerLoop') 
    rootNode.addObject('VisualStyle', displayFlags="showBehaviorModels showCollisionModels")

    # Get material properties from config
    young_modulus = config['material'].get('youngs_modulus', 5000)
    poisson_ratio = config['material'].get('poissons_ratio', 0.25)
    total_mass = config['material'].get('density', 10)

    # Calculate Lamé parameters
    mu = young_modulus / (2 * (1 + poisson_ratio))
    lam = young_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    print(f"Using mu={mu}, lambda={lam}")
    mu_lam_str = f"{mu} {lam}"

    # Get mesh filename from config
    mesh_filename = config['mesh'].get('filename', 'mesh/beam_615.msh')

    # Create high resolution solution node
    exactSolution = rootNode.addChild('HighResSolution', activated=True)
    exactSolution.addObject('MeshGmshLoader', name='grid', filename=mesh_filename)
    surface_topo = exactSolution.addObject('TetrahedronSetTopologyContainer', name='triangleTopo', src='@grid')
    MO1 = exactSolution.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@grid')
    
    # Add system components
    mass = exactSolution.addObject('MeshMatrixMass', totalMass=total_mass, name="SparseMass", topology="@triangleTopo")
    
    # Get solver parameters from config
    rayleighStiffness = config['physics'].get('rayleigh_stiffness', 0.1)
    rayleighMass = config['physics'].get('rayleigh_mass', 0.1)
    
    solver = exactSolution.addObject('EulerImplicitSolver', name="ODEsolver", 
                                   rayleighStiffness=rayleighStiffness, 
                                   rayleighMass=rayleighMass)
    
    linear_solver = exactSolution.addObject('CGLinearSolver', 
                                          template="CompressedRowSparseMatrixMat3x3d",
                                          iterations=config['physics'].get('solver_iterations', 1000), 
                                          tolerance=config['physics'].get('solver_tolerance', 1e-10), 
                                          threshold=config['physics'].get('solver_threshold', 1e-10), 
                                          warmStart=True)
    
    fem = exactSolution.addObject('TetrahedronHyperelasticityFEMForceField',
                                name="FEM", 
                                materialName="NeoHookean", 
                                ParameterSet=mu_lam_str)
    
    # Get constraint box from config
    fixed_box_coords = config['constraints'].get('fixed_box', [-0.01, -0.01, -0.02, 1.01, 0.01, 0.02])
    fixed_box = exactSolution.addObject('BoxROI', 
                                      name='ROI',
                                      box=fixed_box_coords, 
                                      drawBoxes=True)
    
    exactSolution.addObject('FixedConstraint', indices="@ROI.indices")
    
    # Add a force field box for applying forces - similar to simulation_GNN_plate
    # First, define a default force box (will be updated at runtime)
    force_box_coords = config['constraints'].get('force_box', [4.99, -0.01, -0.01, 5.01, 1.01, 1.01])
    force_box = exactSolution.addObject('BoxROI',
                                    name='ForceROI',
                                    box=force_box_coords, 
                                    drawBoxes=True)
    
    # Create an initial force field with zero force (will be updated at runtime)
    cff = exactSolution.addObject('ConstantForceField',
                                name='ExternalForce',
                                indices="@ForceROI.indices",
                                forces=["0 0 0"],
                                showArrowSize=0.1,
                                showColor="0.2 0.2 0.8 1")
    
    # Rest of the scene creation as before...
    # Add visual model for the exact solution
    exact_visual = exactSolution.addChild("visual")
    exact_visual.addObject('OglModel', src='@../DOFs', color='0 1 0 1')
    exact_visual.addObject('BarycentricMapping', input='@../DOFs', output='@./')


    # ----- CREATE MODAL MODEL (Without solver, driven by modal coordinates) -----
    modalModel = rootNode.addChild('ModalModel', activated=True)
    modalModel.addObject('MeshGmshLoader', name='grid', filename=mesh_filename)
    modal_topo = modalModel.addObject('TetrahedronSetTopologyContainer', name='triangleTopo', src='@grid')
    MO2 = modalModel.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@grid')
    
    # Mass is needed for modal projection
    modal_mass = modalModel.addObject('MeshMatrixMass', totalMass=total_mass, name="SparseMass", topology="@triangleTopo")
    
    # No solvers or force fields for modal model - it will be driven by controller
    
    # Apply same constraints as exact model
    modal_fixed = modalModel.addObject('BoxROI', 
                                   name='ROI',
                                   box=fixed_box_coords, 
                                   drawBoxes=False)  # Hide duplicate boxes
    
    modalModel.addObject('FixedConstraint', indices="@ROI.indices")
    
    # Add visual model for the modal solution
    modal_visual = modalModel.addChild("visual")
    modal_visual.addObject('OglModel', src='@../DOFs', color='1 0 0 1')  # Red color to distinguish
    modal_visual.addObject('BarycentricMapping', input='@../DOFs', output='@./')

    # After creating the modal model, add a neural model
    neuralModel = rootNode.addChild('NeuralModel', activated=True)
    neuralModel.addObject('MeshGmshLoader', name='grid', filename=mesh_filename)
    neural_topo = neuralModel.addObject('TetrahedronSetTopologyContainer', name='triangleTopo', src='@grid')
    MO3 = neuralModel.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@grid')
    
    # Apply same constraints as exact model
    neural_fixed = neuralModel.addObject('BoxROI', 
                                  name='ROI',
                                  box=fixed_box_coords, 
                                  drawBoxes=False)  # Hide duplicate boxes
    
    neuralModel.addObject('FixedConstraint', indices="@ROI.indices")
    
    # Add visual model for neural solution
    neural_visual = neuralModel.addChild("visual")
    neural_visual.addObject('OglModel', src='@../DOFs', color='0 0 1 1')  # Blue color for neural model
    neural_visual.addObject('BarycentricMapping', input='@../DOFs', output='@./')
    
    num_modes = config.get('model', {}).get('latent_dim', 5)
    
    print(f"Using {num_modes} modes for modal analysis")
    
    
    # Pass num_modes to the controller
    controller = AnimationStepController(rootNode, 
                                     mass=mass, 
                                     fem=fem,
                                     linear_solver=linear_solver,
                                     surface_topo=surface_topo,
                                     MO1=MO1,
                                     MO2=MO2,
                                     MO3=MO3,
                                     modal_mass=modal_mass,
                                     fixed_box=fixed_box,
                                     force_box=force_box,
                                     force_field=cff,
                                     exact_visual=exact_visual,
                                     modal_visual=modal_visual,
                                     neural_visual=neural_visual,
                                     directory=directory, 
                                     checkpoint_dir='checkpoints',
                                     num_modes=num_modes,
                                     sample=sample,
                                     key=key, 
                                     young_modulus=young_modulus,
                                     poisson_ratio=poisson_ratio,
                                     density=total_mass,
                                     mesh_filename=mesh_filename,
                                     dt=rootNode.dt,  # Pass time step to controller
                                     **kwargs)
    
    rootNode.addObject(controller)
    return rootNode, controller



if __name__ == "__main__":
    import Sofa.Gui
    from tqdm import tqdm
    import yaml
    import argparse
    
    # Add argument parser
    parser = argparse.ArgumentParser(description='SOFA Matrix Creation')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--gui', action='store_true', help='Enable GUI mode')
    parser.add_argument('--steps', type=int, default=10, help='Number of steps to run in headless mode')
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {args.config}")
    except Exception as e:
        print(f"Error loading config from {args.config}: {str(e)}")
        print("Using default configuration")
        config = None
    
    # Required plugins
    required_plugins = [
        "Sofa.GL.Component.Rendering3D",
        "Sofa.GL.Component.Shader",
        "Sofa.Component.StateContainer",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.IO.Mesh",
        "Sofa.Component.MechanicalLoad",
        "Sofa.Component.Engine.Select",
        "Sofa.Component.SolidMechanics.FEM.Elastic",
        "MultiThreading",
        "SofaMatrix",
        "Sofa.Component.SolidMechanics.FEM.HyperElastic"
    ]

    # Import all required plugins
    for plugin in required_plugins:
        SofaRuntime.importPlugin(plugin)

    # Create simulation directory
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    output_dir = os.path.join('modal_data', timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Get mesh filename from config
    mesh_filename = config.get('mesh', {}).get('filename', 'mesh/beam_615.msh') if config else "mesh/beam_615.msh"

    # Setup and run simulation
    root = Sofa.Core.Node("root")
    rootNode, controller = createScene(
        root,
        config=config,
        directory=timestamp,
        sample=0,
        key=(0, 0, 0)  # default key values
    )

    # Initialize simulation
    Sofa.Simulation.init(root)
    controller.save = True

    if args.gui:
        Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
        Sofa.Gui.GUIManager.createGUI(root, __file__)
        Sofa.Gui.GUIManager.SetDimension(800, 600)
        Sofa.Gui.GUIManager.MainLoop(root)
        Sofa.Gui.GUIManager.closeGUI()
    else:
        for _ in tqdm(range(args.steps), desc="Simulation progress"):
            Sofa.Simulation.animate(root, root.dt.value)

    controller.close()