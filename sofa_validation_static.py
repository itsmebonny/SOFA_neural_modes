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
        """Apply random force to the simulation using the approach from GNN_plate"""
        self.bad_sample = False
        
        self.hardreset()
        
        # Reset velocities to zero
        if hasattr(self.MO1, "velocity"):
            self.MO1.velocity.value = [[0, 0, 0]] * len(self.MO1.rest_position.value)
        
        # Reset acceleration if available
        if hasattr(self.MO1, "acceleration"):
            self.MO1.acceleration.value = [[0, 0, 0]] * len(self.MO1.rest_position.value)
        
        # Reset force accumulators in the solver
        solver = self.exactSolution.getObject('ODEsolver')
        if solver is not None:
            solver.reset()
        
        # Clear any dynamic state in your linear solver
        if hasattr(self.linear_solver, 'cleanup'):
            self.linear_solver.cleanup()
        
        # Get mesh extents for reference
        positions = np.array(self.MO1.rest_position.value)
        x_min, y_min, z_min = np.min(positions, axis=0)
        x_max, y_max, z_max = np.max(positions, axis=0)
        print(f"Mesh extents: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}], z=[{z_min}, {z_max}]")
        
        # Generate random direction for force
        self.z = np.random.uniform(-1, 1)
        self.phi = np.random.uniform(0, 2*np.pi)
        self.versor = np.array([np.sqrt(1 - self.z**2) * np.cos(self.phi), 
                            np.sqrt(1 - self.z**2) * np.sin(self.phi), 
                            self.z])
                            
        # Scale force based on material stiffness
        self.magnitude = np.random.uniform(0.5, 1)
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
        try:
            if self.force_field is not None:
                self.exactSolution.removeObject(self.force_field)
                print("Removed old force field")
        except Exception as e:
            print(f"Error removing force field: {str(e)}")
        
        # Create a new force field with object reference like in GNN_plate
        force_str = f"{self.externalForce[0]} {self.externalForce[1]} {self.externalForce[2]}"
        try:
            self.force_field = self.exactSolution.addObject('ConstantForceField', 
                                                    name='ExternalForce', 
                                                    indices=indices_forces,
                                                    totalForce=self.externalForce,
                                                    showArrowSize=0.1,
                                                    showColor="0.2 0.2 0.8 1")  # Reference to mechanical object
            self.force_field.init()
            print("Created new force field")
        except Exception as e:
            print(f"Error creating force field: {str(e)}")
        
        self.start_time = process_time()
    def onAnimateEndEvent(self, event):
        
        
        # Get matrices from SOFA
        self.mass_matrix = self.mass.assembleMMatrix()
        self.stiffness_matrix = self.fem.assembleKMatrix()
        
        # Try loading neural model if not already loaded
        if self.neural_model is None and not hasattr(self, 'neural_model_loaded'):
            self._load_neural_model()
            
        # Solve eigenvalue problem if not already done
        if self.eigenvectors is None or self.eigenvalues is None:
            success = self.compute_eigenmodes()
            if not success:
                print("ERROR: Failed to compute eigenmodes - skipping modal projection")
                return  # Exit early if we couldn't compute eigenmodes
        
        # Get current displacement from MO1 (exact solution)
        U_exact = self.compute_displacement(self.MO1)
        
        # Project to modal coordinates
        q = self.project_to_modal_coords(U_exact)
        
        # Store for later use in update_neural_model
        self.current_modal_coords = q
        
        # 1. Reconstruct displacement using direct modal coordinates
        U_modal = self.reconstruct_from_modal_coords(q)
        
        # 2. Predict displacement using neural network if available
        U_neural = self.predict_with_neural_model(q)
        
        # Update mechanical objects
        self.update_modal_model(U_modal)  # Update standard modal model
        
        # Update neural model if available
        if U_neural is not None:
            # Check if we have a second mechanical object for neural model
            if hasattr(self, 'MO3') and self.MO3 is not None:
                self.update_neural_model(U_neural)
            else:
                print("No separate MO for neural model, skipping neural visualization")
        
        # Compute errors between exact and approximated solutions
        modal_error = np.linalg.norm(U_exact - U_modal) / np.linalg.norm(U_exact)
        print(f"Modal reconstruction error: {modal_error:.6f}")
        
        if U_neural is not None:
            # Neural model is a correction to modal model
            full_neural_solution = U_modal + U_neural
            neural_error = np.linalg.norm(U_exact - full_neural_solution) / np.linalg.norm(U_exact)
            print(f"Neural model error: {neural_error:.6f}")
                
        # Save matrices and metadata as before
        matrices_dir = 'matrices'
        os.makedirs(matrices_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # # Save matrices to standard location
        # np.save(f'{matrices_dir}/mass_matrix_{timestamp}.npy', self.mass_matrix)
        # np.save(f'{matrices_dir}/stiffness_matrix_{timestamp}.npy', self.stiffness_matrix)
        
        # # Save eigenvectors and eigenvalues
        # np.save(f'{matrices_dir}/eigenvectors_{timestamp}.npy', self.eigenvectors)
        # np.save(f'{matrices_dir}/eigenvalues_{timestamp}.npy', self.eigenvalues)
        
        # # Save modal coordinates
        # np.save(f'{matrices_dir}/modal_coordinates_{timestamp}.npy', q)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'mesh_file': self.mesh_filename,
            'young_modulus': self.young_modulus,
            'poisson_ratio': self.poisson_ratio,
            'density': self.density,
            'size': self.mass_matrix.shape[0],
            'num_modes': self.num_modes,
            'modal_error': float(modal_error),
            'neural_error': float(neural_error) if U_neural is not None else None
        }
        
        with open(f'{matrices_dir}/metadata_{timestamp}.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Matrices and data saved to {matrices_dir} with timestamp {timestamp}")
        self.end_time = process_time()

        print(f"Time taken: {self.end_time - self.start_time:.2f} seconds")


    def hardreset(self):
        """Perform a complete hard reset of the simulation state"""
        print("Performing hard reset of simulation...")
        
        # Use SOFA's built-in reset function on the root node
        Sofa.Simulation.reset(self.root)
        
        # Reset all mechanical objects explicitly (probably redundant but safe)
        self.MO1.position.value = self.MO1.rest_position.value
        self.MO2.position.value = self.MO2.rest_position.value
        self.MO3.position.value = self.MO3.rest_position.value
        
        # Clear cached matrices and eigenmodes to force recomputation
        self.mass_matrix = None 
        self.stiffness_matrix = None
        self.eigenvectors = None
        self.eigenvalues = None
        
        # Reset force field
        self.force_field = None
        
        # Clear any neural model predictions
        self.current_modal_coords = None
        
        print("Simulation hard reset complete")

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
            print(f"Eigenvalues range: [{self.eigenvalues[0]:.5f}, {self.eigenvalues[-1]:.5f}]")
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
        # If we've already tried loading and failed, don't try again
        if hasattr(self, 'neural_model_loaded') and self.neural_model_loaded:
            return self.neural_model is not None
        
        self.neural_model_loaded = True  # Mark that we've tried loading
        
        try:
            # Check if we have the mass matrix
            if self.mass_matrix is None:
                print("Mass matrix not available yet, deferring neural model loading")
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
            
            # Create model with same dimensions
            latent_dim = checkpoint.get('latent_dim', self.num_modes)
            output_dim = self.mass_matrix.shape[0]
            
            # Create the model
            self.neural_model = Net(
                latent_dim=latent_dim,
                output_dim=output_dim,
                hid_layers=checkpoint.get('hid_layers', 5),
                hid_dim=checkpoint.get('hid_dim', 64)
            ).to(self.device).double()
            
            # Load model weights
            self.neural_model.load_state_dict(checkpoint['model_state_dict'])
            self.neural_model.eval()  # Set to evaluation mode
            
            print(f"Neural model loaded successfully: {latent_dim} → {output_dim}")
            return True
        
        except Exception as e:
            print(f"Error loading neural model: {str(e)}")
            print(traceback.format_exc())
            self.neural_model = None
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
    
    solver = exactSolution.addObject('StaticSolver', name="ODEsolver", newton_iterations="30", printLog=False)
    
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
    
    num_modes = config.get('modal_analysis', {}).get('num_modes', 
               config.get('model', {}).get('latent_dim', 5))
    
    print(f"Using {num_modes} modes for modal analysis")
    
    
    # Pass num_modes to the controller
    controller = AnimationStepController(rootNode, 
                                     mass=mass, 
                                     fem=fem,
                                     linear_solver=linear_solver,
                                     surface_topo=surface_topo,
                                     MO1=MO1,  # Exact solution MO
                                     MO2=MO2,  # Modal model MO
                                     MO3=MO3,  # Neural model MO 
                                     modal_mass=modal_mass,
                                     fixed_box=fixed_box,
                                     force_box=force_box,
                                     force_field=cff,
                                     exact_visual=exact_visual,
                                     modal_visual=modal_visual,
                                     neural_visual=neural_visual,
                                     directory=directory, 
                                     checkpoint_dir='checkpoints',
                                     num_modes=num_modes,  # Pass the number of modes
                                     sample=sample,
                                     key=key, 
                                     young_modulus=young_modulus,
                                     poisson_ratio=poisson_ratio,
                                     density=total_mass,
                                     mesh_filename=mesh_filename,
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