import Sofa
import SofaRuntime
import numpy as np 
import os
from Sofa import SofaDeformable
from time import process_time, time
import datetime
from sklearn.preprocessing import MinMaxScaler
from training.train_sofa import Routine, load_config
# add network path to the python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../network'))

import json
import torch

import os
import json
import datetime
import numpy as np

import glob
import traceback
from scipy import sparse
from scipy.sparse.linalg import eigsh

import matplotlib.pyplot as plt



class AnimationStepController(Sofa.Core.Controller):
    def __init__(self, node, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.fem = kwargs.get('fem')
        self.linear_solver = kwargs.get('linear_solver')
        self.surface_topo = kwargs.get('surface_topo')
        self.MO1 = kwargs.get('MO1')
        self.fixed_box = kwargs.get('fixed_box')
        self.exactSolution = kwargs.get('exactSolution')
        self.cff = kwargs.get('cff') # Keep this if createScene adds an initial one




        self.key = kwargs.get('key')
        self.iteration = kwargs.get("sample")
        self.start_time = 0
        self.root = node
        self.save = True

        # --- Add list to store results per substep ---
        # Expanded tuple: (ForceMag, RealE, PredE, LinModesE, SOFALinE,
        #                  L2Err_Pred_vs_Real, RMSE_Pred_vs_Real, MSE_Pred_vs_Real,
        #                  L2Err_Lin_vs_Real,  RMSE_Lin_vs_Real,  MSE_Lin_vs_Real,
        #                  L2Err_Lin_vs_SOFALin, RMSE_Lin_vs_SOFALin, MSE_Lin_vs_SOFALin)
        self.substep_results = []
        self.all_z_coords = []    # Stores numpy arrays of modal coordinates (z)

        # --- End Add list ---

        self.num_substeps = kwargs.get('num_substeps', 1)
        self.current_substep = 0
        self.current_main_step = 0
        self.max_main_steps = kwargs.get('max_main_steps', 20)

        # --- Define Fixed Force Target Magnitude ---
        # self.target_force_direction = np.array([-1.0, 0.0, 0.0]) # REMOVED
        self.target_force_magnitude = 50
        # self.target_force_vector = self.target_force_direction * self.target_force_magnitude # REMOVED
        self.current_main_step_direction = np.zeros(3) # Initialize direction
        print(f"Target Max Force Magnitude: {self.target_force_magnitude}")
        # --- End Fixed Force ---
        
        self.directory = kwargs.get('directory')
        # Add material properties to controller
        self.young_modulus = kwargs.get('young_modulus', 5000)
        self.poisson_ratio = kwargs.get('poisson_ratio', 0.25)
        self.density = kwargs.get('density', 10)
        self.volume = kwargs.get('volume', 1)
        self.total_mass = kwargs.get('total_mass', 10)
        self.mesh_filename = kwargs.get('mesh_filename', 'unknown')
        print(f"Using directory: {self.directory}")
        print(f"Material properties: E={self.young_modulus}, nu={self.poisson_ratio}, rho={self.density}")
        
        # Add eigenmodes visualization parameters
        self.show_modes = kwargs.get('show_modes', True)
        self.current_mode_index = -1  # Start with -1 to compute matrices first
        self.mode_animation_step = 0
        self.mode_animation_steps = kwargs.get('steps_per_mode', 100)  # Steps to animate each mode
        self.mode_scale = kwargs.get('mode_scale', 50.0)  # Scaling factor for eigenmodes
        self.num_modes_to_show = kwargs.get('num_modes_to_show', 5)
        self.modes_computed = False
        self.eigenvectors = None
        self.eigenvalues = None
        self.original_positions = None
        self.transition_steps = 20  # Steps to transition between modes
        self.pause_steps = 30  # Steps to pause at maximum amplitude

        # --- Add lists for Deformation Gradient Differences ---
        self.grad_diff_lin_modes_list = []
        self.grad_diff_nn_pred_list = []
        self.grad_diff_sofa_linear_list = []
        # --- End lists for Deformation Gradient Differences ---

      


    def onSimulationInitDoneEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the scene graph initialisation.
        """
        self.inputs = []
        self.outputs = []
        cfg = load_config(args.config)
        print(f"Loaded config from {args.config}")

        # --- Instantiate Routine ---
        try:
            print("Instantiating Routine...")
            # Pass the loaded config dictionary to Routine
            self.routine = Routine(cfg)
            print("Routine instantiated successfully.")
        except Exception as e:
            print(f"Error instantiating Routine: {e}")
            traceback.print_exc() # Print detailed traceback
            sys.exit(1)
        # --- End Routine Instantiation ---

        checkpoint_dir_rel = cfg.get('training', {}).get('checkpoint_dir', 'checkpoints')

        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..')) # Go one level up from sofa_scripts
        checkpoint_dir_abs = os.path.join(project_root, checkpoint_dir_rel) # Join with project root

        # Define the specific checkpoint file name (e.g., 'best_sofa.pt')
        checkpoint_filename = 'best_sofa.pt' # Or read from config if specified differently
        best_checkpoint_path = os.path.join(checkpoint_dir_abs, checkpoint_filename)

        print(f"Attempting to load best checkpoint from: {best_checkpoint_path}")
        if os.path.exists(best_checkpoint_path):
            try:
                self.routine.load_checkpoint(best_checkpoint_path)
                print("Successfully loaded best model checkpoint.")
                # Ensure model is in evaluation mode after loading
                self.routine.model.eval()
            except Exception as e:
                print(f"Error loading checkpoint {best_checkpoint_path}: {e}")
                print("Proceeding without loaded model weights.")
        else:
            print(f"Warning: Best checkpoint file not found at {best_checkpoint_path}. Using initialized model.")
        # --- End Checkpoint Loading ---

        # Extract necessary data from Routine instance
        self.linear_modes = self.routine.linear_modes # This should be a torch tensor
        self.original_positions = np.copy(self.MO1.position.value) # Store original positions

        # --- Prepare for Saving (if enabled) ---
        # Use the directory name passed during scene creation or default
        self.directory = self.root.directory_name.value if hasattr(self.root, 'directory_name') else datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_base_dir = 'modal_data' # Or read from config
        if self.save:
            if not os.path.exists(output_base_dir):
                os.mkdir(output_base_dir)
            self.output_subdir = os.path.join(output_base_dir, self.directory)
            if not os.path.exists(self.output_subdir):
                os.makedirs(self.output_subdir)
            print(f"Data saving enabled. Output directory: {self.output_subdir}")
        # --- End Saving Prep ---

        self.sampled = False # Reset sampling flag

        surface = self.surface_topo
        self.idx_surface = surface.triangles.value.reshape(-1)
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Store the original positions for mode animation
        if self.show_modes and self.MO1:
            self.original_positions = np.copy(self.MO1.position.value)
            print(f"Stored original positions with shape {self.original_positions.shape}")


    def onAnimateBeginEvent(self, event):
        """
        Called by SOFA's animation loop before each physics step.
        Applies incremental force based on substep, with a random direction per main step.
        """
        # --- Check if starting a new MAIN step ---
        if self.current_substep == 0:
            # Reset positions for all models
            rest_pos = self.MO1.rest_position.value
            self.MO1.position.value = rest_pos

            print(f"\n--- Starting Main Step {self.current_main_step + 1} ---")

            # --- Generate a new random direction for this main step ---
            random_vec = np.random.randn(3) # Generate random vector from normal distribution
            norm = np.linalg.norm(random_vec)
            if norm < 1e-9: # Avoid division by zero if vector is near zero
                self.current_main_step_direction = np.array([1.0, 0.0, 0.0]) # Default direction
            else:
                self.current_main_step_direction = random_vec / norm # Normalize to get unit vector
            print(f"  New Random Force Direction: {self.current_main_step_direction}")
            # --- End Generate Random Direction ---

        # --- Calculate and apply force for the CURRENT substep ---
        # Force ramps from F/N to F (where F is the target magnitude * direction)
        # Ensure substep index is within [0, num_substeps-1] for calculation
        substep_fraction = (self.current_substep % self.num_substeps + 1) / self.num_substeps
        current_force_magnitude = self.target_force_magnitude * substep_fraction # Store magnitude for analysis
        incremental_force = self.current_main_step_direction * current_force_magnitude # Apply magnitude to current direction

        current_substep_in_main_step = (self.current_substep % self.num_substeps) + 1
        print(f"  Main Step {self.current_main_step + 1}, Substep {current_substep_in_main_step}/{self.num_substeps}: Applying force = {incremental_force}")


        if self.cff is not None:
            try:
                self.exactSolution.removeObject(self.cff)
            except Exception as e:
                print(f"Warning: Error removing CFF (Exact): {e}")
            finally:
                 self.cff = None # Clear the reference

        # --- Create and add new CFFs ---
        try:
            # Exact Solution
            force_roi_exact = self.exactSolution.getObject('ForceROI')
            if force_roi_exact is None: raise ValueError("ForceROI (Exact) not found.")
            self.cff = self.exactSolution.addObject('ConstantForceField',
                               name="CFF_Exact_Step",
                               indices="@ForceROI.indices",
                               totalForce=incremental_force.tolist(),
                               showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
            self.cff.init()

            # --- Initialize Parent Nodes (if needed after adding objects) ---
            # self.exactSolution.init() # May not be needed if cff.init() is sufficient
            # self.linearSolution.init() # May not be needed if cff_linear.init() is sufficient
            # --- End Initialization ---

            # Store the magnitude applied in this step for analysis in onAnimateEndEvent
            self.last_applied_force_magnitude = current_force_magnitude
            # print(f"  Substep {(self.current_substep % self.num_substeps) + 1}/{self.num_substeps}: Applied force mag = {current_force_magnitude:.4f}")

        except Exception as e:
            print(f"ERROR: Failed to create/add/init ConstantForceField(s): {e}")
            traceback.print_exc()
            self.cff = None
            self.cff_linear = None
            if self.root: self.root.animate = False

        self.start_time = process_time()



    def onAnimateEndEvent(self, event):
        # --- Get Real SOFA (Hyperelastic) State (MO1) ---
        pos_real_all_dofs = self.MO1.position.value
        rest_pos_all_dofs = self.MO1.rest_position.value
        u_real_all_dofs = pos_real_all_dofs - rest_pos_all_dofs
        
        energy_real = np.nan
        z_real = np.zeros(self.linear_modes.shape[1] if self.linear_modes is not None else 0) # Default z_real

        try:
            energy_real = self.computeInternalEnergy(u_real_all_dofs)
        except Exception as e:
            print(f"Warning: Could not compute real energy: {e}")

        try:
            z_real = self.computeModalCoordinates(u_real_all_dofs)
            self.all_z_coords.append(np.copy(z_real)) # Store a copy
        except Exception as e:
            print(f"Warning: Could not compute modal coordinates: {e}")
            # z_real remains default, self.all_z_coords does not get an entry for this step if error

        # --- Save z_real and energy_real ---
        if self.save: # Only save if saving is enabled
            try:
                num_modes = self.linear_modes.shape[1] if self.linear_modes is not None else 0
                save_folder = os.path.join("z_dataset", f"{num_modes}_modes")
                os.makedirs(save_folder, exist_ok=True)
                
                # Determine the next index for saving files
                existing_data_files = glob.glob(os.path.join(save_folder, "data_*.npz"))
                next_index = len(existing_data_files)
                
                # --- Save modal data (z and energy) ---
                data_filename = os.path.join(save_folder, f"data_{next_index:04d}.npz")
                np.savez(data_filename, 
                    z=z_real, 
                    energy=energy_real)
                # print(f"Saved modal data to {data_filename}")

                # --- Save displacement data ---
                displacement_filename = os.path.join(save_folder, f"displacement_{next_index:04d}.npz")
                np.savez(displacement_filename,
                    u_flat=u_real_all_dofs.flatten())
                # print(f"Saved displacement data to {displacement_filename}")

            except Exception as e:
                print(f"Error saving data: {e}")
                traceback.print_exc()


        _substep_idx_this_main_step = self.current_substep # This is the one that just ran (0 to N-1)
        _main_step_idx_this_main_step = self.current_main_step # This is the one that just ran (0 to M-1)

        # Now, determine the *next* substep and main_step
        next_substep_idx = _substep_idx_this_main_step + 1
        next_main_step_idx = _main_step_idx_this_main_step

        if next_substep_idx >= self.num_substeps:
            next_substep_idx = 0
            next_main_step_idx = _main_step_idx_this_main_step + 1
        
        # The stopping condition:
        if next_main_step_idx >= self.max_main_steps:
            print(f"All {self.max_main_steps} main steps have been processed. Stopping simulation.")
            if self.root: self.root.animate = False
        
        self.current_substep += 1 # This is the global substep counter for the simulation.
        
        current_main_step_effectively_processed = (self.current_substep -1) // self.num_substeps

        if (current_main_step_effectively_processed + 1) >= self.max_main_steps and \
           ((self.current_substep -1) % self.num_substeps == self.num_substeps -1) :
            print(f"All {self.max_main_steps} main steps completed after global substep {self.current_substep-1}. Stopping.")
            if self.root: self.root.animate = False
        self.end_time = process_time()

    def computeModalCoordinates(self, displacement):
        """
        Compute modal coordinates from displacement using the linear modes
        and the mass matrix loaded by the Routine.
        Assumes modes are mass-orthonormalized: z = Modes^T * M * displacement.

        Args:
            displacement: Displacement vector (NumPy array, shape (num_dofs,) or (num_nodes, 3)).

        Returns:
            Modal coordinates as a 1D numpy array (shape (num_modes,)).
        """
        # Ensure displacement is a flattened 1D NumPy array
        if displacement.ndim > 1:
            displacement_flat = displacement.flatten()
        else:
            displacement_flat = displacement

        # --- Get linear_modes as NumPy array ---
        if isinstance(self.routine.linear_modes, torch.Tensor):
            linear_modes_np = self.routine.linear_modes.cpu().numpy()
        else:
            linear_modes_np = self.routine.linear_modes # Assuming it might be NumPy already
        # --- End conversion ---

        # --- Get Mass Matrix from Routine ---
        M_sparse = self.routine.M
        # ---

        # --- Check if Mass Matrix is available ---
        if M_sparse is None or not isinstance(M_sparse, sparse.spmatrix):
            print("Warning: Mass matrix not available from Routine or not sparse. Using simple projection (Modes^T * u).")
            # --- Compute modal coordinates: z = Modes^T * displacement ---
            try:
                modal_coordinates = np.dot(linear_modes_np.T, displacement_flat)
            except ValueError as e:
                print(f"Error during simple modal coordinate calculation: {e}")
                print(f"Shape mismatch? Modes.T: {linear_modes_np.T.shape}, Displacement: {displacement_flat.shape}")
                return np.zeros(linear_modes_np.shape[1])

        else:
            # --- Compute modal coordinates: z = Modes^T * M * displacement ---
            # print("Using mass matrix from Routine for projection: z = Modes^T * M * u") # DEBUG
            try:
                # Ensure shapes are compatible
                # linear_modes_np.T: (num_modes, num_dofs)
                # M_sparse:           (num_dofs, num_dofs) (sparse)
                # displacement_flat:  (num_dofs,)
                # Result:             (num_modes,)

                # Perform sparse multiplication: (Modes.T @ M) @ u
                # Note: linear_modes_np.T is dense, M_sparse is sparse. Result of product is dense.
                modes_t_m = linear_modes_np.T @ M_sparse # Shape: (num_modes, num_dofs) (dense)
                modal_coordinates = modes_t_m @ displacement_flat # Shape: (num_modes,) (dense)

            except ValueError as e:
                print(f"Error during mass matrix modal coordinate calculation: {e}")
                print(f"Shapes - Modes.T: {linear_modes_np.T.shape}, M: {M_sparse.shape}, u: {displacement_flat.shape}")
                return np.zeros(linear_modes_np.shape[1])
            except Exception as e: # Catch other potential errors (e.g., from sparse multiplication)
                print(f"Unexpected error during mass matrix projection: {e}")
                traceback.print_exc()
                return np.zeros(linear_modes_np.shape[1])
        # --- End computation ---

        return modal_coordinates
    
    def computeInternalEnergy(self, displacement):
        """
        Compute internal energy of the system using the Routine's energy calculator.

        Args:
            displacement: Displacement vector of the system (NumPy array).

        Returns:
            Internal energy as a float.
        """
        energy_calculator = self.routine.energy_calculator
        device = self.routine.device # Get the device from the routine instance

        # --- Convert NumPy array to PyTorch tensor ---
        # Ensure correct dtype (likely float64 based on train_sofa.py) and device
        displacement_tensor = torch.tensor(displacement, dtype=torch.float64, device=device)

        # Add batch dimension if the energy calculator expects it (common practice)
        if displacement_tensor.dim() == 2:
             displacement_tensor = displacement_tensor.unsqueeze(0)
        # --- End conversion ---

        # Compute internal energy using the tensor
        # Use torch.no_grad() if gradients are not needed here
        with torch.no_grad():
            internal_energy = energy_calculator(displacement_tensor)

        # If a batch dimension was added, remove it from the result if necessary
        if internal_energy.dim() > 0 and internal_energy.shape[0] == 1:
             internal_energy = internal_energy.squeeze(0)

        return internal_energy.item()
    
    
    

    def close(self):
        print("\n--- Simulation Finished ---")



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
    rootNode.bbox = "-10 -2 -2 10 2 2"

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
    density = config['material'].get('density', 10)
    volume = config['material'].get('volume', 1)
    num_modes_to_show = config['model'].get('latent_dim', 5)
    total_mass = density * volume
    print(f"Using E={young_modulus}, nu={poisson_ratio}, rho={density}, V={volume}, M={total_mass}")

    # Calculate Lamé parameters
    mu = young_modulus / (2 * (1 + poisson_ratio))
    lam = young_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    print(f"Using mu={mu}, lambda={lam}")
    mu_lam_str = f"{mu} {lam}"

    # Get mesh filename from config
    mesh_filename = config['mesh'].get('filename', 'mesh/beam_732.msh')

    # Create high resolution solution node
    exactSolution = rootNode.addChild('HighResSolution2D', activated=True)
    exactSolution.addObject('MeshGmshLoader', name='grid', filename=mesh_filename)
    surface_topo = exactSolution.addObject('TetrahedronSetTopologyContainer', name='triangleTopo', src='@grid')
    MO1 = exactSolution.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@grid')
    
    # Add system components
    # mass = exactSolution.addObject('MeshMatrixMass', totalMass=total_mass, name="SparseMass", topology="@triangleTopo")
    
    # Get solver parameters from config
    rayleighStiffness = config['physics'].get('rayleigh_stiffness', 0.1)
    rayleighMass = config['physics'].get('rayleigh_mass', 0.1)
    
    solver = exactSolution.addObject('StaticSolver', name="ODEsolver", 
                                   newton_iterations=20,
                                   printLog=True)
    
    linear_solver = exactSolution.addObject('CGLinearSolver', 
                                          template="CompressedRowSparseMatrixMat3x3d",
                                          iterations=config['physics'].get('solver_iterations', 1000), 
                                          tolerance=config['physics'].get('solver_tolerance', 1e-6), 
                                          threshold=config['physics'].get('solver_threshold', 1e-6), 
                                          warmStart=True)
    
    fem = exactSolution.addObject('TetrahedronHyperelasticityFEMForceField',
                                name="FEM", 
                                materialName="NeoHookean", 
                                ParameterSet=mu_lam_str)
    # fem = exactSolution.addObject('TetrahedronFEMForceField', name="FEM", youngModulus=5000, poissonRatio=0.4, method="large", updateStiffnessMatrix="false")

    # Add a second beam with linear elasticity to check the difference
    
    # Get constraint box from config
    fixed_box_coords = config['constraints'].get('fixed_box', [-0.01, -0.01, -0.02, 1.01, 0.01, 0.02])
    fixed_box = exactSolution.addObject('BoxROI', 
                                      name='ROI',
                                      box=" ".join(str(x) for x in fixed_box_coords), 
                                      drawBoxes=True)
    exactSolution.addObject('FixedConstraint', indices="@ROI.indices")

    force_box_coords = config['constraints'].get('force_box', [0.01, -0.01, -0.02, 10.1, 1.01, 1.02])
    force_box = exactSolution.addObject('BoxROI',
                                        name='ForceROI',
                                        box=" ".join(str(x) for x in force_box_coords), 
                                        drawBoxes=True)
    cff = exactSolution.addObject('ConstantForceField', indices="@ForceROI.indices", totalForce=[0, 0, 0], showArrowSize=0.1, showColor="0.2 0.2 0.8 1")

    

    
    # Add visual model
    visual = exactSolution.addChild("visual")
    visual.addObject('OglModel', src='@../DOFs', color='0 1 0 1')
    visual.addObject('BarycentricMapping', input='@../DOFs', output='@./')


    # Create and add controller with all components
    controller = AnimationStepController(rootNode,
                                        exactSolution=exactSolution,
                                        fem=fem, # Hyperelastic FEM
                                        linear_solver=linear_solver,
                                        surface_topo=surface_topo,
                                        MO1=MO1, # Real SOFA solution
                                        fixed_box=fixed_box,
                                        directory=directory,
                                        sample=sample,
                                        key=key,
                                        young_modulus=young_modulus,
                                        poisson_ratio=poisson_ratio,
                                        density=density,
                                        volume=volume,
                                        total_mass=total_mass,
                                        mesh_filename=mesh_filename,
                                        num_modes_to_show=num_modes_to_show,
                                        # cff=cff, # REMOVED - Controller manages CFFs
                                        **kwargs)
    rootNode.addObject(controller)

    return rootNode, controller


if __name__ == "__main__":
    import Sofa.Gui
    from tqdm import tqdm
    import yaml
    import argparse
    import traceback
    import time # Import time for headless loop

    # Add argument parser
    parser = argparse.ArgumentParser(description='SOFA Validation with Neural Modes and Substeps')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--gui', action='store_true', help='Enable GUI mode')
    parser.add_argument('--steps', type=int, default=None, help='Number of MAIN steps to run (overrides config)') # Renamed from substeps
    parser.add_argument('--num-substeps', type=int, default=None, help='Number of substeps per main step (overrides config)')

    args = parser.parse_args()

    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {args.config}")
    except Exception as e:
        print(f"Error loading config from {args.config}: {str(e)}")
        print("Using default configuration")
        config = None # Or define a minimal default config dict here

    # Determine number of main steps and substeps
    # Command line args override config, otherwise use config default or fallback default
    max_main_steps = args.steps if args.steps is not None else config.get('simulation', {}).get('steps', 20)
    num_substeps = args.num_substeps if args.num_substeps is not None else config.get('physics', {}).get('num_substeps', 1)

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

    # Setup and run simulation
    root = Sofa.Core.Node("root")
    # Pass determined steps to createScene kwargs for the controller
    rootNode, controller = createScene(
        root,
        config=config,
        directory=timestamp,
        sample=0,
        key=(0, 0, 0),
        num_substeps=num_substeps,      # Pass determined value
        max_main_steps=max_main_steps   # Pass determined value
    )

    # Initialize simulation
    Sofa.Simulation.init(root)
    controller.save = True # Ensure saving is enabled if needed for plots

    # --- Run Simulation ---
    if args.gui:
        print(f"Starting GUI mode. Substeps ({num_substeps}) managed by controller.")
        Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
        Sofa.Gui.GUIManager.createGUI(root, __file__)
        Sofa.Gui.GUIManager.SetDimension(1000, 800)
        Sofa.Gui.GUIManager.MainLoop(root) # Controller handles substeps internally
        Sofa.Gui.GUIManager.closeGUI()
    else:
        print(f"Starting headless mode for {max_main_steps} main steps with {num_substeps} substeps each.")
        # Use root.animate flag controlled by the controller to stop
        root.animate = True
        step_count = 0
        # We need a loop that runs until the controller stops it or a max iteration limit
        max_total_iterations = max_main_steps * num_substeps * 1.1 # Safety limit
        pbar = tqdm(total=max_main_steps, desc="Main Steps Progress")
        last_main_step = -1

        while root.animate.value and step_count < max_total_iterations:
            Sofa.Simulation.animate(root, root.dt.value)
            step_count += 1
            # Update progress bar when a main step completes
            if controller.current_main_step > last_main_step:
                 pbar.update(controller.current_main_step - last_main_step)
                 last_main_step = controller.current_main_step
            # Optional small sleep to prevent 100% CPU if simulation is very fast
            # time.sleep(0.001)

        pbar.close()
        if step_count >= max_total_iterations:
            print("Warning: Reached maximum total iterations safety limit.")

    # Close is called regardless of GUI/headless mode
    controller.close()