import Sofa
import SofaRuntime
import numpy as np 
import os
from Sofa import SofaDeformable
from time import process_time, time
import datetime
from sklearn.preprocessing import MinMaxScaler
from training.train_sofa import Routine, load_config # Assuming train_sofa.py is in training/
# add network path to the python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../network')) # If network.py is in ../network
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Add project root for training.train_sofa

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
        self.cff_exact = None # Will be set later
        # self.cff = kwargs.get('cff') # Controller will manage CFF creation

 

        self.key = kwargs.get('key')
        self.iteration = kwargs.get("sample")
        self.start_time = 0
        self.root = node
        self.save = True

        self.substep_results = []
        self.all_z_coords = []    # Stores numpy arrays of actual modal coordinates (z_actual) from SOFA solution
        self.z0_vals = []
        self.scaled_z0_vals = []
        self.actual_z0_vals = []        
        self.num_substeps = kwargs.get('num_substeps', 1)
        self.current_substep = 0
        self.current_main_step = 0
        self.max_main_steps = kwargs.get('max_main_steps', 20)

        # --- New parameter for scaling modal coordinates 'z' ---
        self.max_z_amplitude_scale = kwargs.get('max_z_amplitude_scale', 1000) # Tune this value
        print(f"Max Z Amplitude Scale (for random z generation): {self.max_z_amplitude_scale}")
        # --- End New Parameter ---
        
        # Store the z pattern for the current main step and applied z for current substep
        self.base_z_pattern_for_main_step = None
        self.current_applied_z = None # This will be used for NN prediction
        self.last_applied_force_magnitude = 0.0 # Norm of the applied distributed force Phi*z

        self.directory = kwargs.get('directory')
        self.young_modulus = kwargs.get('young_modulus', 5000)
        self.poisson_ratio = kwargs.get('poisson_ratio', 0.25)
        self.density = kwargs.get('density', 10)
        self.volume = kwargs.get('volume', 1)
        self.total_mass = kwargs.get('total_mass', 10)
        self.mesh_filename = kwargs.get('mesh_filename', 'unknown')
        print(f"Using directory: {self.directory}")
        print(f"Material properties: E={self.young_modulus}, nu={self.poisson_ratio}, rho={self.density}")
        
        self.show_modes = kwargs.get('show_modes', True)
        self.current_mode_index = -1
        self.mode_animation_step = 0
        self.mode_animation_steps = kwargs.get('steps_per_mode', 100)
        self.mode_scale = kwargs.get('mode_scale', 50.0)
        self.num_modes_to_show = kwargs.get('num_modes_to_show', 5)
        self.modes_computed = False
        self.eigenvectors = None
        self.eigenvalues = None
        self.original_positions = None
        self.transition_steps = 20
        self.pause_steps = 30

        self.force_verification_done = False 

      


    def onSimulationInitDoneEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the scene graph initialisation.
        """
        self.inputs = []
        self.outputs = []
        
        # Find the config file relative to this script's location or project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        default_config_path = os.path.join(project_root, 'configs', 'paper.yaml') # Adjust if your default is elsewhere

        # Use global args if available (e.g., from __main__), otherwise use a default path
        config_path_from_args = default_config_path # Default
        if 'args' in globals() and hasattr(args, 'config'):
            config_path_from_args = args.config
            if not os.path.isabs(config_path_from_args):
                 config_path_from_args = os.path.join(project_root, config_path_from_args)

        try:
            cfg = load_config(config_path_from_args)
            print(f"Controller loaded config from {config_path_from_args}")
        except FileNotFoundError:
            print(f"Config file not found at {config_path_from_args}. Attempting default project structure.")
            cfg = load_config(default_config_path) # Fallback
            print(f"Controller loaded config from {default_config_path}")


        # --- Instantiate Routine ---
        try:
            print("Instantiating Routine...")
            self.routine = Routine(cfg)
            print("Routine instantiated successfully.")
        except Exception as e:
            print(f"Error instantiating Routine: {e}")
            traceback.print_exc() 
            sys.exit(1)
        # --- End Routine Instantiation ---

        checkpoint_dir_rel = cfg.get('training', {}).get('checkpoint_dir', 'checkpoints')
        checkpoint_dir_abs = os.path.join(project_root, checkpoint_dir_rel) 

        checkpoint_filename = 'best_sofa.pt' 
        best_checkpoint_path = os.path.join(checkpoint_dir_abs, checkpoint_filename)

        print(f"Attempting to load best checkpoint from: {best_checkpoint_path}")
        if os.path.exists(best_checkpoint_path):
            try:
                self.routine.load_checkpoint(best_checkpoint_path)
                print("Successfully loaded best model checkpoint.")
                self.routine.model.eval()
            except Exception as e:
                print(f"Error loading checkpoint {best_checkpoint_path}: {e}")
                print("Proceeding without loaded model weights.")
        else:
            print(f"Warning: Best checkpoint file not found at {best_checkpoint_path}. Using initialized model.")
        
        self.linear_modes_np = self.routine.linear_modes.cpu().numpy() # Store as NumPy
        self.original_positions = np.copy(self.MO1.position.value) 

        self.directory = self.root.directory_name.value if hasattr(self.root, 'directory_name') else datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_base_dir = 'modal_data' 
        if self.save:
            if not os.path.exists(output_base_dir):
                os.mkdir(output_base_dir)
            self.output_subdir = os.path.join(output_base_dir, self.directory)
            if not os.path.exists(self.output_subdir):
                os.makedirs(self.output_subdir)
            print(f"Data saving enabled. Output directory: {self.output_subdir}")
        
        self.sampled = False 

        surface = self.surface_topo
        self.idx_surface = surface.triangles.value.reshape(-1)
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if self.show_modes and self.MO1:
            # self.original_positions already stored
            print(f"Stored original positions with shape {self.original_positions.shape}")
        
        # Get ForceROI indices once
        self.force_roi_indices_exact = self.exactSolution.getObject('ForceROI').indices.value


    def onAnimateBeginEvent(self, event):
        print(f"===================================================")
        print(f"\n--- Animation Step {self.current_substep} ---")
        # Check if the indices are valid
        num_modes = self.linear_modes_np.shape[1]

        # Calculate current substep within the main step
        substep_in_main_step = self.current_substep % self.num_substeps
        
        if substep_in_main_step == 0:  # Start of a new main step
            self.MO1.position.value = self.MO1.rest_position.value
            print(f"\n--- Starting Main Step {self.current_main_step} ---")
            num_modes_to_use = random.randint(1, num_modes)
            self.modes_to_use = random.sample(range(num_modes), num_modes_to_use)
            print(f"  New modes to use: {self.modes_to_use}")

            base_z_coeffs = np.random.choice([-1, 1], size=len(self.modes_to_use))
            self.base_z_pattern_for_main_step = base_z_coeffs * self.max_z_amplitude_scale
            print(f"  Base Z pattern for main step (norm): {np.linalg.norm(self.base_z_pattern_for_main_step):.4f}")

        # Calculate force for the CURRENT substep based on modal coordinates
        substep_fraction = (substep_in_main_step + 1) / self.num_substeps
        current_amplitude_scale = self.max_z_amplitude_scale * substep_fraction
        self.current_applied_z = self.base_z_pattern_for_main_step * current_amplitude_scale

        # Compute distributed forces: F = Phi * z
        current_step_distributed_forces = self.linear_modes_np[:, self.modes_to_use] @ self.current_applied_z

        # Reshape forces to (num_nodes, 3)
        self.force_in_newton = current_step_distributed_forces.reshape(-1, 3)

        # Compute the total force magnitude as the sum of the magnitudes of the forces at each node
        node_force_magnitudes = np.linalg.norm(self.force_in_newton, axis=1)
        self.last_applied_force_magnitude_newton = np.sum(node_force_magnitudes)
        print(f"  Substep {substep_in_main_step}/{self.num_substeps-1}: Total Force Magnitude: {self.last_applied_force_magnitude_newton:.4f}")

        # Store the force magnitude for plotting
        if not hasattr(self, 'all_force_magnitudes_newton'):
            self.all_force_magnitudes_newton = []
        self.all_force_magnitudes_newton.append(self.last_applied_force_magnitude_newton)

        # Debug: Print force statistics
        print(f"  Norm of Force in Newtons: {np.linalg.norm(self.force_in_newton):.4f}")
        print(f"  Norm of Applied z components: {np.linalg.norm(self.current_applied_z):.4f}")
        print(f"  Applied z components (first few): {self.current_applied_z[:min(len(self.current_applied_z), 5)]}")
        
        # Rest of the method remains the same...
        forces_reshaped = current_step_distributed_forces.reshape(-1, 3)
        
        # Extract forces for the ROI nodes
        forces_for_roi_exact_nodes = forces_reshaped[self.force_roi_indices_exact]

        # Remove previous CFFs
        if self.cff_exact is not None:
            try: self.exactSolution.removeObject(self.cff_exact)
            except Exception as e: print(f"Warning: Error removing CFF (Exact): {e}")
            finally: self.cff_exact = None
        
      

        # Create and add new CFFs using 'forces' attribute
        try:
            # Exact Solution
            self.cff_exact = self.exactSolution.addObject('ConstantForceField',
                            name="CFF_Exact_Modal",
                            indices=self.force_roi_indices_exact.tolist(),
                            forces=forces_for_roi_exact_nodes.tolist(),
                            showArrowSize=1, showColor="0.2 0.2 0.8 1")
            if self.cff_exact: self.cff_exact.init()

   
        except Exception as e:
            print(f"ERROR: Failed to create/add/init ConstantForceField(s) with modal forces: {e}")
            traceback.print_exc()
            self.cff_exact = None
            if self.root: self.root.animate = False

        self.start_time = process_time()

    def onAnimateEndEvent(self, event):
        # Calculate current substep within main step FIRST
        substep_in_main_step = self.current_substep % self.num_substeps
        current_main_step_computed = self.current_substep // self.num_substeps
        
        real_solution = self.MO1.position.value.copy() - self.MO1.rest_position.value.copy()
        
        # Compute modal coordinates from the ACTUAL SOFA solutions
        z_nonlinear = self.computeModalCoordinates(real_solution)  # From hyperelastic solution
        
        # Get displacement from the real (hyperelastic) solution
        pos_real_all_dofs = self.MO1.position.value
        rest_pos_all_dofs = self.MO1.rest_position.value
        u_real_all_dofs = pos_real_all_dofs - rest_pos_all_dofs
        
        # Initialize energy and modal coordinates
        energy_real = np.nan
        
        # Compute internal energy - fix the tensor dimension issue
        try:
            # Ensure displacement is flattened for energy calculation
            u_flat = u_real_all_dofs.flatten()
            energy_real = self.computeInternalEnergy(u_flat)
        except Exception as e:
            print(f"Warning: Could not compute real energy: {e}")
            traceback.print_exc()

        # Store z_nonlinear for statistics (if valid)
        if z_nonlinear is not None and not np.isnan(z_nonlinear).any():
            self.all_z_coords.append(np.copy(z_nonlinear))
        print(f"  =============================================================================")
        print(f"Substep {substep_in_main_step}/{self.num_substeps-1} of Main Step {current_main_step_computed}:")
        print(f"  Applied Z: {self.current_applied_z[:min(5, len(self.current_applied_z))]}...")
        print(f"  Actual Z (nonlinear): {z_nonlinear[:min(5, len(z_nonlinear))] if z_nonlinear is not None else 'None'}...")
        print(f"  Energy (real): {'NaN' if np.isnan(energy_real) else f'{energy_real:.4f}'}")
        print(f"  ================================================================================")
        
        # Save data if conditions are met
        if (self.save and 
            z_nonlinear is not None and 
            not np.isnan(z_nonlinear).any() and 
            not np.isnan(energy_real)):
            
            try:
                num_modes = self.routine.linear_modes.shape[1] 
                save_folder = os.path.join("bunny_dataset", f"{num_modes}_modes")
                os.makedirs(save_folder, exist_ok=True)
                
                # Determine the next index for saving files
                existing_data_files = glob.glob(os.path.join(save_folder, "data_*.npz"))
                next_index = len(existing_data_files)
                
                # --- Save modal data (ACTUAL z from SOFA, not applied z) ---
                data_filename = os.path.join(save_folder, f"data_{next_index:04d}.npz")
                np.savez(data_filename, 
                    z=z_nonlinear,  # Save the ACTUAL modal coordinates, not applied
                    energy=energy_real)
                
                # --- Save displacement data ---
                displacement_filename = os.path.join(save_folder, f"displacement_{next_index:04d}.npz")
                np.savez(displacement_filename,
                    u_flat=u_real_all_dofs.flatten())
                
                print(f"  Saved: z_shape={z_nonlinear.shape}, energy={energy_real:.4f}")
                
            except Exception as e:
                print(f"Error saving data: {e}")
                traceback.print_exc()
        else:
            reason = []
            if not self.save: reason.append("save disabled")
            if z_nonlinear is None: reason.append("z_nonlinear is None")
            if z_nonlinear is not None and np.isnan(z_nonlinear).any(): reason.append("z_nonlinear has NaN")
            if np.isnan(energy_real): reason.append("energy is NaN")
            print(f"  Skipping save: {', '.join(reason)}")

        # Update counters
        self.current_substep += 1
        
        # Update main step counter
        self.current_main_step = self.current_substep // self.num_substeps
        
        # Check stopping condition
        if self.current_main_step >= self.max_main_steps:
            print(f"All {self.max_main_steps} main steps completed. Stopping.")
            if self.root: 
                self.root.animate = False
        
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

    max_z_amplitude_scale = config['simulation'].get('max_z_amplitude_scale', 1000)

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
                                   newton_iterations=100,
                                   printLog=True)
    
    linear_solver = exactSolution.addObject('CGLinearSolver', 
                                          template="CompressedRowSparseMatrixMat3x3d",
                                          iterations=config['physics'].get('solver_iterations', 1000), 
                                          tolerance=config['physics'].get('solver_tolerance', 1e-6), 
                                          threshold=config['physics'].get('solver_threshold', 1e-6), 
                                          warmStart=False)
    
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
                                        max_z_amplitude_scale=max_z_amplitude_scale,
                                        **kwargs) # Pass any additional kwargs
    rootNode.addObject(controller)

    return rootNode, controller


if __name__ == "__main__":
    import Sofa.Gui
    from tqdm import tqdm
    import yaml
    import argparse
    import traceback
    import random
    import time # Import time for headless loop

    # Add argument parser
    parser = argparse.ArgumentParser(description='SOFA Validation with Neural Modes and Substeps')
    parser.add_argument('--config', type=str, default='configs/paper.yaml', help='Path to config file')
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