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

        # Add a placeholder for the force field created/removed each step
        self.current_torsion_ff = None

        self.key = kwargs.get('key')

        self.iteration = kwargs.get("sample")
        self.start_time = 0
        self.root = node
        self.save = True
        self.l2_error, self.MSE_error = [], []
        self.l2_deformation, self.MSE_deformation = [], []
        self.RMSE_error, self.RMSE_deformation = [], []

        # --- Add lists to store individual energies ---
        self.real_energies = []
        self.predicted_energies = []
        self.linear_energies = []

        self.num_substeps = kwargs.get('num_substeps', 1)
        self.total_twist_magnitude_for_step = 0.0 # Target magnitude for the main step
        self.current_substep = 0
        self.current_main_step = 0
        self.max_main_steps = kwargs.get('max_main_steps', 20)


        
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
        Manages substep torsion application by removing/adding TorsionForceField.
        """
        # --- Check if starting a new MAIN step ---
        if self.current_substep == 0:
            self.MO1.position.value = self.MO1.rest_position.value
            max_twist_magnitude = 50.0 # Adjust as needed (Torque units)
            self.total_twist_magnitude_for_step = max_twist_magnitude * (self.current_main_step + 1) / self.max_main_steps
            print(f"Main Step {self.current_main_step + 1}: Target Twist Torque = {self.total_twist_magnitude_for_step:.2f}")

        # --- Calculate incremental torque for the CURRENT substep ---
        incremental_torque = self.total_twist_magnitude_for_step * (self.current_substep + 1) / self.num_substeps

        # --- Remove existing TorsionForceField if it exists ---
        if self.current_torsion_ff is not None:
            try:
                # Check if parent node still exists and contains the object
                if self.exactSolution and self.current_torsion_ff in self.exactSolution.objects:
                     self.exactSolution.removeObject(self.current_torsion_ff)
                # else:
                #      print(f"Warning: TorsionFF parent node invalid or object already removed.")
            except Exception as e:
                print(f"Warning: Error removing TorsionFF: {e}")
            self.current_torsion_ff = None # Clear reference regardless
        # --- End Remove ---

        # --- Create and add a new TorsionForceField ---
        try:
            # Define axis and origin (could be stored in self if constant)
            torsion_axis_val = [1.0, 0.0, 0.0]
            # If not, you might need to retrieve it or hardcode it here
            beam_end_x = 0.0 # Example value, ensure this is correct
            torsion_origin_val = [beam_end_x, 0.0, 0.0]

            # Ensure the TorsionROI component exists (it should, added in createScene)
            torsion_roi = self.exactSolution.getObject('TorsionROI')
            if torsion_roi is None:
                raise ValueError("TorsionROI object not found in exactSolution node.")

            # Create the new TorsionForceField object
            self.current_torsion_ff = self.exactSolution.addObject(
                'TorsionForceField',
                name="TorsionFF_Step", # Use a potentially unique name if helpful
                indices="@TorsionROI.indices", # Reference indices via link path
                axis=torsion_axis_val,
                origin=torsion_origin_val,
                torque=incremental_torque # Set the calculated torque
            )

            # print(f"  Substep {self.current_substep + 1}: Initializing exactSolution node.")
            self.exactSolution.init() # Initialize the node containing the new TorsionFF
            # --- End Initialization ---

            print(f"  Substep {self.current_substep + 1}/{self.num_substeps}: Applied Torsion Torque = {incremental_torque:.2f}")

        except Exception as e:
            print(f"ERROR: Failed to create/add/init TorsionForceField: {e}")
            traceback.print_exc()
            self.current_torsion_ff = None # Ensure reference is None if creation failed
            if self.root: self.root.animate = False # Stop simulation

        self.start_time = process_time()
    
    def onAnimateEndEvent(self, event):
        """
        Called by SOFA's animation loop after each physics step.
        Manages substep counting and triggers analysis at the end of a main step.
        """
        self.current_substep += 1

        # --- Check if the MAIN step is complete ---
        if self.current_substep >= self.num_substeps:
            print(f"\n--- Main Step {self.current_main_step + 1} Completed ---")
            # --- Perform Analysis (existing code) ---
            real_solution = self.MO1.position.value.copy() - self.MO1.rest_position.value.copy()
            real_energy = self.computeInternalEnergy(real_solution)
            print(f"Real energy: {real_energy}")
            self.real_energies.append(real_energy)

            z = self.computeModalCoordinates(real_solution)
            print(f"Modal coordinates: {z}")

            if np.isnan(z).any():
                # ... (existing NaN handling) ...
                predicted_energy = float('nan')
                linear_energy = float('nan')
                self.l2_error.append(float('nan'))
                self.RMSE_error.append(float('nan'))
                self.MSE_error.append(float('nan'))
            else:
                z_th = torch.tensor(z, dtype=torch.float64, device=self.routine.device).unsqueeze(0)
                linear_modes_th = self.routine.linear_modes.to(self.routine.device)
                l_th = torch.matmul(linear_modes_th, z_th.T).squeeze()
                with torch.no_grad():
                    y_th = self.routine.model(z_th).squeeze()
                u_pred_th = l_th + y_th
                num_nodes = self.MO1.position.value.shape[0]
                linear_energy = float('nan')
                predicted_energy = float('nan')

                try:
                    l_th_reshaped = l_th.reshape(num_nodes, 3)
                    linear_energy = self.computeInternalEnergy(l_th_reshaped.cpu().numpy())
                    print(f"Linear energy (l):   {linear_energy}")

                    u_pred_reshaped = u_pred_th.reshape(num_nodes, 3)
                    predicted_energy = self.computeInternalEnergy(u_pred_reshaped.cpu().numpy())
                    print(f"Predicted energy (l+y): {predicted_energy}")

                except RuntimeError as e:
                    # ... (existing error handling) ...
                    print(f"Error reshaping displacement: {e}")
                    linear_energy = float('nan')
                    predicted_energy = float('nan')
                    self.l2_error.append(float('nan'))
                    self.RMSE_error.append(float('nan'))
                else:
                    # ... (existing error calculations) ...
                    real_solution_reshaped = real_solution.reshape(num_nodes, 3)
                    prediction_np = u_pred_reshaped.detach().cpu().numpy()
                    self.l2_error.append(np.linalg.norm(real_solution_reshaped - prediction_np))
                    self.RMSE_error.append(np.sqrt(np.mean((real_solution_reshaped - prediction_np)**2)))

                # Compute energy error
                if not np.isnan(real_energy) and not np.isnan(predicted_energy):
                    self.MSE_error.append(real_energy - predicted_energy)
                else:
                    self.MSE_error.append(float('nan'))
            # --- End Analysis ---

            # Store energies (always append, even if NaN)
            self.predicted_energies.append(predicted_energy)
            self.linear_energies.append(linear_energy)

            # --- Reset for the next MAIN step ---
            self.current_substep = 0
            self.current_main_step += 1
            print("---------------------------------------\n")

            # --- Optional: Stop simulation after N main steps (for headless/GUI consistency) ---
            # Note: This might feel abrupt in GUI mode.
            if not args.gui and self.current_main_step >= self.max_main_steps:
                 print(f"Reached maximum main steps ({self.max_main_steps}). Stopping simulation.")
                 if self.root: self.root.animate = False # Stop the animation loop

        # If not the end of a main step, do nothing here.
        self.end_time = process_time() # Track time

    def computeModalCoordinates(self, displacement):
        """
        Compute modal coordinates from displacement using the linear modes.
        Projects displacement onto the linear modes: z = Modes^T * displacement.

        Args:
            displacement: Displacement vector of the system (NumPy array, shape (num_dofs,) or (num_nodes, 3)).

        Returns:
            Modal coordinates as a 1D numpy array (shape (num_modes,)).
        """
        # Ensure displacement is a flattened 1D NumPy array
        if displacement.ndim > 1:
            displacement_flat = displacement.flatten()
        else:
            displacement_flat = displacement

        # --- Convert linear_modes tensor to NumPy array ---
        # Ensure it's on CPU before converting
        if isinstance(self.linear_modes, torch.Tensor):
            linear_modes_np = self.linear_modes.cpu().numpy()
        else:
            # Assuming it might already be a NumPy array in some cases
            linear_modes_np = self.linear_modes
        # --- End conversion ---

        # Check shapes for debugging
        # print(f"Shape of linear_modes_np: {linear_modes_np.shape}") # Should be (num_dofs, num_modes)
        # print(f"Shape of displacement_flat: {displacement_flat.shape}") # Should be (num_dofs,)

        # --- Compute modal coordinates: z = Modes^T * displacement ---
        try:
            # Use np.dot for matrix multiplication
            modal_coordinates = np.dot(linear_modes_np.T, displacement_flat)
        except ValueError as e:
            print(f"Error during modal coordinate calculation: {e}")
            print(f"Shape mismatch? Modes.T: {linear_modes_np.T.shape}, Displacement: {displacement_flat.shape}")
            # Return zeros or raise error depending on desired behavior
            num_modes = linear_modes_np.shape[1]
            return np.zeros(num_modes)
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
        print(f"Displacement tensor shape: {displacement_tensor.shape}")
        with torch.no_grad():
            internal_energy = energy_calculator(displacement_tensor)

        # If a batch dimension was added, remove it from the result if necessary
        if internal_energy.dim() > 0 and internal_energy.shape[0] == 1:
             internal_energy = internal_energy.squeeze(0)

        return internal_energy.item()
    
    
    

    def close(self):
        """
        Called when the simulation is closing. Calculates final statistics and generates plots.
        """
        print("\n--- Simulation Finished ---")

        # Convert lists to numpy arrays for nan-aware calculations
        real_np = np.array(self.real_energies, dtype=float)
        pred_np = np.array(self.predicted_energies, dtype=float)
        linear_np = np.array(self.linear_energies, dtype=float)
        l2_np = np.array(self.l2_error, dtype=float)
        rmse_np = np.array(self.RMSE_error, dtype=float)
        energy_diff_np = real_np - pred_np # Calculate difference for stats

        # --- Calculate Statistics (Ignoring NaNs) ---
        print("\n--- Statistics (NaNs Ignored) ---")
        print(f"Mean Energy Difference (Real - Predicted): {np.nanmean(energy_diff_np):.4f}")
        print(f"Max Energy Difference (Real - Predicted):  {np.nanmax(energy_diff_np):.4f}")
        print(f"Min Energy Difference (Real - Predicted):  {np.nanmin(energy_diff_np):.4f}")
        print(f"Mean Absolute Energy Error |Real - Predicted|: {np.nanmean(np.abs(energy_diff_np)):.4f}")

        print(f"\nMean L2 Error:   {np.nanmean(l2_np):.4f}")
        print(f"Max L2 Error:    {np.nanmax(l2_np):.4f}")
        print(f"Min L2 Error:    {np.nanmin(l2_np):.4f}")

        print(f"\nMean RMSE Error: {np.nanmean(rmse_np):.4f}")
        print(f"Max RMSE Error:  {np.nanmax(rmse_np):.4f}")
        print(f"Min RMSE Error:  {np.nanmin(rmse_np):.4f}")
        print("-----------------------------------\n")


        # --- Plotting ---
        num_steps = len(real_np)
        steps = np.arange(num_steps)

        # Define output directory for plots
        plot_dir = self.output_subdir if self.save else "." # Save in output dir if saving is enabled
        if self.save and not os.path.exists(plot_dir):
             os.makedirs(plot_dir)

        # 1. Energy Comparison Plot
        plt.figure(figsize=(10, 6))
        plt.plot(steps, real_np, label='Real Energy (SOFA)', marker='o', linestyle='-', markersize=4)
        plt.plot(steps, pred_np, label='Predicted Energy (l+y)', marker='x', linestyle='--', markersize=4)
        plt.plot(steps, linear_np, label='Linear Energy (l)', marker='s', linestyle=':', markersize=4)
        plt.xlabel('Simulation Step')
        plt.ylabel('Internal Energy')
        plt.title('Energy Comparison Over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "energy_comparison.png")) # Save the plot
        # plt.show() # Optionally show plot interactively

        # 2. Absolute Energy Error Plot
        plt.figure(figsize=(10, 6))
        plt.plot(steps, np.abs(energy_diff_np), label='Absolute Energy Error |Real - Predicted|', marker='d', linestyle='-', color='red')
        plt.xlabel('Simulation Step')
        plt.ylabel('Absolute Energy Error')
        plt.title('Absolute Energy Error Over Time')
        plt.legend()
        plt.grid(True)
        plt.yscale('log') # Use log scale if errors vary widely
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "absolute_energy_error.png")) # Save the plot
        # plt.show()

        # 3. Geometric Error Plot (L2)
        plt.figure(figsize=(10, 6))
        plt.plot(steps, l2_np, label='L2 Geometric Error', marker='^', linestyle='-', color='green')
        plt.xlabel('Simulation Step')
        plt.ylabel('L2 Norm of Displacement Error')
        plt.title('L2 Geometric Error Over Time')
        plt.legend()
        plt.grid(True)
        plt.yscale('log') # Use log scale if errors vary widely
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "l2_error.png")) # Save the plot
        # plt.show()

        # 4. Geometric Error Plot (RMSE)
        plt.figure(figsize=(10, 6))
        plt.plot(steps, rmse_np, label='RMSE Geometric Error', marker='v', linestyle='-', color='purple')
        plt.xlabel('Simulation Step')
        plt.ylabel('RMSE of Displacement Error')
        plt.title('RMSE Geometric Error Over Time')
        plt.legend()
        plt.grid(True)
        plt.yscale('log') # Use log scale if errors vary widely
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "rmse_error.png")) # Save the plot
        # plt.show()

        print(f"Plots saved to {plot_dir}")
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

    # --- Constraints ---
    # Fixed end
    fixed_box_coords = config['constraints'].get('fixed_box', [-0.01, -0.01, -0.02, 0.1, 1.01, 1.02]) # Adjusted x_max for fixed end
    fixed_box = exactSolution.addObject('BoxROI',
                                      name='FixedROI',
                                      box=" ".join(str(x) for x in fixed_box_coords),
                                      drawBoxes=True)
    exactSolution.addObject('FixedConstraint', indices="@FixedROI.indices")

    # --- ROI for Torsion Application (End Face) ---
    # Assuming beam end is at x=10, y=[-0.5, 0.5], z=[-0.5, 0.5] - ADJUST THESE
    beam_end_x = 10.0
    beam_half_y = 0.5
    beam_half_z = 0.5
    roi_thickness = 0.1 # Thickness of ROI in x-direction

    torsion_roi_coords = [beam_end_x - roi_thickness, -beam_half_y - 0.01, -beam_half_z - 0.01,
                          beam_end_x + roi_thickness,  beam_half_y + 0.01,  beam_half_z + 0.01]
    torsion_roi = exactSolution.addObject('BoxROI', name='TorsionROI', drawBoxes=True,
                                          box=" ".join(map(str, torsion_roi_coords)))

    # --- Torsion Force Field ---
    torsion_axis_val = [1.0, 0.0, 0.0] # Twist around X-axis
    torsion_origin_val = [beam_end_x, 0.0, 0.0] # Center of the end face

    # torsion_ff = exactSolution.addObject('TorsionForceField',
    #                                      name='TorsionFF',
    #                                      # 'object' link is implicit to parent's MechanicalObject
    #                                      indices="@TorsionROI.indices", # Apply to nodes in the ROI
    #                                      axis=torsion_axis_val,         
    #                                      origin=torsion_origin_val,       
    #                                      torque=0.0)                    
    # --- End Torsion ---

    # --- Visual Model ---
    visual = exactSolution.addChild("visual")
    visual.addObject('OglModel', src='@../DOFs', color='0 1 0 1')
    visual.addObject('BarycentricMapping', input='@../DOFs', output='@./')

    # --- Controller Setup ---
    num_substeps = config['physics'].get('num_substeps', 1)
    max_main_steps = config['simulation'].get('steps', 20)

    controller = AnimationStepController(rootNode,
                                         exactSolution=exactSolution,
                                         fem=fem,
                                         linear_solver=linear_solver,
                                         surface_topo=surface_topo,
                                         MO1=MO1,
                                         fixed_box=fixed_box,
                                        #  torsion_ff=torsion_ff, # Pass the TorsionForceField
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
    os.makedirs(output_dir, exist_ok=True)

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