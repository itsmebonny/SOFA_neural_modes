import Sofa
import SofaRuntime
import numpy as np 
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from Sofa import SofaDeformable
from time import process_time, time
import datetime
from sklearn.preprocessing import MinMaxScaler
from training.train_sofa import Routine, load_config # Assuming train_sofa.py is in training/
# add network path to the python path




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
        # self.cff = kwargs.get('cff') # Controller will manage CFF creation

                # --- Linear Solution Components ---
        self.linearSolution = kwargs.get('linearSolution')
        self.MO2 = kwargs.get('MO2') # MechObj for linear
        self.linearFEM = kwargs.get('linearFEM') # Linear FEM ForceField
        
        # CFFs will be created per step
        self.cff_exact = None 
        self.cff_linear = None 

        self.MO_LinearModes = kwargs.get('MO_LinearModes') # MechObj for Linear Modes Viz
        self.MO_NeuralPred = kwargs.get('MO_NeuralPred')   # MechObj for Neural Pred Viz
        self.visual_LM = kwargs.get('visual_LM') # Visual for Linear Modes
        self.visual_NP = kwargs.get('visual_NP') # Visual for Neural Pred

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
        self.max_main_steps = kwargs.get('max_main_steps', 10)

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
        default_config_path = os.path.join(project_root, 'configs', 'default.yaml') # Adjust if your default is elsewhere

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

        checkpoint_filename = 'bunny_sofa_dataset.pt' 
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
        self.force_roi_indices_linear = self.linearSolution.getObject('ForceROI').indices.value


    def onAnimateBeginEvent(self, event):

        match self.current_main_step:
            case 0 : 
                self.modes_to_use = [0] # list of indices to use for modal coordinates
            case 1 :
                self.modes_to_use = [1] # list of indices to use for modal coordinates
            case 2 : 
                self.modes_to_use = [2] # list of indices to use for modal coordinates
            case 3 :
                self.modes_to_use = [3] # list of indices to use for modal coordinates
            case 4 :
                self.modes_to_use = [4] # list of indices to use for modal coordinates
            case 5 :
                self.modes_to_use = [5] # list of indices to use for modal coordinates
            case 6 :
                self.modes_to_use = [6] # list of indices to use for modal coordinates
            case 7 :
                self.modes_to_use = [2,3] # list of indices to use for modal coordinates
            case 8 :
                self.modes_to_use = [3,4] # list of indices to use for modal coordinates
            case _ :
                self.modes_to_use = [2,3,4,5,6] # list of indices to use for modal coordinates

        #check if the indices are valid
        if any(idx >= self.linear_modes_np.shape[1] for idx in self.modes_to_use):
            print(f"Error: One or more mode indices {self.modes_to_use} exceed available modes ({self.linear_modes_np.shape[1]}).")
            return

        if self.current_substep == 0: # Start of a new main step
            rest_pos = self.MO1.rest_position.value
            self.MO1.position.value = rest_pos
            if self.MO2: self.MO2.position.value = rest_pos
            if self.MO_LinearModes: self.MO_LinearModes.position.value = rest_pos
            if self.MO_NeuralPred: self.MO_NeuralPred.position.value = rest_pos
            print(f"\n--- Starting Main Step {self.current_main_step + 1} ---")

            #base_z_coeffs = 0.5*np.ones(len(self.modes_to_use)) + np.random.rand(len(self.modes_to_use)) - 0.5 # Random coefficients in [0.5, 1]
            base_z_coeffs = np.ones(len(self.modes_to_use))  

            self.base_z_pattern_for_main_step = base_z_coeffs * self.max_z_amplitude_scale
            print(f"  Base Z pattern for main step (norm): {np.linalg.norm(self.base_z_pattern_for_main_step):.4f}")
            print(f"  Base Z pattern components (first 7 values): {self.base_z_pattern_for_main_step[:min(len(self.modes_to_use),7)]}")


        # Calculate force for the CURRENT substep based on modal coordinates
        substep_fraction = (self.current_substep % self.num_substeps + 1) / self.num_substeps

        current_amplitude_scale = self.max_z_amplitude_scale * substep_fraction
        self.current_applied_z = self.base_z_pattern_for_main_step * current_amplitude_scale

        # Compute distributed forces: F = Phi * z
        current_step_distributed_forces = self.linear_modes_np[:, self.modes_to_use] @ self.current_applied_z  # (num_dofs,)

        # Reshape forces to (num_nodes, 3)
        self.force_in_newton = current_step_distributed_forces.reshape(-1, 3)

        # Compute the total force magnitude as the sum of the magnitudes of the forces at each node
        node_force_magnitudes = np.linalg.norm(self.force_in_newton, axis=1)  # Magnitude of force at each node
        self.last_applied_force_magnitude_newton = np.sum(node_force_magnitudes)  # Total force magnitude
        print(f"  Total Force Magnitude (sum of node magnitudes): {self.last_applied_force_magnitude_newton:.4f}")

        # Store the force magnitude for plotting
        if not hasattr(self, 'all_force_magnitudes_newton'):
            self.all_force_magnitudes_newton = []
        self.all_force_magnitudes_newton.append(self.last_applied_force_magnitude_newton)

        # Debug: Print force statistics
        print(f"  Norm of Force in Newtons: {np.linalg.norm(self.force_in_newton):.4f}")
        print(f"  Total Force Magnitude (sum of node magnitudes): {self.last_applied_force_magnitude_newton:.4f}")
        print(f"  Norm of Applied z components: {np.linalg.norm(self.current_applied_z):.4f}")
        forces_reshaped = current_step_distributed_forces.reshape(-1, 3) # (num_nodes, 3)
        
        # Extract forces for the ROI nodes
        forces_for_roi_exact_nodes = forces_reshaped[self.force_roi_indices_exact]
        forces_for_roi_linear_nodes = forces_reshaped[self.force_roi_indices_linear]

        # print(f"  Substep {substep_fraction*100:.1f}%: Applied Z norm={np.linalg.norm(self.current_applied_z):.4f}, Applied Force Norm={self.last_applied_force_magnitude:.4e}")

        # Remove previous CFFs
        if self.cff_exact is not None:
            try: self.exactSolution.removeObject(self.cff_exact)
            except Exception as e: print(f"Warning: Error removing CFF (Exact): {e}")
            finally: self.cff_exact = None
        
        if self.cff_linear is not None:
            try: self.linearSolution.removeObject(self.cff_linear)
            except Exception as e: print(f"Warning: Error removing CFF (Linear): {e}")
            finally: self.cff_linear = None

        # Create and add new CFFs using 'forces' attribute

        #Try with only one force
        try:
            # Exact Solution
            self.cff_exact = self.exactSolution.addObject('ConstantForceField',
                               name="CFF_Exact_Modal",
                               indices=self.force_roi_indices_exact.tolist(), # Use stored indices
                               forces=forces_for_roi_exact_nodes.tolist(), # Apply per-node forces
                               showArrowSize=1, showColor="0.2 0.2 0.8 1")
            if self.cff_exact: self.cff_exact.init()

            # Linear Solution
            self.cff_linear = self.linearSolution.addObject('ConstantForceField',
                               name="CFF_Linear_Modal",
                               indices=self.force_roi_indices_linear.tolist(), # Use stored indices
                               forces=forces_for_roi_linear_nodes.tolist(), # Apply per-node forces
                               showArrowSize=0.0) 
            if self.cff_linear: self.cff_linear.init()
            
        except Exception as e:
            print(f"ERROR: Failed to create/add/init ConstantForceField(s) with modal forces: {e}")
            traceback.print_exc()
            self.cff_exact = None
            self.cff_linear = None
            if self.root: self.root.animate = False

        self.start_time = process_time()

    def onAnimateEndEvent(self, event):
        real_solution = self.MO1.position.value.copy() - self.MO1.rest_position.value.copy()
        linear_solution = self.MO2.position.value.copy() - self.MO2.rest_position.value.copy()
        z_nonlinear=self.computeModalCoordinates(real_solution)
        z_linear = self.computeModalCoordinates(linear_solution)

        np.set_printoptions(precision=2, suppress=True)
        print("z_linear    = ", z_linear)   
        print("z_nonlinear = ", z_nonlinear)
        print("=================================")
        np.set_printoptions(precision=4, suppress=False)
        
        if (self.num_substeps - self.current_substep <= 2):
            print("=============> making prediction")
            try:
                if self.current_applied_z is not None and len(self.current_applied_z) > 0:
                    self.z0_vals.append(float(self.current_applied_z[0]))
                else:
                    self.z0_vals.append(np.nan)

                # current_force_magnitude is now self.last_applied_force_magnitude (norm of Phi*z)
                real_solution = self.MO1.position.value.copy() - self.MO1.rest_position.value.copy()
                linear_solution = self.MO2.position.value.copy() - self.MO2.rest_position.value.copy()
                real_energy = self.computeInternalEnergy(real_solution)

                # Compute actual modal coordinates from SOFA's real solution
                
                z_actual_sofa = self.computeModalCoordinates(real_solution) # This is z_actual
                #z_actual_sofa[0:4] = torch.zeros(4)
                print("z_actual_sofa = ", z_actual_sofa)
                if z_actual_sofa is not None and len(z_actual_sofa) > 0:
                    self.actual_z0_vals.append(float(z_actual_sofa[0]))
                else:
                    self.actual_z0_vals.append(np.nan)
                if z_actual_sofa is not None and not np.isnan(z_actual_sofa).any():
                    self.all_z_coords.append(z_actual_sofa.copy()) # Store z_actual
                print(f"  Random z pattern norm: {np.linalg.norm(self.base_z_pattern_for_main_step):.4f}, Actual z norm: {np.linalg.norm(z_actual_sofa):.4f}")
                print(f"  Random z components (first few): {self.current_applied_z[:min(len(self.current_applied_z),7)]}")

                # Divide current_applied_z by eigenvalues
                if self.routine.eigenvalues is not None and self.current_applied_z is not None:
                    num_eigenvalues_to_use = min(len(self.routine.eigenvalues), len(self.current_applied_z))
                    eigenvalues_truncated = self.routine.eigenvalues[:num_eigenvalues_to_use]
                    safe_eigenvalues = np.where(np.abs(eigenvalues_truncated) < 1e-9, 1e-9, eigenvalues_truncated)
                    z_scaled = self.current_applied_z[:num_eigenvalues_to_use] / np.sqrt(safe_eigenvalues)
                    self.scaled_z0_vals.append(float(z_scaled[0]) if len(z_scaled) > 0 else np.nan)
                    print(f"  Scaled z components (first few): {z_scaled[:min(len(z_scaled),7)]}")
                else:
                    self.scaled_z0_vals.append(np.nan)
                    print("  Eigenvalues not available, cannot scale z.")
                print(f"  Actual z components (first few): {z_actual_sofa[:min(len(z_actual_sofa),7)]}")

                sofa_linear_energy = float('nan')
                linear_solution_sofa_reshaped = None
                if self.MO2 and self.linearFEM:
                    linear_solution_sofa = self.MO2.position.value.copy() - self.MO2.rest_position.value.copy()
                    sofa_linear_energy = self.computeInternalEnergy(linear_solution_sofa)
                    try:
                        linear_solution_sofa_reshaped = linear_solution_sofa.reshape(self.MO1.position.value.shape[0], 3)
                    except ValueError as e:
                        print(f"  Warning: Could not reshape SOFA linear solution: {e}")
                        linear_solution_sofa_reshaped = None
                
                predicted_energy = float('nan')
                linear_energy_modes = float('nan')
                l2_err_pred_real, rmse_pred_real, mse_pred_real = float('nan'), float('nan'), float('nan')
                l2_err_lin_real, rmse_lin_real, mse_lin_real = float('nan'), float('nan'), float('nan')
                l2_err_lin_sofa, rmse_lin_sofa, mse_lin_sofa = float('nan'), float('nan'), float('nan')
                
                l_th_reshaped_np, u_pred_reshaped_np, real_solution_reshaped = None, None, None

                # Use z_actual_sofa (computed from SOFA's real solution) for NN prediction
                if z_actual_sofa is None or np.isnan(z_actual_sofa).any():
                    print(f"  Warning: NaN or None detected in z_actual_sofa for force norm {self.last_applied_force_magnitude:.4f}. Skipping NN prediction.")
                else:
                    # Ensure z_actual_sofa is a 1D array before converting to tensor
                    if z_actual_sofa.ndim > 1:
                        z_actual_sofa_flat = z_actual_sofa.flatten() # Or handle error if shape is unexpected
                    else:
                        z_actual_sofa_flat = z_actual_sofa


                    z_for_nn_th = torch.tensor(z_actual_sofa_flat, dtype=torch.float64, device=self.routine.device).unsqueeze(0)
                    # Ensure modes_to_use matches the latent_dim of z_actual_sofa
                    latent_dim_for_z = len(z_actual_sofa_flat)

                    # Check if routine.linear_modes has enough columns
                    if self.routine.linear_modes.shape[1] < latent_dim_for_z:
                        print(f"  Error: linear_modes has {self.routine.linear_modes.shape[1]} modes, but z_actual_sofa has {latent_dim_for_z} components. Truncating z_actual_sofa for prediction.")
                        # Option 1: Truncate z_actual_sofa (if this makes sense for the model)
                        # z_for_nn_th = z_for_nn_th[:, :self.routine.linear_modes.shape[1]]
                        # latent_dim_for_z = self.routine.linear_modes.shape[1]
                        # Option 2: Pad modes (less likely to be correct unless model expects this)
                        # Option 3: Skip prediction or error out
                        # For now, let's assume we might need to truncate z if modes are fewer,
                        # or more likely, ensure latent_dim_for_z doesn't exceed available modes.
                        # The safer approach is to ensure z_actual_sofa's length matches the expected latent_dim.
                        # The computeModalCoordinates should return z of length routine.latent_dim.
                        # If z_actual_sofa_flat is longer than routine.latent_dim, it implies an issue upstream.
                        # However, if routine.latent_dim is the target, use that.
                        if latent_dim_for_z > self.routine.latent_dim:
                            print(f"  Warning: z_actual_sofa ({latent_dim_for_z}) is longer than routine.latent_dim ({self.routine.latent_dim}). Truncating z_actual_sofa.")
                            z_for_nn_th = z_for_nn_th[:, :self.routine.latent_dim]
                            latent_dim_for_z = self.routine.latent_dim # Update latent_dim_for_z to match
                        elif latent_dim_for_z < self.routine.latent_dim:
                            print(f"  Warning: z_actual_sofa ({latent_dim_for_z}) is shorter than routine.latent_dim ({self.routine.latent_dim}). This might be unexpected.")
                            # Potentially pad z_for_nn_th with zeros if model expects routine.latent_dim
                            # z_padding = torch.zeros((1, self.routine.latent_dim - latent_dim_for_z), dtype=z_for_nn_th.dtype, device=z_for_nn_th.device)
                            # z_for_nn_th = torch.cat((z_for_nn_th, z_padding), dim=1)
                            # latent_dim_for_z = self.routine.latent_dim # Update latent_dim_for_z

                    # Use up to latent_dim_for_z modes, or all available modes if fewer
                    num_available_modes = self.routine.linear_modes.shape[1]
                    modes_to_select = min(latent_dim_for_z, num_available_modes)

                    modes_to_use = self.routine.linear_modes[:, :modes_to_select].to(self.routine.device)
                    
                    modes_used = self.routine.linear_modes[:, :modes_to_select].to(self.routine.device)
                    
                    # Adjust z_for_nn_th if it was longer than available modes
                    if z_for_nn_th.shape[1] > modes_to_select:
                        z_for_nn_th = z_for_nn_th[:, :modes_to_select]

                    # l_th is Phi * z_input_to_NN (which is now z_actual_sofa)
                    l_th = torch.matmul(modes_used, z_for_nn_th.T).squeeze() # Squeeze if z_for_nn_th was unsqueezed

                    with torch.no_grad():
                        y_th = self.routine.model(z_for_nn_th).squeeze() # Squeeze if z_for_nn_th was unsqueezed
                    u_pred_th = l_th + y_th

                    num_nodes = self.MO1.position.value.shape[0]
                    try:
                        l_th_reshaped = l_th.reshape(num_nodes, 3)
                        l_th_reshaped_np = l_th_reshaped.cpu().numpy()

                        u_pred_reshaped = u_pred_th.reshape(num_nodes, 3)
                        u_pred_reshaped_np = u_pred_reshaped.cpu().numpy()

                        real_solution_reshaped = real_solution.reshape(num_nodes, 3)

                        linear_energy_modes = self.computeInternalEnergy(l_th_reshaped_np) # Energy of Phi*z_input_NN
                        predicted_energy = self.computeInternalEnergy(u_pred_reshaped_np)  # Energy of (Phi*z_input_NN + NN(z_input_NN))

                        diff_pred_real = real_solution_reshaped - u_pred_reshaped_np
                        l2_err_pred_real = np.linalg.norm(diff_pred_real)
                        mse_pred_real = np.mean(diff_pred_real**2)
                        rmse_pred_real = np.sqrt(mse_pred_real)

                        diff_lin_real = real_solution_reshaped - l_th_reshaped_np
                        l2_err_lin_real = np.linalg.norm(diff_lin_real)
                        mse_lin_real = np.mean(diff_lin_real**2)
                        rmse_lin_real = np.sqrt(mse_lin_real)

                        if linear_solution_sofa_reshaped is not None:
                            diff_lin_sofa = linear_solution_sofa_reshaped - l_th_reshaped_np
                            l2_err_lin_sofa = np.linalg.norm(diff_lin_sofa)
                            mse_lin_sofa = np.mean(diff_lin_sofa**2)
                            rmse_lin_sofa = np.sqrt(mse_lin_sofa)
                    except (RuntimeError, ValueError) as e:
                        print(f"  Error during prediction processing/reshaping/error calc: {e}")
                
                if self.original_positions is not None:
                    # print("  Debug: original_positions is not None. Updating visualization MOs.")
                    rest_pos = self.original_positions
                    if self.MO_LinearModes is not None and l_th_reshaped_np is not None:
                        # print(f"  Debug: Updating MO_LinearModes.position. l_th_reshaped_np shape: {l_th_reshaped_np.shape}")
                        self.MO_LinearModes.position.value = rest_pos + l_th_reshaped_np
                    elif self.MO_LinearModes is None:
                        print("  Debug: MO_LinearModes is None.")
                    elif l_th_reshaped_np is None:
                        print("  Debug: l_th_reshaped_np is None, cannot update MO_LinearModes.")

                    if self.visual_LM is not None and l_th_reshaped_np is not None:
                        # This might be redundant if MO_LinearModes directly drives the visual,
                        # but included for completeness if visual_LM has its own position.
                        # print(f"  Debug: Updating visual_LM.position. l_th_reshaped_np shape: {l_th_reshaped_np.shape}")
                        self.visual_LM.position.value = rest_pos + l_th_reshaped_np
                    elif self.visual_LM is None:
                        print("  Debug: visual_LM is None.")
                    # No need for another l_th_reshaped_np is None check if covered by MO_LinearModes

                    if self.MO_NeuralPred is not None and u_pred_reshaped_np is not None:
                        # print(f"  Debug: Updating MO_NeuralPred.position. u_pred_reshaped_np shape: {u_pred_reshaped_np.shape}")
                        self.MO_NeuralPred.position.value = rest_pos + u_pred_reshaped_np
                    elif self.MO_NeuralPred is None:
                        print("  Debug: MO_NeuralPred is None.")
                    elif u_pred_reshaped_np is None:
                        print("  Debug: u_pred_reshaped_np is None, cannot update MO_NeuralPred.")

                    if self.visual_NP is not None and u_pred_reshaped_np is not None:
                        # Similar to visual_LM, might be redundant.
                        # print(f"  Debug: Updating visual_NP.position. u_pred_reshaped_np shape: {u_pred_reshaped_np.shape}")
                        self.visual_NP.position.value = rest_pos + u_pred_reshaped_np
                    elif self.visual_NP is None:
                        print("  Debug: visual_NP is None.")
                    # No need for another u_pred_reshaped_np is None check
                else:
                    print("  Debug: self.original_positions is None. Skipping visualization MO updates.")
                
                self.substep_results.append((
                    self.last_applied_force_magnitude, real_energy, predicted_energy, linear_energy_modes, sofa_linear_energy,
                    l2_err_pred_real, rmse_pred_real, mse_pred_real,
                    l2_err_lin_real, rmse_lin_real, mse_lin_real,
                    l2_err_lin_sofa, rmse_lin_sofa, mse_lin_sofa
                ))
                #print mse between linear modes and real solution
                if real_solution_reshaped is not None:
                    mse_lin_sofa = np.mean((real_solution_reshaped - l_th_reshaped_np)**2)
                    print(f"  MSE between Linear Modes and Real Solution: {mse_lin_sofa:.4e}")
                else:
                    print("  Linear solution not available for MSE calculation.")
                # print mse between neural prediction and real solution
                mse_pred_real = np.mean((real_solution_reshaped - u_pred_reshaped_np)**2)
                print(f"  MSE between Neural Prediction and Real Solution: {mse_pred_real:.4e}")
                # print mse between linear modes and neural prediction
                if l_th_reshaped_np is not None:
                    mse_lin_sofa = np.mean((u_pred_reshaped_np - l_th_reshaped_np)**2)
                    print(f"  MSE between Linear Modes and Neural Prediction: {mse_lin_sofa:.4e}")
                else:
                    print("  Linear modes not available for MSE calculation.")

            except Exception as e:
             print(f"ERROR during analysis in onAnimateEndEvent: {e}")
             traceback.print_exc()
             self.substep_results.append((self.last_applied_force_magnitude, float('nan'), float('nan'), float('nan'), float('nan'), 
                                          float('nan'), float('nan'), float('nan'),
                                          float('nan'), float('nan'), float('nan'),
                                          float('nan'), float('nan'), float('nan')))

        self.current_substep += 1
        if (self.current_substep % self.num_substeps) == 0:
            self.current_substep = 0
            self.current_main_step += 1
            print(f"--- Main Step {self.current_main_step} Completed (Total Substeps: {self.current_substep}) ---")
            if not args.gui and self.current_main_step >= self.max_main_steps:
                 print(f"Reached maximum main steps ({self.max_main_steps}). Stopping simulation.")
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

        if isinstance(self.routine.linear_modes, torch.Tensor):
            linear_modes_np_for_z_calc = self.routine.linear_modes.cpu().numpy()
        else:
            linear_modes_np_for_z_calc = self.routine.linear_modes 
        
        M_sparse = self.routine.M
        
        if M_sparse is None or not isinstance(M_sparse, sparse.spmatrix):
            print("Warning: Mass matrix not available. Using simple projection (Modes^T * u).")
            try:
                # Ensure correct dimensions for dot product
                # linear_modes_np_for_z_calc.T should be (num_modes, num_dofs)
                # displacement_flat should be (num_dofs,)
                if linear_modes_np_for_z_calc.T.shape[1] != displacement_flat.shape[0]:
                    print(f"Shape mismatch for simple projection: Modes.T {linear_modes_np_for_z_calc.T.shape}, Disp {displacement_flat.shape}")
                    # Try to use the latent_dim from routine if modes matrix is larger
                    L_dim = self.routine.latent_dim
                    if linear_modes_np_for_z_calc.shape[1] > L_dim :
                         modal_coordinates = np.dot(linear_modes_np_for_z_calc[:,:L_dim].T, displacement_flat)
                    else: # Fallback or error
                         return np.zeros(L_dim) # Or raise error
                else:
                    modal_coordinates = np.dot(linear_modes_np_for_z_calc.T, displacement_flat)

            except ValueError as e:
                print(f"Error during simple modal coordinate calculation: {e}")
                return np.zeros(linear_modes_np_for_z_calc.shape[1]) # num_modes
        else:
            try:
                modes_t_m = linear_modes_np_for_z_calc.T @ M_sparse 
                modal_coordinates = modes_t_m @ displacement_flat 
            except ValueError as e:
                print(f"Error during mass matrix modal coordinate calculation: {e}")
                return np.zeros(linear_modes_np_for_z_calc.shape[1])
            except Exception as e: 
                print(f"Unexpected error during mass matrix projection: {e}")
                traceback.print_exc()
                return np.zeros(linear_modes_np_for_z_calc.shape[1])
        return modal_coordinates
    
    def computeInternalEnergy(self, displacement):
        energy_calculator = self.routine.energy_calculator
        device = self.routine.device 

        displacement_tensor = torch.tensor(displacement, dtype=torch.float64, device=device)

        # Energy calculator expects batch: (batch_size, num_dofs) or (batch_size, num_nodes, 3)
        # If displacement_tensor is (num_dofs) or (num_nodes,3), add batch dim
        if displacement_tensor.dim() == 1: # (num_dofs)
             displacement_tensor = displacement_tensor.unsqueeze(0)
        elif displacement_tensor.dim() == 2 and displacement_tensor.shape[1] == 3 : # (num_nodes, 3)
             displacement_tensor = displacement_tensor.reshape(1,-1) # to (1, num_dofs) if calculator expects flat
             # Or, if calculator handles (batch, nodes, 3) directly:
             # displacement_tensor = displacement_tensor.unsqueeze(0) 
        # Assuming energy_calculator takes (batch_size, num_dofs)

        with torch.no_grad():
            internal_energy = energy_calculator(displacement_tensor)

        if internal_energy.dim() > 0 and internal_energy.shape[0] == 1:
             internal_energy = internal_energy.squeeze(0)
        return internal_energy.item()
    
    
    

    def close(self):
        """
        Called when the simulation is closing. Calculates final statistics and generates plots,
        including average energy vs. force magnitude.
        """
        print("\n--- Simulation Finished ---")

        if not self.substep_results:
            print("No results collected. Skipping analysis and plotting.")
            return

        # Ensure force magnitudes are available
        if not hasattr(self, 'all_force_magnitudes_newton') or not self.all_force_magnitudes_newton:
            print("No force magnitudes collected. Skipping analysis and plotting.")
            return

        # Check for NaN in MO1 or MO2
        if self.MO1 is not None and np.isnan(self.MO1.position.value).any():
            print("MO1 contains NaN values. Skipping saving results.")
            return
        if self.MO2 is not None and np.isnan(self.MO2.position.value).any():
            print("MO2 contains NaN values. Skipping saving results.")
            return

        # Add force magnitudes to the results
        for i, force_mag in enumerate(self.all_force_magnitudes_newton):
            if i < len(self.substep_results):
                self.substep_results[i] = (force_mag,) + self.substep_results[i]

        result_columns = [
            'ForceMag', 'AppliedForceNorm', 'RealE', 'PredE', 'LinearModesE', 'SOFALinearE',
            'L2Err_Pred_Real', 'RMSE_Pred_Real', 'MSE_Pred_Real',
            'L2Err_Lin_Real', 'RMSE_Lin_Real', 'MSE_Lin_Real',
            'L2Err_Lin_SOFALin', 'RMSE_Lin_SOFALin', 'MSE_Lin_SOFALin'
        ]

        try:
            import pandas as pd
            df = pd.DataFrame(self.substep_results, columns=result_columns)
            avg_results = df.groupby('ForceMag').mean().reset_index()
            avg_results = avg_results.sort_values(by='ForceMag')

            print("\n--- Average Results per Force Magnitude ---")
            cols_to_print = ['ForceMag', 'RealE', 'PredE', 'LinearModesE', 'SOFALinearE',
                            'RMSE_Pred_Real', 'MSE_Pred_Real',
                            'RMSE_Lin_Real', 'MSE_Lin_Real',
                            'RMSE_Lin_SOFALin', 'MSE_Lin_SOFALin']
            print(avg_results[cols_to_print].to_string(index=False, float_format="%.4e"))
            print("-------------------------------------------\n")

            force_mags_plot = avg_results['ForceMag'].values
            avg_real_e = avg_results['RealE'].values
            avg_pred_e = avg_results['PredE'].values
            avg_linear_modes_e = avg_results['LinearModesE'].values
            avg_sofa_linear_e = avg_results['SOFALinearE'].values
            avg_rmse_pred_real = avg_results['RMSE_Pred_Real'].values
            avg_rmse_lin_real = avg_results['RMSE_Lin_Real'].values  # Ensure this column exists in avg_results
            avg_rmse_lin_sofa = avg_results['RMSE_Lin_SOFALin'].values  # Ensure this column exists in avg_results
            # Ensure the required variables are computed
            avg_mse_pred_real = avg_results['MSE_Pred_Real'].values 
            avg_mse_lin_real = avg_results['MSE_Lin_Real'].values 
            avg_mse_lin_sofa = avg_results['MSE_Lin_SOFALin'].values 


            # Plotting code uses force_mags_plot for x-axis
            plot_dir = self.output_subdir if self.save else "."
            if self.save and not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            plt.figure(figsize=(10, 6))
            plt.plot(force_mags_plot, avg_real_e, label='Avg Real Energy (SOFA Hyperelastic)', marker='o', linestyle='-')
            plt.plot(force_mags_plot, avg_pred_e, label='Avg Predicted Energy (l+y)', marker='x', linestyle='--')
            plt.plot(force_mags_plot, avg_linear_modes_e, label='Avg Linear Modes Energy (l)', marker='s', linestyle=':')
            plt.plot(force_mags_plot, avg_sofa_linear_e, label='Avg SOFA Linear Energy', marker='d', linestyle='-.')
            plt.xlabel('Applied Force Magnitude (N)')  # Updated X-axis label
            plt.ylabel('Average Internal Energy')
            plt.title('Average Energy vs. Applied Force Magnitude')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "avg_energy_vs_force_magnitude.png"))
            plt.close()

            print(f"Plots saved to {plot_dir}")
            print("Closing simulation")


            plt.figure(figsize=(10, 6))
            # ... (log scale energy plot, ensure valid_indices use correct arrays) ...
            valid_indices_real = avg_real_e > 0; valid_indices_pred = avg_pred_e > 0
            valid_indices_linear_modes = avg_linear_modes_e > 0; valid_indices_sofa_linear = avg_sofa_linear_e > 0
            if np.any(valid_indices_real): plt.plot(force_mags_plot[valid_indices_real], avg_real_e[valid_indices_real], label='Avg Real Energy', marker='o')
            if np.any(valid_indices_pred): plt.plot(force_mags_plot[valid_indices_pred], avg_pred_e[valid_indices_pred], label='Avg Predicted Energy', marker='x', linestyle='--')
            if np.any(valid_indices_linear_modes): plt.plot(force_mags_plot[valid_indices_linear_modes], avg_linear_modes_e[valid_indices_linear_modes], label='Avg Linear Modes Energy', marker='s', linestyle=':')
            if np.any(valid_indices_sofa_linear): plt.plot(force_mags_plot[valid_indices_sofa_linear], avg_sofa_linear_e[valid_indices_sofa_linear], label='Avg SOFA Linear Energy', marker='d', linestyle='-.')
            plt.xlabel('Applied Force Magnitude (N)'); plt.ylabel('Average Internal Energy (log scale)')
            plt.title('Average Energy vs. Applied Force Norm (Log Scale)'); plt.yscale('log')
            plt.legend(); plt.grid(True, which="both", ls="--"); plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "avg_energy_vs_force_norm_log.png"))
            plt.close()

            # ... (RMSE and MSE plots, ensure x-axis is force_mags_plot and labels are updated) ...
            plt.figure(figsize=(10, 6))
            plt.plot(force_mags_plot, avg_rmse_pred_real, label='RMSE: Pred (l+y) vs Real (MO1)', marker='^')
            plt.plot(force_mags_plot, avg_rmse_lin_real, label='RMSE: LinModes (l) vs Real (MO1)', marker='v', linestyle='--')
            plt.plot(force_mags_plot, avg_rmse_lin_sofa, label='RMSE: LinModes (l) vs SOFALin (MO2)', marker='<', linestyle=':')
            plt.xlabel('Applied Force Magnitude (N)'); plt.ylabel('Average RMSE')
            plt.title('Average RMSE vs. Applied Force Norm'); plt.legend(); plt.grid(True); plt.yscale('log'); plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "avg_rmse_vs_force_norm.png"))
            plt.close()


            plt.figure(figsize=(10, 6))
            plt.plot(force_mags_plot, avg_mse_pred_real, label='MSE: Pred (l+y) vs Real (MO1)', marker='^')
            plt.plot(force_mags_plot, avg_mse_lin_real, label='MSE: LinModes (l) vs Real (MO1)', marker='v', linestyle='--')
            plt.plot(force_mags_plot, avg_mse_lin_sofa, label='MSE: LinModes (l) vs SOFALin (MO2)', marker='<', linestyle=':')
            plt.xlabel('Applied Force Magnitude (N)'); plt.ylabel('Average MSE')
            plt.title('Average MSE vs. Applied Force Norm'); plt.legend(); plt.grid(True); plt.yscale('log'); plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "avg_mse_vs_force_norm.png"))
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.plot(force_mags_plot, np.abs(self.z0_vals[:len(force_mags_plot)]), label='|z| (First Value)', marker='o')
            plt.plot(force_mags_plot, np.abs(self.scaled_z0_vals[:len(force_mags_plot)]), label='|z_scaled| (First Value)', marker='x', linestyle='--')
            plt.plot(force_mags_plot, np.abs(self.actual_z0_vals[:len(force_mags_plot)]), label='|z_actual| (First Value)', marker='s', linestyle=':')
            plt.xlabel('Applied Force Magnitude (N)')
            plt.ylabel('|z| (First Value) (log scale)')
            plt.title('|z|, |z_scaled|, and |z_actual| vs. Applied Force Magnitude')
            plt.yscale('symlog')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "z_vs_force_magnitude_semilogy.png"))
            plt.close()


            print(f"Plots saved to {plot_dir}")
            print("Closing simulation")

        except ImportError:
            print("Warning: pandas not found. Cannot compute average results.")



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
    volume = config['material'].get('volume', 1) # Make sure this is in config or handled
    num_modes_to_show = config['model'].get('latent_dim', 5) # For controller, not directly used in scene
    total_mass = density * volume # Ensure volume is correctly sourced
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
    MO1 = exactSolution.addObject('MechanicalObject', name='MO1', template='Vec3d', src='@grid')
    
    # Add system components
    # mass = exactSolution.addObject('MeshMatrixMass', totalMass=total_mass, name="SparseMass", topology="@triangleTopo")
    
    # Get solver parameters from config
    rayleighStiffness = config['physics'].get('rayleigh_stiffness', 0.1)
    rayleighMass = config['physics'].get('rayleigh_mass', 0.1)
    
    solver = exactSolution.addObject('StaticSolver', name="ODEsolver", 
                                   newton_iterations=20,
                                   printLog=False)
    
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

    
    #NOTES: plot some relative errors between linear modes and fem divided by the norm of biggest displacement
    # 
    
    # Add visual model
    visual = exactSolution.addChild("visual")
    visual.addObject('MeshOBJLoader', name='surface_mesh', filename='mesh/bunny.obj')
    visual.addObject('OglModel', name='visual', src='@surface_mesh', color='0 1 0 1')
    visual.addObject('BarycentricMapping', input='@../MO1', output='@./visual')
    visual.addObject('VisualModelOBJExporter', filename="neuralModes-groundtruth", exportEveryNumberOfSteps=50) 

    # Add a second model beam with TetrahedronFEMForceField, which is linear
    # --- Add Linear Solution Node ---
    linearSolution = rootNode.addChild('LinearSolution', activated=True)
    linearSolution.addObject('MeshGmshLoader', name='grid', filename=mesh_filename)
    linearSolution.addObject('TetrahedronSetTopologyContainer', name='triangleTopo', src='@grid')
    MO2 = linearSolution.addObject('MechanicalObject', name='MO2', template='Vec3d', src='@grid') # Named MO2

    # Add system components (similar to exactSolution)
    linearSolution.addObject('StaticSolver', name="ODEsolver",
                           newton_iterations=20,
                           #absolute_residual_tolerance_threshold=1e-5,
                           #relative_residual_tolerance_threshold=1e-5,
                           printLog=False) # Maybe less logging for this one

    linearSolution.addObject('CGLinearSolver',
                           template="CompressedRowSparseMatrixMat3x3d",
                           iterations=config['physics'].get('solver_iterations', 1000),
                           tolerance=config['physics'].get('solver_tolerance', 1e-6),
                           threshold=config['physics'].get('solver_threshold', 1e-6),
                           warmStart=True)

    # Use the linear FEM force field
    linearFEM = linearSolution.addObject('TetrahedronFEMForceField', # Store reference
                           name="LinearFEM",
                           youngModulus=young_modulus,
                           poissonRatio=poisson_ratio,
                           method="small") # needs to be set to small for true linear solution 


    # Add constraints (same as exactSolution)
    linearSolution.addObject('BoxROI',
                           name='ROI',
                           box=" ".join(str(x) for x in fixed_box_coords),
                           drawBoxes=False) # Maybe hide this box
    linearSolution.addObject('FixedConstraint', indices="@ROI.indices")

    linearSolution.addObject('BoxROI',
                           name='ForceROI',
                           box=" ".join(str(x) for x in force_box_coords),
                           drawBoxes=False) # Maybe hide this box
    # Add a CFF to the linear model as well, controlled separately if needed, or linked
    # For now, just add it so the structure is parallel. It won't be actively controlled by the current controller.

    # Add visual model for the linear solution (optional, maybe different color)
    visual = linearSolution.addChild("visual")
    visual.addObject('VisualStyle', displayFlags='showWireframe')
    visual.addObject('MeshOBJLoader', name='surface_mesh', filename='mesh/bunny.obj')
    visual.addObject('OglModel', name='visual', src='@surface_mesh', color='0 0.6 0.95 1') # cyan color
    visual.addObject('BarycentricMapping', input='@../MO2', output='@./visual')
    # --- End Linear Solution Node ---


    # --- Add Node for Linear Modes Visualization Only ---
    linearModesViz = rootNode.addChild('LinearModesViz', activated=True)
    linearModesViz.addObject('MeshGmshLoader', name='grid', filename=mesh_filename)
    linearModesViz.addObject('TetrahedronSetTopologyContainer', name='topo', src='@grid')
    MO_LinearModes = linearModesViz.addObject('MechanicalObject', name='MO_LinearModes', template='Vec3d', src='@grid')

    # Add visual model for the reduced model 
    visualLinearModes = linearModesViz.addChild("visualLinearModes")
    visualLinearModes.addObject('VisualStyle', displayFlags='showWireframe')
    visual_LM = visualLinearModes.addObject('MeshOBJLoader', name='surface_mesh', filename='mesh/bunny.obj')
    visualLinearModes.addObject('OglModel', name='visual', src='@surface_mesh', color='1 0 0 1') # red color
    visualLinearModes.addObject('BarycentricMapping', input='@../MO_LinearModes', output='@./visual')
    # --- End Linear Modes Viz Node ---


    # --- Add Node for Neural Prediction Visualization Only ---
    neuralPredViz = rootNode.addChild('NeuralPredViz', activated=True)
    neuralPredViz.addObject('MeshGmshLoader', name='grid', filename=mesh_filename)
    neuralPredViz.addObject('TetrahedronSetTopologyContainer', name='topo', src='@grid')
    MO_NeuralPred = neuralPredViz.addObject('MechanicalObject', name='MO_NeuralPred', template='Vec3d', src='@grid')
    # Add visual model
    visualNeuralPred = neuralPredViz.addChild("visualNeuralPred")
    visualNeuralPred.addObject('MeshOBJLoader', name='surface_mesh', filename='mesh/bunny.obj')
    visual_NP = visualNeuralPred.addObject('OglModel', name='visual', src='@surface_mesh', color='1 0 1 1') # Magenta color
    visualNeuralPred.addObject('BarycentricMapping', input='@../MO_NeuralPred', output='@./visual')
    visualNeuralPred.addObject('VisualModelOBJExporter', filename="neuralModes-prediction", exportEveryNumberOfSteps=50) 
    # --- End Neural Pred Viz Node ---


    # Create and add controller with all components
    controller_kwargs = {
        'exactSolution': exactSolution, 'fem': fem, 'linear_solver': linear_solver,
        'surface_topo': surface_topo, 'MO1': MO1, 'fixed_box': fixed_box,
        'linearSolution': linearSolution, 'MO2': MO2, 'linearFEM': linearFEM,
        'MO_LinearModes': MO_LinearModes, 'MO_NeuralPred': MO_NeuralPred,
    #    'visual_LM': visual_LM, 'visual_NP': visual_NP, # Corrected visual names
        'directory': directory, 'sample': sample, 'key': key,
        'young_modulus': young_modulus, 'poisson_ratio': poisson_ratio,
        'density': density, 'volume': volume, 'total_mass': total_mass,
        'mesh_filename': mesh_filename, 'num_modes_to_show': num_modes_to_show,
        # Pass through kwargs from createScene call, which might include num_substeps, max_main_steps, max_z_amplitude_scale
    }
    controller_kwargs.update(kwargs) # Add kwargs passed to createScene

    controller = AnimationStepController(rootNode, **controller_kwargs)
    rootNode.addObject(controller)
    return rootNode, controller


if __name__ == "__main__":
    import Sofa.Gui
    from tqdm import tqdm
    import yaml
    import argparse # Keep argparse here
    import traceback
    import time

    parser = argparse.ArgumentParser(description='SOFA Validation with Modal Forces')
    parser.add_argument('--config', type=str, default='configs/paper.yaml', help='Path to config file')
    parser.add_argument('--gui', action='store_true', help='Enable GUI mode')
    parser.add_argument('--steps', type=int, default=None, help='Number of MAIN steps to run (overrides config)')
    parser.add_argument('--num-substeps', type=int, default=None, help='Number of substeps per main step (overrides config)')
    # Add new argument for max_z_amplitude_scale
    parser.add_argument('--max-z-scale', type=float, default=None, help='Max amplitude scale for z components (overrides config)')


    args = parser.parse_args() # Define args globally for onSimulationInitDoneEvent

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {args.config}")
    except Exception as e:
        print(f"Error loading config from {args.config}: {str(e)}. Using default configuration values where needed.")
        config = {} # Initialize empty config, rely on get() defaults

    max_main_steps = args.steps if args.steps is not None else config.get('simulation', {}).get('steps', 20)
    num_substeps = args.num_substeps if args.num_substeps is not None else config.get('physics', {}).get('num_substeps', 1)
    # Get max_z_amplitude_scale from args or config
    max_z_amplitude_scale_val = args.max_z_scale if args.max_z_scale is not None else config.get('simulation', {}).get('max_z_amplitude_scale', 1.0)

    required_plugins = [
        "Sofa.GL.Component.Rendering3D", "Sofa.GL.Component.Shader", "Sofa.Component.StateContainer",
        "Sofa.Component.ODESolver.Backward", "Sofa.Component.LinearSolver.Direct", "Sofa.Component.IO.Mesh",
        "Sofa.Component.MechanicalLoad", "Sofa.Component.Engine.Select", "Sofa.Component.SolidMechanics.FEM.Elastic",
        "MultiThreading", "SofaMatrix", "Sofa.Component.SolidMechanics.FEM.HyperElastic"
    ]
    for plugin in required_plugins: SofaRuntime.importPlugin(plugin)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    # Ensure output_dir is correctly formed if self.directory is used by controller for saving
    # The controller forms its own output_subdir based on self.directory.
    # The timestamp passed to createScene as 'directory' will be used by controller.

    root = Sofa.Core.Node("root")
    # Pass determined steps and new scale to createScene kwargs for the controller
    rootNode, controller = createScene(
        root,
        config=config,
        directory=timestamp, # This will be used by controller for its output_subdir
        sample=0,
        key=(0, 0, 0),
        num_substeps=num_substeps,      
        max_main_steps=max_main_steps,
        max_z_amplitude_scale=max_z_amplitude_scale_val # Pass the new parameter
    )

    # Initialize simulation
    Sofa.Simulation.init(root)
    # controller.save = True # This is already default true in controller

    if args.gui:
        print(f"Starting GUI mode. Max Z Scale: {max_z_amplitude_scale_val}, Substeps ({num_substeps}) managed by controller.")
        Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
        Sofa.Gui.GUIManager.createGUI(root, __file__)
        Sofa.Gui.GUIManager.SetDimension(1000, 800)
        Sofa.Gui.GUIManager.MainLoop(root) 
        Sofa.Gui.GUIManager.closeGUI()
    else:
        print(f"Starting headless mode. Max Z Scale: {max_z_amplitude_scale_val}, {max_main_steps} main steps with {num_substeps} substeps each.")
        root.animate = True
        step_count = 0
        max_total_iterations = max_main_steps * num_substeps * 1.2 # Safety limit
        pbar = tqdm(total=max_main_steps, desc="Main Steps Progress")
        last_main_step = -1

        while root.animate.value and step_count < max_total_iterations:
            Sofa.Simulation.animate(root, root.dt.value)
            step_count += 1
            if controller.current_main_step > last_main_step:
                 pbar.update(controller.current_main_step - last_main_step)
                 last_main_step = controller.current_main_step
        pbar.close()
        if step_count >= max_total_iterations:
            print("Warning: Reached maximum total iterations safety limit.")

    controller.close()
