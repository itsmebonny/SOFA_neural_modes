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

                # --- Linear Solution Components ---
        self.linearSolution = kwargs.get('linearSolution')
        self.MO2 = kwargs.get('MO2') # MechObj for linear
        self.linearFEM = kwargs.get('linearFEM') # Linear FEM ForceField
        self.cff_linear = None # Placeholder for linear solution CFF

        self.MO_LinearModes = kwargs.get('MO_LinearModes') # MechObj for Linear Modes Viz
        self.MO_NeuralPred = kwargs.get('MO_NeuralPred')   # MechObj for Neural Pred Viz
        self.visual_LM = kwargs.get('visual_LM') # Visual for Linear Modes
        self.visual_NP = kwargs.get('visual_NP') # Visual for Neural Pred



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
        self.target_force_magnitude = 1e6
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
            if self.MO2: self.MO2.position.value = rest_pos # Reset linear model 
            if self.MO_LinearModes: self.MO_LinearModes.position.value = rest_pos # Reset viz model
            if self.MO_NeuralPred: self.MO_NeuralPred.position.value = rest_pos # Reset viz model

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

        print(f"  Substep {substep_fraction}/{self.num_substeps}: Applying force = {incremental_force}")

        if self.cff is not None:
            try:
                self.exactSolution.removeObject(self.cff)
            except Exception as e:
                print(f"Warning: Error removing CFF (Exact): {e}")
            finally:
                 self.cff = None # Clear the reference
        # Linear Solution
        if self.cff_linear is not None:
            try:
                self.linearSolution.removeObject(self.cff_linear)
            except Exception as e:
                print(f"Warning: Error removing CFF (Linear): {e}")
            finally:
                 self.cff_linear = None # Clear the reference
        # --- End Remove CFFs ---


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

            # Linear Solution
            force_roi_linear = self.linearSolution.getObject('ForceROI')
            if force_roi_linear is None: raise ValueError("ForceROI (Linear) not found.")
            self.cff_linear = self.linearSolution.addObject('ConstantForceField',
                               name="CFF_Linear_Step",
                               indices="@ForceROI.indices",
                               totalForce=incremental_force.tolist(),
                               showArrowSize=0.0) # Hide arrow for linear model
            self.cff_linear.init()

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
        """
        Called by SOFA's animation loop after each physics step.
        Performs analysis and stores results for the completed substep.
        """
        try:
            current_force_magnitude = self.last_applied_force_magnitude # This should be set in onAnimateBeginEvent
            real_solution_disp = self.MO1.position.value.copy() - self.MO1.rest_position.value.copy()
            linear_solution_sofa_disp = self.MO2.position.value.copy() - self.MO2.rest_position.value.copy()
            real_energy = self.computeInternalEnergy(real_solution_disp)
            
            z = self.computeModalCoordinates(linear_solution_sofa_disp)
            print(f"  Modal coordinates (z) computed with shape {z.shape} for linear solution.")
            print(f"  Z-coordinates: {z}")
            if z is not None and not np.isnan(z).any():
                self.all_z_coords.append(z.copy())

            sofa_linear_energy = float('nan')
            linear_solution_sofa_reshaped = None
            if self.MO2 and self.linearFEM:
                sofa_linear_energy = self.computeInternalEnergy(linear_solution_sofa_disp)
                try:
                    linear_solution_sofa_reshaped = linear_solution_sofa_disp.reshape(self.MO1.position.value.shape[0], 3)
                except ValueError as e:
                    print(f"  Warning: Could not reshape SOFA linear solution: {e}")

            predicted_energy, linear_energy_modes = float('nan'), float('nan')
            l2_err_pred_real, rmse_pred_real, mse_pred_real = float('nan'), float('nan'), float('nan')
            l2_err_lin_real, rmse_lin_real, mse_lin_real = float('nan'), float('nan'), float('nan')
            l2_err_lin_sofa, rmse_lin_sofa, mse_lin_sofa = float('nan'), float('nan'), float('nan')
            l_th_reshaped_np, u_pred_reshaped_np, real_solution_reshaped = None, None, None

            if z is None or np.isnan(z).any():
                print(f"  Warning: NaN or None detected in modal coordinates (z). Skipping NN prediction.")
            else:
                z_th = torch.tensor(z, dtype=torch.float64, device=self.routine.device).unsqueeze(0)
                if z_th.shape[1] != self.routine.latent_dim:
                     print(f"Warning: z_th shape {z_th.shape} does not match routine.latent_dim {self.routine.latent_dim}. Adjusting z_th.")
                     if z_th.shape[1] > self.routine.latent_dim:
                         z_th = z_th[:, :self.routine.latent_dim]
                     else: 
                         padding = torch.zeros((1, self.routine.latent_dim - z_th.shape[1]), dtype=z_th.dtype, device=z_th.device)
                         z_th = torch.cat((z_th, padding), dim=1)

                modes_to_use = self.routine.linear_modes[:, :self.routine.latent_dim].to(self.routine.device)
                l_th = torch.matmul(modes_to_use, z_th.T).squeeze()
                with torch.no_grad():
                    y_th = self.routine.model(z_th).squeeze()
                u_pred_th = l_th + y_th
                num_nodes_mo1 = self.MO1.position.value.shape[0]  # Get num_nodes from MO1
                try:
                    l_th_reshaped = l_th.reshape(num_nodes_mo1, 3)
                    l_th_reshaped_np = l_th_reshaped.cpu().numpy()
                    u_pred_reshaped = u_pred_th.reshape(num_nodes_mo1, 3)
                    u_pred_reshaped_np = u_pred_reshaped.cpu().numpy()
                    real_solution_reshaped = real_solution_disp.reshape(num_nodes_mo1, 3)

                    # Compute energies
                    linear_energy_modes = self.computeInternalEnergy(l_th_reshaped_np)
                    predicted_energy = self.computeInternalEnergy(u_pred_reshaped_np)

                    # Compute errors with respect to the real solution
                    diff_pred_real = real_solution_reshaped - u_pred_reshaped_np
                    l2_err_pred_real = np.linalg.norm(diff_pred_real)
                    mse_pred_real = np.mean(diff_pred_real**2)
                    rmse_pred_real = np.sqrt(mse_pred_real)

                    diff_lin_real = real_solution_reshaped - l_th_reshaped_np
                    l2_err_lin_real = np.linalg.norm(diff_lin_real)
                    mse_lin_real = np.mean(diff_lin_real**2)
                    rmse_lin_real = np.sqrt(mse_lin_real)

                    if linear_solution_sofa_reshaped is not None:
                        diff_lin_sofa_real = real_solution_reshaped - linear_solution_sofa_reshaped
                        l2_err_lin_sofa_real = np.linalg.norm(diff_lin_sofa_real)
                        mse_lin_sofa_real = np.mean(diff_lin_sofa_real**2)
                        rmse_lin_sofa_real = np.sqrt(mse_lin_sofa_real)
                except (RuntimeError, ValueError) as e:
                    print(f"  Error during prediction processing/reshaping/error calc: {e}")

            F_real_np, F_lm_pred_np, F_nn_pred_np, F_sofa_linear_np = None, None, None, None
            norm_diff_F_lm, norm_diff_F_nn, norm_diff_F_sl = float('nan'), float('nan'), float('nan')

            if hasattr(self.routine, 'energy_calculator') and \
               self.routine.energy_calculator is not None and \
               hasattr(self.routine.energy_calculator, 'compute_deformation_gradients'):
                
                calc_F = self.routine.energy_calculator.compute_deformation_gradients
                device = self.routine.device
                dtype = torch.float64 if not hasattr(self.routine, 'dtype') else self.routine.dtype
                
                # Determine num_nodes and dim for reshaping
                # Assuming MO1 is representative for num_nodes
                num_nodes_for_grad = self.MO1.rest_position.value.shape[0]
                spatial_dim = 3 # Assuming 3D

                def get_F_from_disp(disp_np_array):
                    if disp_np_array is None: return None
                    disp_flat = disp_np_array.flatten()
                    disp_tensor = torch.tensor(disp_flat, device=device, dtype=dtype)
                    
                    # Reshape to  [num_nodes, dim]
                    try:
                        # disp_tensor is 1D (num_dofs,). Reshape to (num_nodes, dim).
                        disp_tensor_reshaped = disp_tensor.view(num_nodes_for_grad, spatial_dim)
                    except RuntimeError as e_reshape:
                        print(f"  Error reshaping displacement to ({num_nodes_for_grad}, {spatial_dim}): {e_reshape}")
                        print(f"  Original disp_tensor shape: {disp_tensor.shape}")
                        return None

                    # Add batch dimension for calc_F, as it likely expects (batch_size, num_nodes, dim)
                    F_tensor = calc_F(disp_tensor_reshaped)
                    if F_tensor.dim() == 4 and F_tensor.shape[0] == 1: # If calc_F returned batched output
                        F_tensor = F_tensor.squeeze(0) # Remove batch dimension from F_tensor for consistency
                    return F_tensor.cpu().numpy()

                try:
                    F_real_np = get_F_from_disp(real_solution_disp)

                    if l_th_reshaped_np is not None and F_real_np is not None:
                        F_lm_pred_np = get_F_from_disp(l_th_reshaped_np)
                        if F_lm_pred_np is not None and F_lm_pred_np.shape == F_real_np.shape:
                            diff_F_lm = F_real_np - F_lm_pred_np
                            norm_diff_F_lm = np.mean(np.linalg.norm(diff_F_lm, ord='fro', axis=(-2,-1)))
                        elif F_lm_pred_np is not None: print(f"Shape mismatch F_real vs F_lm_pred: {F_real_np.shape} vs {F_lm_pred_np.shape}")
                    
                    if u_pred_reshaped_np is not None and F_real_np is not None:
                        F_nn_pred_np = get_F_from_disp(u_pred_reshaped_np)
                        if F_nn_pred_np is not None and F_nn_pred_np.shape == F_real_np.shape:
                            diff_F_nn = F_real_np - F_nn_pred_np
                            norm_diff_F_nn = np.mean(np.linalg.norm(diff_F_nn, ord='fro', axis=(-2,-1)))
                        elif F_nn_pred_np is not None: print(f"Shape mismatch F_real vs F_nn_pred: {F_real_np.shape} vs {F_nn_pred_np.shape}")
                    
                    if linear_solution_sofa_disp is not None and F_real_np is not None:
                        F_sofa_linear_np = get_F_from_disp(linear_solution_sofa_disp)
                        if F_sofa_linear_np is not None and F_sofa_linear_np.shape == F_real_np.shape:
                            diff_F_sl = F_real_np - F_sofa_linear_np
                            norm_diff_F_sl = np.mean(np.linalg.norm(diff_F_sl, ord='fro', axis=(-2,-1)))
                        elif F_sofa_linear_np is not None: print(f"Shape mismatch F_real vs F_sofa_linear: {F_real_np.shape} vs {F_sofa_linear_np.shape}")
                except Exception as e_fgrad:
                    print(f"  ERROR calculating deformation gradients or their differences: {e_fgrad}")
                    traceback.print_exc()
            else:
                print("  Skipping deformation gradient calculation: energy_calculator or method not available.")

            self.grad_diff_lin_modes_list.append(norm_diff_F_lm)
            self.grad_diff_nn_pred_list.append(norm_diff_F_nn)
            self.grad_diff_sofa_linear_list.append(norm_diff_F_sl)
            
            if self.original_positions is not None:
                rest_pos = self.original_positions
                if self.MO_LinearModes is not None and l_th_reshaped_np is not None:
                    self.MO_LinearModes.position.value = rest_pos + l_th_reshaped_np
                if self.MO_NeuralPred is not None and u_pred_reshaped_np is not None:
                    self.MO_NeuralPred.position.value = rest_pos + u_pred_reshaped_np
            
            self.substep_results.append((
                current_force_magnitude, real_energy, predicted_energy, linear_energy_modes, sofa_linear_energy,
                l2_err_pred_real, rmse_pred_real, mse_pred_real,
                l2_err_lin_real, rmse_lin_real, mse_lin_real,
                l2_err_lin_sofa_real, rmse_lin_sofa_real, mse_lin_sofa_real ))
            

        except Exception as e:
            print(f"ERROR during analysis in onAnimateEndEvent: {e}")
            traceback.print_exc()
            # Ensure lists are appended to even on error to maintain consistent lengths
            self.substep_results.append((
                self.last_applied_force_magnitude, 
                float('nan'), float('nan'), float('nan'), float('nan'), 
                float('nan'), float('nan'), float('nan'), 
                float('nan'), float('nan'), float('nan'),  
                float('nan'), float('nan'), float('nan')  
            ))
            self.grad_diff_lin_modes_list.append(float('nan'))
            self.grad_diff_nn_pred_list.append(float('nan'))
            self.grad_diff_sofa_linear_list.append(float('nan'))

        self.current_substep += 1
        if (self.current_substep % self.num_substeps) == 0:
            self.current_main_step += 1
            print(f"--- Main Step {self.current_main_step} Completed (Total Substeps: {self.current_substep}) ---")
            # Access args globally if __main__ defines it
            if 'args' in globals() and not args.gui and self.current_main_step >= self.max_main_steps:
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
        if not self.substep_results:
            print("No results collected. Skipping analysis and plotting.")
            return

        result_columns = [
            'ForceMag', 'RealE', 'PredE', 'LinearModesE', 'SOFALinearE',
            'L2Err_Pred_Real', 'RMSE_Pred_Real', 'MSE_Pred_Real',
            'L2Err_Lin_Real', 'RMSE_Lin_Real', 'MSE_Lin_Real',
            'L2Err_SOFALin_Real', 'RMSE_SOFALin_Real', 'MSE_SOFALin_Real'
        ]
        try:
            import pandas as pd
            df = pd.DataFrame(self.substep_results, columns=result_columns)

            # Add new columns for gradient differences
            # Ensure lengths match, especially if simulation ended early or with errors
            num_entries_df = len(df)
            df['GradDiff_LM'] = pd.Series(self.grad_diff_lin_modes_list[:num_entries_df])
            df['GradDiff_NN'] = pd.Series(self.grad_diff_nn_pred_list[:num_entries_df])
            df['GradDiff_SL'] = pd.Series(self.grad_diff_sofa_linear_list[:num_entries_df])
            
            avg_results = df.groupby('ForceMag').mean().reset_index()
            avg_results = avg_results.sort_values(by='ForceMag')

            print("\n--- Average Results per Force Magnitude ---")
            cols_to_print = ['ForceMag', 'RealE', 'PredE', 'LinearModesE', 'SOFALinearE',
                             'RMSE_Pred_Real', 'MSE_Pred_Real',
                             'RMSE_Lin_Real', 'MSE_Lin_Real',
                             'RMSE_SOFALin_Real', 'MSE_SOFALin_Real',
                             'GradDiff_LM', 'GradDiff_NN', 'GradDiff_SL'] # Added new cols
            
            # Filter out columns that might not exist if all values were NaN (though groupby().mean() should handle NaNs)
            cols_to_print_existing = [col for col in cols_to_print if col in avg_results.columns]
            print(avg_results[cols_to_print_existing].to_string(index=False, float_format="%.4e"))
            print("-------------------------------------------\n")

            force_mags_plot = avg_results['ForceMag'].values
            avg_real_e = avg_results['RealE'].values
            avg_pred_e = avg_results['PredE'].values
            avg_linear_modes_e = avg_results['LinearModesE'].values
            avg_sofa_linear_e = avg_results['SOFALinearE'].values
            avg_rmse_pred_real = avg_results['RMSE_Pred_Real'].values
            avg_mse_pred_real = avg_results['MSE_Pred_Real'].values
            avg_rmse_lin_real = avg_results['RMSE_Lin_Real'].values
            avg_mse_lin_real = avg_results['MSE_Lin_Real'].values
            avg_rmse_lin_sofa = avg_results['RMSE_SOFALin_Real'].values
            avg_mse_lin_sofa = avg_results['MSE_SOFALin_Real'].values
            
            # Extract new averaged gradient differences
            avg_grad_diff_lm = avg_results['GradDiff_LM'].values if 'GradDiff_LM' in avg_results else np.full_like(force_mags_plot, float('nan'))
            avg_grad_diff_nn = avg_results['GradDiff_NN'].values if 'GradDiff_NN' in avg_results else np.full_like(force_mags_plot, float('nan'))
            avg_grad_diff_sl = avg_results['GradDiff_SL'].values if 'GradDiff_SL' in avg_results else np.full_like(force_mags_plot, float('nan'))

            plot_dir = self.output_subdir if self.save else "."
            if self.save and not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            # ... (Existing Energy, RMSE, MSE plots remain the same) ...
            # 1. Average Energy vs. Force Magnitude Plot (Linear Scale)
            plt.figure(figsize=(10, 6))
            plt.plot(force_mags_plot, avg_real_e, label='Avg Real Energy (SOFA Hyperelastic)', marker='o', linestyle='-')
            plt.plot(force_mags_plot, avg_pred_e, label='Avg Predicted Energy (l+y)', marker='x', linestyle='--')
            plt.plot(force_mags_plot, avg_linear_modes_e, label='Avg Linear Modes Energy (l)', marker='s', linestyle=':')
            plt.plot(force_mags_plot, avg_sofa_linear_e, label='Avg SOFA Linear Energy', marker='d', linestyle='-.')
            plt.xlabel('Applied Force Magnitude'); plt.ylabel('Average Internal Energy')
            plt.title('Average Energy vs. Applied Force Magnitude'); plt.legend(); plt.grid(True); plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "avg_energy_vs_force.png")); plt.close()

            # 1b. Average Energy vs. Force Magnitude Plot (Log Scale)
            plt.figure(figsize=(10, 6))
            valid_indices_real = avg_real_e > 0; valid_indices_pred = avg_pred_e > 0
            valid_indices_linear_modes = avg_linear_modes_e > 0; valid_indices_sofa_linear = avg_sofa_linear_e > 0
            if np.any(valid_indices_real): plt.plot(force_mags_plot[valid_indices_real], avg_real_e[valid_indices_real], label='Avg Real Energy', marker='o')
            if np.any(valid_indices_pred): plt.plot(force_mags_plot[valid_indices_pred], avg_pred_e[valid_indices_pred], label='Avg Predicted Energy', marker='x', linestyle='--')
            if np.any(valid_indices_linear_modes): plt.plot(force_mags_plot[valid_indices_linear_modes], avg_linear_modes_e[valid_indices_linear_modes], label='Avg Linear Modes Energy', marker='s', linestyle=':')
            if np.any(valid_indices_sofa_linear): plt.plot(force_mags_plot[valid_indices_sofa_linear], avg_sofa_linear_e[valid_indices_sofa_linear], label='Avg SOFA Linear Energy', marker='d', linestyle='-.')
            plt.xlabel('Applied Force Magnitude'); plt.ylabel('Average Internal Energy (log scale)')
            plt.title('Average Energy vs. Applied Force Magnitude (Log Scale)'); plt.yscale('log')
            plt.legend(); plt.grid(True, which="both", ls="--"); plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "avg_energy_vs_force_log.png")); plt.close()

            # 2. RMSE Errors vs Force Magnitude
            plt.figure(figsize=(10, 6))
            plt.plot(force_mags_plot, avg_rmse_pred_real, label='RMSE: Pred (l+y) vs Real (MO1)', marker='^')
            plt.plot(force_mags_plot, avg_rmse_lin_real, label='RMSE: LinModes (l) vs Real (MO1)', marker='v', linestyle='--')
            plt.plot(force_mags_plot, avg_rmse_lin_sofa, label='RMSE: SOFALin (MO2) vs Real (MO1)', marker='<', linestyle=':')
            plt.xlabel('Applied Force Magnitude'); plt.ylabel('Average RMSE')
            plt.title('Average RMSE vs. Applied Force Magnitude'); plt.legend(); plt.grid(True); plt.yscale('log'); plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "avg_rmse_vs_force.png")); plt.close()

            # 3. MSE Errors vs Force Magnitude
            plt.figure(figsize=(10, 6))
            plt.plot(force_mags_plot, avg_mse_pred_real, label='MSE: Pred (l+y) vs Real (MO1)', marker='^')
            plt.plot(force_mags_plot, avg_mse_lin_real, label='MSE: LinModes (l) vs Real (MO1)', marker='v', linestyle='--')
            plt.plot(force_mags_plot, avg_mse_lin_sofa, label='MSE: SOFALin (MO2) vs Real (MO1)', marker='<', linestyle=':')
            plt.xlabel('Applied Force Magnitude'); plt.ylabel('Average MSE')
            plt.title('Average MSE vs. Applied Force Magnitude'); plt.legend(); plt.grid(True); plt.yscale('log'); plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "avg_mse_vs_force.png")); plt.close()

            # --- New Plot for Deformation Gradient Differences ---
            plt.figure(figsize=(12, 7))
            if not np.all(np.isnan(avg_grad_diff_lm)): # Check if there's any valid data to plot
                 plt.plot(force_mags_plot, avg_grad_diff_lm, label='Avg ||F_real - F_LMpred||', marker='o', linestyle='-')
            if not np.all(np.isnan(avg_grad_diff_nn)):
                 plt.plot(force_mags_plot, avg_grad_diff_nn, label='Avg ||F_real - F_NNpred||', marker='x', linestyle='--')
            if not np.all(np.isnan(avg_grad_diff_sl)):
                 plt.plot(force_mags_plot, avg_grad_diff_sl, label='Avg ||F_real - F_SOFALinear||', marker='s', linestyle=':')
            
            plt.xlabel('Applied Force Magnitude (N)')
            plt.ylabel('Avg. Frobenius Norm Diff. of Def. Gradients')
            plt.title('Average Deformation Gradient Difference vs. Applied Force')
            plt.legend()
            plt.grid(True, which="both", ls="--")
            plt.yscale('log') # Log scale often useful for error metrics
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "avg_grad_diff_vs_force_magnitude.png"))
            plt.close()
            print(f"Deformation gradient difference plot saved to {os.path.join(plot_dir, 'avg_grad_diff_vs_force_magnitude.png')}")
            # --- End New Plot ---

            print(f"All plots saved to {plot_dir}")

        except ImportError:
            print("Warning: pandas not found. Cannot compute average results or plot.")
        except Exception as e_close:
            print(f"Error during close method processing: {e_close}")
            traceback.print_exc()
        finally:
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

    # Add a second model beam with TetrahedronFEMForceField, which is linear
    # --- Add Linear Solution Node ---
    linearSolution = rootNode.addChild('LinearSolution', activated=True)
    linearSolution.addObject('MeshGmshLoader', name='grid', filename=mesh_filename)
    linearSolution.addObject('TetrahedronSetTopologyContainer', name='triangleTopo', src='@grid')
    MO2 = linearSolution.addObject('MechanicalObject', name='MO2', template='Vec3d', src='@grid') # Named MO2

    # Add system components (similar to exactSolution)
    linearSolution.addObject('StaticSolver', name="ODEsolver",
                           newton_iterations=20,
                           printLog=True) # Maybe less logging for this one

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
                           method="small") 

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
    linearSolution.addObject('ConstantForceField', indices="@ForceROI.indices", totalForce=[0, 0, 0], showArrowSize=0.0)

    # Add visual model for the linear solution (optional, maybe different color)
    visualLinear = linearSolution.addChild("visualLinear")
    visualLinear.addObject('OglModel', src='@../MO2', color='0 0 1 1') # Blue color
    visualLinear.addObject('BarycentricMapping', input='@../MO2', output='@./')
    # --- End Linear Solution Node ---


    # --- Add Node for Linear Modes Visualization Only ---
    linearModesViz = rootNode.addChild('LinearModesViz', activated=True)
    linearModesViz.addObject('MeshGmshLoader', name='grid', filename=mesh_filename)
    linearModesViz.addObject('TetrahedronSetTopologyContainer', name='topo', src='@grid')
    MO_LinearModes = linearModesViz.addObject('MechanicalObject', name='MO_LinearModes', template='Vec3d', src='@grid')
    # Add visual model
    visualLinearModes = linearModesViz.addChild("visualLinearModes")
    visualLinearModes.addObject('OglModel', src='@../MO_LinearModes', color='1 1 0 1') # Yellow color
    visualLinearModes.addObject('BarycentricMapping', input='@../MO_LinearModes', output='@./')
    # --- End Linear Modes Viz Node ---


    # --- Add Node for Neural Prediction Visualization Only ---
    neuralPredViz = rootNode.addChild('NeuralPredViz', activated=True)
    neuralPredViz.addObject('MeshGmshLoader', name='grid', filename=mesh_filename)
    neuralPredViz.addObject('TetrahedronSetTopologyContainer', name='topo', src='@grid')
    MO_NeuralPred = neuralPredViz.addObject('MechanicalObject', name='MO_NeuralPred', template='Vec3d', src='@grid')
    # Add visual model
    visualNeuralPred = neuralPredViz.addChild("visualNeuralPred")
    visualNeuralPred.addObject('OglModel', src='@../MO_NeuralPred', color='1 0 1 1') # Magenta color
    visualNeuralPred.addObject('BarycentricMapping', input='@../MO_NeuralPred', output='@./')
    # --- End Neural Pred Viz Node ---


    # Create and add controller with all components
    controller = AnimationStepController(rootNode,
                                        exactSolution=exactSolution,
                                        fem=fem, # Hyperelastic FEM
                                        linear_solver=linear_solver,
                                        surface_topo=surface_topo,
                                        MO1=MO1, # Real SOFA solution
                                        fixed_box=fixed_box,
                                        linearSolution=linearSolution, # Pass linear node
                                        MO2=MO2, # SOFA Linear MechObj
                                        linearFEM=linearFEM, # Pass linear FEM FF
                                        MO_LinearModes=MO_LinearModes, # Pass Linear Modes Viz MechObj
                                        MO_NeuralPred=MO_NeuralPred,   # Pass Neural Pred Viz MechObj
                                        visualLinearModes=visualLinearModes, # Pass Linear Modes Viz
                                        visualNeuralPred=visualNeuralPred, # Pass Neural Pred Viz
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