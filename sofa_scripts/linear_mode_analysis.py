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
np.random.seed(0)  # Set seed for reproducibility
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



        self.MO_LinearModes = kwargs.get('MO_LinearModes') # MechObj for Linear Modes Viz
        self.visual_LM = kwargs.get('visual_LM') # Visual for Linear Modes



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
        self.target_force_magnitude = 1e7
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

        # --- For Reconstruction Analysis ---
        self.perform_reconstruction_analysis = kwargs.get('perform_reconstruction_analysis', True)
        self.max_modes_reconstruction = kwargs.get('num_modes_to_show', 5)
        self.reconstruction_analysis_data = [] # Stores dicts: {'ForceMag': fm, 'NumModes_k': k, 'RMSE_Reconstruction': rmse}
        # --- End Reconstruction Analysis Init ---


      


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

        # Extract necessary data from Routine instance
        self.linear_modes = self.routine.linear_modes # This should be a torch tensor
        self.routine.latent_dim 
        self.original_positions = np.copy(self.MO1.position.value) # Store original positions

        # --- Prepare for Saving (if enabled) ---
        # Use the directory name passed during scene creation or default
        self.directory = str(self.routine.latent_dim) + "_modes"

        output_base_dir = 'linear_analysis' # Or read from config
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
            if self.MO_LinearModes: self.MO_LinearModes.position.value = rest_pos # Reset viz model

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
        self.last_applied_force_magnitude = current_force_magnitude

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
            real_energy = self.computeInternalEnergy(real_solution_disp)
            
            # --- Primary Modal Analysis (using all routine.latent_dim modes) ---
            z = self.computeModalCoordinates(real_solution_disp) # This uses all modes in self.routine.linear_modes
            if z is not None and not np.isnan(z).any():
                self.all_z_coords.append(z.copy())

            linear_energy_modes = float('nan')
            l2_err_lin_real, rmse_lin_real, mse_lin_real = float('nan'), float('nan'), float('nan')
            l_th_reshaped_np, real_solution_reshaped = None, None

            if z is None or np.isnan(z).any():
                print(f"  Warning: NaN or None detected in modal coordinates (z) for primary analysis.")
            else:
                # Ensure z matches routine.latent_dim for the primary l_th calculation
                # computeModalCoordinates projects onto ALL modes in self.routine.linear_modes.
                # For l_th, we typically use up to routine.latent_dim, which might be less than total modes.
                z_for_l_th = z 
                if len(z_for_l_th) > self.routine.latent_dim:
                    z_for_l_th = z_for_l_th[:self.routine.latent_dim]
                elif len(z_for_l_th) < self.routine.latent_dim:
                    # This case should ideally not happen if computeModalCoordinates uses linear_modes up to latent_dim
                    # OR if self.routine.linear_modes itself is already truncated to latent_dim.
                    # Assuming self.routine.linear_modes has at least latent_dim columns.
                    print(f"  Warning: z_for_l_th length {len(z_for_l_th)} is less than latent_dim {self.routine.latent_dim}. Padding with zeros for l_th.")
                    z_primary_padded = np.zeros(self.routine.latent_dim)
                    z_primary_padded[:len(z_for_l_th)] = z_for_l_th
                    z_for_l_th = z_primary_padded
                
                # Convert z_for_l_th to tensor for matmul
                z_th_tensor = torch.tensor(z_for_l_th, dtype=torch.float64, device=self.routine.device) # Shape [latent_dim]
                
                # Use modes up to latent_dim for the primary linear reconstruction l_th
                # Assuming self.routine.linear_modes is [num_dofs, total_modes_available]
                modes_for_l_th_torch = self.routine.linear_modes[:, :self.routine.latent_dim].to(self.routine.device) # Shape [num_dofs, latent_dim]
                
                l_th = torch.matmul(modes_for_l_th_torch, z_th_tensor).squeeze() # Shape [num_dofs]

                num_nodes_mo1 = self.MO1.position.value.shape[0]
                try:
                    l_th_reshaped = l_th.reshape(num_nodes_mo1, 3)
                    l_th_reshaped_np = l_th_reshaped.cpu().numpy()
                    real_solution_reshaped = real_solution_disp.reshape(num_nodes_mo1, 3) # Reshape real solution here
                    linear_energy_modes = self.computeInternalEnergy(l_th_reshaped_np)
                    diff_lin_real = real_solution_reshaped - l_th_reshaped_np
                    l2_err_lin_real = np.linalg.norm(diff_lin_real)
                    mse_lin_real = np.mean(diff_lin_real**2)
                    rmse_lin_real = np.sqrt(mse_lin_real)
                except (RuntimeError, ValueError) as e:
                    print(f"  Error during primary linear reconstruction processing: {e}")
            # --- End Primary Modal Analysis ---

            # --- Displacement Reconstruction Analysis with Varying Number of Modes ---
            if self.perform_reconstruction_analysis and hasattr(self.routine, 'linear_modes') and self.routine.linear_modes is not None:
                if isinstance(self.routine.linear_modes, torch.Tensor):
                    linear_modes_np_all = self.routine.linear_modes.cpu().numpy() # All available modes
                else: # Assuming it's already NumPy
                    linear_modes_np_all = self.routine.linear_modes 

                M_sparse = self.routine.M if hasattr(self.routine, 'M') else None
                num_total_available_modes = linear_modes_np_all.shape[1]
                
                # Ensure real_solution_reshaped is available for RMSE calculation
                if real_solution_reshaped is None:
                    try:
                        real_solution_reshaped = real_solution_disp.reshape(self.MO1.position.value.shape[0], 3)
                    except ValueError as e_reshape_real:
                        print(f"  Error reshaping real_solution_disp for reconstruction analysis: {e_reshape_real}. Skipping reconstruction.")
                        # Skip reconstruction for this step if real solution can't be reshaped
                        linear_modes_np_all = None # To skip the loop below

                if linear_modes_np_all is not None: # Proceed only if modes and reshaped real solution are ready
                    max_k_to_consider = min(num_total_available_modes, self.max_modes_reconstruction)
                    
                    k_values_to_test = sorted(list(set(
                        [1] + \
                        list(range(5, max_k_to_consider + 1, 5)) + \
                        ([max_k_to_consider] if max_k_to_consider not in list(range(5, max_k_to_consider + 1, 5)) and max_k_to_consider != 1 else [])
                    )))
                    # Ensure k_values are valid and do not exceed available modes
                    k_values_to_test = [k_val for k_val in k_values_to_test if 0 < k_val <= num_total_available_modes]


                    real_solution_flat = real_solution_disp.flatten()

                    for k_modes in k_values_to_test:
                        modes_k = linear_modes_np_all[:, :k_modes] # Select first k modes

                        # Compute modal coordinates z_k for these k modes
                        z_k_reconstruction = np.zeros(k_modes) # Initialize
                        if M_sparse is not None and isinstance(M_sparse, sparse.spmatrix):
                            try:
                                modes_t_m_k = modes_k.T @ M_sparse
                                z_k_reconstruction = modes_t_m_k @ real_solution_flat
                            except Exception as e_proj:
                                print(f"  Error projecting with M for k={k_modes} in reconstruction: {e_proj}. Using simple projection.")
                                z_k_reconstruction = modes_k.T @ real_solution_flat # Fallback
                        else: 
                            try:
                                z_k_reconstruction = modes_k.T @ real_solution_flat
                            except ValueError as e_simple_proj_k:
                                print(f"  Error during simple projection for k={k_modes}: {e_simple_proj_k}")
                                print(f"  Shapes - modes_k.T: {modes_k.T.shape}, real_solution_flat: {real_solution_flat.shape}")
                                continue # Skip this k_modes if projection fails
                        
                        u_reconstructed_k_flat = modes_k @ z_k_reconstruction
                        try:
                            u_reconstructed_k_reshaped = u_reconstructed_k_flat.reshape(self.MO1.position.value.shape[0], 3)
                            diff_reconstruction = real_solution_reshaped - u_reconstructed_k_reshaped
                            mse_reconstruction = np.mean(diff_reconstruction**2)
                            rmse_reconstruction = np.sqrt(mse_reconstruction)

                            self.reconstruction_analysis_data.append({
                                'ForceMag': current_force_magnitude,
                                'NumModes_k': k_modes,
                                'RMSE_Reconstruction': rmse_reconstruction
                            })
                        except ValueError as e_reshape_rec:
                            print(f"  Error reshaping reconstructed displacement for k={k_modes}: {e_reshape_rec}")
            # --- End Reconstruction Analysis ---
            
            if self.original_positions is not None:
                rest_pos = self.original_positions
                if self.MO_LinearModes is not None and l_th_reshaped_np is not None:
                    self.MO_LinearModes.position.value = rest_pos + l_th_reshaped_np
            
            self.substep_results.append((
                current_force_magnitude, real_energy, linear_energy_modes,
                l2_err_lin_real, rmse_lin_real, mse_lin_real))
            
        except Exception as e:
            print(f"ERROR during analysis in onAnimateEndEvent: {e}")
            traceback.print_exc()
            self.substep_results.append((
                self.last_applied_force_magnitude, 
                float('nan'), float('nan'),
                float('nan'), float('nan'), float('nan')
            ))

        self.current_substep += 1
        if (self.current_substep % self.num_substeps) == 0:
            self.current_main_step += 1
            print(f"--- Main Step {self.current_main_step} Completed (Total Substeps: {self.current_substep}) ---")
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
            print("No results collected for primary analysis. Skipping analysis and plotting.")
            # Still check if reconstruction data exists, as it might have been collected
            # even if primary substep_results are empty (though unlikely with current flow)
        else:
            try:
                import pandas as pd
                result_columns = [
                    'ForceMag', 'RealE', 'LinearModesE',
                    'L2Err_Lin_Real', 'RMSE_Lin_Real', 'MSE_Lin_Real'
                ]
                df = pd.DataFrame(self.substep_results, columns=result_columns)

                avg_results = df.groupby('ForceMag').mean().reset_index()
                avg_results = avg_results.sort_values(by='ForceMag')

                print("\n--- Average Results per Force Magnitude (Primary Analysis) ---")
                cols_to_print = [
                    'ForceMag', 'RealE', 'LinearModesE',
                    'RMSE_Lin_Real', 'MSE_Lin_Real'
                ]
                cols_to_print_existing = [col for col in cols_to_print if col in avg_results.columns]
                print(avg_results[cols_to_print_existing].to_string(index=False, float_format="%.4e"))
                print("--------------------------------------------------------------\n")

                force_mags_plot = avg_results['ForceMag'].values
                avg_real_e = avg_results['RealE'].values
                avg_linear_modes_e = avg_results['LinearModesE'].values
                avg_rmse_lin_real = avg_results['RMSE_Lin_Real'].values
                avg_mse_lin_real = avg_results['MSE_Lin_Real'].values

                plot_dir = self.output_subdir if self.save else "."
                if self.save and not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)

                num_modes_primary_analysis = None
                if hasattr(self, 'routine') and hasattr(self.routine, 'latent_dim'):
                    num_modes_primary_analysis = self.routine.latent_dim
                
                num_modes_str_primary = f" (Primary Lin. Recon. using {num_modes_primary_analysis} modes)" if num_modes_primary_analysis is not None else ""


                # 1. Average Energy vs. Force Magnitude Plot (Linear Scale)
                plt.figure(figsize=(10, 6))
                plt.plot(force_mags_plot, avg_real_e, label='Avg Real Energy (SOFA Hyperelastic)', marker='o', linestyle='-')
                plt.plot(force_mags_plot, avg_linear_modes_e, label=f'Avg Linear Modes Energy (l, {num_modes_primary_analysis} modes)', marker='s', linestyle=':')
                plt.xlabel('Applied Force Magnitude'); plt.ylabel('Average Internal Energy')
                plt.title(f'Average Energy vs. Applied Force Magnitude{num_modes_str_primary}')
                plt.legend(); plt.grid(True); plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, "avg_energy_vs_force.png")); plt.close()

                # 1b. Average Energy vs. Force Magnitude Plot (Log Scale)
                plt.figure(figsize=(10, 6))
                valid_indices_real = avg_real_e > 0; 
                valid_indices_linear_modes = avg_linear_modes_e > 0; 
                if np.any(valid_indices_real): plt.plot(force_mags_plot[valid_indices_real], avg_real_e[valid_indices_real], label='Avg Real Energy', marker='o')
                if np.any(valid_indices_linear_modes): plt.plot(force_mags_plot[valid_indices_linear_modes], avg_linear_modes_e[valid_indices_linear_modes], label=f'Avg Linear Modes Energy (l, {num_modes_primary_analysis} modes)', marker='s', linestyle=':')
                plt.xlabel('Applied Force Magnitude'); plt.ylabel('Average Internal Energy (log scale)')
                plt.title(f'Average Energy vs. Applied Force Magnitude (Log Scale){num_modes_str_primary}')
                plt.yscale('log')
                plt.legend(); plt.grid(True, which="both", ls="--"); plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, "avg_energy_vs_force_log.png")); plt.close()

                # 2. RMSE Errors vs Force Magnitude
                plt.figure(figsize=(10, 6))
                plt.plot(force_mags_plot, avg_rmse_lin_real, label=f'RMSE: LinModes (l, {num_modes_primary_analysis} modes) vs Real (MO1)', marker='v', linestyle='--')
                plt.xlabel('Applied Force Magnitude'); plt.ylabel('Average RMSE')
                plt.title(f'Average RMSE vs. Applied Force Magnitude{num_modes_str_primary}')
                plt.legend(); plt.grid(True); plt.yscale('log'); plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, "avg_rmse_vs_force.png")); plt.close()

                # 3. MSE Errors vs Force Magnitude
                plt.figure(figsize=(10, 6))
                plt.plot(force_mags_plot, avg_mse_lin_real, label=f'MSE: LinModes (l, {num_modes_primary_analysis} modes) vs Real (MO1)', marker='v', linestyle='--')
                plt.xlabel('Applied Force Magnitude'); plt.ylabel('Average MSE')
                plt.title(f'Average MSE vs. Applied Force Magnitude{num_modes_str_primary}')
                plt.legend(); plt.grid(True); plt.yscale('log'); plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, "avg_mse_vs_force.png")); plt.close()
                
                print(f"Primary analysis plots saved to {plot_dir}")

            except ImportError:
                print("Warning: pandas not found. Cannot compute average results or plot for primary analysis.")
            except Exception as e_close_primary:
                print(f"Error during primary analysis plotting in close method: {e_close_primary}")
                traceback.print_exc()

        # --- Plotting for Reconstruction Analysis ---
        if self.perform_reconstruction_analysis and self.reconstruction_analysis_data:
            try:
                # Ensure pandas is imported if not already done by primary analysis
                if 'pd' not in locals(): import pandas as pd

                df_reconstruction = pd.DataFrame(self.reconstruction_analysis_data)
                
                if not df_reconstruction.empty:
                    # Average RMSE for each number of modes used in reconstruction, across all forces
                    avg_rmse_per_k = df_reconstruction.groupby('NumModes_k')['RMSE_Reconstruction'].mean().reset_index()
                    avg_rmse_per_k = avg_rmse_per_k.sort_values(by='NumModes_k')

                    # Get total number of available modes for plot title
                    total_modes_available_str = ""
                    if hasattr(self, 'routine') and hasattr(self.routine, 'linear_modes') and self.routine.linear_modes is not None:
                        if isinstance(self.routine.linear_modes, torch.Tensor):
                            total_modes_available_str = f" (Total Avail. Modes: {self.routine.linear_modes.shape[1]})"
                        elif isinstance(self.routine.linear_modes, np.ndarray):
                             total_modes_available_str = f" (Total Avail. Modes: {self.routine.linear_modes.shape[1]})"


                    plt.figure(figsize=(10, 6))
                    plt.plot(avg_rmse_per_k['NumModes_k'], avg_rmse_per_k['RMSE_Reconstruction'], marker='o', linestyle='-')
                    plt.xlabel('Number of Modes (k) for Reconstruction')
                    plt.ylabel('Average RMSE of Displacement Reconstruction')
                    plt.title(f'Reconstruction Accuracy vs. Number of Modes Used{total_modes_available_str}')
                    plt.grid(True, which="both", ls="--")
                    plt.yscale('log') 
                    plt.tight_layout()
                    
                    # Ensure plot_dir is defined (it would be if primary analysis ran)
                    # If primary analysis didn't run, define it here.
                    if 'plot_dir' not in locals():
                        plot_dir = self.output_subdir if self.save else "."
                        if self.save and not os.path.exists(plot_dir):
                            os.makedirs(plot_dir)

                    plt.savefig(os.path.join(plot_dir, "avg_reconstruction_rmse_vs_num_modes.png"))
                    plt.close()
                    print(f"Reconstruction analysis plot saved to {plot_dir}")

                    print("\n--- Average Reconstruction RMSE per Number of Modes ---")
                    print(avg_rmse_per_k.to_string(index=False, float_format="%.4e"))
                    print("-------------------------------------------------------\n")
                else:
                    print("No data for reconstruction analysis plotting.")

            except ImportError:
                 print("Warning: pandas not found. Cannot process or plot reconstruction analysis data.")
            except Exception as e_close_reconstruction:
                print(f"Error during reconstruction analysis plotting in close method: {e_close_reconstruction}")
                traceback.print_exc()
        elif self.perform_reconstruction_analysis:
            print("Reconstruction analysis was enabled, but no data was collected.")
        # --- End Reconstruction Plotting ---

        print("Closing simulation") # This should be the final print


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
    
    fem = exactSolution.addObject('TetrahedronFEMForceField', # Store reference
                           name="LinearFEM",
                           youngModulus=young_modulus,
                           poissonRatio=poisson_ratio,
                           method="small") 
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


    # Create and add controller with all components
    controller = AnimationStepController(rootNode,
                                        exactSolution=exactSolution,
                                        fem=fem, # Hyperelastic FEM
                                        linear_solver=linear_solver,
                                        surface_topo=surface_topo,
                                        MO1=MO1, # Real SOFA solution
                                        fixed_box=fixed_box,
                                        MO_LinearModes=MO_LinearModes, # Pass Linear Modes Viz MechObj
                                        MO_NeuralPred=MO_NeuralPred,   # Pass Neural Pred Viz MechObj
                                        visualLinearModes=visualLinearModes, # Pass Linear Modes Viz
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