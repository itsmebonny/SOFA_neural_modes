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
sys.path.append(os.path.join(os.path.dirname(__file__), "../network"))

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
import torch.optim as optim # For L-BFGS optimizer




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

        self.u_nn_prev_flat_th = None       # Previous NN displacement (DOF vector, PyTorch tensor)
        self.u_nn_prev_prev_flat_th = None  # Second previous NN displacement (DOF vector, PyTorch tensor)
        self.z_nn_current_th = None         # Current optimized latent coordinates (PyTorch tensor)
        self.F_ext_dof_th = None            # External force vector on DOFs (PyTorch tensor)
        self.dt = 0.01                      # Simulation time step, will be updated
        self.M_torch = None                 # Mass matrix as a PyTorch tensor
        self.force_roi_indices_flat = None  # Flattened DOF indices for force application
        self.num_force_roi_nodes = 0        # Number of nodes in ForceROI

        self.MO_LinearModes = kwargs.get('MO_LinearModes') # MechObj for Linear Modes
        self.MO_NeuralPred = kwargs.get('MO_NeuralPred')   # MechObj for Neural Pred Viz
        self.visual_LM = kwargs.get('visual_LM') # Visual for Linear Modes
        self.visual_NP = kwargs.get('visual_NP') # Visual for Neural Pred

        self.simulation_time = 0.0 # Initialize simulation time
        self.time_list = []        # List to store time at each step
        self.timestep_counter = 0  # Initialize timestep counter

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
        self.target_force_magnitude = 1000
        self.current_main_step_direction = np.zeros(3) # Initialize direction
        self.last_applied_force_magnitude = 0.0 # Initialize the attribute here
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
        self.current_period_timestep_counter = 0 # Counter for timesteps within the current force period


        # --- Add lists for Deformation Gradient Differences ---
        self.grad_diff_lin_modes_list = []
        self.grad_diff_nn_pred_list = []
        self.grad_diff_sofa_linear_list = []
        # --- End lists for Deformation Gradient Differences ---
        self.force_change_interval = 500 # Number of timesteps for each force period
      


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
        checkpoint_filename = 'best_sofa_dataset.pt' # Or read from config if specified differently
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

        self.dt = self.root.dt.value # Get actual dt
        num_dofs = self.MO1.rest_position.value.size # Total number of DOFs

        # Initialize previous displacements to zero or from a static solve if available
        self.u_nn_prev_flat_th = torch.zeros(num_dofs, dtype=torch.float64, device=self.routine.device)
        self.u_nn_prev_prev_flat_th = torch.zeros(num_dofs, dtype=torch.float64, device=self.routine.device)
        # Initialize z_nn_current_th (e.g., zeros or from initial modal projection)
        self.z_nn_current_th = torch.zeros(self.routine.latent_dim, dtype=torch.float64, device=self.routine.device)


        # Prepare mass matrix as PyTorch tensor (assuming self.routine.M is scipy sparse)
        if self.routine.M is not None:
            M_coo = self.routine.M.tocoo()
            M_indices = torch.tensor(np.vstack((M_coo.row, M_coo.col)), dtype=torch.long, device=self.routine.device)
            M_values = torch.tensor(M_coo.data, dtype=torch.float64, device=self.routine.device)
            self.M_torch = torch.sparse_coo_tensor(M_indices, M_values, M_coo.shape, device=self.routine.device)
        else:
            print("ERROR: Mass matrix M not found in routine. Cannot proceed with dynamic NN objective.")
            if self.root: self.root.animate = False # Stop simulation
            return

        # Prepare ForceROI information for distributing total force
        force_roi_exact_obj = self.exactSolution.getObject('ForceROI')
        if force_roi_exact_obj:
            roi_node_indices = force_roi_exact_obj.indices.value
            self.num_force_roi_nodes = len(roi_node_indices)
            if self.num_force_roi_nodes > 0:
                # Create a flat list of DOF indices for these nodes
                self.force_roi_indices_flat = []
                for node_idx in roi_node_indices:
                    self.force_roi_indices_flat.extend([node_idx * 3, node_idx * 3 + 1, node_idx * 3 + 2])
            else:
                print("Warning: ForceROI is empty.")
        else:
            print("Warning: ForceROI not found in exactSolution.")
        # Initialize previous displacements to zero or from a static solve if available
        self.u_nn_prev_flat_th = torch.zeros(num_dofs, dtype=torch.float64, device=self.routine.device)
        self.u_nn_prev_prev_flat_th = torch.zeros(num_dofs, dtype=torch.float64, device=self.routine.device)
        # Initialize z_nn_current_th (e.g., zeros or from initial modal projection)
        self.z_nn_current_th = torch.zeros(self.routine.latent_dim, dtype=torch.float64, device=self.routine.device)
        # Initialize F_ext_dof_th to a zero tensor
        self.F_ext_dof_th = torch.zeros(num_dofs, dtype=torch.float64, device=self.routine.device)
        print(f"Initialized F_ext_dof_th with shape {self.F_ext_dof_th.shape}")
        self.optimization_start_step_in_period = 100

        
        # Store the original positions for mode animation
        if self.show_modes and self.MO1:
            self.original_positions = np.copy(self.MO1.position.value)
            print(f"Stored original positions with shape {self.original_positions.shape}")


    def onAnimateBeginEvent(self, event):
        """
        Called by SOFA's animation loop before each physics step.
        Applies a force with a random direction, changing every `force_change_interval` timesteps.
        """
        self.timestep_counter += 1

        # Check if it's time to start a new force period
        if (self.timestep_counter - 1) % self.force_change_interval == 0:
            # This block executes at the beginning of each new force period

            if self.current_main_step >= self.max_main_steps:
                # We have completed all planned force periods
                print(f"All {self.max_main_steps} force periods completed. Stopping simulation.")
                if self.root: self.root.animate = False
                return # Stop further processing for this event

            self.current_main_step += 1 # Increment to mark the start of a new period

            # Reset positions for all models to their rest positions
            rest_pos = self.MO1.rest_position.value
            self.MO1.position.value = np.copy(rest_pos)
            if self.MO2: self.MO2.position.value = np.copy(rest_pos)
            if self.MO_LinearModes: self.MO_LinearModes.position.value = np.copy(rest_pos)
            if self.MO_NeuralPred: self.MO_NeuralPred.position.value = np.copy(rest_pos)
            
    

            print(f"\n--- Timestep {self.timestep_counter}: Starting Force Period {self.current_main_step}/{self.max_main_steps} ---")

            # Generate a new random direction for this force period
            random_vec = [0.0, -1.0, 0.0] # Generate random vector
            norm = np.linalg.norm(random_vec)
            if norm < 1e-9: # Avoid division by zero
                self.current_main_step_direction = np.array([1.0, 0.0, 0.0]) # Default direction
            else:
                self.current_main_step_direction = random_vec / norm # Normalize
            
            # Set the force vector for the current period
            self.current_period_force_vector = self.current_main_step_direction * self.target_force_magnitude
            self.last_applied_force_magnitude = self.target_force_magnitude # Store magnitude for analysis

            # --- Prepare F_ext_dof_th for the objective function for this period ---
            num_dofs_force = self.MO1.rest_position.value.size # Ensure this matches M_torch dimensions
            F_ext_dof_np = np.zeros(num_dofs_force, dtype=np.float64)
            if self.force_roi_indices_flat and self.num_force_roi_nodes > 0:
                force_per_node_vec = self.current_period_force_vector / self.num_force_roi_nodes
                # Apply force_per_node_vec to the DOFs in force_roi_indices_flat
                # This assumes force_roi_indices_flat correctly maps to F_ext_dof_np
                node_indices_in_roi = self.exactSolution.getObject('ForceROI').indices.value
                for i, node_idx in enumerate(node_indices_in_roi):
                    if node_idx * 3 + 2 < num_dofs_force: # Check bounds
                         F_ext_dof_np[node_idx*3 : node_idx*3+3] = force_per_node_vec
                    else:
                        print(f"Warning: Node index {node_idx} out of bounds for F_ext_dof_np.")

                # This line updates self.F_ext_dof_th when the force changes
                self.F_ext_dof_th = torch.tensor(F_ext_dof_np, device=self.routine.device, dtype=torch.float64)

            # --- End F_ext_dof_th preparation ---



            print(f"  New Random Force Direction: {self.current_main_step_direction}")
            print(f"  Applying force (mag: {self.last_applied_force_magnitude:.2e}, vec: {self.current_period_force_vector}) for the next {self.force_change_interval} timesteps.")

        # If simulation has been flagged to stop, do not proceed
        if not (self.root and self.root.animate.value):
            return

        # --- Remove existing ConstantForceFields before adding new ones ---
        # This ensures that forces don't accumulate if objects are not properly cleared.
        if self.cff is not None:
            try:
                self.exactSolution.removeObject(self.cff)
            except Exception as e:
                # print(f"Warning: Error removing CFF (Exact): {e}") # Can be verbose
                pass # Object might have been already removed or scene graph changed
            finally:
                 self.cff = None # Clear the reference

        if self.cff_linear is not None:
            try:
                self.linearSolution.removeObject(self.cff_linear)
            except Exception as e:
                # print(f"Warning: Error removing CFF (Linear): {e}") # Can be verbose
                pass
            finally:
                 self.cff_linear = None # Clear the reference
        # --- End Remove CFFs ---

        # --- Create and add new CFFs with the current period's force vector ---
        # This force vector remains constant for `self.force_change_interval` timesteps.
        try:
            # Exact Solution
            force_roi_exact = self.exactSolution.getObject('ForceROI')
            if force_roi_exact is None: raise ValueError("ForceROI (Exact) not found in exactSolution node.")
            self.cff = self.exactSolution.addObject('ConstantForceField',
                               name="CFF_Exact_Managed", # Use a distinct name
                               indices="@ForceROI.indices", # Reference the ROI in the same node
                               totalForce=self.current_period_force_vector.tolist(),
                               showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
            if self.cff: self.cff.init() # Initialize the newly added component

            # Linear Solution
            force_roi_linear = self.linearSolution.getObject('ForceROI')
            if force_roi_linear is None: raise ValueError("ForceROI (Linear) not found in linearSolution node.")
            self.cff_linear = self.linearSolution.addObject('ConstantForceField',
                               name="CFF_Linear_Managed", # Use a distinct name
                               indices="@ForceROI.indices", # Reference the ROI in the same node
                               totalForce=self.current_period_force_vector.tolist(),
                               showArrowSize=0.0) # Hide arrow for linear model
            if self.cff_linear: self.cff_linear.init() # Initialize

        except Exception as e:
            print(f"ERROR: Failed to create/add/init ConstantForceField(s): {e}")
            traceback.print_exc()
            # If CFF creation fails, stop the simulation to prevent incorrect results
            if self.root: self.root.animate = False

        self.start_time = process_time()



    def onAnimateEndEvent(self, event):
        """
        Called by SOFA's animation loop after each physics step.
        Performs analysis and stores results for the completed substep.
        """
        try:
            self.current_period_timestep_counter += 1
            self.simulation_time += self.root.dt.value # Increment simulation time
            self.time_list.append(self.simulation_time) # Store current time

            current_force_magnitude = self.last_applied_force_magnitude
            real_solution_disp_np = self.MO1.position.value.copy() - self.MO1.rest_position.value.copy()
            linear_solution_sofa_disp_np = self.MO2.position.value.copy() - self.MO2.rest_position.value.copy()
            
            real_energy = self.computeInternalEnergy(real_solution_disp_np)
            sofa_linear_energy = float('nan')
            if self.MO2 and self.linearFEM:
                sofa_linear_energy = self.computeInternalEnergy(linear_solution_sofa_disp_np)

            # Initialize placeholders for NN and Linear Modes predictions
            u_pred_nn_flat_th = torch.zeros_like(self.u_nn_prev_flat_th) # Default to zeros
            l_th_flat_for_report = torch.zeros_like(self.u_nn_prev_flat_th) # Default to zeros
            
            # --- NN Dynamic Prediction ---
            if self.routine and self.M_torch is not None and self.F_ext_dof_th is not None:
                device = self.routine.device
                real_solution_disp_th = torch.tensor(real_solution_disp_np.flatten(), dtype=torch.float64, device=device)

                # self.optimization_start_step_in_period should be defined in __init__, e.g., = 3

                if self.current_period_timestep_counter < self.optimization_start_step_in_period:
                    # --- Bootstrapping phase: Use SOFA solutions to guide NN state before optimization kicks in ---
                    # u_pred_nn_flat_th is set to the real solution.
                    # z_nn_current_th is projected from the SOFA linear solution.
                    # l_th_flat_for_report is the linear part from this projected z.
                    # The history (u_nn_prev, u_nn_prev_prev) will be populated by these "real" values later.

                    u_pred_nn_flat_th = real_solution_disp_th.clone() # NN "output" forced to match real solution for history

                    # Project SOFA linear solution to get z_nn_current_th as an estimate
                    z_from_sofa_linear_np = self.computeModalCoordinates(linear_solution_sofa_disp_np)
                    if z_from_sofa_linear_np is None or np.isnan(z_from_sofa_linear_np).any():
                        # print(f"Warning: Projection of SOFA linear solution for z_nn_current_th resulted in None/NaN at timestep {self.current_period_timestep_counter}. Using zeros.")
                        z_from_sofa_linear_np = np.zeros(self.routine.latent_dim)
                    
                    current_z_tensor = torch.tensor(z_from_sofa_linear_np, dtype=torch.float64, device=device)

                    # Ensure z_nn_current_th (which is self.z_nn_current_th) has the correct latent_dim for model consistency
                    if current_z_tensor.shape[0] != self.routine.latent_dim:
                        new_z = torch.zeros(self.routine.latent_dim, dtype=current_z_tensor.dtype, device=device)
                        common_dim = min(current_z_tensor.shape[0], self.routine.latent_dim)
                        new_z[:common_dim] = current_z_tensor[:common_dim]
                        self.z_nn_current_th = new_z
                    else:
                        self.z_nn_current_th = current_z_tensor
                    
                    # Calculate l_th_flat_for_report based on this z_nn_current_th
                    # Ensure modes used for l_th_flat_for_report match the dimension of z_nn_current_th (self.routine.latent_dim)
                    # This assumes self.routine.linear_modes has at least self.routine.latent_dim columns.
                    modes_for_report = self.routine.linear_modes[:, :self.routine.latent_dim].to(device, dtype=torch.float64)
                    l_th_flat_for_report = torch.matmul(modes_for_report, self.z_nn_current_th)

                else: # Timestep >= self.optimization_start_step_in_period: Perform optimization
                    # Initial guess for z for the optimizer is self.z_nn_current_th (from previous step's projection or optimization)
                    z_guess_th = self.z_nn_current_th.clone().detach().requires_grad_(True)
                    
                    # Ensure z_guess_th has the correct latent_dim for the model input
                    if z_guess_th.shape[0] != self.routine.latent_dim:
                        new_z_guess = torch.zeros(self.routine.latent_dim, dtype=z_guess_th.dtype, device=device)
                        common_dim = min(z_guess_th.shape[0], self.routine.latent_dim)
                        new_z_guess[:common_dim] = z_guess_th[:common_dim]
                        z_guess_th = new_z_guess.requires_grad_(True)

                    optimizer = optim.LBFGS([z_guess_th], lr=1.0, max_iter=15, line_search_fn="strong_wolfe")

                    # History terms for the objective function come from self.u_nn_prev_flat_th and self.u_nn_prev_prev_flat_th
                    u_prev_for_opt = self.u_nn_prev_flat_th.clone().detach()
                    u_prev_prev_for_opt = self.u_nn_prev_prev_flat_th.clone().detach()
                    F_ext_for_opt = self.F_ext_dof_th.clone().detach() # External force for current step

                    def closure():
                        optimizer.zero_grad()
                        loss = self.objective_function(z_guess_th, u_prev_for_opt, u_prev_prev_for_opt, F_ext_for_opt)
                        loss.backward()
                        return loss
                    
                    optimizer.step(closure)
                    self.z_nn_current_th = z_guess_th.clone().detach() # Update z with optimized value
                    print(f"  Optimized z_nn_current_th: {self.z_nn_current_th.cpu().numpy().round(3)}")


                    # Reconstruct displacement u_pred_nn_flat_th and l_th_flat_for_report from optimized z
                    with torch.no_grad():
                        modes_final = self.routine.linear_modes[:, :self.routine.latent_dim].to(device, dtype=torch.float64)
                        l_th_flat_for_report = torch.matmul(modes_final, self.z_nn_current_th)
                        
                        y_th_final_flat = self.routine.model(self.z_nn_current_th.unsqueeze(0)).squeeze(0)
                        u_pred_nn_flat_th = l_th_flat_for_report + y_th_final_flat
                    
                # --- This history update applies to both cases (bootstrapping and optimization) ---
                # It stores the u_pred_nn_flat_th determined for the *current* step (either real_solution or optimized NN prediction)
                # into the history to be used as u_nn_prev_flat_th in the *next* step's optimization.
                self.u_nn_prev_prev_flat_th = self.u_nn_prev_flat_th.clone().detach()
                self.u_nn_prev_flat_th = u_pred_nn_flat_th.clone().detach()

                # else: # Timestep >= 3: Perform optimization
                #     z_guess_th = self.z_nn_current_th.clone().detach().requires_grad_(True)
                #     if z_guess_th.shape[0] != self.routine.latent_dim: # Adjust shape
                #         new_z_guess = torch.zeros(self.routine.latent_dim, dtype=z_guess_th.dtype, device=device)
                #         common_dim = min(z_guess_th.shape[0], self.routine.latent_dim); new_z_guess[:common_dim] = z_guess_th[:common_dim]
                #         z_guess_th = new_z_guess.requires_grad_(True)

                #     optimizer = optim.LBFGS([z_guess_th], lr=1.0, max_iter=15, line_search_fn="strong_wolfe") # Reduced max_iter

                #     u_prev_for_opt = self.u_nn_prev_flat_th.clone().detach()
                #     u_prev_prev_for_opt = self.u_nn_prev_prev_flat_th.clone().detach()
                #     F_ext_for_opt = self.F_ext_dof_th.clone().detach()

                #     def closure():
                #         optimizer.zero_grad()
                #         loss = self.objective_function(z_guess_th, u_prev_for_opt, u_prev_prev_for_opt, F_ext_for_opt)
                #         loss.backward()
                #         return loss
                #     optimizer.step(closure)
                #     self.z_nn_current_th = z_guess_th.clone().detach()
                #     print(f"  Optimized z_nn_current_th: {self.z_nn_current_th}")

                #     with torch.no_grad():
                #         latent_dim_final = self.z_nn_current_th.shape[0]
                #         modes_final = self.routine.linear_modes[:, :latent_dim_final].to(device, dtype=torch.float64)
                #         l_th_flat_for_report = torch.matmul(modes_final, self.z_nn_current_th)
                #         y_th_final_flat = self.routine.model(self.z_nn_current_th.unsqueeze(0)).squeeze(0)
                #         u_pred_nn_flat_th = l_th_flat_for_report + y_th_final_flat
                    
                #     self.u_nn_prev_prev_flat_th = self.u_nn_prev_flat_th.clone().detach()
                    # self.u_nn_prev_flat_th = u_pred_nn_flat_th.clone().detach()
            else:
                missing_components = []
                if not self.routine:
                    missing_components.append("routine")
                if self.M_torch is None:
                    missing_components.append("M_torch")
                if self.F_ext_dof_th is None:
                    missing_components.append("F_ext_dof_th")
                print(f"  Skipping NN dynamic prediction. Not available: {', '.join(missing_components)}.")
            # --- End NN Dynamic Prediction ---

            # Convert predictions to NumPy for reporting and visualization
            num_nodes_mo1 = self.MO1.position.value.shape[0]
            u_pred_reshaped_np = u_pred_nn_flat_th.cpu().numpy().reshape(num_nodes_mo1, 3)
            l_th_reshaped_np = l_th_flat_for_report.cpu().numpy().reshape(num_nodes_mo1, 3)
            real_solution_reshaped_np = real_solution_disp_np.reshape(num_nodes_mo1, 3)
            linear_solution_sofa_reshaped_np = linear_solution_sofa_disp_np.reshape(num_nodes_mo1, 3)


            # Compute energies
            predicted_energy = self.computeInternalEnergy(u_pred_reshaped_np)
            linear_energy_modes = self.computeInternalEnergy(l_th_reshaped_np)

            # Compute errors
            diff_pred_real = real_solution_reshaped_np - u_pred_reshaped_np
            l2_err_pred_real = np.linalg.norm(diff_pred_real)
            mse_pred_real = np.mean(diff_pred_real**2)
            rmse_pred_real = np.sqrt(mse_pred_real)

            diff_lin_modes_real = real_solution_reshaped_np - l_th_reshaped_np
            l2_err_lin_real = np.linalg.norm(diff_lin_modes_real) # Error of NN's linear part vs Real
            mse_lin_real = np.mean(diff_lin_modes_real**2)
            rmse_lin_real = np.sqrt(mse_lin_real)

            diff_sofa_lin_real = real_solution_reshaped_np - linear_solution_sofa_reshaped_np
            l2_err_sofa_lin_real = np.linalg.norm(diff_sofa_lin_real) # Error of SOFA Linear vs Real
            mse_sofa_lin_real = np.mean(diff_sofa_lin_real**2)
            rmse_sofa_lin_real = np.sqrt(mse_sofa_lin_real)
            
            # Deformation Gradients (using the new predictions)
            F_real_np, F_lm_pred_np, F_nn_pred_np, F_sofa_linear_np = None, None, None, None
            norm_diff_F_lm, norm_diff_F_nn, norm_diff_F_sl = float('nan'), float('nan'), float('nan')

            if hasattr(self.routine, 'energy_calculator') and \
               self.routine.energy_calculator is not None and \
               hasattr(self.routine.energy_calculator, 'compute_deformation_gradients'):
                
                calc_F = self.routine.energy_calculator.compute_deformation_gradients
                device_grad = self.routine.device
                dtype_grad = torch.float64 # Assuming float64 for consistency
                
                num_nodes_for_grad = num_nodes_mo1
                spatial_dim = 3

                def get_F_from_disp(disp_np_array_flat): # Expects flat (num_dofs)
                    if disp_np_array_flat is None: return None
                    disp_tensor = torch.tensor(disp_np_array_flat, device=device_grad, dtype=dtype_grad)
                    try:
                        disp_tensor_reshaped = disp_tensor.view(num_nodes_for_grad, spatial_dim)
                    except RuntimeError as e_reshape:
                        print(f"  Error reshaping for F: {e_reshape}. Shape: {disp_tensor.shape}")
                        return None
                    F_tensor = calc_F(disp_tensor_reshaped) # Expects (num_nodes, dim)
                    if F_tensor.dim() == 4 and F_tensor.shape[0] == 1: F_tensor = F_tensor.squeeze(0)
                    return F_tensor.cpu().numpy()

                try:
                    F_real_np = get_F_from_disp(real_solution_disp_np.flatten())
                    if F_real_np is not None:
                        if l_th_reshaped_np is not None:
                            F_lm_pred_np = get_F_from_disp(l_th_reshaped_np.flatten())
                            if F_lm_pred_np is not None and F_lm_pred_np.shape == F_real_np.shape:
                                norm_diff_F_lm = np.mean(np.linalg.norm(F_real_np - F_lm_pred_np, ord='fro', axis=(-2,-1)))
                        if u_pred_reshaped_np is not None:
                            F_nn_pred_np = get_F_from_disp(u_pred_reshaped_np.flatten())
                            if F_nn_pred_np is not None and F_nn_pred_np.shape == F_real_np.shape:
                                norm_diff_F_nn = np.mean(np.linalg.norm(F_real_np - F_nn_pred_np, ord='fro', axis=(-2,-1)))
                        if linear_solution_sofa_disp_np is not None:
                            F_sofa_linear_np = get_F_from_disp(linear_solution_sofa_disp_np.flatten())
                            if F_sofa_linear_np is not None and F_sofa_linear_np.shape == F_real_np.shape:
                                norm_diff_F_sl = np.mean(np.linalg.norm(F_real_np - F_sofa_linear_np, ord='fro', axis=(-2,-1)))
                except Exception as e_fgrad:
                    print(f"  ERROR calculating deformation gradients: {e_fgrad}"); traceback.print_exc()
            
            self.grad_diff_lin_modes_list.append(norm_diff_F_lm)
            self.grad_diff_nn_pred_list.append(norm_diff_F_nn)
            self.grad_diff_sofa_linear_list.append(norm_diff_F_sl)
            
            # Update visualizations
            if self.original_positions is not None:
                rest_pos = self.original_positions
                if self.MO_LinearModes is not None and l_th_reshaped_np is not None:
                    self.MO_LinearModes.position.value = rest_pos + l_th_reshaped_np
                if self.MO_NeuralPred is not None and u_pred_reshaped_np is not None:
                    self.MO_NeuralPred.position.value = rest_pos + u_pred_reshaped_np
            
            # Store results for plotting
            self.substep_results.append((
                self.simulation_time, # Add current time
                current_force_magnitude, real_energy, predicted_energy, linear_energy_modes, sofa_linear_energy,
                l2_err_pred_real, rmse_pred_real, mse_pred_real,
                l2_err_lin_real, rmse_lin_real, mse_lin_real, # Errors for NN's linear part
                l2_err_sofa_lin_real, rmse_sofa_lin_real, mse_sofa_lin_real # Errors for SOFA Linear FEM
            ))
            if self.z_nn_current_th is not None:
                 self.all_z_coords.append(self.z_nn_current_th.cpu().numpy().copy()) # Store optimized z

        except Exception as e:
            print(f"ERROR during analysis in onAnimateEndEvent: {e}")
            traceback.print_exc()
            current_time_for_error = self.time_list[-1] if self.time_list else 0.0
            self.substep_results.append((
                current_time_for_error,
                self.last_applied_force_magnitude if hasattr(self, 'last_applied_force_magnitude') else float('nan'),
                float('nan'), float('nan'), float('nan'), float('nan'),
                float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                float('nan'), float('nan'), float('nan')
            ))
            self.grad_diff_lin_modes_list.append(float('nan'))
            self.grad_diff_nn_pred_list.append(float('nan'))
            self.grad_diff_sofa_linear_list.append(float('nan'))
            if self.all_z_coords and self.routine: # Append zeros if z_coords list exists
                 self.all_z_coords.append(np.zeros(self.routine.latent_dim))


        # This counter is for the overall simulation, not per force period
        # self.current_substep += 1 # This was from the old logic, self.timestep_counter is now the global step
        
        # Check for stopping condition based on main steps (managed in onAnimateBeginEvent)
        # if 'args' in globals() and not args.gui and self.current_main_step >= self.max_main_steps:
        #     if self.root: self.root.animate = False # Controller will stop based on this

        self.end_time = process_time()

        
    # --- Full new method for the Neural Network Objective Function ---
    def objective_function(self, z_th_opt, u_nn_prev_flat_th, u_nn_prev_prev_flat_th, F_ext_dof_th):
        """
        Objective function for optimizing the latent coordinates z_th.
        J(z) = InertialTerm(u_nn(z)) + ElasticEnergy(u_nn(z)) - Work_ExternalForce(u_nn(z)) (+ DampingTerm(u_nn(z)))
        """
        # Ensure z_th_opt requires gradients for the optimizer
        z_th_opt.requires_grad_(True)

        # 1. Predict current displacement u_curr_nn_flat_th from z_th_opt
        # Ensure modes_to_use matches the dimension of z_th_opt
        latent_dim_current = z_th_opt.shape[0]
        modes_to_use = self.routine.linear_modes[:, :latent_dim_current].to(self.routine.device, dtype=torch.float64)

        # l_th is (num_dofs,)
        l_th_flat = torch.matmul(modes_to_use, z_th_opt)

        # y_th is (num_dofs,)
        # Model input z_th_opt needs to be (batch_size, latent_dim) = (1, latent_dim_current)
        y_th_flat = self.routine.model(z_th_opt.unsqueeze(0)).squeeze(0)
        
        u_curr_nn_flat_th = l_th_flat + y_th_flat # Shape: (num_dofs,)

   

        # 3. Calculate Inertial Term: 0.5 * u_ddot^T * M * u_ddot
        # Use finite difference scheme with previous displacements
        # u_ddot = (u_curr - 2*u_prev + u_prev_prev) / dt^2
        if self.dt == 0:
            inertial_term = torch.tensor(0.0, device=self.routine.device, dtype=torch.float64)
        else:
            u_ddot_fd_th = (u_curr_nn_flat_th - 2 * u_nn_prev_flat_th + u_nn_prev_prev_flat_th)
            inertial_force_th = torch.sparse.mm(self.M_torch, u_ddot_fd_th.unsqueeze(1)).squeeze(1)
            inertial_term = torch.dot(u_ddot_fd_th, inertial_force_th) / (2 * self.dt**2)

        # 4. Calculate Elastic Strain Energy: E_elastic(u_nn(z))
        # energy_calculator expects displacement in shape (batch_size, num_nodes, 3) or (num_nodes, 3)
        # u_curr_nn_flat_th is (num_dofs,). Reshape it.
        num_nodes = self.MO1.rest_position.value.shape[0] # Or from self.routine.num_nodes
        try:
            u_curr_nn_reshaped_th = u_curr_nn_flat_th.view(1, num_nodes, 3) # Add batch dim
            elastic_energy = self.routine.energy_calculator(u_curr_nn_reshaped_th).squeeze()
        except Exception as e:
            print(f"Error in energy_calculator with u_curr_nn: {e}")
            elastic_energy = torch.tensor(0.0, device=self.routine.device, dtype=torch.float64)


        # 5. Calculate External Force Work Term: - F_ext_dof · u_curr_nn
        # F_ext_dof_th is (num_dofs,), u_curr_nn_flat_th is (num_dofs,)
        work_external_term = -torch.dot(F_ext_dof_th, u_curr_nn_flat_th)

        # 6. Damping Term (Placeholder - can be mass-proportional, stiffness-proportional, etc.)
        # Example: Mass-proportional damping: 0.5 * alpha_damping * v_curr^T * M * v_curr
        # v_curr = (u_curr - u_prev) / dt
        damping_term = torch.tensor(0.0, device=self.routine.device, dtype=torch.float64)
        # if self.dt > 0:
        #     alpha_damping = 0.01 # Example damping coefficient
        #     v_curr_nn_flat_th = (u_curr_nn_flat_th - u_nn_prev_flat_th) / self.dt
        #     damping_force_th = torch.sparse.mm(self.M_torch, v_curr_nn_flat_th.unsqueeze(1)).squeeze(1)
        #     damping_term = 0.5 * alpha_damping * torch.dot(v_curr_nn_flat_th, damping_force_th)


        # 7. Total Objective
        total_objective = inertial_term + elastic_energy + work_external_term + damping_term
        
        print(f"  Objective: {total_objective.item():.4e} (Inertial: {inertial_term.item():.3e}, Elastic: {elastic_energy.item():.3e}, Work: {work_external_term.item():.3e}, Damping: {damping_term.item():.3e})")
        return total_objective
    # --- End Objective Function ---

    def objective_function_linear_modes(self, z_lm_th_opt, u_lm_prev_flat_th, u_lm_prev_prev_flat_th, F_ext_dof_th):
        """
        Objective function for optimizing the latent coordinates z_lm_th for a purely linear modal model.
        J(z) = InertialTerm(u_lm(z)) + ElasticEnergy(u_lm(z)) - Work_ExternalForce(u_lm(z)) (+ DampingTerm(u_lm(z)))
        """
        # Ensure z_lm_th_opt requires gradients for the optimizer
        z_lm_th_opt.requires_grad_(True)

        # 1. Predict current displacement u_curr_lm_flat_th from z_lm_th_opt (Linear Modes Only)
        latent_dim_current = z_lm_th_opt.shape[0]
        # Ensure modes_to_use matches the dimension of z_lm_th_opt
        modes_to_use = self.routine.linear_modes[:, :latent_dim_current].to(self.routine.device, dtype=torch.float64)

        # u_lm_th is (num_dofs,)
        u_curr_lm_flat_th = torch.matmul(modes_to_use, z_lm_th_opt) # Linear reconstruction

        # 2. Calculate acceleration u_ddot_lm_flat_th
        if self.dt == 0:
            u_ddot_lm_flat_th = torch.zeros_like(u_curr_lm_flat_th)
        else:
            u_ddot_lm_flat_th = (u_curr_lm_flat_th - 2 * u_lm_prev_flat_th + u_lm_prev_prev_flat_th) / (self.dt**2)

        # 3. Calculate Inertial Term: 0.5 * u_ddot^T * M * u_ddot
        inertial_force_th = torch.sparse.mm(self.M_torch, u_ddot_lm_flat_th.unsqueeze(1)).squeeze(1)
        inertial_term = 0.5 * torch.dot(u_ddot_lm_flat_th, inertial_force_th)

        # 4. Calculate Elastic Strain Energy: E_elastic(u_lm(z))
        num_nodes = self.MO1.rest_position.value.shape[0]
        try:
            # energy_calculator expects displacement in shape (batch_size, num_nodes, 3)
            u_curr_lm_reshaped_th = u_curr_lm_flat_th.view(1, num_nodes, 3)
            # Note: This will use the routine's energy calculator. If this is hyperelastic,
            # it will compute hyperelastic energy for a purely modal displacement.
            # If a specific linear elastic energy calculator is desired for this objective,
            # it would need to be provided and used here.
            elastic_energy = self.routine.energy_calculator(u_curr_lm_reshaped_th).squeeze()
        except Exception as e:
            print(f"Error in energy_calculator with u_curr_lm: {e}")
            elastic_energy = torch.tensor(0.0, device=self.routine.device, dtype=torch.float64)

        # 5. Calculate External Force Work Term: - F_ext_dof · u_curr_lm
        work_external_term = -torch.dot(F_ext_dof_th, u_curr_lm_flat_th)

        # 6. Damping Term (Placeholder)
        damping_term = torch.tensor(0.0, device=self.routine.device, dtype=torch.float64)
        # if self.dt > 0:
        #     alpha_damping = 0.01 # Example damping coefficient
        #     v_curr_lm_flat_th = (u_curr_lm_flat_th - u_lm_prev_flat_th) / self.dt
        #     damping_force_th = torch.sparse.mm(self.M_torch, v_curr_lm_flat_th.unsqueeze(1)).squeeze(1)
        #     damping_term = 0.5 * alpha_damping * torch.dot(v_curr_lm_flat_th, damping_force_th)

        # 7. Total Objective
        total_objective_lm = inertial_term + elastic_energy + work_external_term + damping_term
        
        # print(f"  LM Objective: {total_objective_lm.item():.4e} (Inertial: {inertial_term.item():.3e}, Elastic: {elastic_energy.item():.3e}, Work: {work_external_term.item():.3e})")
        return total_objective_lm
    # --- End Linear Modes Objective Function ---




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
            'Time', # Added 'Time' column
            'ForceMag', 'RealE', 'PredE', 'LinearModesE', 'SOFALinearE',
            'L2Err_Pred_Real', 'RMSE_Pred_Real', 'MSE_Pred_Real',
            'L2Err_Lin_Real', 'RMSE_Lin_Real', 'MSE_Lin_Real',
            'L2Err_SOFALin_Real', 'RMSE_SOFALin_Real', 'MSE_SOFALin_Real'
        ]
        try:
            import pandas as pd
            df = pd.DataFrame(self.substep_results, columns=result_columns)

            # Add new columns for gradient differences
            num_entries_df = len(df)
            df['GradDiff_LM'] = pd.Series(self.grad_diff_lin_modes_list[:num_entries_df])
            df['GradDiff_NN'] = pd.Series(self.grad_diff_nn_pred_list[:num_entries_df])
            df['GradDiff_SL'] = pd.Series(self.grad_diff_sofa_linear_list[:num_entries_df])
            
            # Calculate and print average results (this part remains)
            avg_results = df.groupby('ForceMag').mean().reset_index()
            avg_results = avg_results.sort_values(by='ForceMag')

            print("\n--- Average Results per Force Magnitude ---")
            cols_to_print = ['ForceMag', 'RealE', 'PredE', 'LinearModesE', 'SOFALinearE',
                             'RMSE_Pred_Real', 'MSE_Pred_Real',
                             'RMSE_Lin_Real', 'MSE_Lin_Real',
                             'RMSE_SOFALin_Real', 'MSE_SOFALin_Real',
                             'GradDiff_LM', 'GradDiff_NN', 'GradDiff_SL']
            
            cols_to_print_existing = [col for col in cols_to_print if col in avg_results.columns]
            print(avg_results[cols_to_print_existing].to_string(index=False, float_format="%.4e"))
            print("-------------------------------------------\n")

            # --- Data for plotting against timesteps ---
            timesteps_plot = df.index.values # Use DataFrame index for timesteps

            real_e_ts = df['RealE'].values
            pred_e_ts = df['PredE'].values
            linear_modes_e_ts = df['LinearModesE'].values
            sofa_linear_e_ts = df['SOFALinearE'].values

            rmse_pred_real_ts = df['RMSE_Pred_Real'].values
            mse_pred_real_ts = df['MSE_Pred_Real'].values
            rmse_lin_real_ts = df['RMSE_Lin_Real'].values
            mse_lin_real_ts = df['MSE_Lin_Real'].values
            rmse_sofa_lin_real_ts = df['RMSE_SOFALin_Real'].values
            mse_sofa_lin_real_ts = df['MSE_SOFALin_Real'].values
            
            grad_diff_lm_ts = df['GradDiff_LM'].values if 'GradDiff_LM' in df.columns else np.full_like(timesteps_plot, float('nan'), dtype=float)
            grad_diff_nn_ts = df['GradDiff_NN'].values if 'GradDiff_NN' in df.columns else np.full_like(timesteps_plot, float('nan'), dtype=float)
            grad_diff_sl_ts = df['GradDiff_SL'].values if 'GradDiff_SL' in df.columns else np.full_like(timesteps_plot, float('nan'), dtype=float)

            plot_dir = self.output_subdir if self.save else "."
            if self.save and not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            # 1. Energy vs. Timestep Plot (Linear Scale)
            plt.figure(figsize=(10, 6))
            plt.plot(timesteps_plot, real_e_ts, label='Real Energy (SOFA Hyperelastic)', linestyle='-')
            plt.plot(timesteps_plot, pred_e_ts, label='Predicted Energy (l+y)', linestyle='--')
            plt.plot(timesteps_plot, linear_modes_e_ts, label='Linear Modes Energy (l)', linestyle=':')
            plt.plot(timesteps_plot, sofa_linear_e_ts, label='SOFA Linear Energy', linestyle='-.')
            plt.xlabel('Timestep'); plt.ylabel('Internal Energy')
            plt.title('Energy vs. Timestep'); plt.legend(); plt.grid(True); plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "energy_vs_timestep.png")); plt.close()

            # 1b. Energy vs. Timestep Plot (Log Scale)
            plt.figure(figsize=(10, 6))
            valid_indices_real = real_e_ts > 0; valid_indices_pred = pred_e_ts > 0
            valid_indices_linear_modes = linear_modes_e_ts > 0; valid_indices_sofa_linear = sofa_linear_e_ts > 0
            if np.any(valid_indices_real): plt.plot(timesteps_plot[valid_indices_real], real_e_ts[valid_indices_real], label='Real Energy', linestyle='-')
            if np.any(valid_indices_pred): plt.plot(timesteps_plot[valid_indices_pred], pred_e_ts[valid_indices_pred], label='Predicted Energy', linestyle='--')
            if np.any(valid_indices_linear_modes): plt.plot(timesteps_plot[valid_indices_linear_modes], linear_modes_e_ts[valid_indices_linear_modes], label='Linear Modes Energy', linestyle=':')
            if np.any(valid_indices_sofa_linear): plt.plot(timesteps_plot[valid_indices_sofa_linear], sofa_linear_e_ts[valid_indices_sofa_linear], label='SOFA Linear Energy', linestyle='-.')
            plt.xlabel('Timestep'); plt.ylabel('Internal Energy (log scale)')
            plt.title('Energy vs. Timestep (Log Scale)'); plt.yscale('log')
            plt.legend(); plt.grid(True, which="both", ls="--"); plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "energy_vs_timestep_log.png")); plt.close()

            # 2. RMSE Errors vs Timestep
            plt.figure(figsize=(10, 6))
            plt.plot(timesteps_plot, rmse_pred_real_ts, label='RMSE: Pred (l+y) vs Real (MO1)')
            plt.plot(timesteps_plot, rmse_lin_real_ts, label='RMSE: LinModes (l) vs Real (MO1)', linestyle='--')
            plt.plot(timesteps_plot, rmse_sofa_lin_real_ts, label='RMSE: SOFALin (MO2) vs Real (MO1)', linestyle=':')
            plt.xlabel('Timestep'); plt.ylabel('RMSE')
            plt.title('RMSE vs. Timestep'); plt.legend(); plt.grid(True); plt.yscale('log'); plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "rmse_vs_timestep.png")); plt.close()

            # 3. MSE Errors vs Timestep
            plt.figure(figsize=(10, 6))
            plt.plot(timesteps_plot, mse_pred_real_ts, label='MSE: Pred (l+y) vs Real (MO1)')
            plt.plot(timesteps_plot, mse_lin_real_ts, label='MSE: LinModes (l) vs Real (MO1)', linestyle='--')
            plt.plot(timesteps_plot, mse_sofa_lin_real_ts, label='MSE: SOFALin (MO2) vs Real (MO1)', linestyle=':')
            plt.xlabel('Timestep'); plt.ylabel('MSE')
            plt.title('MSE vs. Timestep'); plt.legend(); plt.grid(True); plt.yscale('log'); plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "mse_vs_timestep.png")); plt.close()

            # --- Plot for Deformation Gradient Differences vs Timestep ---
            plt.figure(figsize=(12, 7))
            if not np.all(np.isnan(grad_diff_lm_ts)):
                 plt.plot(timesteps_plot, grad_diff_lm_ts, label='||F_real - F_LMpred||', linestyle='-')
            if not np.all(np.isnan(grad_diff_nn_ts)):
                 plt.plot(timesteps_plot, grad_diff_nn_ts, label='||F_real - F_NNpred||', linestyle='--')
            if not np.all(np.isnan(grad_diff_sl_ts)):
                 plt.plot(timesteps_plot, grad_diff_sl_ts, label='||F_real - F_SOFALinear||', linestyle=':')
            
            plt.xlabel('Timestep')
            plt.ylabel('Frobenius Norm Diff. of Def. Gradients')
            plt.title('Deformation Gradient Difference vs. Timestep')
            plt.legend()
            plt.grid(True, which="both", ls="--")
            plt.yscale('log') 
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "grad_diff_vs_timestep.png"))
            plt.close()
            print(f"Deformation gradient difference plot saved to {os.path.join(plot_dir, 'grad_diff_vs_timestep.png')}")
            # --- End Plot ---

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
    rootNode.dt = config['physics'].get('dt', 0.001)
    rootNode.gravity = [0, 0, 0]
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
    mass = exactSolution.addObject('MeshMatrixMass', totalMass=total_mass, name="SparseMass", topology="@triangleTopo")
    
    # Get solver parameters from config
    rayleighStiffness = config['physics'].get('rayleigh_stiffness', 0.1)
    rayleighMass = config['physics'].get('rayleigh_mass', 0.1)
    
    exactSolution.addObject('EulerImplicitSolver', name="ODEsolver", rayleighStiffness=rayleighStiffness, rayleighMass=rayleighMass)

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

    force_box_coords = config['constraints'].get('force_box_1', [9.91, -0.01, -0.02, 10.1, 1.01, 1.02])
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

    linearSolution.addObject('MeshMatrixMass', totalMass=total_mass, name="SparseMass", topology="@triangleTopo")


    # Add system components (similar to exactSolution)
    linearSolution.addObject('EulerImplicitSolver', name="ODEsolver", rayleighStiffness=rayleighStiffness, rayleighMass=rayleighMass)
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