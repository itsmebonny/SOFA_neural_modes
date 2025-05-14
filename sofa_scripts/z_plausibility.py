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
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import torch

import os
import json
import datetime
import numpy as np

from scipy.stats import pearsonr

import glob
import traceback
from scipy import sparse
from scipy.sparse.linalg import eigsh
np.random.seed(0)  # Set seed for reproducibility
import matplotlib.pyplot as plt



class AnimationStepController(Sofa.Core.Controller):
    def __init__(self, node, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.surface_topo = kwargs.get('surface_topo')
        self.MO1 = kwargs.get('MO1')
        self.fixed_box = kwargs.get('fixed_box')
        self.exactSolution = kwargs.get('exactSolution') # Retained as it's in original __init__
        self.visual_model = kwargs.get('visual_model')

        self.key = kwargs.get('key')
        self.iteration = kwargs.get("sample")
        self.start_time = 0
        self.root = node
        self.save = True

        self.energy_values = [] 
        self.all_z_coords = []
        self.volume_values = []

        self.current_main_step = 0
        self.max_main_steps = kwargs.get('max_main_steps', 200) 
        
        self.directory = kwargs.get('directory')
        self.mesh_filename = kwargs.get('mesh_filename', 'unknown')
        
        self.original_positions = None
        self.linear_modes_np = None 
        self.latent_dim = 0       
        self.current_z_for_step = None 
        self.z_scale_factor = kwargs.get('z_scale_factor', 50.0) # New configurable scale factor for z
        print(f"Using Z scale factor: {self.z_scale_factor}")

        self.output_subdir = "z_plausibility_analysis"


      


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
        self.linear_modes_np = self.routine.linear_modes # This should be a torch tensor
        self.latent_dim = self.routine.latent_dim
        self.original_positions = np.copy(self.MO1.position.value) # Store original positions

        # --- Prepare for Saving (if enabled) ---
        # Use the directory name passed during scene creation or default
        self.directory = str(self.routine.latent_dim) + "_modes"

        output_base_dir = 'z_debug' # Or read from config
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
        



    def onAnimateBeginEvent(self, event):
        """
        Called by SOFA's animation loop before each physics step.
        Applies incremental force based on substep, with a random direction per main step.
        """
        if self.original_positions is None or self.linear_modes_np is None:
            print("Error: Original positions or linear_modes_np not initialized. Skipping step.")
            if self.root: self.root.animate = False
            return

        z = np.random.rand(self.latent_dim) * 2 - 1 # Random values between -1 and 1
        z = z * self.z_scale_factor # Apply the new scale factor 
        self.current_z_for_step = z.copy()
        self.all_z_coords.append(z.copy()) # Store the current z for analysis
        
        displacement_flat = np.dot(self.linear_modes_np, z)
        
        num_nodes = self.original_positions.shape[0]
        spatial_dim = self.original_positions.shape[1]

        try:
            displacement_reshaped = displacement_flat.reshape(num_nodes, spatial_dim)
        except ValueError as e:
            print(f"Error reshaping displacement: {e}. Using zero displacement.")
            displacement_reshaped = np.zeros_like(self.original_positions)

        self.MO1.position.value = self.original_positions + displacement_reshaped
        
        if hasattr(self, 'visual_model') and self.visual_model is not None:
            self.visual_model.position.value = self.MO1.position.value 

        self.start_time = process_time()



    def onAnimateEndEvent(self, event):
        """
        Called by SOFA's animation loop after each physics step.
        Performs analysis, stores results, and saves step-specific data to disk.
        """
        displacement = self.MO1.position.value - self.original_positions
        energy = self.computeInternalEnergy(displacement)
        self.energy_values.append(energy) # Store energy for this step

        volume = self.computeVolume(displacement)
        self.volume_values.append(volume)

        if self.save:
            # Define the specific output directory for z_amplitude data
            # self.output_subdir is like 'z_debug/5_modes'
            # self.z_scale_factor is the amplitude scale
            z_amplitude_data_dir = os.path.join(self.output_subdir, f"z_amplitude_{self.z_scale_factor}")
            os.makedirs(z_amplitude_data_dir, exist_ok=True)

            step_number = self.current_main_step # Use current step number for filenames

            # Save displacement as .npy
            disp_filename = os.path.join(z_amplitude_data_dir, f"step_{step_number}_displacement.npy")
            np.save(disp_filename, displacement)

            # Prepare data for JSON
            # self.current_z_for_step was set in onAnimateBeginEvent
            step_data = {
                "step": step_number,
                "z_coords": self.current_z_for_step.tolist() if self.current_z_for_step is not None else None,
                "energy": energy,
                "volume": volume,
                "z_scale_factor": self.z_scale_factor,
                "displacement_file": os.path.basename(disp_filename) # Relative path to displacement
            }

            # Save other data as .json
            json_filename = os.path.join(z_amplitude_data_dir, f"step_{step_number}_data.json")
            try:
                with open(json_filename, 'w') as f:
                    json.dump(step_data, f, indent=4)
            except Exception as e:
                print(f"Error saving JSON data for step {step_number}: {e}")

        self.current_main_step += 1


        self.end_time = process_time()


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
    
    def computeVolume(self, displacement):
        """
        Compute the volume of the deformed mesh using the displacement vector.

        Args:
            displacement: Displacement vector of the system (NumPy array).

        Returns:
            Volume as a float.
        """
        if not hasattr(self.routine, 'energy_calculator') or not hasattr(self.routine.energy_calculator, '_compute_deformed_volume'):
            print("Error: _compute_deformed_volume method not found in routine.energy_calculator.")
            return float('nan')

        device = self.routine.device # Get the device from the routine instance

        # Convert NumPy array to PyTorch tensor
        displacement_tensor = torch.tensor(displacement, dtype=torch.float64, device=device)

        # Add batch dimension if the energy calculator expects it
        # Assuming _compute_deformed_volume might expect similar input shape as energy_calculator
        if displacement_tensor.dim() == 2: # (N, D)
             displacement_tensor = displacement_tensor.unsqueeze(0) # (1, N, D)
        elif displacement_tensor.dim() == 1: # Flat (N*D)
            num_nodes = self.original_positions.shape[0]
            spatial_dim = self.original_positions.shape[1]
            try:
                displacement_tensor = displacement_tensor.reshape(1, num_nodes, spatial_dim)
            except Exception as e:
                print(f"Error reshaping flat displacement in computeVolume: {e}")
                return float('nan')


        # Compute volume using the tensor
        # Use torch.no_grad() if gradients are not needed here
        with torch.no_grad():
            volume_value = self.routine.energy_calculator._compute_deformed_volume(displacement_tensor.squeeze(0))
        
        # If a batch dimension was added, remove it from the result if necessary
        if hasattr(volume_value, 'dim') and volume_value.dim() > 0 and volume_value.shape[0] == 1: # Check if it's a tensor
             volume_value = volume_value.squeeze(0)

        return volume_value.item()
    

    def close(self):
        print("\n--- Simulation Finished ---")
        # if not self.all_z_coords or not self.energy_values or len(self.all_z_coords) != len(self.energy_values):
        #     print("Insufficient or mismatched data for Z vs Energy analysis. Skipping.")
        #     if self.all_z_coords:
        #         all_z_coords_np_partial = np.array(self.all_z_coords)
        #         if all_z_coords_np_partial.size > 0:
        #             print("\n--- Z Coordinate Statistics (Partial) ---")
        #             print(f"Shape of z array: {all_z_coords_np_partial.shape}")
        #     print("Closing simulation")
        #     return

        all_z_coords_np = np.array(self.all_z_coords)
        energy_values_np = np.array(self.energy_values)

        valid_indices = ~np.isnan(energy_values_np)
        if all_z_coords_np.ndim == 2: # Ensure all_z_coords_np is 2D before using np.any with axis=1
            valid_indices = valid_indices & ~np.any(np.isnan(all_z_coords_np), axis=1)
        elif all_z_coords_np.ndim == 1 and all_z_coords_np.size > 0: # Handle case where only one z sample might be NaN
             valid_indices = valid_indices & ~np.isnan(all_z_coords_np)


        if not np.any(valid_indices):
            print("No valid energy or z values found after NaN filtering. Skipping Z vs Energy analysis.")
            print("Closing simulation")
            return
            
        all_z_coords_np = all_z_coords_np[valid_indices]
        energy_values_np = energy_values_np[valid_indices]

        if all_z_coords_np.size == 0 or energy_values_np.size == 0:
            print("No valid data after NaN filtering. Skipping Z vs Energy analysis.")
            print("Closing simulation")
            return
        
        # Ensure all_z_coords_np is 2D for subsequent analysis if it became 1D after filtering
        if all_z_coords_np.ndim == 1 and self.latent_dim > 0:
            if all_z_coords_np.size == self.latent_dim : # A single valid z sample
                 all_z_coords_np = all_z_coords_np.reshape(1, self.latent_dim)
            else: # Data shape mismatch after filtering, problematic for further analysis
                print(f"Warning: Z data shape is {all_z_coords_np.shape} after filtering, expected 2D. Skipping some Z-specific analyses.")
                # Fallback or skip Z-specific parts if shape is not (N, latent_dim)

        plots_subdir = os.path.join(self.output_subdir, 'plots')
        if self.save and not os.path.exists(plots_subdir):
            os.makedirs(plots_subdir, exist_ok=True)

        # --- Analysis of Z for Plausible Energies ---
        plausible_energy_threshold = 1000.0
        plausible_indices = energy_values_np < plausible_energy_threshold
        
        if np.any(plausible_indices) and all_z_coords_np.ndim == 2: # Ensure 2D for indexing
            z_plausible = all_z_coords_np[plausible_indices]
            # energy_plausible = energy_values_np[plausible_indices] # Not directly used below
            print(f"\n--- Analysis of Z for Plausible Energies (Energy < {plausible_energy_threshold}) ---")
            print(f"Number of plausible samples: {z_plausible.shape[0]} out of {len(energy_values_np)}")
            if z_plausible.shape[0] > 0: # z_plausible will be 2D if all_z_coords_np was
                print("Statistics for Z components in plausible samples:")
                print(f"{'Mode':<5} {'Mean':<10} {'Std Dev':<10} {'Min':<10} {'Max':<10}")
                for i in range(z_plausible.shape[1]): # Iterate over latent dimensions
                    mode_values = z_plausible[:, i]
                    mean_val = np.mean(mode_values)
                    std_val = np.std(mode_values)
                    min_val = np.min(mode_values)
                    max_val = np.max(mode_values)
                    print(f"{i:<5} {mean_val:<10.4f} {std_val:<10.4f} {min_val:<10.4f} {max_val:<10.4f}")
                
                print("\nSuggested Z component ranges based on plausible energy samples:")
                for i in range(z_plausible.shape[1]):
                    min_val = np.min(z_plausible[:, i])
                    max_val = np.max(z_plausible[:, i])
                    print(f"  Z[{i}]: Min ~ {min_val:.3f}, Max ~ {max_val:.3f}")
            print("---------------------------------------------------------------------\n")
        elif not np.any(plausible_indices):
            print(f"\nNo samples found with energy < {plausible_energy_threshold}.")
            print("Consider reducing 'z_scale_factor' if energies are consistently too high.")
            print("---------------------------------------------------------------------\n")
        elif all_z_coords_np.ndim != 2:
            print(f"\nSkipping plausible energy Z analysis as Z data is not 2D (shape: {all_z_coords_np.shape}).")

        # --- Volume Analysis ---
        print("\n--- Volume Analysis ---")
        volume_values_np = np.array(self.volume_values)
        
        # Filter NaN values from volume and corresponding Z coordinates
        valid_volume_indices = ~np.isnan(volume_values_np)
        volume_values_np_filtered = volume_values_np[valid_volume_indices]
        
        # Ensure all_z_coords_np is also filtered by valid_volume_indices
        # and is 2D before proceeding with Z-specific analysis
        z_coords_for_volume_analysis = None
        if all_z_coords_np.ndim == 2 and all_z_coords_np.shape[0] == len(volume_values_np):
            z_coords_for_volume_analysis = all_z_coords_np[valid_volume_indices]
        elif all_z_coords_np.ndim == 1 and len(all_z_coords_np) == len(volume_values_np) and self.latent_dim > 0:
             # Handle case where all_z_coords_np might be 1D if only one sample was collected
             # and needs reshaping if it corresponds to a single multi-dimensional z vector
            if all_z_coords_np.size == self.latent_dim and np.sum(valid_volume_indices) == 1:
                z_coords_for_volume_analysis = all_z_coords_np[valid_volume_indices].reshape(1, self.latent_dim)
            elif np.sum(valid_volume_indices) > 0 : # Multiple valid samples, but original was 1D (unlikely for multi-dim z)
                # This case is less likely if z is multi-dimensional.
                # If z is 1D (latent_dim=1), then this is fine.
                if self.latent_dim == 1:
                    z_coords_for_volume_analysis = all_z_coords_np[valid_volume_indices].reshape(-1, 1)
                else:
                    print("Warning: Z data shape mismatch for volume analysis after NaN filtering.")
        else:
            print("Warning: Z data not available or shape mismatch for detailed volume-Z analysis.")


        if volume_values_np_filtered.size == 0:
            print("No valid volume values after NaN filtering. Skipping volume analysis.")
        else:
            # Define a threshold for "large" volume deviations (e.g., 5%)
            # Assuming a baseline volume around which deviations are measured.
            # If an "original" or "expected" volume is known, use that. Otherwise, use mean/median of observed.
            # For now, let's use a fixed baseline or mean of observed as an example.
            # If self.routine.energy_calculator has an initial volume, that would be ideal.
            # Using a placeholder baseline_volume for now.
            baseline_volume = 10.0 # Placeholder - replace with actual expected/initial volume if available

            volume_deviation_percentage = 10.0 # Allow larger deviation for broader analysis
            max_volume = baseline_volume * (1 + volume_deviation_percentage / 100)
            min_volume = baseline_volume * (1 - volume_deviation_percentage / 100)
            
            # Identify indices where volume is within the acceptable range
            acceptable_volume_indices = np.where((volume_values_np_filtered >= min_volume) & (volume_values_np_filtered <= max_volume))[0]
            out_of_range_indices = np.where((volume_values_np_filtered < min_volume) | (volume_values_np_filtered > max_volume))[0]

            print(f"Number of valid volume values: {len(volume_values_np_filtered)}")
            print(f"Baseline volume for deviation: {baseline_volume:.4f}")
            print(f"Acceptable volume range ({volume_deviation_percentage}% deviation): [{min_volume:.4f}, {max_volume:.4f}]")
            print(f"Number of volume values within acceptable range: {len(acceptable_volume_indices)}")
            print(f"Number of volume values outside acceptable range: {len(out_of_range_indices)}")

            if len(out_of_range_indices) > 0 and len(out_of_range_indices) < 20: # Print details if not too many
                print("Indices and values of out-of-range volume values (from filtered list):")
            for i in out_of_range_indices[:20]: # Limit printing
                print(f"  Index (in filtered): {i}, Volume: {volume_values_np_filtered[i]:.4f}")
            
            # --- Statistics of Z for Acceptable Volumes ---
            if z_coords_for_volume_analysis is not None and z_coords_for_volume_analysis.ndim == 2 and len(acceptable_volume_indices) > 0:
                z_acceptable_volume = z_coords_for_volume_analysis[acceptable_volume_indices]
                
                if z_acceptable_volume.shape[0] > 0:
                    print(f"\n--- Statistics of Z Components for Samples with Acceptable Volume (Original Modes) ---")
                    print(f"Number of samples with acceptable volume: {z_acceptable_volume.shape[0]}")
                    print(f"{'Mode':<5} {'Mean':<10} {'Std Dev':<10} {'Min':<10} {'Max':<10}")
                    num_latent_dims = z_acceptable_volume.shape[1] # Get latent_dim from the data
                    for i in range(num_latent_dims): 
                        mode_values = z_acceptable_volume[:, i]
                        mean_val = np.mean(mode_values)
                        std_val = np.std(mode_values)
                        min_val = np.min(mode_values)
                        max_val = np.max(mode_values)
                        print(f"{i:<5} {mean_val:<10.4f} {std_val:<10.4f} {min_val:<10.4f} {max_val:<10.4f}")
                        
                    print("\nSuggested Z component ranges (original modes) based on acceptable volume samples:")
                    for i in range(num_latent_dims):
                        min_val = np.min(z_acceptable_volume[:, i])
                        max_val = np.max(z_acceptable_volume[:, i])
                        print(f"  Z[{i}]: Min ~ {min_val:.3f}, Max ~ {max_val:.3f}")

                    # --- PCA Analysis of Z for Acceptable Volumes ---
                    if z_acceptable_volume.shape[0] > num_latent_dims : # Need more samples than features for meaningful PCA
                        print(f"\n--- PCA Analysis of Z Components for Samples with Acceptable Volume ---")
                        try:
                            pca = PCA(n_components=num_latent_dims)
                            z_pca_transformed = pca.fit_transform(z_acceptable_volume)
                            
                            print("Explained variance ratio by each principal component:")
                            for i, variance_ratio in enumerate(pca.explained_variance_ratio_):
                                print(f"  PC{i}: {variance_ratio:.4f} (Cumulative: {np.sum(pca.explained_variance_ratio_[:i+1]):.4f})")

                            print("\nPlausible ranges along Principal Components (PC):")
                            for i in range(num_latent_dims):
                                pc_values = z_pca_transformed[:, i]
                                print(f"  PC{i}: Min ~ {np.min(pc_values):.3f}, Max ~ {np.max(pc_values):.3f}, Mean ~ {np.mean(pc_values):.3f}, StdDev ~ {np.std(pc_values):.3f}")
                            
                            # You can store pca.mean_ and pca.components_ if you want to transform new Z vectors later:
                            # new_z_transformed = pca.transform(new_z_original.reshape(1, -1))
                            # Then check if new_z_transformed components are within the PC ranges.

                            if self.save:
                                # Plot explained variance
                                plt.figure(figsize=(10, 6))
                                plt.bar(range(num_latent_dims), pca.explained_variance_ratio_, alpha=0.7, label='Individual explained variance')
                                plt.plot(range(num_latent_dims), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--', color='r', label='Cumulative explained variance')
                                plt.xlabel('Principal Component Index')
                                plt.ylabel('Explained Variance Ratio')
                                plt.title('PCA Explained Variance for Z (Acceptable Volume Samples)')
                                plt.xticks(range(num_latent_dims))
                                plt.legend(loc='best')
                                plt.grid(True, linestyle=':', alpha=0.7)
                                plt.tight_layout()
                                pca_var_path = os.path.join(plots_subdir, "z_acceptable_vol_pca_variance.png")
                                try: plt.savefig(pca_var_path); print(f"PCA variance plot saved to {pca_var_path}")
                                except Exception as e: print(f"Error saving PCA variance plot: {e}")
                                plt.close()

                                # Scatter plot of first two PCs if num_latent_dims >= 2
                                if num_latent_dims >= 2:
                                    plt.figure(figsize=(8, 7))
                                    plt.scatter(z_pca_transformed[:, 0], z_pca_transformed[:, 1], alpha=0.5, s=15)
                                    plt.xlabel('Principal Component 1')
                                    plt.ylabel('Principal Component 2')
                                    plt.title('Plausible Z (Acceptable Volume) in PCA Space (First 2 PCs)')
                                    plt.axhline(0, color='grey', lw=0.5); plt.axvline(0, color='grey', lw=0.5)
                                    plt.grid(True, linestyle=':', alpha=0.5)
                                    plt.tight_layout()
                                    pca_scatter_path = os.path.join(plots_subdir, "z_acceptable_vol_pca_scatter.png")
                                    try: plt.savefig(pca_scatter_path); print(f"PCA scatter plot saved to {pca_scatter_path}")
                                    except Exception as e: print(f"Error saving PCA scatter plot: {e}")
                                    plt.close()

                        except Exception as e_pca:
                            print(f"Error during PCA analysis for acceptable volume Z: {e_pca}")
                    elif z_acceptable_volume.shape[0] > 0 :
                        print("\nNot enough samples for full PCA analysis of Z for acceptable volumes (samples <= features).")
                else:
                    print("No Z data available for samples with acceptable volume (after filtering).")
            elif z_coords_for_volume_analysis is None:
                print("\nSkipping Z statistics for acceptable volumes as Z data was not appropriately filtered or available for volume analysis.")
            elif len(acceptable_volume_indices) == 0:
                print("\nNo samples found with volume in the acceptable range. Cannot compute Z statistics for them.")
            
            # Use volume_values_np_filtered for plotting
            if self.save and len(volume_values_np_filtered) > 0: # Check if there's anything to plot
                plt.figure(figsize=(10, 6))
                plt.plot(volume_values_np_filtered, color='purple', label='Volume')
                plt.axhline(y=min_volume, color='red', linestyle='--', label=f'Min Acceptable Vol ({min_volume:.2f})')
                plt.axhline(y=max_volume, color='blue', linestyle='--', label=f'Max Acceptable Vol ({max_volume:.2f})')
                plt.axhline(y=baseline_volume, color='green', linestyle=':', label=f'Baseline Vol ({baseline_volume:.2f})')
                plt.title('Volume Values Over Time (NaNs Filtered)')
                plt.xlabel('Step (after NaN filter)'); plt.ylabel('Volume')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
                volume_plot_path = os.path.join(plots_subdir, "volume_plot_filtered.png")
                try: plt.savefig(volume_plot_path); print(f"Filtered volume plot saved to {volume_plot_path}")
                except Exception as e: print(f"Error saving filtered volume plot: {e}")
                plt.close()
            elif self.save:
                print("Skipping volume plot as no valid volume data after NaN filtering.")
        print("-------------------------------------------\n")
        # The original plot block is removed from here as it's now conditional and uses filtered data.
        # The user's original plot block started after the $SELECTION_PLACEHOLDER$
        # Ensure the next lines in the original script correctly use `volume_values_np_filtered` if they intend to plot.
        # However, the provided original script has its plot block *after* the selection.
        # The plot block below the selection should be updated or removed if this new plot is sufficient.
        # For now, I will assume the user wants to keep the original plot block as well,
        # but it should ideally use `volume_values_np_filtered`.
        # To avoid conflict, I've named the new plot "volume_plot_filtered.png".
        # The original plot used `volume_values_np` which might contain NaNs.
        # And it used `min_volume` and `max_volume` which are now defined inside this block.
        # To make the original plot block work correctly if it's kept, it would need access to these
        # `min_volume` and `max_volume` or redefine them.
        # For clarity, I will assume the user wants THIS plot to be the primary volume plot.
        # The original plot block after the selection will be plotting `volume_values_np` (which could have NaNs)
        # and its own `min_volume`, `max_volume` if they are redefined there.
        # The current structure means the original plot block will use the `min_volume` and `max_volume`
        # calculated in this new block if `volume_values_np_filtered` was not empty.
        
        # Plot volume values
        if self.save:
            plt.figure(figsize=(10, 6))
            plt.plot(volume_values_np, color='purple')
            plt.axhline(y=min_volume, color='red', linestyle='--', label=f'Min Volume ({min_volume:.2f})')
            plt.axhline(y=max_volume, color='red', linestyle='--', label=f'Max Volume ({max_volume:.2f})')
            plt.title('Volume Values Over Time')
            plt.xlabel('Step'); plt.ylabel('Volume')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
            volume_plot_path = os.path.join(plots_subdir, "volume_plot.png")
            try: plt.savefig(volume_plot_path); print(f"Volume plot saved to {volume_plot_path}")
            except Exception as e: print(f"Error saving volume plot: {e}")
            plt.close()
        print("-------------------------------------------\n")
        # --- Existing Z Coordinate Statistics (Overall) ---
        if all_z_coords_np.ndim == 2:
            print("\n--- Overall Z Coordinate Statistics ---")
            num_latent_dims = all_z_coords_np.shape[1]
            print(f"{'Mode':<5} {'Mean':<10} {'Std Dev':<10} {'Min':<10} {'Max':<10}")
            for i in range(num_latent_dims):
                mode_values = all_z_coords_np[:, i]
                mean_val = np.mean(mode_values)
                std_val = np.std(mode_values)
                min_val = np.min(mode_values)
                max_val = np.max(mode_values)
                print(f"{i:<5} {mean_val:<10.4f} {std_val:<10.4f} {min_val:<10.4f} {max_val:<10.4f}")
            print("-------------------------------------------\n")
        # ... (print energy stats and save histogram as before) ...
        if self.save:
            plt.figure(figsize=(10, 6))
            plt.hist(energy_values_np, bins=50, color='skyblue', edgecolor='black')
            plt.title('Histogram of Collected Energies')
            plt.xlabel('Energy'); plt.ylabel('Frequency')
            plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
            hist_path = os.path.join(plots_subdir, "energy_histogram.png")
            try: plt.savefig(hist_path); print(f"Energy histogram saved to {hist_path}")
            except Exception as e: print(f"Error saving energy histogram: {e}")
            plt.close()

        # --- Existing Scatter Plots and Correlation Analysis ---
        print("\n--- Energy vs. Individual Z Components (Scatter & Correlation) ---")
        correlations = []
        if all_z_coords_np.ndim == 2:
            for i in range(num_latent_dims):
                z_component = all_z_coords_np[:, i]
                if self.save:
                    plt.figure(figsize=(8, 5))
                    plt.scatter(z_component, energy_values_np, alpha=0.5, s=10)
                    plt.title(f'Energy vs. Z Component {i}'); plt.xlabel(f'Z[{i}]'); plt.ylabel('Energy')
                    plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
                    plot_path = os.path.join(plots_subdir, f"energy_vs_z{i}_scatter.png")
                    try: plt.savefig(plot_path)
                    except Exception as e: print(f"Error saving plot energy_vs_z{i}_scatter.png: {e}")
                    plt.close()
                try:
                    correlation, _ = pearsonr(z_component, energy_values_np)
                    correlations.append(correlation)
                    print(f"Pearson Correlation (Energy vs. Z[{i}]): {correlation:.4f}")
                except Exception as e_corr:
                    print(f"Could not compute correlation for Z[{i}]: {e_corr}")
                    correlations.append(float('nan'))
            if self.save and correlations:
                plt.figure(figsize=(max(10, num_latent_dims * 0.5), 6))
                plt.bar(range(num_latent_dims), correlations, color='coral')
                plt.title('Pearson Correlation: Energy vs. Z Components'); plt.xlabel('Z Component Index'); plt.ylabel('Correlation Coefficient')
                plt.xticks(range(num_latent_dims)); plt.grid(True, axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
                corr_plot_path = os.path.join(plots_subdir, "energy_z_correlations_bar.png")
                try: plt.savefig(corr_plot_path); print(f"Correlations bar plot saved to {corr_plot_path}")
                except Exception as e: print(f"Error saving correlations bar plot: {e}")
                plt.close()
        print("-------------------------------------------\n")

        # --- NEW: Multiple Linear Regression ---
        if all_z_coords_np.ndim == 2 and all_z_coords_np.shape[0] > all_z_coords_np.shape[1]: # More samples than features
            print("\n--- Multiple Linear Regression (Energy ~ Z) ---")
            try:
                lin_reg_model = LinearRegression(fit_intercept=True)
                lin_reg_model.fit(all_z_coords_np, energy_values_np)
                print(f"Intercept (c0): {lin_reg_model.intercept_:.4e}")
                print("Coefficients (c_i for Z_i):")
                for i, coef in enumerate(lin_reg_model.coef_):
                    print(f"  Z[{i}]: {coef:.4e}")
                
                # Optional: Plot coefficients
                if self.save:
                    plt.figure(figsize=(max(10, num_latent_dims * 0.5), 6))
                    plt.bar(range(num_latent_dims), lin_reg_model.coef_, color='mediumseagreen')
                    plt.title('Linear Regression Coefficients: Energy vs. Z Components')
                    plt.xlabel('Z Component Index'); plt.ylabel('Coefficient Value')
                    plt.xticks(range(num_latent_dims)); plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    lr_coef_path = os.path.join(plots_subdir, "energy_z_linreg_coeffs.png")
                    try: plt.savefig(lr_coef_path); print(f"Linear regression coefficients plot saved to {lr_coef_path}")
                    except Exception as e: print(f"Error saving linreg coeffs plot: {e}")
                    plt.close()

            except Exception as e_lr:
                print(f"Error during linear regression: {e_lr}")
            print("-------------------------------------------\n")

        # --- NEW: Compare Z Distributions for Low vs. High Energy ---
        if all_z_coords_np.ndim == 2 and len(energy_values_np) > 20: # Need enough data
            print("\n--- Z Component Distributions for Low vs. High Energy States ---")
            try:
                # Define low/high energy groups (e.g., bottom/top 10%)
                percentile_threshold = 10 
                low_energy_threshold = np.percentile(energy_values_np, percentile_threshold)
                high_energy_threshold = np.percentile(energy_values_np, 100 - percentile_threshold)

                z_low_energy = all_z_coords_np[energy_values_np <= low_energy_threshold]
                z_high_energy = all_z_coords_np[energy_values_np >= high_energy_threshold]

                if z_low_energy.shape[0] > 1 and z_high_energy.shape[0] > 1: # Need data in both groups
                    for i in range(num_latent_dims):
                        data_to_plot = [z_low_energy[:, i], z_high_energy[:, i]]
                        if self.save:
                            plt.figure(figsize=(6, 7)) # Adjusted for better boxplot display
                            bp = plt.boxplot(data_to_plot, patch_artist=True, widths=0.6)
                            colors = ['lightblue', 'lightcoral']
                            for patch, color in zip(bp['boxes'], colors):
                                patch.set_facecolor(color)
                            plt.xticks([1, 2], [f'Low Energy\n(Bottom {percentile_threshold}%)', f'High Energy\n(Top {percentile_threshold}%)'])
                            plt.ylabel(f'Value of Z[{i}]')
                            plt.title(f'Distribution of Z[{i}] for Low vs. High Energy States')
                            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                            plt.tight_layout()
                            comp_box_path = os.path.join(plots_subdir, f"z{i}_low_vs_high_energy_boxplot.png")
                            try: plt.savefig(comp_box_path)
                            except Exception as e: print(f"Error saving Z[{i}] comparison boxplot: {e}")
                            plt.close()
                    print(f"Saved Z component comparison boxplots for low/high energy states to {plots_subdir}")
                else:
                    print("Not enough distinct samples in low/high energy groups for Z distribution comparison.")
            except Exception as e_dist_comp:
                print(f"Error during Z distribution comparison: {e_dist_comp}")
            print("-------------------------------------------\n")


        # --- Existing: Z Vectors for Lowest Energies ---
        print("\n--- Z Vectors for Lowest Energies ---")
        # ... (print lowest energy z vectors as before) ...
        num_lowest_to_show = min(10, len(energy_values_np)) 
        if num_lowest_to_show > 0:
            sorted_indices = np.argsort(energy_values_np)
            print(f"Top {num_lowest_to_show} Z vectors with the lowest energy:")
            for k_idx in range(num_lowest_to_show):
                actual_idx = sorted_indices[k_idx]
                z_vec = all_z_coords_np[actual_idx]
                e_val = energy_values_np[actual_idx]
                z_str = ", ".join([f"{val:.3f}" for val in z_vec])
                print(f"  Rank {k_idx+1}: Energy = {e_val:.4e}, Z = [{z_str}]")
        else:
            print("Not enough data to show lowest energy Z vectors.")
        print("-------------------------------------------\n")
        
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
    visual_model = visual.addObject('OglModel', src='@../DOFs', color='0 1 0 1')
    visual.addObject('BarycentricMapping', input='@../DOFs', output='@./')

    # Add a second model beam with TetrahedronFEMForceField, which is linear



    

    # Create and add controller with all components
    controller = AnimationStepController(rootNode,
                                        exactSolution=exactSolution,
                                        surface_topo=surface_topo,
                                        MO1=MO1, # Real SOFA solution
                                        visual_model=visual_model,
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