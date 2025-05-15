import pyvista as pv
import numpy as np
import json
import os
import glob
import argparse

import torch # NEW

# --- Imports for loading the model and config ---
import sys
try:
    # This assumes that the script is run from a location where 'training' module is accessible
    # or that SOFA_neural_modes directory is in PYTHONPATH.
    from training.train_sofa import Routine, ResidualNet, load_config 
    # SOFANeoHookeanModel will be imported by Routine from tests.solver
except ImportError as e:
    print(f"ImportError: {e}. Attempting to add project root to sys.path.")
    # Try to add the parent directory of 'tests' to sys.path if script is in 'tests'
    # This is a common structure where 'tests' and 'training' are subdirectories of the project root.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_candidate = os.path.dirname(script_dir) 
    sys.path.insert(0, project_root_candidate)
    try:
        from training.train_sofa import Routine, ResidualNet, load_config
    except ImportError as e2:
        print(f"Failed to import Routine, ResidualNet, load_config even after path adjustment: {e2}")
        print("Please ensure that the SOFA_neural_modes project directory is in your PYTHONPATH,")
        print("or run this script from the SOFA_neural_modes project root directory.")
        sys.exit(1)


def load_simulation_data(data_directory):
    """
    Loads all step data (JSON and displacement NPY) from a given directory.

    Args:
        data_directory (str): Path to the directory containing step_{i}_data.json
                              and step_{i}_displacement.npy files.

    Returns:
        list: A list of dictionaries, where each dictionary contains
              'step_number', 'z_coords', 'energy', 'volume',
              'z_scale_factor', and 'displacement' (NumPy array) for a step.
              Returns an empty list if no data is found.
    """
    json_files = sorted(glob.glob(os.path.join(data_directory, "step_*_data.json")))
    
    all_step_data = []
    if not json_files:
        print(f"No JSON data files found in {data_directory}")
        return all_step_data

    for json_file_path in json_files:
        try:
            with open(json_file_path, 'r') as f:
                step_meta = json.load(f)
            
            step_number = step_meta.get("step")
            displacement_filename = step_meta.get("displacement_file")

            if step_number is None or displacement_filename is None:
                print(f"Warning: Skipping {json_file_path}, missing 'step' or 'displacement_file' key.")
                continue

            displacement_path = os.path.join(data_directory, displacement_filename)
            if not os.path.exists(displacement_path):
                print(f"Warning: Displacement file not found for step {step_number}: {displacement_path}")
                continue
            
            displacement_array = np.load(displacement_path)
            
            all_step_data.append({
                "step_number": step_number,
                "z_coords": step_meta.get("z_coords"),
                "energy": step_meta.get("energy"),
                "volume": step_meta.get("volume"),
                "z_scale_factor": step_meta.get("z_scale_factor"),
                "displacement": displacement_array
            })
        except Exception as e:
            print(f"Error loading data for {json_file_path}: {e}")
            
    # Sort by step number just in case glob didn't return them perfectly sorted by number
    all_step_data.sort(key=lambda x: x["step_number"])
    print(f"Loaded data for {len(all_step_data)} steps from {data_directory}")
    return all_step_data


def main():
    parser = argparse.ArgumentParser(description="Visualize Z plausibility simulation data using a trained Neural Network.")
    parser.add_argument("--data_dir", type=str, default="z_debug/1_modes/z_amplitude_300.0",
                        help="Path to the specific z_amplitude_{scale} directory containing step_N_data.json files.")
    parser.add_argument("--mesh_path", type=str, default="mesh/beam_732.msh",
                        help="Path to the original .msh mesh file.")
    parser.add_argument("--config_path", type=str, default="configs/default.yaml", # NEW
                        help="Path to the training configuration YAML file.")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/best_sofa.pt", # NEW
                        help="Path to the trained model checkpoint (.pt file).")
    parser.add_argument("--gif_path", type=str, default="z_debug/1_modes/output.gif", # Default to None
                        help="Optional path to save the animation as a GIF.")
    parser.add_argument("--fps", type=int, default=1,
                        help="Frames per second for the GIF.")
    parser.add_argument("--loop_gif", type=int, default=0,
                        help="Number of times the GIF should loop (0 for infinite).")
    parser.add_argument("--interactive", action="store_true",
                        help="Show interactive plotter window.")
    parser.add_argument("--camera_position", nargs='+', type=float, default=None,
                        help="Camera position for PyVista (e.g., x y z for isometric or 9 floats for full spec). Example: 1 1 1")

    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}"); return
    if not os.path.isfile(args.mesh_path):
        print(f"Error: Mesh file not found: {args.mesh_path}"); return
    if not os.path.isfile(args.config_path):
        print(f"Error: Config file not found: {args.config_path}"); return
    # Checkpoint path is optional if user wants to see untrained model (though less useful)

    # --- Load Config and Initialize Routine (which includes the model structure) ---
    print(f"Loading configuration from: {args.config_path}")
    cfg = load_config(args.config_path)
    
    print("Initializing SOFA/NN Routine...")
    engine = Routine(cfg)
    engine.model.eval() # Set model to evaluation mode
    print(f"Routine initialized. Model latent dim: {engine.latent_dim}, Output dim: {engine.output_dim}")

    # --- Load the trained model weights ---
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading model checkpoint from: {args.checkpoint_path}")
        engine.load_checkpoint(args.checkpoint_path) # load_checkpoint sets model.eval()
        engine.model.eval() # Explicitly set again
        print("Model checkpoint loaded successfully.")
    else:
        print(f"Warning: Checkpoint path '{args.checkpoint_path}' not provided or not found.")
        print("Proceeding with the (potentially untrained) model initialized by Routine.")
        print("The visualization will show the model's current state.")

    # Load the original mesh
    try:
        original_mesh = pv.read(args.mesh_path)
        if not isinstance(original_mesh, pv.UnstructuredGrid):
            original_mesh = original_mesh.cast_to_unstructured_grid()
        print(f"Successfully loaded mesh: {args.mesh_path} with {original_mesh.n_points} points and {original_mesh.n_cells} cells.")
        original_points = original_mesh.points.copy()
        if original_points.shape[0] * original_points.shape[1] != engine.output_dim:
            print(f"Error: Mesh DOFs ({original_points.shape[0] * original_points.shape[1]}) "
                  f"do not match model output dimension ({engine.output_dim}).")
            return
    except Exception as e:
        print(f"Error loading mesh {args.mesh_path}: {e}"); return

    # Load simulation step metadata (z_coords, etc.)
    simulation_steps_metadata = load_simulation_data(args.data_dir)
    if not simulation_steps_metadata:
        print("No simulation metadata loaded. Exiting."); return

    # --- Pre-computation pass for displacements using the NN and for color limits ---
    all_predicted_displacements_for_clim_calc = []
    computed_displacements_for_animation = []

    print("Pre-computing displacements using the loaded model...")
    for step_idx, step_meta in enumerate(simulation_steps_metadata):
        z_coords_list = step_meta.get("z_coords")

        if z_coords_list is None:
            print(f"  Warning: Skipping step {step_meta['step_number']} for displacement calculation: 'z_coords' not found.")
            computed_displacements_for_animation.append(None) # Placeholder for skipped steps
            continue
        
        if len(z_coords_list) != engine.latent_dim:
            print(f"  Warning: Skipping step {step_meta['step_number']}. z_coords length {len(z_coords_list)} "
                  f"does not match model latent_dim {engine.latent_dim}.")
            computed_displacements_for_animation.append(None)
            continue

        z_tensor = torch.tensor(z_coords_list, device=engine.device, dtype=engine.linear_modes.dtype)
        
        # Ensure correct linear modes are used (first latent_dim modes)
        active_linear_modes = engine.linear_modes[:, :engine.latent_dim] # Shape: [num_dofs, latent_dim]

        with torch.no_grad():
            # u_linear = torch.matmul(z_tensor, active_linear_modes.T) # z:[L], A.T:[L,D] -> u_lin:[D]
            u_linear = torch.einsum('l,dl->d', z_tensor, active_linear_modes) # More explicit: z_l * A_dl -> u_d
            u_correction = engine.model(z_tensor) # model output is [num_dofs]
            u_total_tensor = u_linear + u_correction
        
        predicted_displacement_np = u_total_tensor.cpu().numpy()
        computed_displacements_for_animation.append(predicted_displacement_np)

        # Check shape for color limit calculation
        if predicted_displacement_np.size == engine.output_dim: # MODIFIED HERE
             # Reshape to [num_nodes, dim] if it's flat [num_dofs]
            disp_reshaped = predicted_displacement_np.reshape(original_points.shape)
            all_predicted_displacements_for_clim_calc.extend(np.linalg.norm(disp_reshaped, axis=1))
        else:
            # This case should ideally not happen if z_coords length matches latent_dim
            print(f"  Warning: Size mismatch for predicted displacement at step {step_meta['step_number']}. "
                  f"Expected total DOFs {engine.output_dim}, but got array of shape {predicted_displacement_np.shape} (size {predicted_displacement_np.size}).")


    clim = None
    if all_predicted_displacements_for_clim_calc:
        min_mag = np.min(all_predicted_displacements_for_clim_calc)
        max_mag = np.max(all_predicted_displacements_for_clim_calc)
        if max_mag - min_mag < 1e-7 : max_mag = min_mag + 1e-7 # Avoid zero or too small range
        clim = [min_mag, max_mag]
        print(f"Predicted displacement magnitude color range: {clim}")
    else:
        print("Warning: Could not determine color limits for displacement magnitude.")
        clim = [0, 1e-6] # Default small range
    # --- End Pre-computation pass ---

    # Setup PyVista plotter
    off_screen_plotting = bool(args.gif_path) and not args.interactive
    plotter = pv.Plotter(off_screen=off_screen_plotting, window_size=[1024, 768])
    
    if args.camera_position:
        if len(args.camera_position) == 3:
            plotter.camera_position = args.camera_position
            plotter.camera.Azimuth(30)
            plotter.camera.Elevation(20)
        elif len(args.camera_position) == 9:
             plotter.camera_position = [
                 tuple(args.camera_position[0:3]),
                 tuple(args.camera_position[3:6]),
                 tuple(args.camera_position[6:9])
             ]
        else:
            print("Warning: Invalid camera_position argument. Use 3 or 9 floats.")

    if args.gif_path:
        # Ensure directory for GIF exists
        gif_dir = os.path.dirname(args.gif_path)
        if gif_dir and not os.path.exists(gif_dir):
            os.makedirs(gif_dir, exist_ok=True)
        plotter.open_gif(args.gif_path, fps=args.fps, loop=args.loop_gif)
        print(f"Preparing to write GIF to {args.gif_path} at {args.fps} FPS.")

    # Animation loop
    for i, step_meta in enumerate(simulation_steps_metadata):
        current_step_number = step_meta['step_number']
        print(f"Visualizing step {current_step_number}...")
        
        displacement_np = computed_displacements_for_animation[i]
        
        if displacement_np is None:
            print(f"  Skipping step {current_step_number} due to earlier processing error (e.g., z_coords missing/mismatch).")
            if args.gif_path: # Write a blank or static frame to keep GIF timing
                plotter.clear_actors()
                plotter.add_text(f"Step: {current_step_number}\nData Error", position="center", font_size=12)
                plotter.write_frame()
            continue

        # Reshape flat displacement [num_dofs] to [num_nodes, dim]
        try:
            displacement_reshaped = displacement_np.reshape(original_points.shape)
        except ValueError as e:
            print(f"  Skipping step {current_step_number}: Error reshaping displacement. Expected {original_points.shape}, "
                  f"got flat array of size {displacement_np.size}. Error: {e}")
            if args.gif_path:
                plotter.clear_actors()
                plotter.add_text(f"Step: {current_step_number}\nReshape Error", position="center", font_size=12)
                plotter.write_frame()
            continue
            
        deformed_mesh = original_mesh.copy()
        deformed_mesh.points = original_points + displacement_reshaped
        
        disp_magnitude = np.linalg.norm(displacement_reshaped, axis=1)
        deformed_mesh["Displacement Magnitude"] = disp_magnitude

        plotter.clear_actors()
        plotter.add_mesh(deformed_mesh, scalars="Displacement Magnitude", cmap="viridis", clim=clim,
                         show_edges=False, scalar_bar_args={'title': 'Disp. Mag.'})

        # Add text annotations from metadata
        energy_val = step_meta.get('energy', float('nan'))
        volume_val = step_meta.get('volume', float('nan'))
        z_scale_val = step_meta.get('z_scale_factor', float('nan'))
        z_coords_val = step_meta.get('z_coords', [])

        energy_color = "red" if not np.isnan(energy_val) and energy_val > 500 else "green" # Example condition
        volume_color = "green" if not np.isnan(volume_val) and 0.95 <= (volume_val / (engine.energy_calculator._compute_mesh_volume().item() + 1e-9)) <= 1.05 else "red" # Example condition

        plotter.add_text(f"Step: {current_step_number}", position="upper_left", font_size=10, color="black", shadow=True)
        plotter.add_text(f"Energy (NN): {energy_val:.2e}", position=[0.02, 0.92], font_size=10, color=energy_color, shadow=True, viewport=True)
        plotter.add_text(f"Volume (NN): {volume_val:.3f}", position=[0.02, 0.88], font_size=10, color=volume_color, shadow=True, viewport=True)
        plotter.add_text(f"Z-Scale (NN): {z_scale_val:.2f}", position=[0.02, 0.84], font_size=10, color="black", shadow=True, viewport=True)
        
        z_coords_str = "N/A"
        if z_coords_val:
            z_coords_str = np.array2string(np.array(z_coords_val), precision=2, floatmode='fixed', max_line_width=30, threshold=engine.latent_dim +1)
        plotter.add_text(f"Z (meta): {z_coords_str}", position=[0.02, 0.76], font_size=8, color="black", shadow=True, viewport=True)


        if args.interactive and not off_screen_plotting:
            plotter.show(auto_close=False, interactive_update=True)
            if i == 0: plotter.camera.reset_clipping_range()
        
        if args.gif_path:
            plotter.write_frame()

    if args.interactive and not off_screen_plotting:
        print("Displaying final frame. Close window to exit.")
        plotter.show()
    elif not args.gif_path and not args.interactive:
        print("No GIF path provided and not interactive. Displaying first valid frame and exiting.")
        # Show the first valid frame if nothing else is happening
        first_valid_idx = -1
        for idx, disp in enumerate(computed_displacements_for_animation):
            if disp is not None:
                first_valid_idx = idx
                break
        
        if first_valid_idx != -1:
            step_meta = simulation_steps_metadata[first_valid_idx]
            displacement_np = computed_displacements_for_animation[first_valid_idx]
            try:
                displacement_reshaped = displacement_np.reshape(original_points.shape)
                deformed_mesh = original_mesh.copy()
                deformed_mesh.points = original_points + displacement_reshaped
                disp_magnitude = np.linalg.norm(displacement_reshaped, axis=1)
                deformed_mesh["Displacement Magnitude"] = disp_magnitude
                plotter.clear_actors()
                plotter.add_mesh(deformed_mesh, scalars="Displacement Magnitude", cmap="viridis", clim=clim, show_edges=False, scalar_bar_args={'title': 'Disp. Mag.'})
                
                # Re-add text for the static frame
                energy_val = step_meta.get('energy', float('nan'))
                volume_val = step_meta.get('volume', float('nan'))
                z_scale_val = step_meta.get('z_scale_factor', float('nan'))
                z_coords_val = step_meta.get('z_coords', [])
                energy_color = "red" if not np.isnan(energy_val) and energy_val > 500 else "green"
                volume_color = "green" if not np.isnan(volume_val) and 0.95 <= (volume_val / (engine.energy_calculator._compute_mesh_volume().item() + 1e-9)) <= 1.05 else "red"
                plotter.add_text(f"Step: {step_meta['step_number']}", position="upper_left", font_size=10, color="white", shadow=True)
                plotter.add_text(f"Energy (meta): {energy_val:.2e}", position=[0.02, 0.92], font_size=10, color=energy_color, shadow=True, viewport=True)
                plotter.add_text(f"Volume (meta): {volume_val:.3f}", position=[0.02, 0.88], font_size=10, color=volume_color, shadow=True, viewport=True)
                plotter.add_text(f"Z-Scale (meta): {z_scale_val:.2f}", position=[0.02, 0.84], font_size=10, color="white", shadow=True, viewport=True)
                z_coords_str = "N/A"
                if z_coords_val: z_coords_str = np.array2string(np.array(z_coords_val), precision=2, floatmode='fixed', max_line_width=30, threshold=engine.latent_dim +1)
                plotter.add_text(f"Z (meta): {z_coords_str}", position=[0.02, 0.76], font_size=8, color="white", shadow=True, viewport=True)
                plotter.show()
            except Exception as e:
                print(f"Error displaying static frame: {e}")
        else:
            print("No valid steps found to display statically.")


    plotter.close()
    print("Visualization finished.")

if __name__ == "__main__":
    # Example usage:
    # python tests/z_visualization_NN.py \
    #   --data_dir z_debug/10_modes/z_amplitude_10.0 \
    #   --mesh_path mesh/beam_732.msh \
    #   --config_path configs/default.yaml \
    #   --checkpoint_path checkpoints/best_sofa.pt \
    #   --gif_path z_debug/10_modes/output_nn_animation.gif \
    #   --fps 5 \
    #   --interactive
    main()