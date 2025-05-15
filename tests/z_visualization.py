import pyvista as pv
import numpy as np
import json
import os
import glob
import argparse

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
    parser = argparse.ArgumentParser(description="Visualize Z plausibility simulation data.")
    parser.add_argument("--data_dir", type=str, default="z_debug/1_modes/z_amplitude_300.0",
                        help="Path to the specific z_amplitude_{scale} directory containing step data.")
    parser.add_argument("--mesh_path", type=str, default="mesh/beam_732.msh",
                        help="Path to the original .msh mesh file.")
    parser.add_argument("--gif_path", type=str, default="z_debug/1_modes/output_animation.gif",
                        help="Optional path to save the animation as a GIF.")
    parser.add_argument("--fps", type=int, default=1,
                        help="Frames per second for the GIF.")
    parser.add_argument("--loop_gif", type=int, default=0,
                        help="Number of times the GIF should loop (0 for infinite).")
    parser.add_argument("--interactive", action="store_true",
                        help="Show interactive plotter window instead of just saving GIF.")
    parser.add_argument("--camera_position", nargs='+', type=float, default=None,
                        help="Camera position for PyVista (e.g., x y z for isometric). Example: 1 1 1")


    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        return
    if not os.path.isfile(args.mesh_path):
        print(f"Error: Mesh file not found: {args.mesh_path}")
        return

    # Load the original mesh
    try:
        original_mesh = pv.read(args.mesh_path)
        # Ensure it's UnstructuredGrid for consistent point access
        if not isinstance(original_mesh, pv.UnstructuredGrid):
            original_mesh = original_mesh.cast_to_unstructured_grid()
        print(f"Successfully loaded mesh: {args.mesh_path} with {original_mesh.n_points} points.")
    except Exception as e:
        print(f"Error loading mesh {args.mesh_path}: {e}")
        return

    # Load all simulation step data
    simulation_steps = load_simulation_data(args.data_dir)
    if not simulation_steps:
        print("No simulation data loaded. Exiting.")
        return

    # Setup PyVista plotter
    # If saving GIF and not interactive, use off_screen=True
    off_screen_plotting = bool(args.gif_path) and not args.interactive
    plotter = pv.Plotter(off_screen=off_screen_plotting, window_size=[1024, 768])
    
    if args.camera_position:
        # PyVista expects camera position as a list of tuples or lists
        # e.g. [(pos_x, pos_y, pos_z), (focal_x, focal_y, focal_z), (view_up_x, view_up_y, view_up_z)]
        # For simplicity, we'll just set the position and let PyVista auto-orient if only 3 floats are given.
        if len(args.camera_position) == 3:
            plotter.camera_position = args.camera_position
            plotter.camera.Azimuth(30) # Example adjustments
            plotter.camera.Elevation(20)
        elif len(args.camera_position) == 9: # Full camera spec
             plotter.camera_position = [
                 tuple(args.camera_position[0:3]),
                 tuple(args.camera_position[3:6]),
                 tuple(args.camera_position[6:9])
             ]
        else:
            print("Warning: Invalid camera_position argument. Use 3 (position) or 9 (pos, focal, viewup) floats.")


    # Prepare for GIF
    if args.gif_path:
        plotter.open_gif(args.gif_path, fps=args.fps, loop=args.loop_gif)
        print(f"Preparing to write GIF to {args.gif_path} at {args.fps} FPS.")

    # Store original points for repeated use
    original_points = original_mesh.points.copy()
    
    # Determine a consistent color range for displacement magnitude
    all_displacements_mag = []
    for step_data in simulation_steps:
        displacement = step_data["displacement"]
        if displacement.shape[0] == original_points.shape[0] and displacement.shape[1] == original_points.shape[1]:
            all_displacements_mag.extend(np.linalg.norm(displacement, axis=1))
        else:
            print(f"Warning: Displacement shape mismatch for step {step_data['step_number']}. Expected {original_points.shape}, got {displacement.shape}")


    clim = None
    if all_displacements_mag:
        min_mag = np.min(all_displacements_mag) if all_displacements_mag else 0
        max_mag = np.max(all_displacements_mag) if all_displacements_mag else 1
        if max_mag < 1e-6 : max_mag = 1e-6 # Avoid zero range
        clim = [min_mag, max_mag]
        print(f"Displacement magnitude color range: {clim}")


    # Animation loop
    for i, step_data in enumerate(simulation_steps):
        print(f"Processing step {step_data['step_number']}...")
        
        displacement = step_data["displacement"]
        
        # Ensure displacement array matches mesh points
        if displacement.shape[0] != original_points.shape[0] or \
           displacement.shape[1] != original_points.shape[1]:
            print(f"  Skipping step {step_data['step_number']}: Displacement shape {displacement.shape} "
                  f"does not match original mesh points shape {original_points.shape}.")
            continue

        deformed_mesh = original_mesh.copy() # Work on a copy
        deformed_mesh.points = original_points + displacement
        
        # Compute displacement magnitude for coloring
        disp_magnitude = np.linalg.norm(displacement, axis=1)
        deformed_mesh["Displacement Magnitude"] = disp_magnitude

        plotter.clear_actors() # Clear previous mesh and text
        
        plotter.add_mesh(deformed_mesh, scalars="Displacement Magnitude", cmap="viridis", clim=clim,
                         show_edges=False, scalar_bar_args={'title': 'Disp. Mag.'})

        # Add text annotations
        # Add text annotations with conditional coloring
        energy_color = "red" if step_data['energy'] > 500 else "green"
        volume_color = "green" if 9.7 <= step_data['volume'] <= 10.3 else "red"

        plotter.add_text(f"Step: {step_data['step_number']}", position="upper_left", font_size=10, color="black", shadow=True)
        plotter.add_text(f"Energy: {step_data['energy']:.2f}", position=[0.05, 700], font_size=10, color=energy_color, shadow=True)
        plotter.add_text(f"Volume: {step_data['volume']:.2f}", position=[0.05, 680], font_size=10, color=volume_color, shadow=True)
        plotter.add_text(f"Z-Scale: {step_data['z_scale_factor']:.2f}", position=[0.05, 660], font_size=10, color="black", shadow=True)
        plotter.add_text(f"Z-Coords: {np.array2string(np.array(step_data['z_coords']), precision=2, floatmode='fixed', max_line_width=50)}",
            position=[0.05, 620], font_size=10, color="black", shadow=True)
        if args.interactive and not off_screen_plotting:
            plotter.show(auto_close=False, interactive_update=True) # Render and keep window open
            if i == 0: plotter.camera.reset_clipping_range() # Adjust clipping on first frame
        
        if args.gif_path:
            plotter.write_frame()

    if args.interactive and not off_screen_plotting:
        print("Displaying final frame. Close window to exit.")
        plotter.show() # Keep the last frame visible until closed
    elif not args.gif_path and not args.interactive:
        print("No GIF path provided and not interactive. Displaying first frame and exiting.")
        # Show the first frame if nothing else is happening
        if simulation_steps:
            # (Re-plot the first valid frame for a static view)
            step_data = simulation_steps[0]
            displacement = step_data["displacement"]
            if displacement.shape[0] == original_points.shape[0] and displacement.shape[1] == original_points.shape[1]:
                deformed_mesh = original_mesh.copy()
                deformed_mesh.points = original_points + displacement
                disp_magnitude = np.linalg.norm(displacement, axis=1)
                deformed_mesh["Displacement Magnitude"] = disp_magnitude
                plotter.clear_actors()
                plotter.add_mesh(deformed_mesh, scalars="Displacement Magnitude", cmap="viridis", clim=clim, show_edges=False, scalar_bar_args={'title': 'Disp. Mag.'})
                # Add text annotations with conditional coloring
                energy_color = "red" if step_data['energy'] > 500 else "green"
                volume_color = "green" if 9.7 <= step_data['volume'] <= 10.3 else "red"

                plotter.add_text(f"Step: {step_data['step_number']}", position="upper_left", font_size=10, color="black", shadow=True)
                plotter.add_text(f"Energy: {step_data['energy']:.4e}", position=[0.05, 5.9], font_size=10, color=energy_color, shadow=True)
                plotter.add_text(f"Volume: {step_data['volume']:.4e}", position=[0.05, 5.8], font_size=10, color=volume_color, shadow=True)
                plotter.add_text(f"Z-Scale: {step_data['z_scale_factor']:.2f}", position=[0.05, 5.7], font_size=10, color="black", shadow=True)
                plotter.add_text(f"Z-Coords: {np.array2string(np.array(step_data['z_coords']), precision=2, floatmode='fixed', max_line_width=50)}",
                 position=[0.05, 5.6], font_size=10, color="black", shadow=True)
                plotter.show()


    plotter.close()
    print("Visualization finished.")

if __name__ == "__main__":
    # Example usage:
    # python tests/z_visualization.py \
    #   --data_dir path_to/z_plausibility_output/YYYYMMDD_HHMMSS_or_L_modes/z_amplitude_X.X \
    #   --mesh_path path_to/your_mesh.msh \
    #   --gif_path output_animation.gif \
    #   --fps 10 \
    #   --interactive
    main()