import os
import numpy as np
import torch
import argparse
import logging
import matplotlib.pyplot as plt
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io, plot
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import ufl
from dolfinx.io import gmshio
import pyvista
from training.solver import ModernFEMSolver, NeoHookeanEnergyModel

import time
import sys

from tqdm import tqdm


from training.train import setup_logger, Net, load_config, Routine

import traceback


class ForceInferenceModel(torch.nn.Module):
    """Neural network that infers forces from latent vector"""
    def __init__(self, input_dim, output_dim, hidden_layers):
        super(ForceInferenceModel, self).__init__()
        
        # Build network architecture with increasing complexity
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.GELU())
            prev_dim = hidden_dim
            
        # Output layer for force prediction
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        
        self.model = torch.nn.Sequential(*layers)

        # layers[-1].weight.data.normal_(0.0, 0.02)
        # layers[-1].bias.data.fill_(0.0)
        
    def forward(self, x):
        return self.model(x)


    
    

class FEMDataGenerator:
    """Generates training data using FEniCSx simulations with various force configurations"""
    def __init__(self, mesh_path, cfg, output_dir="force_data", visualize=False):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logger with explicit configuration to ensure messages are shown
        self.logger = logging.getLogger("force_data_generator")
        self.logger.setLevel(logging.INFO)
        
        # Check if handler already exists to avoid duplicates
        if not self.logger.handlers:
            # Add console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.ERROR)
            formatter = logging.Formatter('[%(asctime)s] %(name)s %(levelname)s: %(message)s', 
                                        datefmt='%m/%d %H:%M:%S')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # Add file handler
            file_handler = logging.FileHandler(os.path.join(output_dir, 'fem_solver.log'))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Ensure messages propagate up to the root logger
        self.logger.propagate = True
        
        # Material parameters - use proper keys from config
        self.E = cfg.get("material", {}).get("youngs_modulus", 1.0e5)  # Young's modulus
        self.nu = cfg.get("material", {}).get("poissons_ratio", 0.49)   # Poisson's ratio
        
        # Derived material parameters
        self.mu = self.E / (2.0 * (1.0 + self.nu))  # Shear modulus
        self.lmbda = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))  # First Lamé parameter
        
        self.logger.info(f"Material parameters: E={self.E}, nu={self.nu}, mu={self.mu}, lambda={self.lmbda}")
        
        # Setup mesh
        self._setup_mesh(mesh_path)
        
        # Setup function spaces - using the correct API
        self.V = fem.functionspace(self.domain, ("CG", 1, (3,)))  # Displacement space
        self.V_force = fem.functionspace(self.domain, ("CG", 1, (3,)))  # Force space for visualization
        
        # Setup boundary conditions
        self._setup_boundary_conditions()
        
        # Store material parameters
        self.cfg = cfg
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create Routine object for energy calculations
        self.routine = Routine(cfg)

        # Create Neo-Hookean energy model
        mesh_coords = self.domain.geometry.x
        # Convert to tensor
        self.coordinates_tensor = torch.tensor(mesh_coords, device=self.device, dtype=torch.float64)
        
        # Get element connectivity
        tdim = self.domain.topology.dim
        self.domain.topology.create_connectivity(tdim, 0)  # Create connectivity for cells to vertices
        
        # Get cell-vertex connectivity as a numpy array of indices
        cell_vertex_conn = self.domain.topology.connectivity(tdim, 0)
        elements_list = []
        
        # Create elements array by extracting cell-vertex connectivity
        for i in range(self.domain.topology.index_map(tdim).size_local):
            vertices = cell_vertex_conn.links(i)
            elements_list.append(vertices)
        
        # Convert to tensor
        self.elements_tensor = torch.tensor(elements_list, device=self.device, dtype=torch.long)
        
        # Now create the NeoHookeanEnergyModel
        self.neohookean_model = NeoHookeanEnergyModel(
            coordinates=self.coordinates_tensor,
            elements=self.elements_tensor,
            young_modulus=self.E,
            poisson_ratio=self.nu,
            device=self.device,
            precompute=True
        )
                

        self.visualize = visualize
        if self.visualize:
            self._setup_visualization()

        self.logger.info(f"FEM Data Generator initialized with mesh containing {self.domain.topology.index_map(0).size_local} nodes")


    def _setup_visualization(self):
        """Setup interactive visualization for generated data"""
        try:
            # Get mesh coordinates and connectivity
            coords = self.domain.geometry.x
            
            # Create PyVista grid
            self.viz_plotter = pyvista.Plotter(shape=(1, 2), 
                                              title="Force-Displacement Generation",
                                              window_size=[1200, 600], 
                                              off_screen=False)
            
            # Get mesh data from dolfinx
            topology, cell_types, x = plot.vtk_mesh(self.domain)
            grid = pyvista.UnstructuredGrid(topology, cell_types, x)
            
            # Add initial mesh to left viewport
            self.viz_plotter.subplot(0, 0)
            self.viz_plotter.add_text("Applied Force", position="upper_edge", font_size=10)
            self.mesh_actor_left = self.viz_plotter.add_mesh(grid, color='lightblue', show_edges=True)
            
            # Add deformed mesh placeholder to right viewport
            self.viz_plotter.subplot(0, 1)
            self.viz_plotter.add_text("Resulting Displacement", position="upper_edge", font_size=10)
            self.mesh_actor_right = self.viz_plotter.add_mesh(grid, color='lightblue', show_edges=True)
            
            # Add info text
            self.info_actor = self.viz_plotter.add_text("Initializing...", position=(0.02, 0.02), font_size=10)
            
            # Link camera views for easier comparison
            self.viz_plotter.link_views()
            
            # Show the window without blocking
            self.viz_plotter.show(interactive=False, auto_close=False)
            
            self.logger.info("Visualization initialized for data generation")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize visualization: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.visualize = False
    
    def _update_visualization(self, force_params, u_solution):
        """Update visualization with current force and displacement"""
        if not hasattr(self, 'viz_plotter') or not self.visualize:
            return
            
        try:
            # Get mesh and displacement
            topology, cell_types, x = plot.vtk_mesh(self.domain)
            grid = pyvista.UnstructuredGrid(topology, cell_types, x)
            
            # Create force and displacement data for visualization
            u_array = u_solution.x.array.copy()
            force_array = np.zeros_like(u_array)
            
            # Fill force array based on force type
            coords = self.domain.geometry.x
            
            if force_params["type"] == "distributed":
                region = force_params["region"]
                direction = np.array(force_params["direction"])
                
                for i, coord in enumerate(coords):
                    if coord[0] >= region["x_min"] and coord[0] <= region["x_max"]:
                        node_idx = i * 3
                        force_array[node_idx:node_idx+3] = force_params["magnitude"] * direction
                        
            elif force_params["type"] == "torque":
                axis = np.array(force_params["axis"])
                beam_center_y = (self.y_min + self.y_max) / 2
                beam_center_z = (self.z_min + self.z_max) / 2
                
                for i, coord in enumerate(coords):
                    # Skip fixed boundary
                    if np.isclose(coord[0], self.x_min):
                        continue
                        
                    # Calculate r vector (distance from beam axis)
                    r_vec = np.array([0, coord[1] - beam_center_y, coord[2] - beam_center_z])
                    
                    # Skip nodes on the axis
                    if np.linalg.norm(r_vec) < 1e-10:
                        continue
                    
                    # Scale by distance from fixed end
                    scale = (coord[0] - self.x_min) / (self.x_max - self.x_min)
                    
                    # Force is perpendicular to both r_vec and axis
                    force_dir = np.cross(axis, r_vec)
                    if np.linalg.norm(force_dir) > 1e-10:
                        force_dir = force_dir / np.linalg.norm(force_dir)
                        # Set force
                        node_idx = i * 3
                        force_array[node_idx:node_idx+3] = force_params["magnitude"] * scale * np.linalg.norm(r_vec) * force_dir
            
            # Update force visualization (left viewport)
            self.viz_plotter.subplot(0, 0)
            self.viz_plotter.remove_actor(self.mesh_actor_left)
            
            # Set force data on grid
            grid.point_data["Force"] = force_array.reshape(-1, 3)
            grid["force_magnitude"] = np.linalg.norm(force_array.reshape(-1, 3), axis=1)
            
            # Add new force visualization
            self.mesh_actor_left = self.viz_plotter.add_mesh(
                grid, 
                scalars="force_magnitude",
                cmap="viridis",
                show_edges=True,
                clim=[0, np.max(grid["force_magnitude"]) if np.max(grid["force_magnitude"]) > 0 else 1.0]
            )
            
            # Update displacement visualization (right viewport)
            self.viz_plotter.subplot(0, 1)
            self.viz_plotter.remove_actor(self.mesh_actor_right)
            
            # Set displacement data
            grid.point_data["Displacement"] = u_array.reshape(-1, 3)
            grid["displacement_magnitude"] = np.linalg.norm(u_array.reshape(-1, 3), axis=1)
            
            # Create warped mesh
            warped = grid.warp_by_vector("Displacement", factor=1.0)
            
            # Add new displacement visualization
            self.mesh_actor_right = self.viz_plotter.add_mesh(
                warped, 
                scalars="displacement_magnitude",
                cmap="plasma",
                show_edges=True,
                clim=[0, np.max(grid["displacement_magnitude"]) if np.max(grid["displacement_magnitude"]) > 0 else 1.0]
            )
            
            # Update info text
            info_text = (
                f"Force Type: {force_params['type']}\n"
                f"Magnitude: {force_params['magnitude']:.2f}\n"
                f"Max Displacement: {np.max(grid['displacement_magnitude']):.4f}\n"
                f"Mean Displacement: {np.mean(grid['displacement_magnitude']):.4f}\n"
            )
            
            self.viz_plotter.subplot(0, 0)
            self.viz_plotter.remove_actor(self.info_actor)
            self.info_actor = self.viz_plotter.add_text(info_text, position=(0.02, 0.02), font_size=10)
            
            # Render updated scene
            self.viz_plotter.update()
            
        except Exception as e:
            self.logger.error(f"Visualization update error: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def close_visualization(self):
        """Close visualization window"""
        if hasattr(self, 'viz_plotter') and self.viz_plotter is not None:
            self.viz_plotter.close()

    def _setup_mesh(self, mesh_path):
        """Load mesh file and set up domain"""
        # Check file extension
        if mesh_path.endswith(".xdmf"):
            with io.XDMFFile(MPI.COMM_WORLD, mesh_path, "r") as xdmf:
                self.domain = xdmf.read_mesh(name="Grid")
        elif mesh_path.endswith(".msh"):
            self.domain, self.cell_tags, self.facet_tags = gmshio.read_from_msh(mesh_path, MPI.COMM_WORLD, gdim=3)
        else:
            raise ValueError(f"Unsupported mesh format: {mesh_path}")
        
        # Get mesh dimensions
        coords = self.domain.geometry.x
        self.x_min, self.x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
        self.y_min, self.y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
        self.z_min, self.z_max = np.min(coords[:, 2]), np.max(coords[:, 2])
        
        self.logger.info(f"Mesh bounds: X: [{self.x_min:.2f}, {self.x_max:.2f}], " 
                        f"Y: [{self.y_min:.2f}, {self.y_max:.2f}], "
                        f"Z: [{self.z_min:.2f}, {self.z_max:.2f}]")
    
    def _setup_boundary_conditions(self):
        """Set up fixed boundary conditions"""
        # Fixed boundary at x_min (one end of beam)
        def fixed_boundary(x):
            return np.isclose(x[0], self.x_min)
        
        self.boundary_dofs = fem.locate_dofs_geometrical(self.V, fixed_boundary)
        self.bc = fem.dirichletbc(np.zeros(3), self.boundary_dofs, self.V)


    def _apply_external_forces(self, f_array, force_params):
        """Apply ONLY external forces to array based on force parameters - no reaction forces"""
        coords = self.domain.geometry.x
        magnitude = force_params["magnitude"]
        
        if force_params["type"] == "distributed":
            region = force_params["region"]
            direction = np.array(force_params["direction"])
            
            for i, coord in enumerate(coords):
                if coord[0] >= region["x_min"] and coord[0] <= region["x_max"]:
                    # Skip fixed boundary nodes (critical for correct external forces)
                    if np.isclose(coord[0], self.x_min):
                        continue
                        
                    node_idx = i * 3
                    # Apply force to the node
                    f_array[node_idx:node_idx+3] = magnitude * direction
        
        elif force_params["type"] == "torque":
            axis = np.array(force_params["axis"])
            beam_center_y = (self.y_min + self.y_max) / 2
            beam_center_z = (self.z_min + self.z_max) / 2
            
            for i, coord in enumerate(coords):
                # Skip fixed boundary nodes (critical for correct external forces)
                if np.isclose(coord[0], self.x_min):
                    continue
                    
                # Calculate r vector (distance from beam axis)
                r_vec = np.array([0, coord[1] - beam_center_y, coord[2] - beam_center_z])
                
                # Skip nodes on the axis
                if np.linalg.norm(r_vec) < 1e-10:
                    continue
                
                # Scale by distance from fixed end
                scale = (coord[0] - self.x_min) / (self.x_max - self.x_min)
                
                # Force is perpendicular to both r_vec and axis
                force_dir = np.cross(axis, r_vec)
                if np.linalg.norm(force_dir) > 1e-10:
                    force_dir = force_dir / np.linalg.norm(force_dir)
                    # Set force
                    node_idx = i * 3
                    f_array[node_idx:node_idx+3] = magnitude * scale * np.linalg.norm(r_vec) * force_dir
        
        # Explicitly ensure no forces at fixed boundary 
        fixed_boundary_mask = np.isclose(coords[:, 0], self.x_min)
        fixed_nodes = np.where(fixed_boundary_mask)[0]
        for node in fixed_nodes:
            node_idx = node * 3
            f_array[node_idx:node_idx+3] = 0.0
            
        return f_array

    def _assemble_tangent_stiffness(self, u):
        """Assemble tangent stiffness matrix for Neo-Hookean material"""
        # Test and trial functions
        v = ufl.TestFunction(self.V)
        du = ufl.TrialFunction(self.V)
        
        # Kinematics
        d = len(u)
        I = ufl.Identity(d)
        F = I + ufl.grad(u)
        C = F.T * F
        J = ufl.det(F)
        
        # Neo-Hookean strain energy
        psi = (self.mu/2) * (ufl.tr(C) - 3) - self.mu * ufl.ln(J) + (self.lmbda/2) * (ufl.ln(J))**2
        
        # First variation of strain energy (directional derivative)
        F_var = ufl.derivative(psi * ufl.dx, u, v)
        
        # Second variation (Hessian) - this is our tangent stiffness
        J_form = ufl.derivative(F_var, u, du)
        
        return J_form
    

    def generate_force_displacement_pairs(self, num_samples=100, force_types=["distributed", "torque"]):
        """Generate pairs of forces and resulting displacements, storing only applied external forces"""
        self.logger.info(f"Generating {num_samples} force-displacement pairs")
        self.logger.info(f"Using force types: {force_types}")
        
        force_data = []
        displacement_data = []
        latent_vectors = []  # Store latent vectors
        nodal_forces_data = []  # Will store ONLY the applied external forces, not reactions
        fixed_dof_masks = []  # Initialize the list for fixed DOF masks - THIS WAS MISSING
        
        modes = self.routine.linear_modes
        modes_tensor = torch.tensor(modes, dtype=torch.float64, device=self.device).T
        
        type_counts = {force_type: 0 for force_type in force_types}
        failed_samples = 0

        # Find fixed boundary nodes ONCE
        fixed_boundary_mask = np.isclose(self.domain.geometry.x[:, 0], self.x_min)
        fixed_nodes = np.where(fixed_boundary_mask)[0]
        fixed_dofs = np.concatenate([fixed_nodes*3, fixed_nodes*3+1, fixed_nodes*3+2])

            
            
        i = 0
        total_attempts = 0
        
        # Create progress bar with a maximum for total samples
        pbar = tqdm(total=num_samples, desc="Generating force-displacement pairs")
        
        while i < num_samples and total_attempts < num_samples * 2:  # Limit total attempts
            total_attempts += 1
            self.logger.info(f"\n=== Generating sample {i+1}/{num_samples} (attempt {total_attempts}) ===")

            # Randomly select force type
            force_type = np.random.choice(force_types)

            # Generate random force parameters with more modest magnitudes
            force_magnitude = np.random.uniform(20.0, 100.0)  # Reduced magnitudes
            self.logger.info(f"Selected force type: {force_type}, magnitude: {force_magnitude:.2f}")
        
            # Generate force parameters based on type
            if force_type == "distributed":
                # Distributed force on a region
                force_dir = np.random.uniform(-1, 1, 3)
                force_dir = force_dir / np.linalg.norm(force_dir)
                
                # Region parameters (simple box for now)
                x_min = np.random.uniform(0.5 * self.x_max, 0.9 * self.x_max)
                
                self.logger.info(f"Distributed force in region x ≥ {x_min:.2f}, direction {force_dir}")
                
                force_params = {
                    "type": "distributed",
                    "magnitude": force_magnitude,
                    "direction": force_dir,
                    "region": {"x_min": x_min, "x_max": self.x_max}
                }
                
            elif force_type == "torque":
                # Torque around an axis
                axis_dir = np.array([0, 0, 1])  # Start with simple z-axis rotation
                if i > num_samples // 2:  # More complex rotations for later samples
                    axis_dir = np.random.uniform(-1, 1, 3)
                    axis_dir = axis_dir / np.linalg.norm(axis_dir)
                
                self.logger.info(f"Torque around axis {axis_dir}")
                
                force_params = {
                    "type": "torque",
                    "magnitude": force_magnitude,
                    "axis": axis_dir
                }
                
            self.logger.info("Solving FEM system for this force...")
            try:
                # *** CRITICAL CHANGE: Create and store the pure external force field first ***
                # Create modified f_array with zeros at fixed DOFs
                f_array = np.zeros(self.domain.geometry.x.shape[0] * 3)
                self._apply_external_forces(f_array, force_params)
                
                # Explicitly ensure forces are zero at fixed boundaries (redundant but safe)
                f_array[fixed_dofs] = 0.0 
                
                # Store this pure external force field
                original_external_forces = f_array.copy()
                
                # Solve for displacement using this force field
                u_solution = self._solve_fem_system(force_params)
                
                # Store data including force mask
                
                
                # Check if solution is valid (not NaN and not too small)
                u_array = u_solution.x.array.copy()
                u_norm = np.linalg.norm(u_array.reshape(-1, 3), axis=1)
                max_disp = np.max(u_norm)
                
                if np.isnan(max_disp) or max_disp < 1e-10:
                    self.logger.warning(f"Invalid solution detected: {'NaN values' if np.isnan(max_disp) else 'Too small displacement'}")
                    failed_samples += 1
                    continue  # Skip this sample and try again
                
                # Properly compute modal amplitudes
                u_tensor = torch.tensor(u_array, dtype=torch.float64, device=self.device)
                z = torch.matmul(modes_tensor, u_tensor)  # [n_dofs] x [n_dofs, n_modes] = [n_modes]
                z_np = z.numpy()
       
                
                # Store force, displacement, latent vector, and THE ORIGINAL external forces
                force_data.append(force_params)
                displacement_data.append(u_array)
                latent_vectors.append(z_np)
                nodal_forces_data.append(original_external_forces)
                fixed_dof_masks.append(fixed_dofs)  # Save fixed DOF indices
                
                # Update counters
                type_counts[force_type] += 1
                i += 1
                
                # Update progress bar with each successful sample
                pbar.update(1)
                pbar.set_postfix({ 
                    'distributed': type_counts['distributed'], 
                    'torque': type_counts['torque'],
                    'failed': failed_samples
                })
                
            except Exception as e:
                self.logger.error(f"Error solving FEM system: {str(e)}")
                failed_samples += 1
                continue  # Skip this sample and try again
        
        # Close the progress bar
        pbar.close()
        
        # Save data
        self.logger.info(f"Data generation complete. Generated {len(force_data)}/{num_samples} valid samples.")
        self.logger.info(f"Failed samples: {failed_samples}")
        self.logger.info(f"Final force type distribution: {type_counts}")
        
        if len(force_data) > 0:
            self.logger.info(f"Saving data to {self.output_dir}...")
            np.save(os.path.join(self.output_dir, "force_data.npy"), force_data)
            np.save(os.path.join(self.output_dir, "displacement_data.npy"), displacement_data)
            np.save(os.path.join(self.output_dir, "latent_vectors.npy"), latent_vectors)
            np.save(os.path.join(self.output_dir, "nodal_forces.npy"), nodal_forces_data)
            np.save(os.path.join(self.output_dir, "fixed_dofs.npy"), fixed_dofs)  # Save just once, not per sample
        else:
            self.logger.error("No valid samples were generated. Cannot save empty data.")
                
        return force_data, displacement_data, latent_vectors, nodal_forces_data
    

    def _solve_fem_system(self, force_params):
        """Solve FEM system with the given force parameters using NonlinearProblem"""
        self.logger.info(f"=========== Solving FEM system for {force_params['type']} force ===========")
        
        # Initialize displacement function
        u = fem.Function(self.V, name="Displacement")
        
        # Create a function for nodal forces
        f_func = fem.Function(self.V, name="NodalForces")
        f_array = np.zeros_like(f_func.x.array)
            
        # Get coordinates of mesh nodes
        coords = self.domain.geometry.x
        
        # Scale down force magnitude for stability
        initial_scale = 0.001  # Start very small
        magnitude = force_params["magnitude"] * initial_scale
        self.logger.info(f"Starting with very small force: {magnitude:.4f} (scale={initial_scale})")
    
        
        # Apply force directly to nodes based on type
        if force_params["type"] == "distributed":
            region = force_params["region"]
            direction = np.array(force_params["direction"])
            self.logger.info(f"Region x_min={region['x_min']:.4f}, x_max={region['x_max']:.4f}")
            self.logger.info(f"Direction: {direction}, |direction|={np.linalg.norm(direction):.4f}")
            
            # Apply force to all nodes in the region
            for i, coord in enumerate(coords):
                if coord[0] >= region["x_min"] and coord[0] <= region["x_max"]:
                    node_idx = i * 3
                    # Apply force to the node
                    f_array[node_idx:node_idx+3] = magnitude * direction
        
        elif force_params["type"] == "torque":
            axis = np.array(force_params["axis"])
            self.logger.info(f"Torque axis: {axis}, |axis|={np.linalg.norm(axis):.4f}")
            
            beam_center_y = (self.y_min + self.y_max) / 2
            beam_center_z = (self.z_min + self.z_max) / 2
            
            for i, coord in enumerate(coords):
                # Skip fixed boundary
                if np.isclose(coord[0], self.x_min):
                    continue
                    
                # Calculate r vector (distance from beam axis)
                r_vec = np.array([0, coord[1] - beam_center_y, coord[2] - beam_center_z])
                
                # Skip nodes on the axis
                if np.linalg.norm(r_vec) < 1e-10:
                    continue
                
                # Scale by distance from fixed end
                scale = (coord[0] - self.x_min) / (self.x_max - self.x_min)
                
                # Force is perpendicular to both r_vec and axis
                force_dir = np.cross(axis, r_vec)
                if np.linalg.norm(force_dir) > 1e-10:
                    force_dir = force_dir / np.linalg.norm(force_dir)
                    # Set force
                    node_idx = i * 3
                    f_array[node_idx:node_idx+3] = magnitude * scale * np.linalg.norm(r_vec) * force_dir
        
        ## Set initial forces
        f_func.x.array[:] = f_array
        
        # Define test and trial functions
        v = ufl.TestFunction(self.V)
        du = ufl.TrialFunction(self.V)
        
        # Kinematics and energy - keeping the same
        d = len(u)
        I = ufl.Identity(d)
        F_tensor = I + ufl.grad(u)
        C = F_tensor.T * F_tensor
        Ic = ufl.tr(C)
        J = ufl.det(F_tensor)
        
        # Neo-Hookean strain energy density
        psi = (self.mu/2) * (Ic - 3) - self.mu * ufl.ln(J) + (self.lmbda/2) * (ufl.ln(J))**2
        
        # Potential energy (internal work)
        internal_energy = psi * ufl.dx
        
        # External work as UFL expression (not Form)
        external_work_expr = ufl.inner(f_func, v) * ufl.dx
        
        # Residual form (weak form) as UFL expression
        residual_expr = ufl.derivative(internal_energy, u, v) - external_work_expr
        
        # Jacobian as UFL expression
        jacobian_expr = ufl.derivative(residual_expr, u, du)
        
        # CHANGE: Gradual force application using MORE continuation steps
        num_steps = 20  # Doubled for better convergence
        force_scales = np.logspace(np.log10(initial_scale), np.log10(1.0), num_steps)
        
        self.logger.info(f"Using {num_steps} continuation steps with force scales: {force_scales}")
        
        # Keep track of last successful solution
        last_successful_u = None
        
        for step, scale in enumerate(force_scales):
            try:
                # Update force magnitude
                actual_magnitude = force_params["magnitude"] * scale
                self.logger.info(f"Step {step+1}/{num_steps}: scale={scale:.6f}, magnitude={actual_magnitude:.4f}")
                
                # Scale the force array
                scale_ratio = scale / (force_scales[step-1] if step > 0 else initial_scale)
                f_func.x.array[:] *= scale_ratio
                
                # NOW convert to Form objects
                residual_form = fem.form(residual_expr)
                jacobian_form = fem.form(jacobian_expr)
                
                # Create nonlinear problem
                problem = fem.petsc.NonlinearProblem(residual_form, u, [self.bc], J=jacobian_form)
                
                # Create solver with IMPROVED settings
                solver = NewtonSolver(MPI.COMM_WORLD, problem)
                solver.convergence_criterion = "residual"
                solver.rtol = 1e-6  # Slightly relaxed
                solver.atol = 1e-6  # Slightly relaxed
                solver.max_it = 50  # INCREASED max iterations
                
                # More aggressive PETSc options for better convergence
                PETSc.Options().setValue("snes_linesearch_type", "bt")
                PETSc.Options().setValue("snes_linesearch_maxstep", "0.5")
                PETSc.Options().setValue("snes_linesearch_damping", "0.5")  # Add damping
                PETSc.Options().setValue("ksp_type", "preonly")
                PETSc.Options().setValue("pc_type", "lu")
                
                # Solve nonlinear problem
                self.logger.info(f"Solving nonlinear system for step {step+1}...")
                n_its, converged = solver.solve(u)
                
                if converged:
                    self.logger.info(f"Step {step+1} converged in {n_its} iterations")
                    
                    # Save the successful solution
                    last_successful_u = u.x.array.copy()
                    
                    # Check solution quality
                    u_norm = np.linalg.norm(u.x.array.reshape(-1, 3), axis=1)
                    max_disp = np.max(u_norm)
                    avg_disp = np.mean(u_norm)
                    self.logger.info(f"Displacement stats: max={max_disp:.6f}, avg={avg_disp:.6f}")
                else:
                    self.logger.warning(f"Step {step+1} did not converge after {n_its} iterations")
                    if last_successful_u is not None:
                        # Restore the last successful solution and try smaller steps
                        u.x.array[:] = last_successful_u
                        # Create finer scales between last successful and current failed scale
                        if step > 0:
                            last_scale = force_scales[step-1]
                            current_scale = scale
                            fine_scales = np.linspace(last_scale, current_scale, 5)[1:]
                            for fine_scale in fine_scales:
                                # Try with a finer step
                                # ... (code to try with finer scale) ...
                                pass  # Skip for now
                        break
                    else:
                        raise RuntimeError("First step did not converge")
                    
            except Exception as e:
                self.logger.error(f"Error in step {step+1}: {str(e)}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                if last_successful_u is not None:
                    self.logger.info("Using results from previous successful step")
                    u.x.array[:] = last_successful_u
                    break
                else:
                    raise RuntimeError(f"First step failed: {str(e)}")
        
        # Final validation
        if last_successful_u is not None:
            # Ensure we're using the last successful solution
            u.x.array[:] = last_successful_u
        
        u_norm = np.linalg.norm(u.x.array.reshape(-1, 3), axis=1)
        max_disp = np.max(u_norm)
        
        if max_disp < 1e-10 or np.isnan(max_disp):
            raise RuntimeError(f"Invalid solution: {'NaN values' if np.isnan(max_disp) else 'Too small displacement'}")
        
        self.logger.info(f"Solution valid. Maximum displacement: {max_disp:.6f}")

        # After solving and before returning u_solution, add:
        if self.visualize and hasattr(self, 'viz_plotter'):
            self._update_visualization(force_params, u)
            
            # Optionally wait a moment to view the result
            import time
            time.sleep(0.5)
        
        return u

    def visualize_force_displacement(self, force_params, displacement, output_file=None):
        """Visualize force and resulting displacement"""
        # Create force function for visualization
        force_func = fem.Function(self.V_force, name="Force")
        force_array = np.zeros_like(force_func.x.array)
        
        # Populate force array based on force type
        coords = self.domain.geometry.x
        
        if force_params["type"] == "point":
            # Find closest node to point location
            loc = np.array(force_params["location"])
            distances = np.linalg.norm(coords - loc, axis=1)
            closest_node = np.argmin(distances)
            
            # Set force at that node
            node_idx = closest_node * 3
            force_array[node_idx:node_idx+3] = force_params["magnitude"] * np.array(force_params["direction"])
            
        elif force_params["type"] == "distributed":
            # Set force in the region
            region = force_params["region"]
            direction = np.array(force_params["direction"])
            
            for i, coord in enumerate(coords):
                if coord[0] >= region["x_min"] and coord[0] <= region["x_max"]:
                    node_idx = i * 3
                    force_array[node_idx:node_idx+3] = force_params["magnitude"] * direction
                    
        elif force_params["type"] == "torque":
            # Apply torque forces
            axis = np.array(force_params["axis"])
            beam_center = np.array([(self.y_min + self.y_max)/2, (self.z_min + self.z_max)/2])
            
            for i, coord in enumerate(coords):
                # Skip fixed boundary
                if np.isclose(coord[0], self.x_min):
                    continue
                    
                # Calculate r vector (distance from beam axis)
                r_vec = np.array([0, coord[1] - beam_center[0], coord[2] - beam_center[1]])
                
                # Skip nodes on the axis
                if np.linalg.norm(r_vec) < 1e-10:
                    continue
                
                # Scale by distance from fixed end
                scale = (coord[0] - self.x_min) / (self.x_max - self.x_min)
                
                # Force is perpendicular to both r_vec and axis
                force_dir = np.cross(axis, r_vec)
                if np.linalg.norm(force_dir) > 1e-10:
                    force_dir = force_dir / np.linalg.norm(force_dir)
                    # Set force
                    node_idx = i * 3
                    force_array[node_idx:node_idx+3] = force_params["magnitude"] * scale * force_dir
        
        # Set force function values
        force_func.x.array[:] = force_array
        
        # Create displacement function
        u = fem.Function(self.V, name="Displacement")
        u.x.array[:] = displacement
        
        # Create plotter with off_screen=True if saving to file
        use_offscreen = output_file is not None
        plotter = pyvista.Plotter(shape=(1, 2), off_screen=use_offscreen)
        
        # Plot force
        plotter.subplot(0, 0)
        topology, cell_types, x = plot.vtk_mesh(self.V_force)
        grid = pyvista.UnstructuredGrid(topology, cell_types, x)
        grid.point_data["Force"] = force_array.reshape(-1, 3)
        grid["force_magnitude"] = np.linalg.norm(force_array.reshape(-1, 3), axis=1)
        
        plotter.add_mesh(grid, scalars="force_magnitude", cmap="viridis")
        plotter.add_title(f"Applied Force ({force_params['type']})")
        
        # Plot displacement
        plotter.subplot(0, 1)
        topology, cell_types, x = plot.vtk_mesh(self.V)
        grid = pyvista.UnstructuredGrid(topology, cell_types, x)
        grid.point_data["Displacement"] = displacement.reshape(-1, 3)
        grid["displacement_magnitude"] = np.linalg.norm(displacement.reshape(-1, 3), axis=1)
        
        warped = grid.warp_by_vector("Displacement", factor=1.0)
        plotter.add_mesh(warped, scalars="displacement_magnitude", cmap="plasma")
        plotter.add_title("Resulting Displacement")
        
        
        plotter.show()



class ForceInferenceTrainer:
    """Trains a neural network to infer forces from displacements"""
    def __init__(self, data_dir="force_data", output_dir="force_models", config_path="configs/default.yaml", mesh_filename=None, coordinates=None, elements=None, input_dim=None, output_dim=None, hidden_dim=128, num_layers=2):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.config_path = config_path
        self.mesh_filename = mesh_filename
        self
        self.coordinates = coordinates
        self.elements = elements
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.hidden_layers = [hidden_dim] * num_layers
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logger with explicit configuration to ensure messages are shown
        self.logger = logging.getLogger("force_trainer")
        self.logger.setLevel(logging.INFO)
        
        # Check if handler already exists to avoid duplicates
        if not self.logger.handlers:
            # Add console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('[%(asctime)s] %(name)s %(levelname)s: %(message)s', 
                                        datefmt='%m/%d %H:%M:%S')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # Add file handler
            file_handler = logging.FileHandler(os.path.join(output_dir, 'fem_solver.log'))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Ensure messages propagate up to the root logger
        self.logger.propagate = True
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Load data
        self._load_data()
        
        # Load config
        self.cfg = load_config(self.config_path)
        
        # Store mesh data
        self.coordinates = coordinates
        self.elements = elements
        
        # Create Routine object for energy calculations
        self.routine = Routine(self.cfg)
        
        # Convert to tensors
        self.displacements_tensor = torch.tensor(self.displacements, dtype=torch.float64, device=self.device)
        self.latent_vectors_tensor = torch.tensor(self.latent_vectors, dtype=torch.float64, device=self.device)
        self.nodal_forces_tensor = torch.tensor(self.nodal_forces, dtype=torch.float64, device=self.device)
        
        # Create force tensors with consistent size
        force_vectors = []
        for force in self.forces:
            if force["type"] == "point":
                # For point forces: [magnitude, loc_x, loc_y, loc_z, dir_x, dir_y, dir_z]
                vec = np.concatenate([
                    [force["magnitude"]], 
                    force["location"],
                    force["direction"]
                ])
            elif force["type"] == "distributed":
                # For distributed forces: [magnitude, dir_x, dir_y, dir_z, region_x_min]
                vec = np.concatenate([
                    [force["magnitude"]], 
                    force["direction"],
                    [force["region"]["x_min"]]
                ])
                # Pad to same length as point forces
                vec = np.pad(vec, (0, 2), 'constant')
            elif force["type"] == "torque":
                # For torque forces: [magnitude, axis_x, axis_y, axis_z]
                vec = np.concatenate([
                    [force["magnitude"]], 
                    force["axis"]
                ])
                # Pad to same length as point forces
                vec = np.pad(vec, (0, 3), 'constant')
            
            # Add force type as one-hot encoding [point, distributed, torque]
            type_onehot = np.zeros(3)
            if force["type"] == "point":
                type_onehot[0] = 1
            elif force["type"] == "distributed":
                type_onehot[1] = 1
            elif force["type"] == "torque":
                type_onehot[2] = 1
                
            # Concatenate with force vector
            vec = np.concatenate([vec, type_onehot])
            
            force_vectors.append(vec)
        
        self.forces_tensor = torch.tensor(np.array(force_vectors), dtype=torch.float64, device=self.device)
        
        # Normalize data
        self._normalize_data()
        
        # Create model
        # self.input_dim = self.latent_vectors_tensor.shape[1]  
        # self.output_dim = self.nodal_forces_flat.shape[1]  # Flattened nodal forces dimension
        self.model = ForceInferenceModel(self.input_dim, self.output_dim, self.hidden_layers).to(self.device)
        self.model.double()  # Use double precision

        self.logger.info(f"Model created with input dim {self.input_dim} and output dim {self.output_dim}")
    
    def _load_data(self):
        """Load force and displacement data"""
        try:
            self.forces = np.load(os.path.join(self.data_dir, "force_data.npy"), allow_pickle=True)
            self.displacements = np.load(os.path.join(self.data_dir, "displacement_data.npy"), allow_pickle=True)
            self.latent_vectors = np.load(os.path.join(self.data_dir, "latent_vectors.npy"), allow_pickle=True)
            self.nodal_forces = np.load(os.path.join(self.data_dir, "nodal_forces.npy"), allow_pickle=True)
            
            # Reshape displacements if needed
            if len(self.displacements.shape) > 2:
                self.displacements = self.displacements.reshape(self.displacements.shape[0], -1)
            
            self.logger.info(f"Loaded {len(self.forces)} force-displacement pairs")
            self.logger.info(f"Displacement shape: {self.displacements.shape}")
            self.logger.info(f"Latent vector shape: {self.latent_vectors.shape}")
            self.logger.info(f"Nodal forces shape: {self.nodal_forces.shape}")
            
            # Add checks for data content
            self.logger.info(f"First displacement: {self.displacements[0]}")
            self.logger.info(f"First latent vector: {self.latent_vectors[0]}")
            self.logger.info(f"First nodal force: {self.nodal_forces[0]}")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _normalize_data(self):
        """Normalize displacement and force data"""
        # Displacement normalization
        self.displacement_mean = torch.mean(self.displacements_tensor, dim=0)
        self.displacement_std = torch.std(self.displacements_tensor, dim=0)
        self.displacement_std[self.displacement_std < 1e-6] = 1.0  # Avoid division by zero
        
        self.displacements_normalized = (self.displacements_tensor - self.displacement_mean) / self.displacement_std
        
        # Latent vector normalization
        self.latent_vectors_mean = torch.mean(self.latent_vectors_tensor, dim=0)
        self.latent_vectors_std = torch.std(self.latent_vectors_tensor, dim=0)
        self.latent_vectors_std[self.latent_vectors_std < 1e-6] = 1.0  # Avoid division by zero
        
        self.latent_vectors_normalized = (self.latent_vectors_tensor - self.latent_vectors_mean) / self.latent_vectors_std
        
        # Flatten nodal forces before normalization
        self.nodal_forces_flat = self.nodal_forces_tensor.reshape(self.nodal_forces_tensor.shape[0], -1)
        self.nodal_forces_flat_mean = torch.mean(self.nodal_forces_flat, dim=0)
        self.nodal_forces_flat_std = torch.std(self.nodal_forces_flat, dim=0)
        self.nodal_forces_flat_std[self.nodal_forces_flat_std < 1e-6] = 1.0  # Avoid division by zero
        
        self.nodal_forces_normalized = (self.nodal_forces_flat - self.nodal_forces_flat_mean) / self.nodal_forces_flat_std
        
        # Also store these with the names used in the train method
        self.nodal_forces_mean = self.nodal_forces_flat_mean
        self.nodal_forces_std = self.nodal_forces_flat_std
        
        self.logger.info("Data normalized")

    def _calculate_reaction_forces(self, external_forces):
        """Calculate physically appropriate reaction forces at fixed boundaries
        
        Args:
            external_forces: External forces tensor [batch_size, num_nodes*3]
            
        Returns:
            Reaction forces tensor [batch_size, num_nodes*3]
        """
        batch_size = external_forces.shape[0]
        reaction_forces = torch.zeros_like(external_forces)
        
        # Reshape for per-node operations
        num_nodes = self.coordinates.shape[0]
        forces_reshaped = external_forces.reshape(batch_size, num_nodes, 3)
        
        # Get fixed boundary nodes
        fixed_nodes = torch.where(self.coordinates[:, 0] < 1e-6)[0]
        num_fixed_nodes = len(fixed_nodes)
        
        # Create a better reaction force distribution:
        # 1. Extract moment of external forces around fixed boundary
        # 2. Apply counteracting moment through appropriate reaction distribution
        
        # Calculate total external force (should be countered by reactions)
        total_ext_force = torch.sum(forces_reshaped, dim=1)  # [batch_size, 3]
        
        # Calculate center of fixed boundary
        fixed_coords = self.coordinates[fixed_nodes]
        boundary_center = torch.mean(fixed_coords, dim=0)
        
        # Calculate moment of external forces around boundary center
        r_vectors = self.coordinates.unsqueeze(0) - boundary_center.unsqueeze(0).unsqueeze(0)  # [1, num_nodes, 3]
        moments = torch.cross(r_vectors, forces_reshaped, dim=2)  # [batch_size, num_nodes, 3]
        total_moment = torch.sum(moments, dim=1)  # [batch_size, 3]
        
        # Apply appropriate reactions:
        # 1. Distribute total force evenly among fixed nodes
        force_per_node = -total_ext_force / num_fixed_nodes  # [batch_size, 3]
        
        # 2. Add moment-countering forces
        for i, node in enumerate(fixed_nodes):
            # Position relative to boundary center
            r = self.coordinates[node] - boundary_center  # [3]
            
            # Add basic reaction force (equal distribution)
            reaction_forces[:, node*3:node*3+3] = force_per_node
            
            # Add moment contribution (if we have enough fixed nodes)
            if num_fixed_nodes > 3:  # Need at least 4 nodes for stable moment distribution
                # Compute direction perpendicular to r and weighted by distance
                r_mag = torch.norm(r) + 1e-10
                
                # Skip nodes too close to center
                if r_mag > 1e-6:
                    # For each batch sample
                    for b in range(batch_size):
                        # Direction that contributes to counteracting the moment
                        moment_dir = torch.cross(r, total_moment[b])
                        
                        # Scale based on position
                        moment_contribution = moment_dir * 0.5 / (num_fixed_nodes * r_mag)
                        
                        # Add to reaction forces
                        reaction_forces[b, node*3:node*3+3] += moment_contribution
        
        return reaction_forces
    
    def train(self, epochs=50, batch_size=32, learning_rate=1e-3, validation_split=0.2,
          force_loss_weight=1.0, displacement_consistency_weight=3.0, 
          youngs_modulus=1e5, poissons_ratio=0.45):
        """Train force inference model with proper physics handling"""
        
        # Split data into training and validation
        n_samples = len(self.displacements_normalized)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val
        
        indices = torch.randperm(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # Extract training and validation data
        train_latent = self.latent_vectors_normalized[train_indices]
        train_nodal_forces = self.nodal_forces_normalized[train_indices]
        train_displacements = self.displacements_normalized[train_indices]
        
        val_latent = self.latent_vectors_normalized[val_indices]
        val_nodal_forces = self.nodal_forces_normalized[val_indices]
        val_displacements = self.displacements_normalized[val_indices]
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(train_latent, train_nodal_forces, train_displacements)
        val_dataset = torch.utils.data.TensorDataset(val_latent, val_nodal_forces, val_displacements)
        
        # Prepare solvers and criteria
        mse_criterion = torch.nn.MSELoss()
        
        # Create energy model and solver
        neohookean_model = NeoHookeanEnergyModel(
            coordinates=self.coordinates,
            elements=self.elements,
            young_modulus=youngs_modulus,
            poisson_ratio=poissons_ratio,
            device=self.device,
            precompute=True
        )

        diff_solver = ModernFEMSolver(
            energy_model=neohookean_model,
            max_iterations=10,
            tolerance=1e-8,
            energy_tolerance=1e-8,
            verbose=False,
            visualize=True,
            filename=self.mesh_filename
        )

        # Set up boundary conditions
        bottom_nodes = torch.where(self.coordinates[:, 0] < 1e-6)[0]
        bottom_dofs = torch.cat([bottom_nodes * 3, bottom_nodes * 3 + 1, bottom_nodes * 3 + 2])
        fixed_values = torch.zeros_like(bottom_dofs, dtype=torch.float32)
        diff_solver.set_fixed_dofs(bottom_dofs, fixed_values)

        max_expected_force = 100.0  # Maximum expected force magnitude
        
        # Metrics tracking
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        metrics_log = {
            'epoch': [], 'train_loss': [], 'val_loss': [], 
            'val_force_mse': [], 'val_displacement_loss': [], 'lr': []
        }
        
        # Check latent vector variance to verify proper modal projection
        latent_vectors_var = torch.var(self.latent_vectors_normalized, dim=0)
        print(f"Latent vector variance per dimension: min={torch.min(latent_vectors_var).item():.6f}, max={torch.max(latent_vectors_var).item():.6f}, mean={torch.mean(latent_vectors_var).item():.6f}")
        
        # Use Adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            self.model.train()
            train_loss = 0.0
            force_mse_total = 0.0  # Initialize counters
            disp_loss_total = 0.0
            
            # Gradually increase displacement weight
            current_disp_weight = displacement_consistency_weight 
            
            # Create data loader
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_latent, batch_nodal_forces, batch_displacements in pbar:
                optimizer.zero_grad()
                
                # 1. Predict external forces
                predicted_external_forces = self.model(batch_latent)

                
                
                
                # 2. Compute force MSE loss
                force_loss = mse_criterion(predicted_external_forces, batch_nodal_forces)
                
                # 3. Physics-based loss if applicable
                displacement_consistency_loss = 0.0
               
                # Denormalize forces and displacements for physics
                predicted_forces_denorm = predicted_external_forces * self.nodal_forces_std + self.nodal_forces_mean
                batch_displacements_denorm = batch_displacements * self.displacement_std + self.displacement_mean

                #Regularization for force magnitude

                
                
                # CRITICAL CHANGE: Zero forces at fixed DOFs
                # This ensures boundary conditions are properly respected
                pred_forces_with_bcs = predicted_forces_denorm.clone()
                for node in bottom_nodes:
                    node_indices = torch.tensor([node*3, node*3+1, node*3+2], device=self.device)
                    for batch_idx in range(pred_forces_with_bcs.shape[0]):
                        pred_forces_with_bcs[batch_idx, node_indices] = 0.0
                
                # Use a smaller subset for physics computation (efficiency)
                solver_batch_size = min(32, pred_forces_with_bcs.shape[0])
                indices = torch.randperm(pred_forces_with_bcs.shape[0])[:solver_batch_size]
                pred_forces_subset = pred_forces_with_bcs[indices]
                disp_gt_subset = batch_displacements_denorm[indices]
                disp_gt_noise = disp_gt_subset + torch.randn_like(disp_gt_subset) * 1e-3
                
                # Solve for displacements - solver handles boundary conditions internally
                predicted_displacements = diff_solver(pred_forces_subset, disp_gt_noise)
                
                # Physics-based loss using weighted displacement
                displacement_consistency_loss = mse_criterion(
                    predicted_displacements, disp_gt_subset
                )
                
                # 4. Combined loss
                total_loss = (force_loss_weight * force_loss + 
                            current_disp_weight * displacement_consistency_loss)
                
                # 5. Backward and optimize
                total_loss.backward()
                optimizer.step()
                
                # Update metrics
                batch_size = batch_latent.size(0)
                train_loss += total_loss.item() * batch_size
                force_mse_total += force_loss.item() * batch_size
                disp_loss_total += displacement_consistency_loss.item() * batch_size
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': total_loss.item(),
                    'force_mse': force_loss.item(),
                    'disp_loss': displacement_consistency_loss.item()
                })
            
            # Normalize loss by dataset size
            train_loss /= len(train_dataset)
            force_mse_total /= len(train_dataset)
            disp_loss_total /= len(train_dataset)
            train_losses.append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_force_mse = 0.0
            val_displacement_loss = 0.0
            
            with torch.no_grad():
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                for batch_latent, batch_nodal_forces, batch_displacements in val_loader:
                    # Predict forces
                    predicted_external_forces = self.model(batch_latent)
                    
                    # Force MSE loss
                    force_mse_loss = mse_criterion(predicted_external_forces, batch_nodal_forces)
                    val_force_mse += force_mse_loss.item() * batch_latent.size(0)
                    
                    # Physics loss if applicable
                    disp_loss = 0.0
                    # Denormalize for physics computations
                    predicted_forces_denorm = predicted_external_forces * self.nodal_forces_std + self.nodal_forces_mean
                    batch_displacements_denorm = batch_displacements * self.displacement_std + self.displacement_mean
                    
                    # CRITICAL CHANGE: Zero forces at fixed DOFs 
                    pred_forces_with_bcs = predicted_forces_denorm.clone()
                    for node in bottom_nodes:
                        node_indices = torch.tensor([node*3, node*3+1, node*3+2], device=self.device)
                        for batch_idx in range(pred_forces_with_bcs.shape[0]):
                            pred_forces_with_bcs[batch_idx, node_indices] = 0.0
                    
                    # Use smaller batch for solver
                    solver_batch_size = min(32, pred_forces_with_bcs.shape[0])
                    pred_forces_subset = pred_forces_with_bcs[:solver_batch_size]
                    disp_gt_subset = batch_displacements_denorm[:solver_batch_size]
                    
                    # Solve for displacements and compute loss
                    predicted_displacements = diff_solver(pred_forces_subset)
                    disp_loss = mse_criterion(predicted_displacements, disp_gt_subset)
                    val_displacement_loss += disp_loss.item() * solver_batch_size
                    
                    # Combined validation loss
                    batch_loss = force_loss_weight * force_mse_loss + current_disp_weight * disp_loss
                    val_loss += batch_loss.item() * batch_latent.size(0)
            
            # Normalize validation metrics
            val_loss /= len(val_dataset)
            val_force_mse /= len(val_dataset)
            if val_displacement_loss > 0:
                val_displacement_loss /= len(val_dataset)
            val_losses.append(val_loss)
            
            # Update metrics log
            metrics_log['epoch'].append(epoch+1)
            metrics_log['train_loss'].append(train_loss)
            metrics_log['val_loss'].append(val_loss)
            metrics_log['val_force_mse'].append(val_force_mse)
            metrics_log['val_displacement_loss'].append(val_displacement_loss)
            metrics_log['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'latent_vectors_mean': self.latent_vectors_mean,
                    'latent_vectors_std': self.latent_vectors_std,
                    'nodal_forces_mean': self.nodal_forces_mean,
                    'nodal_forces_std': self.nodal_forces_std,
                    'displacement_mean': self.displacement_mean,
                    'displacement_std': self.displacement_std,
                    'input_dim': self.input_dim,
                    'output_dim': self.output_dim,
                    'val_metrics': {
                        'loss': val_loss,
                        'force_mse': val_force_mse,
                        'displacement_loss': val_displacement_loss
                    }
                }, os.path.join(self.output_dir, "best_model.pt"))
                
                self.logger.info(f"✅ Epoch {epoch+1}: New best model saved with val_loss: {val_loss:.6f}")
            
            # Print progress
            epoch_time = time.time() - epoch_start_time
            self.logger.info(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s):")
            self.logger.info(f"  Training Loss:        {train_loss:.6f}")
            self.logger.info(f"  Validation Loss:      {val_loss:.6f}")
            self.logger.info(f"  Force MSE:            {val_force_mse:.6f}")
            if val_displacement_loss > 0:
                self.logger.info(f"  Displacement Loss:    {val_displacement_loss:.6f}")
            self.logger.info(f"  Learning Rate:        {optimizer.param_groups[0]['lr']:.6f}")
            
            if epoch > 0 and metrics_log['val_loss'][-2] > 0:
                improvement = (metrics_log['val_loss'][-2] - val_loss) / metrics_log['val_loss'][-2] * 100
                self.logger.info(f"    - Improvement:      {improvement:.2f}%")
        
        # Save final model
        torch.save({
            'epoch': epochs-1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'latent_vectors_mean': self.latent_vectors_mean,
            'latent_vectors_std': self.latent_vectors_std,
            'nodal_forces_mean': self.nodal_forces_mean,
            'nodal_forces_std': self.nodal_forces_std,
            'displacement_mean': self.displacement_mean,
            'displacement_std': self.displacement_std,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'metrics_log': metrics_log
        }, os.path.join(self.output_dir, "final_model.pt"))
        
        # Plot training metrics if available
        if hasattr(self, '_plot_training_metrics'):
            self._plot_training_metrics(metrics_log)
        
        self.logger.info(f"Training complete. Best validation loss: {best_val_loss:.6f}")
        
        return train_losses, val_losses, metrics_log
    

    


    def test_force_inference(self, n_samples=5):
        """Test force inference on random samples"""
        self.model.eval()
        
        # Randomly select samples
        indices = np.random.choice(len(self.displacements_normalized), n_samples, replace=False)
        
        results = []
        
        for idx in indices:
            # Get displacement
            displacement = self.displacements_normalized[idx].unsqueeze(0)
            
            # Infer force
            with torch.no_grad():
                force_pred_norm = self.model(displacement)
                # Denormalize
                force_pred = force_pred_norm * self.force_std + self.force_mean
            
            # Get ground truth
            force_true = self.forces[idx]
            
            # Evaluate prediction - extract type and parameters
            force_vec = force_pred.squeeze().cpu().numpy()
            
            # Get force type from one-hot part (last 3 elements)
            type_onehot = force_vec[-3:]
            force_type_idx = np.argmax(type_onehot)
            force_types = ["point", "distributed", "torque"]
            predicted_type = force_types[force_type_idx]
            
            # Construct predicted force parameters
            if predicted_type == "point":
                predicted_force = {
                    "type": "point",
                    "magnitude": force_vec[0],
                    "location": force_vec[1:4],
                    "direction": force_vec[4:7]
                }
            elif predicted_type == "distributed":
                predicted_force = {
                    "type": "distributed",
                    "magnitude": force_vec[0],
                    "direction": force_vec[1:4],
                    "region": {"x_min": force_vec[4]}
                }
            elif predicted_type == "torque":
                predicted_force = {
                    "type": "torque",
                    "magnitude": force_vec[0],
                    "axis": force_vec[1:4]
                }
            
            # Store result
            results.append({
                "displacement": self.displacements[idx],
                "true_force": force_true,
                "predicted_force": predicted_force
            })
            
            self.logger.info(f"Sample {idx}:")
            self.logger.info(f"  True force type: {force_true['type']}")
            self.logger.info(f"  Predicted force type: {predicted_type}")
            self.logger.info(f"  True magnitude: {force_true['magnitude']:.2f}")
            self.logger.info(f"  Predicted magnitude: {predicted_force['magnitude']:.2f}")
            
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Force inference for physics-informed neural modes')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--generate', action='store_true', help='Generate force-displacement data')
    parser.add_argument('--train', action='store_true', help='Train force inference model')
    parser.add_argument('--integrate', action='store_true', help='Integrate with training')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='force_inference', help='Output directory')
    parser.add_argument('--force_model', type=str, default='force_inference/best_model.pt', 
                        help='Path to trained force model for integration')
    parser.add_argument('--force_loss_weight', type=float, default=1.0, help='Weight for force MSE loss')
    parser.add_argument('--div_p_loss_weight', type=float, default=10.0, help='Weight for div(P) loss')
    
    args = parser.parse_args()
    
    # Setup logger
    setup_logger("force_inference", log_dir=args.output_dir)
    logger = logging.getLogger("force_inference")
    
    # Load config for all operations
    cfg = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")
    
    if args.generate:
        logger.info("Generating force-displacement data...")
        
        # Get material parameters from config using the correct keys
        # material_params = {
        #     "youngs_modulus": cfg.get("material", {}).get("youngs_modulus", 10000),
        #     "poissons_ratio": cfg.get("material", {}).get("poissons_ratio", 0.49),
        #     "density": cfg.get("material", {}).get("density", 1000)
        # }
        
        # Get mesh filename from config
        mesh_filename = cfg.get('mesh', {}).get('filename', 'mesh/beam_732.msh')
        logger.info(f"Using mesh file: {mesh_filename}")
        
        # Create data generator
        data_generator = FEMDataGenerator(mesh_filename, cfg, output_dir=args.output_dir, visualize=True)
        
        # Generate data
        force_data, displacement_data, latent_vectors, nodal_forces = data_generator.generate_force_displacement_pairs(
            num_samples=args.samples
        )
    elif args.train:
        try:
            logger.info("Training force inference model...")
            
            # Load mesh data
            routine = Routine(cfg)
            coordinates = routine.energy_calculator.coordinates
            elements = routine.energy_calculator.elements
            mesh_filename = cfg.get('mesh', {}).get('filename', 'mesh/beam_732.msh')
            input_dim = cfg.get('model', {}).get('latent_dim', 128)
            hid_dim = cfg.get('model', {}).get('hid_dim', 256)
            hid_layers = cfg.get('model', {}).get('hid_layers', 2)
            output_dim = 3 * coordinates.shape[0]  # 3 DOFs per node
            
            # Create trainer
            trainer = ForceInferenceTrainer(data_dir=args.output_dir, output_dir=args.output_dir, config_path=args.config, mesh_filename=mesh_filename, coordinates=coordinates, elements=elements, input_dim=input_dim, output_dim=output_dim, hidden_dim=hid_dim, num_layers=hid_layers)
            
            # Check if data exists
            if hasattr(trainer, 'latent_vectors') and len(trainer.latent_vectors) > 0:
                logger.info(f"Data loaded successfully: {len(trainer.latent_vectors)} samples")
            else:
                logger.error("No training data found. Run with --generate first.")
                sys.exit(1)
                
            # Train model with more debugging
            logger.info("Starting training routine...")
            try:
                result = trainer.train(
                    epochs=cfg.get('training', {}).get('epochs', 50),
                    batch_size=cfg.get('training', {}).get('batch_size', 1),
                    learning_rate=cfg.get('training', {}).get('learning_rate', 1e-3),
                    validation_split=cfg.get('training', {}).get('validation_split', 0.2),
                    force_loss_weight=args.force_loss_weight,
                    displacement_consistency_weight=args.div_p_loss_weight,
                    youngs_modulus=cfg.get('material', {}).get('youngs_modulus', 1e5),  # Pass material properties
                    poissons_ratio=cfg.get('material', {}).get('poissons_ratio', 0.49)  # Pass material properties
                )
                logger.info(f"Training completed successfully: {result}")
            except Exception as e:
                logger.error(f"Error during training: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                
        except Exception as e:
            logger.error(f"Error in training setup: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")