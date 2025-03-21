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

from training.train import setup_logger, Net, load_config, Routine

class ForceInferenceModel(torch.nn.Module):
    """Neural network that infers forces from displacements"""
    def __init__(self, input_dim, output_dim, hidden_layers=[128, 64, 32]):
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
        
    def forward(self, x):
        return self.model(x)

class FEMDataGenerator:
    """Generates training data using FEniCSx simulations with various force configurations"""
    def __init__(self, mesh_path, material_params, output_dir="force_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logger with explicit configuration to ensure messages are shown
        self.logger = logging.getLogger("force_data_generator")
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
        
        # Material parameters - use proper keys from config
        self.E = material_params.get("youngs_modulus", 1.0e5)  # Young's modulus
        self.nu = material_params.get("poissons_ratio", 0.49)   # Poisson's ratio
        
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
        
        self.logger.info(f"FEM Data Generator initialized with mesh containing {self.domain.topology.index_map(0).size_local} nodes")
        
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
        
    def generate_force_displacement_pairs(self, num_samples=100, force_types=["point", "distributed", "torque"]):
        """Generate pairs of forces and resulting displacements"""
        self.logger.info(f"Generating {num_samples} force-displacement pairs")
        self.logger.info(f"Using force types: {force_types}")
        
        force_data = []
        displacement_data = []
        
        type_counts = {force_type: 0 for force_type in force_types}
        
        for i in range(num_samples):
            self.logger.info(f"\n=== Generating sample {i+1}/{num_samples} ===")
            
            # Randomly select force type
            force_type = np.random.choice(force_types)
            type_counts[force_type] += 1
            
            # Generate random force parameters with more modest magnitudes
            force_magnitude = np.random.uniform(100.0, 1000.0)  # Reduced magnitudes
            self.logger.info(f"Selected force type: {force_type}, magnitude: {force_magnitude:.2f}")
            
            if force_type == "point":
                # Point force at random location on free end
                x_loc = np.random.uniform(0.7 * self.x_max, self.x_max)  # Avoid very edge
                y_loc = np.random.uniform(self.y_min + 0.1, self.y_max - 0.1)  # Avoid boundaries
                z_loc = np.random.uniform(self.z_min + 0.1, self.z_max - 0.1)  # Avoid boundaries
                force_dir = np.random.uniform(-1, 1, 3)
                force_dir = force_dir / np.linalg.norm(force_dir)
                
                # For stability, use primarily y-z plane forces initially
                if i < num_samples // 3:
                    force_dir[0] *= 0.2  # Reduce x component
                
                self.logger.info(f"Point force at location [{x_loc:.2f}, {y_loc:.2f}, {z_loc:.2f}], direction {force_dir}")
                
                force_params = {
                    "type": "point",
                    "magnitude": force_magnitude,
                    "location": [x_loc, y_loc, z_loc],
                    "direction": force_dir
                }
                
            elif force_type == "distributed":
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
            
            # Solve FEM system with the force
            self.logger.info("Solving FEM system for this force...")
            u_solution = self._solve_fem_system(force_params)
            
            # Extract nodal values
            u_array = u_solution.x.array.copy()
            
            # Store force and displacement
            force_data.append(force_params)
            displacement_data.append(u_array)
            
            if (i+1) % 10 == 0 or i+1 == num_samples:
                self.logger.info(f"Progress: {i+1}/{num_samples} samples generated")
                self.logger.info(f"Force type distribution: {type_counts}")
        
        # Save data
        self.logger.info(f"Saving data to {self.output_dir}...")
        np.save(os.path.join(self.output_dir, "force_data.npy"), force_data)
        np.save(os.path.join(self.output_dir, "displacement_data.npy"), displacement_data)
        
        self.logger.info(f"Data generation complete. Files saved to {self.output_dir}")
        self.logger.info(f"Final force type distribution: {type_counts}")
        
        return force_data, displacement_data

    def _solve_fem_system(self, force_params):
        """Solve FEM system with the given force parameters - using approach from validation scripts"""
        self.logger.info(f"=========== Solving FEM system for {force_params['type']} force ===========")
        
        # Initialize displacement function
        u = fem.Function(self.V, name="Displacement")
        v = ufl.TestFunction(self.V)
        
        # Log detailed mesh info
        self.logger.info(f"Mesh stats: {self.domain.topology.index_map(0).size_local} vertices, " 
                        f"{self.domain.topology.index_map(3).size_local} cells")
        self.logger.info(f"Function space dim: {self.V.dofmap.index_map.size_local * 3}")
        
        # Kinematics - same approach as validate_twist.py
        d = len(u)
        I = ufl.Identity(d)
        F = I + ufl.grad(u)
        C = F.T * F
        Ic = ufl.tr(C)
        J = ufl.det(F)
        
        # Neo-Hookean strain energy density - same as validation scripts
        psi = (self.mu/2) * (Ic - 3) - self.mu * ufl.ln(J) + (self.lmbda/2) * (ufl.ln(J))**2
        
        # Log material parameters
        self.logger.info(f"Material parameters: E={self.E}, nu={self.nu}, mu={self.mu}, lambda={self.lmbda}")
        
        # Internal work
        internal_work = psi * ufl.dx
        
        # Prepare external work - use a more direct approach like in validate_gravity.py
        x = ufl.SpatialCoordinate(self.domain)
        
        # Use a continuation approach - scale down force even more for initial solve
        initial_scale = 0.001  # Start with 0.1% of requested force
        target_scale = 0.01    # Work up to 1% of requested force
        
        # Scale down force magnitude for stability
        magnitude = force_params["magnitude"] * initial_scale  # Start very small
        self.logger.info(f"Starting with very small force: {magnitude:.4f} (scale={initial_scale})")
        
        if force_params["type"] == "point":
            # Use smoother point force application
            loc = np.array(force_params["location"])
            direction = np.array(force_params["direction"])
            
            # Check if location is within mesh bounds
            in_x_bounds = self.x_min <= loc[0] <= self.x_max
            in_y_bounds = self.y_min <= loc[1] <= self.y_max
            in_z_bounds = self.z_min <= loc[2] <= self.z_max
            if not (in_x_bounds and in_y_bounds and in_z_bounds):
                self.logger.warning(f"Point force location {loc} is outside mesh bounds: "
                                f"x=[{self.x_min:.2f}, {self.x_max:.2f}], "
                                f"y=[{self.y_min:.2f}, {self.y_max:.2f}], "
                                f"z=[{self.z_min:.2f}, {self.z_max:.2f}]")
            
            # Calculate distance from application point (smoother field)
            dist_expr = ufl.sqrt((x[0] - loc[0])**2 + (x[1] - loc[1])**2 + (x[2] - loc[2])**2)
            radius = 0.15 * max(self.x_max - self.x_min, self.y_max - self.y_min, self.z_max - self.z_min)
            
            # Create a smooth field that decays with distance
            smooth_field = ufl.conditional(dist_expr < radius, 
                                        ufl.exp(-4.0 * (dist_expr/radius)**2), 
                                        0.0)
            
            # Create force expression
            force_expr = ufl.as_vector([
                magnitude * direction[0] * smooth_field,
                magnitude * direction[1] * smooth_field,
                magnitude * direction[2] * smooth_field
            ])
            
            self.logger.info(f"Applying point force at {loc} with smooth falloff radius {radius:.4f}")
            self.logger.info(f"Direction: {direction}, |direction|={np.linalg.norm(direction):.4f}")
            
        elif force_params["type"] == "distributed":
            # More details about region
            region = force_params["region"]
            direction = np.array(force_params["direction"])
            self.logger.info(f"Region x_min={region['x_min']:.4f}, x_max={region['x_max']:.4f}")
            self.logger.info(f"Direction: {direction}, |direction|={np.linalg.norm(direction):.4f}")
            
            # Define a smooth transition at region boundary
            region_start = region["x_min"]
            transition_width = 0.05 * (self.x_max - self.x_min)
            
            # Create a smooth indicator function for the region
            # 1 inside region, smoothly decreases to 0 at boundary
            region_indicator = ufl.conditional(x[0] >= region_start + transition_width, 
                                            1.0,
                                            ufl.conditional(x[0] <= region_start, 
                                                        0.0,
                                                        (x[0] - region_start) / transition_width))
            
            # Create force expression
            force_expr = ufl.as_vector([
                magnitude * direction[0] * region_indicator,
                magnitude * direction[1] * region_indicator,
                magnitude * direction[2] * region_indicator
            ])
            
            self.logger.info(f"Applying distributed force in region x >= {region_start} with transition width {transition_width:.4f}")
            
        elif force_params["type"] == "torque":
            # Detailed torque info
            axis = np.array(force_params["axis"])
            self.logger.info(f"Torque axis: {axis}, |axis|={np.linalg.norm(axis):.4f}")
            
            beam_center_y = (self.y_min + self.y_max) / 2
            beam_center_z = (self.z_min + self.z_max) / 2
            
            # Scale by distance from fixed end (linear)
            scale = (x[0] - self.x_min) / (self.x_max - self.x_min)
            
            # Calculate distance from beam axis
            r_loc = ufl.sqrt((x[1] - beam_center_y)**2 + (x[2] - beam_center_z)**2)
            
            # Calculate theta (angle around axis)
            theta = ufl.atan2(x[2] - beam_center_z, x[1] - beam_center_y)
            
            # Force is perpendicular to both r_vec and axis
            if abs(axis[2]) > 0.9:  # Primarily z-axis rotation
                force_dir = ufl.as_vector([0.0, -ufl.sin(theta), ufl.cos(theta)])
            else:
                # For other axes, use a simpler approximation
                force_dir = ufl.as_vector([0.0, -axis[2], axis[1]])
                force_dir_norm = ufl.sqrt(force_dir[0]**2 + force_dir[1]**2 + force_dir[2]**2)
                force_dir = force_dir / (force_dir_norm + 1e-10)  # Add small epsilon to avoid division by zero
            
            # Create force expression
            force_expr = magnitude * scale * r_loc * force_dir
            
            self.logger.info(f"Applying torque around axis {axis} with center at y={beam_center_y:.4f}, z={beam_center_z:.4f}")
        
        # External work using the force expression
        external_work = ufl.inner(force_expr, v) * ufl.dx
        
        # Weak form
        F = ufl.derivative(internal_work, u, v) - external_work
        
        # Compute Jacobian
        J = ufl.derivative(F, u)
        
        # Setup nonlinear problem
        problem = NonlinearProblem(F, u, [self.bc], J=J)
        
        # Create Newton solver with better parameters (more conservative)
        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        solver.convergence_criterion = "incremental"  # Incremental works better for nonlinear problems
        solver.rtol = 1e-4  # More relaxed tolerance
        solver.atol = 1e-4  # More relaxed absolute tolerance
        solver.max_it = 50  # More iterations
        
        # Add more stability with line search
        PETSc.Options().setValue("snes_linesearch_type", "basic")
        
        # Log solver parameters
        self.logger.info(f"Newton solver parameters: rtol={solver.rtol}, atol={solver.atol}, "
                        f"max_it={solver.max_it}, criterion={solver.convergence_criterion}")
        
        # Gradual force application using continuation
        num_continuation_steps = 5
        force_scales = np.linspace(initial_scale, target_scale, num_continuation_steps)
        
        self.logger.info(f"Using {num_continuation_steps} continuation steps with force scales: {force_scales}")
        
        # Solve with continuation
        for step, scale in enumerate(force_scales):
            try:
                # Update force magnitude
                actual_magnitude = force_params["magnitude"] * scale
                self.logger.info(f"Continuation step {step+1}/{num_continuation_steps}: scale={scale:.6f}, magnitude={actual_magnitude:.4f}")
                
                # Create updated force expression with current scale
                if force_params["type"] == "point":
                    force_expr = ufl.as_vector([
                        actual_magnitude * direction[0] * smooth_field,
                        actual_magnitude * direction[1] * smooth_field,
                        actual_magnitude * direction[2] * smooth_field
                    ])
                elif force_params["type"] == "distributed":
                    force_expr = ufl.as_vector([
                        actual_magnitude * direction[0] * region_indicator,
                        actual_magnitude * direction[1] * region_indicator,
                        actual_magnitude * direction[2] * region_indicator
                    ])
                elif force_params["type"] == "torque":
                    force_expr = actual_magnitude * scale * r_loc * force_dir
                
                # Update external work
                external_work = ufl.inner(force_expr, v) * ufl.dx
                
                # Update weak form
                F = ufl.derivative(internal_work, u, v) - external_work
                J = ufl.derivative(F, u)
                
                # Create new problem and solver
                problem = NonlinearProblem(F, u, [self.bc], J=J)
                solver = NewtonSolver(MPI.COMM_WORLD, problem)
                solver.convergence_criterion = "incremental"
                solver.rtol = 1e-4
                solver.atol = 1e-4
                solver.max_it = 50
                
                # Solve
                self.logger.info(f"Starting Newton solver for step {step+1}...")
                n_its, converged = solver.solve(u)
                
                if not converged:
                    self.logger.warning(f"Newton solver did not converge at scale {scale:.6f} after {n_its} iterations")
                    # Try with relaxed parameters before giving up
                    solver.rtol = 1e-3
                    solver.atol = 1e-3
                    solver.max_it = 100
                    self.logger.info("Retrying with more relaxed parameters: "
                                    f"rtol={solver.rtol}, atol={solver.atol}, max_it={solver.max_it}")
                    n_its, converged = solver.solve(u)
                    
                    if not converged:
                        self.logger.error(f"Solver failed with relaxed parameters at scale {scale:.6f}")
                        if step > 0:
                            self.logger.info(f"Using result from previous step (scale={force_scales[step-1]:.6f})")
                            break
                        else:
                            # Generate small displacement for first step failure
                            self.logger.error("First continuation step failed. Using small random displacement.")
                            u.x.array[:] = np.random.normal(0, 1e-5, size=u.x.array.shape)
                            return u
                
                # Print statistics about displacement at this step
                u_array = u.x.array
                u_norm = np.linalg.norm(u_array.reshape(-1, 3), axis=1)
                max_disp = np.max(u_norm)
                avg_disp = np.mean(u_norm)
                self.logger.info(f"Step {step+1} converged in {n_its} iterations. "
                                f"Displacement: max={max_disp:.6f}, avg={avg_disp:.6f}")
                
            except Exception as e:
                self.logger.error(f"Error in continuation step {step+1}: {str(e)}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                if step > 0:
                    self.logger.info(f"Using result from previous successful step")
                    break
                else:
                    # Generate small displacement for first step exception
                    self.logger.error("First continuation step failed with exception. Using small random displacement.")
                    u.x.array[:] = np.random.normal(0, 1e-5, size=u.x.array.shape)
                    return u
        
        # Final check for valid displacement
        u_norm = np.linalg.norm(u.x.array.reshape(-1, 3), axis=1)
        max_disp = np.max(u_norm)
        
        if max_disp < 1e-10:
            self.logger.warning(f"Final displacement is nearly zero (max={max_disp:.2e}). Solution may be invalid.")
        elif np.isnan(max_disp):
            self.logger.error("NaN values detected in displacement. Using small random displacement.")
            u.x.array[:] = np.random.normal(0, 1e-5, size=u.x.array.shape)
        else:
            self.logger.info(f"Final solution looks valid. Maximum displacement: {max_disp:.6f}")
        
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
    def __init__(self, data_dir="force_data", output_dir="force_models"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger("force_inference_trainer")
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Load data
        self._load_data()
        
        # Convert to tensors
        self.displacements_tensor = torch.tensor(self.displacements, dtype=torch.float64, device=self.device)
        
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
        self.input_dim = self.displacements_tensor.shape[1]
        self.output_dim = self.forces_tensor.shape[1]
        self.model = ForceInferenceModel(self.input_dim, self.output_dim).to(self.device)
        self.model.double()  # Use double precision
        
        self.logger.info(f"Model created with input dim {self.input_dim} and output dim {self.output_dim}")
    
    def _load_data(self):
        """Load force and displacement data"""
        try:
            self.forces = np.load(os.path.join(self.data_dir, "force_data.npy"), allow_pickle=True)
            self.displacements = np.load(os.path.join(self.data_dir, "displacement_data.npy"), allow_pickle=True)
            
            # Reshape displacements if needed
            if len(self.displacements.shape) > 2:
                self.displacements = self.displacements.reshape(self.displacements.shape[0], -1)
                
            self.logger.info(f"Loaded {len(self.forces)} force-displacement pairs")
            self.logger.info(f"Displacement shape: {self.displacements.shape}")
            
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
        
        # Force normalization
        self.force_mean = torch.mean(self.forces_tensor, dim=0)
        self.force_std = torch.std(self.forces_tensor, dim=0)
        self.force_std[self.force_std < 1e-6] = 1.0  # Avoid division by zero
        
        self.forces_normalized = (self.forces_tensor - self.force_mean) / self.force_std
        
        self.logger.info("Data normalized")
    
    def train(self, epochs=1000, batch_size=32, learning_rate=1e-3, validation_split=0.2):
        """Train the force inference model"""
        # Split data into training and validation
        n_samples = len(self.displacements_normalized)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val
        
        indices = torch.randperm(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        train_displacements = self.displacements_normalized[train_indices]
        train_forces = self.forces_normalized[train_indices]
        
        val_displacements = self.displacements_normalized[val_indices]
        val_forces = self.forces_normalized[val_indices]
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(train_displacements, train_forces)
        val_dataset = torch.utils.data.TensorDataset(val_displacements, val_forces)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
        
        # Optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
        criterion = torch.nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        self.logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_displacements, batch_forces in train_loader:
                optimizer.zero_grad()
                
                outputs = self.model(batch_displacements)
                loss = criterion(outputs, batch_forces)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_displacements.size(0)
            
            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_displacements, batch_forces in val_loader:
                    outputs = self.model(batch_displacements)
                    loss = criterion(outputs, batch_forces)
                    
                    val_loss += loss.item() * batch_displacements.size(0)
            
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'displacement_mean': self.displacement_mean,
                    'displacement_std': self.displacement_std,
                    'force_mean': self.force_mean,
                    'force_std': self.force_std,
                    'input_dim': self.input_dim,
                    'output_dim': self.output_dim
                }, os.path.join(self.output_dir, "best_model.pt"))
                
            # Log progress
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save final model
        torch.save({
            'epoch': epochs-1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'displacement_mean': self.displacement_mean,
            'displacement_std': self.displacement_std,
            'force_mean': self.force_mean,
            'force_std': self.force_std,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim
        }, os.path.join(self.output_dir, "final_model.pt"))
        
        # Plot loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Force Inference Training Progress')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "training_loss.png"))
        
        self.logger.info(f"Training complete. Best validation loss: {best_val_loss:.6f}")
        
        return train_losses, val_losses
    
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

class ForceLoss(torch.nn.Module):
    """Loss function that uses inferred forces to improve physical consistency"""
    def __init__(self, force_model_path, volume_preservation_weight=0.1,
                 residual_force_weight=1.0, device=None):
        super(ForceLoss, self).__init__()
        
        # Set device
        self.device = device if device is not None else (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # Load force inference model
        self.load_force_model(force_model_path)
        
        # Weights for different loss components
        self.volume_preservation_weight = volume_preservation_weight
        self.residual_force_weight = residual_force_weight
        
    def load_force_model(self, model_path):
        """Load the force inference model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model dimensions
        input_dim = checkpoint['input_dim']
        output_dim = checkpoint['output_dim']
        
        # Create model with the same architecture
        self.force_model = ForceInferenceModel(input_dim, output_dim).to(self.device)
        self.force_model.double()
        
        # Load weights
        self.force_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load normalization parameters
        self.displacement_mean = checkpoint['displacement_mean'].to(self.device)
        self.displacement_std = checkpoint['displacement_std'].to(self.device)
        self.force_mean = checkpoint['force_mean'].to(self.device)
        self.force_std = checkpoint['force_std'].to(self.device)
        
        # Set to evaluation mode
        self.force_model.eval()
        
    def forward(self, displacement, predicted_displacement, energy_calculator):
        """
        Compute loss that balances physical consistency using inferred forces
        
        Args:
            displacement: Ground truth displacement
            predicted_displacement: Predicted displacement from neural model
            energy_calculator: Function to compute elastic energy
            
        Returns:
            Total loss value
        """
        # 1. Compute elastic energy of predicted displacement
        elastic_energy = energy_calculator(predicted_displacement)
        
        # 2. Infer forces from ground truth displacement
        # Normalize displacement
        displacement_flat = displacement.reshape(1, -1)
        displacement_norm = (displacement_flat - self.displacement_mean) / self.displacement_std
        
        with torch.no_grad():
            # Infer force parameters
            force_params_norm = self.force_model(displacement_norm)
            # Denormalize
            force_params = force_params_norm * self.force_std + self.force_mean
        
        # 3. Compute volume preservation term
        # This encourages nearly-incompressible behavior
        volume_loss = self._compute_volume_preservation(predicted_displacement)
        
        # 4. Compute residual force term
        # This ensures the predicted displacement is consistent with inferred forces
        residual_force = self._compute_residual_force(predicted_displacement, force_params)
        
        # 5. Combine loss terms
        total_loss = elastic_energy + \
                    self.volume_preservation_weight * volume_loss + \
                    self.residual_force_weight * residual_force
        
        return total_loss
    
    def _compute_volume_preservation(self, displacement):
        """Compute volume preservation loss term"""
        # This is a placeholder. In a real implementation, we would:
        # 1. Compute the deformation gradient F from displacement
        # 2. Compute det(F) for each element
        # 3. Penalize deviation of det(F) from 1.0
        
        # For now, return a dummy value
        return torch.tensor(0.0, device=self.device)
    
    def _compute_residual_force(self, displacement, force_params):
        """Compute residual force loss term"""
        # This is a placeholder. In a real implementation, we would:
        # 1. Extract the force parameters from the inferred values
        # 2. Compute the expected displacement from these forces using FEM
        # 3. Compute the difference between this expected displacement and the prediction
        
        # For now, return a dummy value
        return torch.tensor(0.0, device=self.device)

def integrate_with_training(routine, force_model_path):
    """Integrate force inference with the training routine"""
    # Create force loss module
    force_loss = ForceLoss(force_model_path, device=routine.device)
    
    # Original loss function
    original_loss_fn = routine.compute_loss
    
    # Override the loss function to include force-based regularization
    def force_regularized_loss(predicted, target, epoch):
        # Compute the original loss
        original_loss = original_loss_fn(predicted, target, epoch)
        
        # Add force-based regularization
        if epoch > 50:  # Start using force loss after some convergence
            force_loss_val = force_loss(target, predicted, routine.energy_calculator)
            return original_loss + 0.01 * force_loss_val
        else:
            return original_loss
    
    # Replace the loss function
    routine.compute_loss = force_regularized_loss
    
    return routine

def main():
    parser = argparse.ArgumentParser(description='Force inference for physics-informed neural modes')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--generate', action='store_true', help='Generate force-displacement data')
    parser.add_argument('--train', action='store_true', help='Train force inference model')
    parser.add_argument('--integrate', action='store_true', help='Integrate with training')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='force_inference', help='Output directory')
    parser.add_argument('--force_model', type=str, default='force_inference/best_model.pt', 
                        help='Path to trained force model for integration')
    
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
        material_params = {
            "youngs_modulus": cfg.get("material", {}).get("youngs_modulus", 10000),
            "poissons_ratio": cfg.get("material", {}).get("poissons_ratio", 0.49),
            "density": cfg.get("material", {}).get("density", 1000)
        }
        
        # Get mesh filename from config
        mesh_filename = cfg.get('mesh', {}).get('filename', 'mesh/beam_732.msh')
        logger.info(f"Using mesh file: {mesh_filename}")
        
        # Create data generator
        data_generator = FEMDataGenerator(mesh_filename, material_params, output_dir=args.output_dir)
        
        # Generate data
        force_data, displacement_data = data_generator.generate_force_displacement_pairs(
            num_samples=args.samples
        )
        
        # Visualize some samples
        for i in range(min(5, args.samples)):
            output_file = os.path.join(args.output_dir, f"sample_{i+1}.png")
            data_generator.visualize_force_displacement(
                force_data[i], displacement_data[i], output_file=output_file
            )
        
    if args.train:
        logger.info("Training force inference model...")
        
        # Create trainer
        trainer = ForceInferenceTrainer(data_dir=args.output_dir, output_dir=args.output_dir)
        
        # Train model
        train_losses, val_losses = trainer.train(epochs=500, batch_size=32)
        
        # Test on some samples
        results = trainer.test_force_inference(n_samples=5)
    
    if args.integrate:
        logger.info("Integrating force inference with training...")
        
        if not os.path.exists(args.force_model):
            logger.error(f"Force model not found at {args.force_model}")
            return
        
        # Load config and create routine
        # (config is already loaded above)
        
        # Create and setup routine
        routine = Routine(cfg)
        
        # Integrate force inference
        routine = integrate_with_training(routine, args.force_model)
        
        # Continue with training as usual
        logger.info("Training with force-based regularization...")
        routine.train(epochs=cfg.get("training", {}).get("epochs", 1000))
if __name__ == "__main__":
    main()