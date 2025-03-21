import os
import torch
import numpy as np
import argparse
from training.train import Routine, load_config, setup_logger
from dolfinx import mesh, fem, plot, io
import ufl
import pyvista

import matplotlib.pyplot as plt
from mpi4py import MPI
from petsc4py import PETSc
import time
import logging
from dolfinx.nls.petsc import NewtonSolver


class DynamicValidator:
    """
    Validates neural plates model with dynamic simulation using finite difference method.
    Can apply stretching force to test beam deformation.
    """
    def __init__(self, routine, num_steps=100, total_time=2.4, 
                 stretch_magnitude=100.0, stretch_direction=[1, 0, 0],
                 force_ramp_time=0.5, damping=0.01, output_dir="stretch_validation"):
        """
        Initialize dynamic validator with stretching force parameters.
        
        Args:
            routine: Trained Routine object with neural model
            num_steps: Number of simulation steps
            total_time: Total simulation time
            stretch_magnitude: Force magnitude (N), default 100.0
            stretch_direction: Direction of stretching force [x, y, z], default along beam axis
            force_ramp_time: Time to ramp up force to full magnitude
            damping: Damping coefficient (0 = no damping, 1 = critically damped)
            output_dir: Directory to save results
        """
        self.routine = routine
        self.total_time = total_time
        self.num_steps = num_steps
        self.dt = total_time / num_steps  # Calculate dt from num_steps
        
        # Stretching force parameters
        self.stretch_magnitude = stretch_magnitude
        self.stretch_direction = np.array(stretch_direction) / np.linalg.norm(stretch_direction)  # Normalize
        self.force_ramp_time = force_ramp_time
        
        # Material parameters
        self.rho = 1000.0  # Density (kg/m³)
        
        # Store original parameters too
        self.damping = damping
        self.output_dir = output_dir
        self.device = routine.device
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger("stretch_validation")
        
        # Get mass matrix (M) from routine
        self.M = routine.M
        
        # Initialize state variables
        self.z_current = torch.zeros(routine.latent_dim, device=self.device, dtype=torch.float64)
        self.z_prev = torch.zeros(routine.latent_dim, device=self.device, dtype=torch.float64)
        self.u_current = torch.zeros(routine.V.dofmap.index_map.size_global * routine.domain.geometry.dim, 
                                    device=self.device, dtype=torch.float64)
        self.u_prev = torch.zeros_like(self.u_current)
        
        # Store solutions for visualization and comparison
        self.fem_solution_history = []
        self.u_history = []
        
        # Get mesh parameters
        self._get_mesh_parameters()
        
        # Initialize visualization
        self.setup_visualization()

    def _get_mesh_parameters(self):
        """Get mesh parameters including volume for gravity calculation"""
        # Get mesh coordinates
        coords = self.routine.domain.geometry.x
        
        # Find bounding box
        x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
        y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
        z_min, z_max = np.min(coords[:, 2]), np.max(coords[:, 2])
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2
        
        # Store these for later use
        self.x_min = x_min
        self.x_max = x_max
        self.y_center = y_center
        self.z_center = z_center
        self.beam_length = x_max - x_min
        
        # Calculate approximate beam volume for gravity calculations
        beam_width = y_max - y_min
        beam_height = z_max - z_min
        self.beam_volume = self.beam_length * beam_width * beam_height
        
        self.logger.info(f"Mesh bounds: X: [{x_min:.4f}, {x_max:.4f}], Y: [{y_min:.4f}, {y_max:.4f}], Z: [{z_min:.4f}, {z_max:.4f}]")
        self.logger.info(f"Mesh center line at y={y_center:.4f}, z={z_center:.4f}")
        self.logger.info(f"Beam dimensions (L×W×H): {self.beam_length:.4f} × {beam_width:.4f} × {beam_height:.4f}")
        self.logger.info(f"Approximate beam volume: {self.beam_volume:.6f} m³")

            
    def compute_stretch_term(self, u, t):
        """Compute the work done by the stretching force at the free end"""
        # Skip if no stretch force
        if self.stretch_magnitude == 0:
            return torch.tensor(0.0, device=self.device, dtype=torch.float64)
            
        # Reshape u to get nodal displacements [n_nodes, 3]
        u_reshaped = u.reshape(-1, 3)
        
        # Get original coordinates of the mesh
        coords = torch.tensor(self.routine.domain.geometry.x, device=self.device, dtype=torch.float64)
        
        # Create stretch direction tensor
        stretch_dir = torch.tensor(self.stretch_direction, device=self.device, dtype=torch.float64)
        
        # Scale magnitude based on ramp time
        current_magnitude = self.stretch_magnitude
        if t < self.force_ramp_time:
            current_magnitude *= (t / self.force_ramp_time)
        
        # Find nodes at the free end (x = x_max)
        free_end_nodes = []
        for i, coord in enumerate(coords):
            if abs(coord[0] - self.x_max) < 1e-4:  # Tolerance for floating point comparison
                free_end_nodes.append(i)
        
        # Compute potential energy from stretching force
        # U = -F·u (negative work done by force)
        stretch_potential = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        
        # Apply force only at free end nodes
        if len(free_end_nodes) > 0:
            for node_idx in free_end_nodes:
                # Displacement at this node
                disp = u_reshaped[node_idx]
                # U = -F·u (negative because work done by external force reduces potential energy)
                stretch_potential -= current_magnitude * torch.dot(stretch_dir, disp)
            
            # Scale by number of nodes to normalize the effect
            stretch_potential /= len(free_end_nodes)
            
        return stretch_potential
                
    def objective_function(self, z, u_current, u_prev, dt, alpha=1.0, t=0.0):
        """
        Objective function to minimize: 
        1/(2*dt²) * ||u(z) - 2u_current + u_{n-1}||_M^2 + E(n(z)) + stretch_term + damping_term
        """
        # Get displacement from latent vector
        l = (self.routine.linear_modes @ z.unsqueeze(1)).squeeze(1)
        y = self.routine.model(z)
        u_next = l + y
        
        # Calculate the temporal term: (u_next - 2*u_current + u_prev) for central difference
        residual = u_next - 2*u_current + u_prev
        
        # Calculate M*residual
        M_petsc_vec = PETSc.Vec().createWithArray(residual.cpu().detach().numpy())
        result_petsc_vec = M_petsc_vec.duplicate()
        self.M.mult(M_petsc_vec, result_petsc_vec)
        M_residual = torch.tensor(result_petsc_vec.getArray(), device=self.device, dtype=torch.float64)
        
        # Calculate norm_M^2 = residual^T * M * residual
        temporal = torch.sum(residual * M_residual) / (2 * dt * dt)
        
        # Energy term
        energy = self.compute_energy(z)

        # Stretching force term
        stretch_term = self.compute_stretch_term(u_next, t)
        
        # Damping term based on velocity 
        damping_term = 0.0
        if self.damping > 0:
            velocity = (u_next - u_current) / dt
            damping_term = self.damping * torch.sum(velocity * velocity)
        
        # Total objective with all terms
        objective = temporal + alpha * energy + stretch_term + damping_term

        z_change_penalty = 1e6 * torch.sum((z - self.z_current)**2)
        objective += z_change_penalty
        
        return objective
    
    def compute_initial_steps_with_fenics(self, num_initial_steps=2):
        """
        Compute initial displacement states using FEniCS with NeoHookean material
        
        Args:
            num_initial_steps: Number of initial steps to compute
            
        Returns:
            u_prev, u_current: Last two displacements as torch tensors
        """
        self.logger.info(f"Computing {num_initial_steps} initial steps using FEniCS with NeoHookean material...")
        
        # Initialize storage for FEM solutions
        self.fem_solution_history = []
        
        # Get domain and function space from routine
        domain = self.routine.domain
        V = self.routine.V
        
        # Run FEM steps
        t = 0.0
        for n in range(num_initial_steps):
            t += self.dt
            self.run_fem_step(t)
            self.logger.info(f"Completed initial FEM step {n+1}/{num_initial_steps}, t={t:.4f}s")
        
        # Check if we have enough solutions
        if len(self.fem_solution_history) >= 2:
            u_prev_array = self.fem_solution_history[-2]
            u_current_array = self.fem_solution_history[-1]
        elif len(self.fem_solution_history) == 1:
            u_prev_array = np.zeros_like(self.fem_solution_history[0])
            u_current_array = self.fem_solution_history[0]
        else:
            # Fallback if no solutions (shouldn't happen)
            u_prev_array = np.zeros((V.dofmap.index_map.size_global * domain.geometry.dim))
            u_current_array = np.zeros((V.dofmap.index_map.size_global * domain.geometry.dim))
        
        # Log max displacement of last step
        max_displacement = np.max(np.linalg.norm(u_current_array.reshape(-1, 3), axis=1))
        self.logger.info(f"Initial steps computed. Max displacement: {max_displacement:.6e}")
        
        # Convert to torch tensors for our simulation
        u_prev_torch = torch.tensor(u_prev_array, device=self.device, dtype=torch.float64)
        u_current_torch = torch.tensor(u_current_array, device=self.device, dtype=torch.float64)
        
        return u_prev_torch, u_current_torch

    def setup_visualization(self):
        """Set up visualization environment for dynamic simulation"""
        self.logger.info("Setting up visualization environment...")
        
        # Set up for visualization similar to twisting_beam.py
        pyvista.set_jupyter_backend("static")
        topology, cell_types, x = plot.vtk_mesh(self.routine.V)
        self.grid = pyvista.UnstructuredGrid(topology, cell_types, x)
        self.x = x  # Store x coordinates for later use
        
        # Create a plotter with 2x1 layout for vertical stacking
        self.plotter = pyvista.Plotter(shape=(2, 1))
        gif_path = os.path.join(self.output_dir, "beam_deformation_stretch.gif")
        self.plotter.open_gif(gif_path)
        
        # Calculate the center of the beam
        center = [(self.x_min + self.x_max) / 2, self.y_center, self.z_center]

        # Set camera for both subplots
        for i in range(2):
            self.plotter.subplot(i, 0)  # Changed to (i, 0) for vertical stacking
            
            # Position camera to see the entire beam from the side
            self.plotter.camera_position = [
                (center[0], center[1] - self.beam_length*2, center[2]),  # Camera position further back
                center,  # Focal point at center of beam
                (0, 0, 1)  # Up direction
            ]
            
            # Apply XZ view for side view
            
            # Use a better zoom level to see the whole beam
            self.plotter.zoom_camera(0.7)
        
        # Create a linear function space for visualization if needed
        V_linear = fem.functionspace(self.routine.domain, ("CG", 1, (3,)))
        self.u_linear = fem.Function(V_linear)
        
        self.logger.info(f"Created visualization grid with {self.grid.n_points} points and {self.grid.n_cells} cells")
        self.logger.info(f"GIF will be saved to {gif_path}")
        
    def visualize_step(self, step, t, z, u, gravity_magnitude):
        """Visualize current simulation step with side-by-side comparison to FEM solution"""
        if self.grid is None or self.plotter is None:
            self.logger.warning(f"Skipping visualization for step {step} (no visualization grid or plotter)")
            return
        
        try:
            # Create a function in the original function space
            u_quadratic = fem.Function(self.routine.V)
            u_quadratic.x.array[:] = u.cpu().numpy()
            
            # Interpolate to the visualization space
            self.u_linear.interpolate(u_quadratic)
            u_linear_np = self.u_linear.x.array
            
            # Create mesh with deformation for NN solution
            nn_grid = self.grid.copy()
            nn_grid.point_data["displacement"] = u_linear_np.reshape((-1, 3))
            nn_grid["displacement_magnitude"] = np.linalg.norm(u_linear_np.reshape((-1, 3)), axis=1)
            nn_warped = nn_grid.warp_by_vector("displacement", factor=1.0)
            
            # Check if we have FEM solution for this step
            has_fem_solution = step < len(self.fem_solution_history)
            
            # Clear the plotter
            self.plotter.clear()
            
            # Add neural network solution (left)
            self.plotter.subplot(0, 0)
            self.plotter.add_mesh(nn_warped, scalars="displacement_magnitude", cmap="viridis", show_edges=True)
            self.plotter.add_title("Neural Network Solution")
            
            # Add FEM solution if available (right)
            self.plotter.subplot(1, 0)
            if has_fem_solution:
                # Create mesh with FEM deformation
                fem_grid = self.grid.copy()
                fem_grid.point_data["displacement"] = self.fem_solution_history[step].reshape((-1, 3))
                fem_grid["displacement_magnitude"] = np.linalg.norm(self.fem_solution_history[step].reshape((-1, 3)), axis=1)
                fem_warped = fem_grid.warp_by_vector("displacement", factor=1.0)
                
                self.plotter.add_mesh(fem_warped, scalars="displacement_magnitude", cmap="viridis", show_edges=True)
                self.plotter.add_title("FEM Reference Solution")
                
                # Compute error metrics
                error = u.cpu().numpy() - self.fem_solution_history[step]
                error_norm = np.linalg.norm(error) / np.linalg.norm(self.fem_solution_history[step])
                max_error = np.max(np.abs(error))
                
                # Add global title with error metrics
                self.plotter.add_text(f"Time: {t:.3f}s, Step: {step}, Rel. Error: {error_norm:.4f}, Max Error: {max_error:.4e}", 
                                    position="upper_edge", font_size=14, color='white')
            else:
                # Display a message in the right subplot when FEM solution is not yet available
                # Fix: Change 'center' to specific coordinates (0.5, 0.5) for center positioning
                self.plotter.add_text("FEM solution not yet available", position=(0.5, 0.5), font_size=14, color='white')
                self.plotter.add_title("FEM Reference Solution (Processing...)")
            
            # Write the frame to the GIF file
            self.plotter.write_frame()
            
            if step % 10 == 0:  # Log less frequently
                self.logger.info(f"Visualization frame added for step {step}, t={t:.3f}s")
                
        except Exception as e:
            self.logger.error(f"Error in visualization for step {step}: {e}")


    # Modify visualize_results to include error metrics compared to FEM
    def visualize_results(self, time_points, displacement_norms, energy_history, gravity_history):
        """Create summary plots of simulation results including comparison with FEM"""
        fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        
        # Plot displacement norm over time
        axes[0].plot(time_points, displacement_norms, label='Neural Network')
        
        # Add FEM solution if available
        fem_time_points = time_points[:len(self.fem_solution_history)]
        if len(fem_time_points) > 0:
            fem_displacement_norms = [np.max(np.linalg.norm(sol.reshape(-1, 3), axis=1)) for sol in self.fem_solution_history]
            axes[0].plot(fem_time_points, fem_displacement_norms, '--', label='FEM Reference')
        
        axes[0].set_ylabel("Displacement Norm")
        axes[0].set_title("Displacement Norm Over Time")
        axes[0].grid(True)
        axes[0].legend()
        
        # Plot energy over time
        axes[1].plot(time_points, energy_history)
        axes[1].set_ylabel("Elastic Energy")
        axes[1].set_title("System Energy Over Time")
        axes[1].grid(True)
        
        # Plot gravity over time (renamed from torque)
        axes[2].plot(time_points, gravity_history)
        axes[2].set_ylabel("Gravity Magnitude (m/s²)")  # Updated from Torque
        axes[2].set_title("Applied Gravity")  # Updated from Applied Torque
        axes[2].grid(True)
        
        # Plot error compared to FEM if available
        if len(self.fem_solution_history) > 0:
            error_metrics = []
            for i, t in enumerate(fem_time_points):
                step_idx = time_points.index(t)
                nn_sol = self.u_history[step_idx] if hasattr(self, 'u_history') else None
                
                if nn_sol is not None:
                    error = nn_sol.cpu().numpy() - self.fem_solution_history[i]
                    rel_error = np.linalg.norm(error) / np.linalg.norm(self.fem_solution_history[i])
                    error_metrics.append(rel_error)
            
            if error_metrics:
                axes[3].plot(fem_time_points, error_metrics)
                axes[3].set_ylabel("Relative Error")
                axes[3].set_xlabel("Time (s)")
                axes[3].set_title("Neural Network vs FEM Relative Error")
                axes[3].grid(True)
        else:
            axes[3].set_xlabel("Time (s)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "simulation_summary.png"))
        plt.close()
    
    def find_optimal_z(self, u_current, u_prev, dt, current_time, num_iters=10):
        """Find optimal latent vector z that minimizes the objective function"""
        # Use extrapolation from previous two z vectors as starting point
        z_extrapolated = self.z_current + (self.z_current - self.z_prev)
        
        # Start optimization from extrapolated z to maintain momentum
        z = z_extrapolated.clone().detach().requires_grad_(True)
        
        # Use L-BFGS for optimization
        optimizer = torch.optim.LBFGS([z], 
                                    lr=1, 
                                    max_iter=10,
                                    line_search_fn='strong_wolfe')
        
        # Weight for energy term
        alpha = 1.0
        
        initial_loss = None
        final_loss = None
        self.logger.info(f"Starting z optimization at t={current_time:.3f}s")
        
        # Track whether each iteration has had its first evaluation
        first_evals = [False] * num_iters
        
        # Optimize
        for i in range(num_iters):
            def closure():
                optimizer.zero_grad()
                # Pass current time to the objective function
                loss = self.objective_function(z, u_current, u_prev, dt, alpha, current_time)
                
                # Only log on the FIRST evaluation of each iteration
                nonlocal first_evals
                if not first_evals[i]:
                    with torch.no_grad():
                        # Get components for logging
                        energy = self.compute_energy(z)
                        stage = "Initial" if i == 0 else "Iteration"
                        
                        # Store initial loss (only on first iteration's first eval)
                        nonlocal initial_loss
                        if i == 0:
                            initial_loss = loss.item()
                            self.logger.info(f"{stage} components at t={current_time:.3f}s, iter {i}: "
                                            f"Energy={energy.item():.3e}, "
                                            f"Total={loss.item():.3e}")
                    
                    # Mark this iteration as having had its first evaluation
                    first_evals[i] = True
                    
                loss.backward()
                return loss
                    
            loss = optimizer.step(closure)
            
            # Print progress every iteration after the first
            if i > 0:
                self.logger.info(f"  z opt iter {i}/{num_iters}, loss: {loss.item():.6e}")
        
        # Final logging after optimization
        with torch.no_grad():
            final_loss = loss.item()
            energy = self.compute_energy(z)
            self.logger.info(f"Final components at t={current_time:.3f}s: "
                            f"Energy={energy.item():.3e}, "
                            f"Total={final_loss:.3e}")
        
        # Report optimization results
        reduction = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss else 0
        self.logger.info(f"z optimization complete: initial loss={initial_loss:.6e}, "
                        f"final loss={final_loss:.6e}, reduction={reduction:.2f}%")
        
        # Get optimal displacement
        with torch.no_grad():
            l = (self.routine.linear_modes @ z.unsqueeze(1)).squeeze(1)
            y = self.routine.model(z)
            u_opt = l + y
            
            # Log physical quantities
            max_disp = torch.max(torch.norm(u_opt.reshape(-1, 3), dim=1)).item()
            self.logger.info(f"Max displacement: {max_disp:.6f}")
            
        return z.detach(), u_opt.detach()
    
    def find_best_latent_vector(self, u_target):
        """Find the latent vector that best reproduces the target displacement using modal coordinates"""
        self.logger.info(f"Computing latent vector through modal projection")
        
        # First compute the modal coordinates through projection
        with torch.no_grad():
            # Get modes in matrix form
            modes = self.routine.linear_modes
            
            # Project displacement onto modes using mass matrix
            # z = Φᵀ·M·u (if modes are mass-normalized)
            
            # Convert u_target to PETSc vector
            u_petsc = PETSc.Vec().createWithArray(u_target.cpu().numpy())
            result_petsc = u_petsc.duplicate()
            
            # Apply mass matrix: M·u_target
            self.M.mult(u_petsc, result_petsc)
            Mu = torch.tensor(result_petsc.getArray(), device=self.device, dtype=torch.float64)
            
            # Project onto modes
            modal_coords = torch.matmul(modes.T, Mu)
            
            # Now find the neural correction
            # Start with the modal coordinates
            z = modal_coords.clone().to(self.device).requires_grad_(True)
            
            # Use a few iterations of optimization to refine
            optimizer = torch.optim.LBFGS([z], lr=1, max_iter=5)
            
            for i in range(5):  # Just a few iterations to refine
                def closure():
                    optimizer.zero_grad()
                    
                    # Get displacement from latent vector
                    l = (modes @ z.unsqueeze(1)).squeeze(1)
                    y = self.routine.model(z)
                    u = l + y
                    
                    # Compute difference
                    loss = torch.sum((u - u_target)**2)
                    loss.backward()
                    return loss
                    
                optimizer.step(closure)
            
            # Get final error
            l = (modes @ z.unsqueeze(1)).squeeze(1)
            y = self.routine.model(z)
            u = l + y
            error = torch.mean((u - u_target)**2).sqrt().item()
            
            self.logger.info(f"Modal projection complete. RMSE: {error:.6e}")
            
            return z.detach()
        
    def run_fem_step(self, t):
        """Run one step of the FEM simulation with stretching force - rewritten to match stretch_beam.py"""
        self.logger.info(f"Running FEM simulation for time t={t:.3f}s")
        
        # Get domain and function space from routine
        domain = self.routine.domain
        V = self.routine.V
        
        # Create or update displacement functions
        if len(self.fem_solution_history) >= 2:
            u = fem.Function(V)
            u_prev = fem.Function(V)
            u.x.array[:] = 2*self.fem_solution_history[-1] - self.fem_solution_history[-2]
            u_prev.x.array[:] = self.fem_solution_history[-1]
        elif len(self.fem_solution_history) == 1:
            u = fem.Function(V)
            u_prev = fem.Function(V)
            u.x.array[:] = self.fem_solution_history[-1]
            u_prev.x.array[:] = 0.0  # Zero for first step
        else:
            u = fem.Function(V)
            u_prev = fem.Function(V)
        
        # Create test function and additional functions for dynamics
        v = ufl.TestFunction(V)
        u_dot = fem.Function(V, name="Velocity")
        u_dot_prev = fem.Function(V, name="Velocity_prev")
        u_ddot = fem.Function(V, name="Acceleration")
        
        # Material parameters - match exactly with stretch_beam.py
        E = 1.0e5  # Young's modulus
        nu = 0.4  # Poisson's ratio
        mu = E / (2.0 * (1.0 + nu))
        lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        rho = 1000.0  # Density
        
        # Newmark-beta parameters - match exactly with stretch_beam.py
        beta = 0.25
        gamma = 0.5
        
        # Define boundary condition (fixed at one end)
        def fixed_boundary(x):
            return np.isclose(x[0], self.x_min)
        
        boundary_dofs = fem.locate_dofs_geometrical(V, fixed_boundary)
        u_D = np.zeros(3, dtype=np.float64)  # 3D zero displacement
        bc = fem.dirichletbc(u_D, boundary_dofs, V)

        # Define the force application function - exactly as in stretch_beam.py
        def apply_force(x, t):
            # Apply force in the x-direction
            force_dir = ufl.as_vector([1.0, 0.0, 0.0])
            
            # Magnitude that varies with time
            magnitude = self.stretch_magnitude
            
            # Ramp up force for the first 2 seconds
            if t < 2.0:
                magnitude *= (t / 2.0)  # Scale magnitude from 0 to 1 over 2 seconds
            
            change_time = 2.5
            if t >= change_time:
                magnitude *= 0.5  # Reduce after change_time
            
            return magnitude * force_dir

        # Define NeoHookean weak form with stretching force - follow stretch_beam.py structure
        def weak_form(u, u_prev, u_dot_prev, u_ddot, v, t, dt, domain, mu, lmbda, beta, gamma):
            # Kinematics
            d = len(u)
            I = ufl.Identity(d)
            F = I + ufl.grad(u)  
            C = F.T * F
            Ic = ufl.tr(C)
            J = ufl.det(F)
            
            # Neo-Hookean strain energy density (incompressible)
            psi = (mu/2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda/2) * (ufl.ln(J))**2
            
            # Internal elastic energy
            internal_work = psi * ufl.dx
            
            # External work from force - applied at the free end
            x = ufl.SpatialCoordinate(domain)
            
            # Define the force direction and magnitude
            force = apply_force(x, t)
            
            # Define the boundary where the force is applied (x = max)
            def end(x):
                return np.isclose(x[0], self.x_max)
            
            # Create a facet integration measure for the end boundary
            from dolfinx import mesh as dmesh
            end_facets = dmesh.locate_entities_boundary(domain, domain.topology.dim-1, end)
            from dolfinx.mesh import meshtags
            mt = meshtags(domain, domain.topology.dim-1, end_facets, np.ones(len(end_facets), dtype=np.int32))
            ds = ufl.Measure("ds", domain=domain, subdomain_data=mt, subdomain_id=1)
            
            # External work due to the force
            external_work = ufl.dot(force, v) * ds
            
            # Newmark-beta for dynamics
            inertia = ufl.dot(rho * u_ddot, v) * ufl.dx
            
            return ufl.derivative(internal_work, u, v) - external_work + inertia
        
        # Newmark-beta update - exactly as in stretch_beam.py
        u_ddot_expr = fem.Expression(u_ddot, V.element.interpolation_points())
        u_ddot.interpolate(u_ddot_expr)
        
        u_expr = fem.Expression(u, V.element.interpolation_points())
        u.interpolate(u_expr)
        
        u_dot_expr = fem.Expression(u_dot, V.element.interpolation_points())
        u_dot.interpolate(u_dot_expr)
        
        u_ddot.x.array[:] = (1 / (beta * self.dt**2)) * (u.x.array - u_prev.x.array) - (1 / (beta * self.dt)) * u_dot_prev.x.array - (1/(2*beta)-1) * u_ddot.x.array
        u_dot.x.array[:] = u_dot_prev.x.array + (1 - gamma) * self.dt * u_ddot.x.array + gamma * self.dt * u_ddot.x.array
        
        # Simple approach as in stretch_beam.py
        try:
            F = weak_form(u, u_prev, u_dot_prev, u_ddot, v, t, self.dt, domain, mu, lmbda, beta, gamma)
            J = ufl.derivative(F, u)  # Jacobian
            problem = fem.petsc.NonlinearProblem(F, u, [bc], J=J)
            
            # Create Newton solver - same parameters as stretch_beam.py
            solver = NewtonSolver(MPI.COMM_WORLD, problem)
            solver.convergence_criterion = "residual"
            solver.rtol = 1e-8  # Match exactly with stretch_beam.py
            solver.max_it = 50
            
            # Solve the nonlinear problem
            num_its, converged = solver.solve(u)
            
            if converged:
                self.logger.info(f"FEM step at t={t:.4f}s: Nonlinear solve complete in {num_its} iterations")
                
                # Calculate maximum displacement
                u_values = u.x.array.reshape(-1, 3)
                max_displacement = np.max(np.linalg.norm(u_values, axis=1))
                self.logger.info(f"  Max displacement: {max_displacement:.4f}")
                
                # Store solution
                self.fem_solution_history.append(u.x.array.copy())
                
                # Update previous solution
                u_prev.x.array[:] = u.x.array
                u_dot_prev.x.array[:] = u_dot.x.array
                
                return True
            else:
                self.logger.error("Newton solver did not converge")
                
        except Exception as e:
            self.logger.error(f"Solver failed: {str(e)}")
        
        # If we get here, use previous solution as fallback
        if len(self.fem_solution_history) > 0:
            self.logger.warning("Using previous solution due to solver error")
            u.x.array[:] = self.fem_solution_history[-1]
            self.fem_solution_history.append(u.x.array.copy())
            return False
        else:
            self.logger.error("No previous solution available and solver failed")
            self.fem_solution_history.append(np.zeros_like(u.x.array))
            return False

    def run_simulation(self):
        """Run the full dynamic simulation with improved stability"""
        self.logger.info("Starting dynamic simulation with stretching force application...")
        
        # Initialize arrays for storing results
        z_history = []
        self.u_history = []  # Store reference to u_history for visualization
        displacement_norms = []
        energy_history = []
        force_history = []
        time_points = []
        
        # Initialize with FEniCS-computed values
        self.u_prev, self.u_current = self.compute_initial_steps_with_fenics(num_initial_steps=3)

        # Find initial latent vectors that best represent these displacements
        self.z_prev = self.find_best_latent_vector(self.u_prev)
        self.z_current = self.find_best_latent_vector(self.u_current)

        # Store initial state
        z_history.append(self.z_current.cpu().numpy())
        self.u_history.append(self.u_current.clone())
        displacement_norms.append(torch.max(torch.norm(self.u_current.reshape(-1, 3), dim=1)).item())
        energy_history.append(self.compute_energy(self.z_current).item())
        
        # Calculate initial force magnitude with ramp-up
        if self.force_ramp_time > 0:
            initial_force = 0.0
        else:
            initial_force = self.stretch_magnitude
        force_history.append(initial_force)
        
        time_points.append(0.0)
        
        # Create plotter for visualization
        last_viz_time = 0.0
        viz_interval = self.dt * 5  # Visualize less frequently (every 5th step)
        
        ## Main simulation loop
        for step in range(1, self.num_steps + 1):
            t = step * self.dt
            
            # Calculate current force magnitude with ramp-up (for logging)
            current_force = self.stretch_magnitude
            if t < 2.0:  # Match the time ramp from stretch_beam.py exactly
                current_force *= (t / 2.0)
            if t >= 2.5:  # Match the force reduction time from stretch_beam.py
                current_force *= 0.5
                
            # Run FEM simulation for this time step - BEFORE neural network
            # This ensures we have good reference solutions
            if step % 2 == 0:  # Only run FEM every other step to save time
                self.run_fem_step(t)
                
            # Find optimal z - but use more modest optimization
            self.z_next, self.u_next = self.find_optimal_z(
                self.u_current, self.u_prev, self.dt, t, num_iters=5  # Reduced iterations
            )
            
            # Compute energy
            energy = self.compute_energy(self.z_next).item()
            
            # Store results
            z_history.append(self.z_next.cpu().numpy())
            self.u_history.append(self.u_next.clone())
            max_disp = torch.max(torch.norm(self.u_next.reshape(-1, 3), dim=1)).item()
            displacement_norms.append(max_disp)
            energy_history.append(energy)
            force_history.append(current_force)
            time_points.append(t)
            
            # Log progress
            if step % 10 == 0:
                self.logger.info(f"Step {step}/{self.num_steps}, Time: {t:.2f}s, Energy: {energy:.4e}, "
                            f"Force: {current_force:.4f}, Max Disp: {max_disp:.4f}")
            
            # Visualize less frequently to avoid memory issues
            if t - last_viz_time >= viz_interval:
                self.visualize_step(len(time_points)-1, t, self.z_next, self.u_next, current_force)
                last_viz_time = t
            
            # Update state for next step
            self.u_prev = self.u_current.clone()
            self.u_current = self.u_next.clone()
            self.z_prev = self.z_current.clone()
            self.z_current = self.z_next.clone()
            
        # Visualize final results
        self.visualize_results(time_points, displacement_norms, energy_history, force_history)
        
        return {
            'time': time_points,
            'z': z_history,
            'displacement_norm': displacement_norms,
            'energy': energy_history
        }


    def compute_energy(self, z):
        """Compute elastic energy for a given latent vector z"""
        with torch.no_grad():
            l = (self.routine.linear_modes @ z.unsqueeze(1)).squeeze(1)
            y = self.routine.model(z)
            u = l + y
            energy = self.routine.energy_calculator(u)
        return energy

    
    
    



def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Dynamic validation with stretching for Neural Plates')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='config file path')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt', help='checkpoint path')
    parser.add_argument('--steps', type=int, default=1000, help='number of simulation steps')
    parser.add_argument('--time', type=float, default=10.0, help='total simulation time')
    parser.add_argument('--force', type=float, default=1000, help='stretching force magnitude (N)')
    parser.add_argument('--damping', type=float, default=0.01, help='damping coefficient')
    parser.add_argument('--output', type=str, default='validation_results', help='output directory')
    args = parser.parse_args()
    
    # Setup logger
    setup_logger("stretch_validation", log_dir=args.output)
    logger = logging.getLogger("stretch_validation")
    
    # Load config and create routine
    cfg = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")
    
    # Create routine and load checkpoint
    routine = Routine(cfg)
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    if os.path.exists(args.checkpoint):
        routine.load_checkpoint(args.checkpoint)
        logger.info("Checkpoint loaded successfully")
    else:
        logger.warning(f"Checkpoint not found at {args.checkpoint}, using untrained model")

  # Create validator and run simulation with stretching
    validator = DynamicValidator(
        routine=routine,
        num_steps=args.steps,
        total_time=args.time,
        stretch_magnitude=args.force,
        stretch_direction=[1, 0, 0],  # Along beam axis (X direction)
        force_ramp_time=0.5,          # Ramp up force over 0.5s
        damping=args.damping,
        output_dir=args.output
    )
    
    # Run simulation
    results = validator.run_simulation()

if __name__ == "__main__":
    main()

