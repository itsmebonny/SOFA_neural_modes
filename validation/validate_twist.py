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
    Can apply torque at the beam end to test twisting behavior.
    """
    def __init__(self, routine, num_steps=100, total_time=2.4, 
                 torque_magnitude=1.0e4, torque_axis=[0, 0, 1],
                 torque_ramp_time=2.0, change_time=2.5, damping=0.01, output_dir="validation_results"):
        """
        Initialize dynamic validator with torque parameters matching twisting_beam.py.
        
        Args:
            routine: Trained Routine object with neural model
            num_steps: Number of simulation steps (default: 100 as in twisting_beam.py)
            total_time: Total simulation time (matches twisting_beam.py)
            torque_magnitude: Maximum torque magnitude to apply (matches twisting_beam.py)
            torque_axis: Axis of rotation for the torque [x, y, z]
            torque_ramp_time: Time to ramp up the torque to full magnitude (matches twisting_beam.py)
            change_time: Time at which torque reduces by 50% (from twisting_beam.py)
            damping: Damping coefficient (0 = no damping, 1 = critically damped)
            output_dir: Directory to save results
        """
        self.routine = routine
        self.total_time = total_time
        self.num_steps = num_steps
        self.dt = total_time / num_steps  # Calculate dt from num_steps
        
        # Torque parameters
        self.torque_magnitude = torque_magnitude
        self.torque_axis = np.array(torque_axis) / np.linalg.norm(torque_axis)  # Normalize axis
        self.torque_ramp_time = torque_ramp_time  # 2.0s to match twisting_beam.py
        self.change_time = change_time  # Time at which torque reduces by 50% (from twisting_beam.py)
        
        # Material parameters from twisting_beam.py
        self.rho = 1000.0  # Density (match twisting_beam.py)
        
        # Store original parameters too
        self.damping = damping
        self.output_dir = output_dir
        self.device = routine.device
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger("dynamic_validator")
        
        # Get mass matrix (M) from routine
        self.M = routine.M
        
        # Initialize state variables
        self.z_current = torch.zeros(routine.latent_dim, device=self.device, dtype=torch.float64)
        self.z_prev = torch.zeros(routine.latent_dim, device=self.device, dtype=torch.float64)
        self.u_current = torch.zeros(routine.V.dofmap.index_map.size_global * routine.domain.geometry.dim, 
                                    device=self.device, dtype=torch.float64)
        self.u_prev = torch.zeros_like(self.u_current)


        self.V = routine.V
        self.u = fem.Function(self.V, name="Displacement")
        self.u_prev = fem.Function(self.V, name="Displacement_prev")
        self.u_dot = fem.Function(self.V, name="Velocity")
        self.u_dot_prev = fem.Function(self.V, name="Velocity_prev")
        self.u_ddot = fem.Function(self.V, name="Acceleration")
        
        # Store solutions for visualization and comparison
        self.fem_solution_history = []
        self.u_history = []
        
        # Get mesh parameters - match twisting_beam.py approach
        self._get_mesh_parameters()
        
        # Initialize visualization
        self.setup_visualization()
        
    def _get_mesh_parameters(self):
        """Get mesh parameters including center line - exactly as in twisting_beam.py"""
        # Get mesh coordinates
        coords = self.routine.domain.geometry.x
        
        # Find bounding box and center line
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
        
        # Calculate the center of the end face (for torque application)
        end_region_nodes = [i for i, coord in enumerate(coords) if coord[0] >= (x_max - 0.1 * self.beam_length)]
        self.end_center = np.mean(coords[end_region_nodes], axis=0) if end_region_nodes else np.array([x_max, y_center, z_center])
        
        self.logger.info(f"Mesh bounds: X: [{x_min:.4f}, {x_max:.4f}], Y: [{y_min:.4f}, {y_max:.4f}], Z: [{z_min:.4f}, {z_max:.4f}]")
        self.logger.info(f"Mesh center line at y={y_center:.4f}, z={z_center:.4f}")
        self.logger.info(f"Actual beam length: {self.beam_length:.4f}")


            
    def compute_torque_term(self, u, t):
        """Compute the virtual work of the applied torque - using same approach as twisting_beam.py"""
        # Skip if no torque
        if self.torque_magnitude == 0:
            return torch.tensor(0.0, device=self.device, dtype=torch.float64)
            
        # Reshape u to get nodal displacements [n_nodes, 3]
        u_reshaped = u.reshape(-1, 3)
        
        # Get original coordinates of the mesh
        coords = torch.tensor(self.routine.domain.geometry.x, device=self.device, dtype=torch.float64)
        
        # Compute torque with corrected center (match twisting_beam.py)
        torque_potential = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        
        for i, (pos, orig_pos) in enumerate(zip(coords + u_reshaped, coords)):
            # Calculate distance from the actual center axis
            r_loc = torch.sqrt((orig_pos[1]-self.y_center)**2 + (orig_pos[2]-self.z_center)**2)
            
            # Direction perpendicular to r_vec and parallel to xy-plane
            theta = torch.atan2(orig_pos[2]-self.z_center, orig_pos[1]-self.y_center)
            force_dir = torch.tensor([0.0, -torch.sin(theta), torch.cos(theta)], 
                                device=self.device, dtype=torch.float64)
            
            # Magnitude with same scaling as twisting_beam.py
            magnitude = self.torque_magnitude * ((orig_pos[0]-self.x_min)/self.beam_length)
            
            if t < 2.0:
                magnitude *= (t / 2.0)
            
            if t >= self.change_time:
                magnitude *= 0.5
            
            torque_potential += magnitude * r_loc * torch.dot(force_dir, u_reshaped[i])
            
        return torque_potential
                

    def compute_volume_preservation(self, u):
        """
        Compute volume preservation term based on deformation gradient determinants
        """
        # Reshape displacement
        u_reshaped = u.reshape(-1, 3)
        
        # Get coordinates
        coords = torch.tensor(self.routine.domain.geometry.x, device=self.device, dtype=torch.float64)
        
        # Initialize storage for J values
        num_elements = self.routine.domain.topology.index_map(self.routine.domain.topology.dim).size_local
        det_J_values = []
        
        # Get mesh connectivity
        tdim = self.routine.domain.topology.dim
        self.routine.domain.topology.create_connectivity(tdim, 0)
        connectivity = self.routine.domain.topology.connectivity(tdim, 0)
        
        # Process elements in batches to save memory
        batch_size = min(500, num_elements)  # Process up to 500 elements at once
        total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        
        # Sample some elements to reduce computational cost if there are many
        if num_elements > 1000:
            sample_indices = torch.randperm(num_elements)[:1000].to(self.device)
        else:
            sample_indices = torch.arange(num_elements).to(self.device)
        
        num_processed = 0
        
        for elem_idx in sample_indices:
            # Get nodes for this element
            nodes = connectivity.links(elem_idx)
            
            if len(nodes) < 4:  # Skip if not enough nodes
                continue
                
            # Get original coordinates and deformed coordinates
            X = coords[nodes]  # Original
            x = X + u_reshaped[nodes]  # Deformed
            
            # Compute deformation gradient F
            # For tetrahedron, can use first 3 edges
            if len(nodes) == 4:  # Tetrahedron
                # Edges in reference configuration
                dX1 = X[1] - X[0]
                dX2 = X[2] - X[0]
                dX3 = X[3] - X[0]
                
                # Edges in deformed configuration
                dx1 = x[1] - x[0]
                dx2 = x[2] - x[0]
                dx3 = x[3] - x[0]
                
                # Create matrices for reference and deformed configurations
                X_mat = torch.stack([dX1, dX2, dX3], dim=1)
                x_mat = torch.stack([dx1, dx2, dx3], dim=1)
                
                # Compute F = x_mat * X_mat^(-1)
                if torch.det(X_mat) > 1e-10:  # Check if invertible
                    F = x_mat @ torch.linalg.inv(X_mat)
                    
                    # Compute determinant of F
                    det_J = torch.det(F)
                    det_J_values.append(det_J)
                    
                    # Compute loss: (J - 1)^2
                    vol_loss = (det_J - 1.0)**2
                    
                    # Add barrier term to prevent negative J
                    if det_J <= 0:
                        vol_loss += 1000.0 * torch.abs(det_J)
                        
                    total_loss += vol_loss
                    num_processed += 1
            
        # Average the loss over processed elements
        if num_processed > 0:
            total_loss /= num_processed
            
            # Log J statistics for debugging
            if len(det_J_values) > 0:
                det_J_tensor = torch.stack(det_J_values)
                min_J = torch.min(det_J_tensor).item()
                max_J = torch.max(det_J_tensor).item()
                mean_J = torch.mean(det_J_tensor).item()
                
                if hasattr(self, 'logger'):
                    self.logger.debug(f"Volume stats - min J: {min_J:.4f}, max J: {max_J:.4f}, mean J: {mean_J:.4f}")
        
        return total_loss
    



    def objective_function(self, z, u_current, u_prev, dt, alpha=1.0, t=0.0):
        """
        Objective function with volume preservation constraint
        """
        # Get displacement from latent vector
        l = (self.routine.linear_modes @ z.unsqueeze(1)).squeeze(1)
        y = self.routine.model(z)
        u_next = l + y
        
        # Existing code for temporal term calculation
        beta = 0.25
        gamma = 0.5
        
        # We need velocity for Newmark - approximate from previous step
        v_current = (u_current - u_prev) / dt
        
        # Newmark-beta residual calculation
        residual = (1/(beta*dt**2))*(u_next - u_current - dt*v_current)
        
        # Apply mass matrix
        M_petsc_vec = PETSc.Vec().createWithArray(residual.cpu().detach().numpy())
        result_petsc_vec = M_petsc_vec.duplicate()
        self.M.mult(M_petsc_vec, result_petsc_vec)
        M_residual = torch.tensor(result_petsc_vec.getArray(), device=self.device, dtype=torch.float64)
        
        # Calculate norm_M^2 = residual^T * M * residual
        temporal = torch.sum(residual * M_residual) / (2 * dt * dt)

        # Energy term
        energy_term = self.compute_energy(z)

       
        
        # Total objective with carefully weighted components
        objective = temporal + energy_term 

   
        
        # Store component values for logging (add volume preservation)
        self.loss_components = {
            'temporal': temporal.item(),
            'energy': energy_term.item(),
            'total': objective.item()
        }
        
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
        
        try:
            # Set up for visualization similar to twisting_beam.py
            pyvista.set_jupyter_backend("static")
            topology, cell_types, x = plot.vtk_mesh(self.routine.V)
            self.grid = pyvista.UnstructuredGrid(topology, cell_types, x)
            self.x = x  # Store x coordinates for later use
            
            # Create a plotter with 1x2 layout for side-by-side comparison
            self.plotter = pyvista.Plotter(shape=(1, 2), window_size=[1920, 1080])
            gif_path = os.path.join(self.output_dir, "beam_deformation.gif")
            self.plotter.open_gif(gif_path)
            
            # Set camera to match twisting_beam.py
            for i in range(2):
                self.plotter.subplot(0, i)
                self.plotter.camera_position = [(20.0, 3.0, 2.0), (0.0, -2.0, 0.0), (0.0, 0.0, 2.0)]
                self.plotter.camera.zoom(1.5)
            
            # Create a linear function space for visualization if needed
            V_linear = fem.functionspace(self.routine.domain, ("CG", 1, (3,)))
            self.u_linear = fem.Function(V_linear)
            
            self.logger.info(f"Created visualization grid with {self.grid.n_points} points and {self.grid.n_cells} cells")
            self.logger.info(f"GIF will be saved to {gif_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to set up visualization: {e}")
            self.logger.warning("Running without visualization capability")
            self.grid = None
            self.plotter = None

    def visualize_step(self, step, t, z, u, torque_magnitude):
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
            
            # Calculate energies for comparison
            nn_energy = self.compute_energy(z).item()
            
            # Get FEM energy if available
            if has_fem_solution:
                # Use the previously computed FEM energy if available
                if hasattr(self, 'fem_energy_history') and step < len(self.fem_energy_history):
                    fem_energy = self.fem_energy_history[step]
                else:
                    # Compute it now if not already computed
                    u_fem = fem.Function(self.routine.V)
                    u_fem.x.array[:] = self.fem_solution_history[step]
                    fem_energy = self.compute_neohookean_energy(u_fem)
            else:
                fem_energy = None
                
            # Clear the plotter
            self.plotter.clear()
            
            # Add neural network solution (left)
            self.plotter.subplot(0, 0)
            self.plotter.add_mesh(nn_warped, scalars="displacement_magnitude", cmap="viridis", show_edges=True, clim=[0, 1])
            
            self.plotter.add_title(f"Neural Network Solution (Energy: {nn_energy:.4e})", font_size=10)
            
            # Add FEM solution if available (right)
            self.plotter.subplot(0, 1)
            if has_fem_solution:
                # Create mesh with FEM deformation
                fem_grid = self.grid.copy()
                fem_grid.point_data["displacement"] = self.fem_solution_history[step].reshape((-1, 3))
                fem_grid["displacement_magnitude"] = np.linalg.norm(self.fem_solution_history[step].reshape((-1, 3)), axis=1)
                fem_warped = fem_grid.warp_by_vector("displacement", factor=1.0)
                
                self.plotter.add_mesh(fem_warped, scalars="displacement_magnitude", cmap="viridis", show_edges=True)
                self.plotter.add_title(f"FEM Reference Solution (Energy: {fem_energy:.4e})", font_size=10)
                
                # Compute error metrics
                error = u.cpu().numpy() - self.fem_solution_history[step]
                error_norm = np.linalg.norm(error) / np.linalg.norm(self.fem_solution_history[step])
                max_error = np.max(np.abs(error))
                
                # Calculate energy difference and ratio
                energy_diff = abs(nn_energy - fem_energy)
                energy_ratio = nn_energy / fem_energy if fem_energy > 0 else float('inf')
                
                # Add global title with error metrics and energy comparison
                self.plotter.add_text(
                    f"Time: {t:.3f}s, Step: {step}, Torque: {torque_magnitude:.2e}\n" 
                    f"Rel. Error: {error_norm:.4f}, Max Error: {max_error:.4e}\n"
                    f"Energy Diff: {energy_diff:.4e}, Energy Ratio: {energy_ratio:.4f}", 
                    position="lower_left", font_size=10, color='black'
                )
            else:
                # Display a message in the right subplot when FEM solution is not yet available
                self.plotter.add_text("FEM solution not yet available", position="center", font_size=14, color='white')
                self.plotter.add_title("FEM Reference Solution (Processing...)")
                
                # Add neural network energy information
                self.plotter.add_text(
                    f"Time: {t:.3f}s, Step: {step}, Torque: {torque_magnitude:.2e}\n" 
                    f"Neural Network Energy: {nn_energy:.4e}", 
                    position="lower_left", font_size=10, color='black'
                )
            
            # Write the frame to the GIF file
            self.plotter.write_frame()
            
            if step % 10 == 0:  # Log less frequently
                self.logger.info(f"Visualization frame added for step {step}, t={t:.3f}s")
                if has_fem_solution:
                    self.logger.info(f"  NN Energy: {nn_energy:.6e}, FEM Energy: {fem_energy:.6e}, Ratio: {energy_ratio:.4f}")
                
        except Exception as e:
            self.logger.error(f"Error in visualization for step {step}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())



    # Modify visualize_results to include error metrics compared to FEM
    def visualize_results(self, time_points, displacement_norms, energy_history, torque_history):
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
        
        # Plot torque over time
        axes[2].plot(time_points, torque_history)
        axes[2].set_ylabel("Torque Magnitude (N·m)")
        axes[2].set_title("Applied Torque")
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
                                max_iter=50,
                                history_size=20,
                                line_search_fn='strong_wolfe')
        
        # Weight for energy term
        alpha = 100
        
        initial_loss = None
        final_loss = None
        initial_components = None
        self.logger.info(f"Starting z optimization at t={current_time:.3f}s")
        
        # Track whether each iteration has had its first evaluation
        first_evals = [False] * num_iters
        
        # Optimize
        for i in range(num_iters):
            def closure():
                optimizer.zero_grad()
                # Pass current time to the objective function for torque calculation
                loss = self.objective_function(z, u_current, u_prev, dt, alpha, current_time)
                
                # Only log on the FIRST evaluation of each iteration
                nonlocal first_evals, initial_components
                if not first_evals[i]:
                    with torch.no_grad():
                        # First convert z to displacement vector before computing torque
                        l = (self.routine.linear_modes @ z.unsqueeze(1)).squeeze(1)
                        y = self.routine.model(z)
                        u = l + y
                        z_change = torch.norm(z - self.z_current).item()
                        
                        # Get components from the stored values
                        components = self.loss_components
                        
                        stage = "Initial" if i == 0 else "Iteration"
                        
                        # Store initial loss (only on first iteration's first eval)
                        nonlocal initial_loss
                        if i == 0:
                            initial_loss = loss.item()
                            initial_components = components
                            self.logger.info(f"{stage} components at t={current_time:.3f}s, iter {i}: ")
                            self.logger.info(f"  Temporal: {components['temporal']:.3e}")
                            self.logger.info(f"  Energy: {components['energy']:.3e}")
                            self.logger.info(f"  Total: {components['total']:.3e}")
                    
                    # Mark this iteration as having had its first evaluation
                    first_evals[i] = True
                    
                loss.backward()
                return loss
                
            loss = optimizer.step(closure)
            
            # Print progress every iteration after the first
            if i % 5 == 0:
                self.logger.info(f"  z opt iter {i}/{num_iters}, loss: {loss.item():.6e}")
        
        # Final logging after optimization
        with torch.no_grad():
            final_loss = self.objective_function(z, u_current, u_prev, dt, alpha, current_time).item()
            final_components = self.loss_components
            
            self.logger.info(f"Final components at t={current_time:.3f}s: ")
            self.logger.info(f"  Temporal: {final_components['temporal']:.3e} (initial: {initial_components['temporal']:.3e})")
            self.logger.info(f"  Energy: {final_components['energy']:.3e} (initial: {initial_components['energy']:.3e})")
            self.logger.info(f"  Total: {final_components['total']:.3e} (initial: {initial_components['total']:.3e})")

        # Report optimization results
        reduction = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss else 0
        self.logger.info(f"z optimization complete: reduction={reduction:.2f}%")
        
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
        
    def plot_energy_vs_torque(self, save_path=None):
        """
        Create a plot comparing neural network energy and FEM energy against applied torque.
        Uses the stored energy histories directly rather than recomputing them.
        
        Args:
            save_path: Optional path to save the figure (defaults to output_dir/energy_torque_comparison.png)
        """
        try:
            import matplotlib.pyplot as plt
            
            # Default save location if not specified
            if save_path is None:
                save_path = os.path.join(self.output_dir, "energy_torque_comparison.png")
            
            # Collect data for plotting
            time_points = []
            torque_values = []
            
            # Calculate time points and torque values
            if hasattr(self, 'fem_energy_history') and self.fem_energy_history:
                num_steps = len(self.fem_energy_history)
                self.logger.info(f"Found {num_steps} FEM energy entries")
                
                for i in range(num_steps):
                    t = (i+1) * self.dt
                    time_points.append(t)
                    
                    # Calculate applied torque at this time
                    torque = self.torque_magnitude
                    if t < self.torque_ramp_time:
                        torque *= (t / self.torque_ramp_time)
                    if t >= self.change_time:
                        torque *= 0.5
                    torque_values.append(torque)
            
            # Check if we have energy histories directly available
            if hasattr(self, 'nn_energy_history') and hasattr(self, 'fem_energy_history'):
                self.logger.info(f"Using stored energy histories directly")
                
                # Use only valid entries from both histories
                valid_steps = min(len(self.nn_energy_history), len(self.fem_energy_history))
                nn_energy = self.nn_energy_history[:valid_steps]
                fem_energy = self.fem_energy_history[:valid_steps]
                
                # Ensure time points match the energy data available
                time_points = time_points[:valid_steps]
                torque_values = torque_values[:valid_steps]
                
                self.logger.info(f"Plotting energy comparison with {valid_steps} data points")
                self.logger.info(f"NN energy range: [{min(nn_energy):.4e}, {max(nn_energy):.4e}]")
                self.logger.info(f"FEM energy range: [{min(fem_energy):.4e}, {max(fem_energy):.4e}]")
            else:
                # Fallback to the older approach if direct histories aren't available
                self.logger.warning("No direct energy histories found, reconstructing from stored data")
                nn_energy = []
                fem_energy = []
                
                # Original code for calculating energy values from stored data
                if hasattr(self, 'u_history') and self.u_history:
                    for i, u in enumerate(self.u_history):
                        if i >= len(self.fem_energy_history):
                            break
                            
                        # Get neural network energy if available
                        if hasattr(self, 'z_history') and i < len(self.z_history) and self.z_history[i] is not None:
                            z = torch.tensor(self.z_history[i], device=self.device, dtype=torch.float64)
                            nn_energy.append(self.compute_energy(z).item())
                        else:
                            # Try to compute from displacement
                            try:
                                z = self.find_best_latent_vector(u)
                                nn_energy.append(self.compute_energy(z).item())
                            except:
                                nn_energy.append(float('nan'))
                        
                        # Get FEM energy
                        self.logger.info(f"NN energy: {nn_energy[-1]}")
                        fem_energy.append(self.fem_energy_history[i])
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            # Plot 1: Energy vs Time with Torque overlay
            ax1.plot(time_points, nn_energy, 'b-', label='Neural Network Energy')
            ax1.plot(time_points, fem_energy, 'r--', label='FEM Energy')
            ax1.set_ylabel('Energy')
            ax1.set_title('Energy vs Time')
            ax1.grid(True)
            ax1.legend(loc='upper left')
            
            # Add torque as second y-axis
            ax1_twin = ax1.twinx()
            ax1_twin.plot(time_points, torque_values, 'g-.', label='Applied Torque')
            ax1_twin.set_ylabel('Torque (N·m)')
            ax1_twin.legend(loc='upper right')
            
            # Plot 2: Energy vs Torque
            ax2.plot(torque_values, nn_energy, 'bo-', label='Neural Network Energy')
            ax2.plot(torque_values, fem_energy, 'ro--', label='FEM Energy')
            ax2.set_xlabel('Applied Torque (N·m)')
            ax2.set_ylabel('Energy')
            ax2.set_title('Energy vs Applied Torque')
            ax2.grid(True)
            ax2.legend()
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
            self.logger.info(f"Energy vs Torque plot saved to {save_path}")
            
            # Also create a direct plotting function for the plotter
            def add_energy_torque_plot_to_plotter(plotter=None):
                """Adds the energy vs torque plot to a PyVista plotter"""
                if plotter is None and hasattr(self, 'plotter') and self.plotter is not None:
                    plotter = self.plotter
                
                if plotter is not None:
                    try:
                        # Create a PyVista chart with the energy vs torque data
                        import pyvista
                        
                        # Save the plot to a temporary file
                        temp_file = os.path.join(self.output_dir, "temp_energy_plot.png")
                        plt.figure(figsize=(8, 6))
                        plt.plot(torque_values, nn_energy, 'bo-', label='Neural Network Energy')
                        plt.plot(torque_values, fem_energy, 'ro--', label='FEM Energy')
                        plt.xlabel('Applied Torque (N·m)')
                        plt.ylabel('Energy')
                        plt.title('Energy vs Applied Torque')
                        plt.grid(True)
                        plt.legend()
                        plt.savefig(temp_file, dpi=150)
                        plt.close()
                        
                        # Add the image to the plotter
                        chart_actor = plotter.add_background_image(temp_file)
                        
                        # Return the actor for potential later removal
                        return chart_actor
                    except Exception as e:
                        self.logger.error(f"Failed to add energy-torque plot to plotter: {e}")
                        return None
                return None
            
            # Store the function as an instance method
            self.add_energy_torque_plot_to_plotter = add_energy_torque_plot_to_plotter
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create energy vs torque plot: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        return False
        
    def run_fem_step(self, t):
        """Run one step of the FEM simulation matching exactly the twisting_beam.py implementation"""
        self.logger.info(f"Running FEM simulation for time t={t:.3f}s")
        
        # We'll use the state variables that were initialized once during __init__
        # This is critical for stability - exactly like in twisting_beam.py
        
        # Newmark-beta parameters - same as twisting_beam.py
        beta = 0.25
        gamma = 0.5
        
        # 1. Newmark-beta update formulas - EXACTLY as in twisting_beam.py
        # Create expressions for interpolation
        u_ddot_expr = fem.Expression(self.u_ddot, self.V.element.interpolation_points())
        self.u_ddot.interpolate(u_ddot_expr)
        
        u_expr = fem.Expression(self.u, self.V.element.interpolation_points())
        self.u.interpolate(u_expr)
        
        u_dot_expr = fem.Expression(self.u_dot, self.V.element.interpolation_points())
        self.u_dot.interpolate(u_dot_expr)
        
        # Update acceleration using Newmark formula
        self.u_ddot.x.array[:] = (1 / (beta * self.dt**2)) * (self.u.x.array - self.u_prev.x.array) - \
                                (1 / (beta * self.dt)) * self.u_dot_prev.x.array - \
                                (1/(2*beta)-1) * self.u_ddot.x.array
        
        # Update velocity using Newmark formula with corrected terms
        self.u_dot.x.array[:] = self.u_dot_prev.x.array + self.dt * ((1 - gamma) * self.u_ddot.x.array + gamma * self.u_ddot.x.array)
        
        # 2. Define the nonlinear problem - same approach as twisting_beam.py
        v = ufl.TestFunction(self.V)
        
        # Define fixed boundary condition
        def fixed_boundary(x):
            return np.isclose(x[0], self.x_min)
            
        boundary_dofs = fem.locate_dofs_geometrical(self.V, fixed_boundary)
        u_D = np.zeros(3, dtype=np.float64)
        bc = fem.dirichletbc(u_D, boundary_dofs, self.V)
        
        # Material parameters - match twisting_beam.py exactly
        E = 1.0e5
        nu = 0.45
        mu = E / (2 * (1 + nu))
        lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
        
        # Create the weak form
        F = self._create_weak_form(self.u, self.u_prev, self.u_dot_prev, self.u_ddot, v, t, 
                                self.dt, self.routine.domain, mu, lmbda, 
                                self.torque_magnitude, self.beam_length, beta, gamma)
        
        J = ufl.derivative(F, self.u)
        problem = fem.petsc.NonlinearProblem(F, self.u, [bc], J=J)
        
        # 3. Create and configure Newton solver - same as twisting_beam.py
        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        solver.convergence_criterion = "residual"
        solver.rtol = 1e-8
        solver.max_it = 50
        
        # 4. Solve the nonlinear problem
        try:
            n_its, converged = solver.solve(self.u)
            
            # Same failure condition as twisting_beam.py - fail immediately if not converged
            if not converged:
                self.logger.error(f"Newton solver did not converge at t={t:.4f}s")
                return False
                
            self.logger.info(f"FEM step at t={t:.4f}s: Nonlinear solve complete in {n_its} iterations, converged: True")
            
            # Calculate maximum displacement
            u_values = self.u.x.array.reshape(-1, 3)
            max_displacement = np.max(np.linalg.norm(u_values, axis=1))
            self.logger.info(f"  Max displacement: {max_displacement:.4f}")
            
            # 5. CRITICAL: Update previous solution for next step
            #    This is exactly like in twisting_beam.py
            self.u_prev.x.array[:] = self.u.x.array
            self.u_dot_prev.x.array[:] = self.u_dot.x.array
            
            # Store solution for later comparison
            self.fem_solution_history.append(self.u.x.array.copy())

            fem_energy = self.compute_neohookean_energy(self.u)
            if not hasattr(self, 'fem_energy_history'):
                self.fem_energy_history = []
            self.fem_energy_history.append(fem_energy)
            self.logger.info(f"  FEM Neo-Hookean energy: {fem_energy:.6e}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error in solver at t={t:.4f}s: {str(e)}")
            return False
            
    def _create_weak_form(self, u, u_prev, u_dot_prev, u_ddot, v, t, dt, domain, mu, lmbda, load_magnitude, beam_length, beta, gamma):
        """Create weak form - exactly as in twisting_beam.py"""
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
        
        # External work from torque - adjusted to use the correct center line
        x = ufl.SpatialCoordinate(domain)
        
        # Calculate distance from actual center axis - exactly as in twisting_beam.py
        r_loc = ufl.sqrt((x[1]-self.y_center)**2 + (x[2]-self.z_center)**2)
        # Direction of torque (tangential)
        theta = ufl.atan2(x[2]-self.z_center, x[1]-self.y_center)
        torque_dir = ufl.as_vector([-0.0, -ufl.sin(theta), ufl.cos(theta)])
        
        # Magnitude that varies with time and position - exactly as in twisting_beam.py
        magnitude = load_magnitude * ((x[0]-self.x_min)/beam_length)
        
        # Ramp up torque for the first 2 seconds
        if t < 2.0:
            magnitude *= (t / 2.0)
        
        if t >= self.change_time:
            magnitude *= 0.5  # Reduce after change_time
        
        external_work = ufl.dot(magnitude * r_loc * torque_dir, v) * ufl.dx
        
        # Newmark-beta for dynamics
        inertia = ufl.dot(self.rho * u_ddot, v) * ufl.dx
        
        return ufl.derivative(internal_work, u, v) - external_work + inertia
    

    def run_simulation(self):
        """Run parallel FEM and neural network simulations for direct comparison"""
        self.logger.info("Starting dynamic simulation with torque application...")
        
        try:
            # Initialize arrays for storing results
            z_history = []
            u_nn_history = []  # Neural network predictions
            u_fem_history = []  # FEM solutions
            displacement_norms_nn = []
            displacement_norms_fem = []
            
            torque_history = []
            time_points = []
            error_history = []  # Track error between NN and FEM


            self.fem_history = []
            self.nn_energy_history = [] # Store neural network energy history
            
            
            # Store initial state
            self.fem_solution_history = [self.u.x.array.copy()]
            
            # Initialize neural network state
            self.z_current = torch.zeros(self.routine.latent_dim, device=self.device, dtype=torch.float64)
            self.z_prev = torch.zeros(self.routine.latent_dim, device=self.device, dtype=torch.float64)
            
            # First compute initial steps using pure FEniCS
            num_initial_steps = 2 # Use 5 initial FEniCS steps
            self.logger.info(f"Computing {num_initial_steps} initial steps using FEniCS with NeoHookean material...")
            
            # Clear solution history before computing initial steps
            self.fem_solution_history = []
            
            # Run the initial FEM steps
            t = 0.0
            for n in range(num_initial_steps):
                t += self.dt
                success = self.run_fem_step(t)
                if not success:
                    self.logger.error(f"Initial FEM step failed at t={t:.4f}s, terminating simulation")
                    return {}
                
                time_points.append(t)
                torque_history.append(self.torque_magnitude * min(t/self.torque_ramp_time, 1.0))
                
                self.logger.info(f"Initial FEM step {n+1}/{num_initial_steps}, t={t:.4f}s completed")
            
            # Convert latest FEM solutions to torch tensors for the neural network
            # CRITICAL FIX: Use different variable names for tensors vs. FEniCS Functions
            u_prev_torch = torch.tensor(self.fem_solution_history[-2], device=self.device, dtype=torch.float64)
            u_current_torch = torch.tensor(self.fem_solution_history[-1], device=self.device, dtype=torch.float64)
            
            # Find initial latent vector through projection
            self.z_prev = self.find_best_latent_vector(u_prev_torch)
            self.z_current = self.find_best_latent_vector(u_current_torch)
            
            # Compute initial nn energy
            energy = self.compute_energy(self.z_current).item()
            self.nn_energy_history.append(energy)
            
            # Store latent vectors
            z_history.extend([None] * (num_initial_steps-1))  # Placeholder for earlier steps
            z_history.append(self.z_current.cpu().numpy())
            
            # Store optimal displacements for visualization
            u_nn = u_current_torch.clone()  # Define u_nn using current torch tensor
            u_nn_history.append(u_nn.clone())
            displacement_norms_nn.append(torch.norm(u_nn).item())
            
            # Compute FEM displacement norms for consistent comparison
            displacement_norms_fem = [np.linalg.norm(sol) for sol in self.fem_solution_history]
            
            # Current torque for logging
            current_torque = self.torque_magnitude * min(t/self.torque_ramp_time, 1.0)
            if t >= self.change_time:
                current_torque *= 0.5
            torque_history.append(current_torque)
            
            
            # Main simulation loop - run both FEM and neural network for comparison
            for n in range(1, self.num_steps):
                t += self.dt
                self.logger.info(f"\n--- Step {n+1}/{self.num_steps}, Time: {t:.4f}s ---")
                
                # 1. Run FEM step (ground truth)
                fem_success = self.run_fem_step(t)
                if not fem_success:
                    self.logger.warning(f"FEM solver failed at t={t:.4f}s but continuing with neural model")
                
                # Get FEM solution
                u_fem_current = torch.tensor(self.u.x.array, device=self.device, dtype=torch.float64) if fem_success else None
                
                # 2. Run neural network prediction (independent from FEM result)
                # Use the previous NN state to predict the next state
                prev_z = self.z_current.clone()
                prev_u_nn = u_nn.clone()
                
                ## Find optimal z for next timestep
                self.logger.info(f"Computing neural network prediction for t={t:.4f}s")
                z_predicted, u_nn_predicted = self.find_optimal_z(
                    u_nn, u_prev_torch, self.dt, t  # Use u_prev_torch here, not self.u_prev
                )
                print(f"z_predicted: {z_predicted}")
                            
                # Update neural network state for next prediction
                u_prev_torch = u_nn.clone()
                u_nn = u_nn_predicted.clone()
                self.z_prev = prev_z
                self.z_current = z_predicted
                
                # 3. Compare and store results
                if fem_success and u_fem_current is not None:
                    # Calculate error between NN and FEM
                    error = torch.norm(u_nn - u_fem_current) / torch.norm(u_fem_current)
                    self.logger.info(f"Relative error between NN and FEM: {error.item():.6f}")
                    
                    # Store FEM results
                    u_fem_history.append(u_fem_current.clone())
                    displacement_norms_fem.append(torch.norm(u_fem_current).item())
                else:
                    error = torch.tensor(float('nan'))
                    u_fem_history.append(None)
                    displacement_norms_fem.append(float('nan'))
                
                # Store time and results for this step
                time_points.append(t)
                z_history.append(z_predicted.cpu().numpy())
                u_nn_history.append(u_nn.clone())
                displacement_norms_nn.append(torch.norm(u_nn).item())
                error_history.append(error.item())
                
                # Compute energy and torque
                energy = self.compute_energy(z_predicted).item()
                self.nn_energy_history.append(energy)
                
                current_torque = self.torque_magnitude
                if t < self.torque_ramp_time:
                    current_torque *= (t / self.torque_ramp_time)
                if t >= self.change_time:
                    current_torque *= 0.5
                torque_history.append(current_torque)
                
                # Log progress every few steps
                if n % 5 == 0:
                    self.logger.info(f"Step {n+1}/{self.num_steps}, Time: {t:.2f}s, Energy: {energy:.4e}")
                    self.logger.info(f"NN max displacement: {torch.max(torch.norm(u_nn.reshape(-1, 3), dim=1)).item():.4e}")
                    if fem_success and u_fem_current is not None:
                        self.logger.info(f"FEM max displacement: {torch.max(torch.norm(u_fem_current.reshape(-1, 3), dim=1)).item():.4e}")
                    self.logger.info(f"Rel. Error: {error.item():.6f}")
                
                # Visualize steps
                if n % 1 == 0:
                    self.visualize_step(n, t, z_predicted, u_nn, current_torque)
            
            # Create error plot
            plt.figure(figsize=(10, 6))
            plt.plot(time_points, error_history, 'o-')
            plt.axhline(y=0.05, color='r', linestyle='--', label='5% error threshold')
            plt.xlabel('Time (s)')
            plt.ylabel('Relative Error')
            plt.title('Neural Network vs FEM Relative Error')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, "nn_vs_fem_error.png"))
            plt.close()
            
            # Visualize final comparison results
            self.visualize_comparison(time_points, displacement_norms_nn, displacement_norms_fem, 
                                    self.fem_history, torque_history, error_history)
            
            return {
                'time': time_points,
                'z': z_history,
                'displacement_norm_nn': displacement_norms_nn,
                'displacement_norm_fem': displacement_norms_fem,
                'nn_energy': self.nn_energy_history,   
                'fem_energy': self.fem_energy_history, 
                'error': error_history
            }
        except Exception as e:
            self.logger.error(f"Simulation error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}
        
        finally:
            # Always create energy vs torque plot, even if simulation failed
            self.logger.info("Creating energy vs torque plot...")
            self.plot_energy_vs_torque()
            
            # If we have a plotter, add the energy-torque plot to the final frame
            if hasattr(self, 'plotter') and self.plotter is not None:
                try:
                    for _ in range(30):  # Hold the final frame for a few seconds
                        self.plotter.update()
                        self.plotter.clear()
                        self.plotter.subplot(0, 0)
                        self.plotter.add_text("Simulation Complete", position="upper_edge", font_size=24, color='white')
                        
                        # Add energy vs torque plot to the right subplot
                        self.plotter.subplot(0, 1)
                        self.add_energy_torque_plot_to_plotter()
                        
                        # Save the final frame
                        self.plotter.write_frame()
                    
                except Exception as e:
                    self.logger.error(f"Error adding final visualization: {e}")
    
    def compute_neohookean_energy(self, u_function):
        """
        Compute the Neo-Hookean strain energy for a given displacement field
        
        Args:
            u_function: FEM displacement function
            
        Returns:
            Total Neo-Hookean strain energy
        """
        # Get domain and material parameters
        domain = self.routine.domain
        
        # Material parameters - match twisting_beam.py exactly
        E = 10000
        nu = 0.49
        mu = E / (2 * (1 + nu))
        lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
        
        # Define strain energy density for Neo-Hookean material
        d = 3  # 3D problem
        I = ufl.Identity(d)
        F = I + ufl.grad(u_function)
        C = F.T * F
        Ic = ufl.tr(C)
        J = ufl.det(F)
        
        # Neo-Hookean strain energy density formula
        psi = (mu/2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda/2) * (ufl.ln(J))**2
        
        # Integrate over domain to get total energy
        energy_form = psi * ufl.dx
        
        # Compute the energy using dolfinx's form assembly
        energy = fem.assemble_scalar(fem.form(energy_form))
        
        return energy

    def visualize_comparison(self, time_points, disp_nn, disp_fem, energy, torque, error):
        """Create comparison plots between neural network and FEM results"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        # Plot displacement comparison
        axes[0].plot(time_points, disp_nn, 'b-', label='Neural Network')
        axes[0].plot(time_points, disp_fem, 'r--', label='FEM Reference')
        axes[0].set_ylabel("Displacement Norm")
        axes[0].set_title("Neural Network vs FEM Displacement")
        axes[0].grid(True)
        axes[0].legend()
        
        # Plot error
        axes[1].plot(time_points, error, 'g-o')
        axes[1].set_ylabel("Relative Error")
        axes[1].set_title("Neural Network vs FEM Relative Error")
        axes[1].grid(True)
        axes[1].axhline(y=0.05, color='r', linestyle='--', label='5% threshold')
        axes[1].legend()
        
        # Plot energy and torque
        ax2 = axes[2].twinx()
        axes[2].plot(time_points, energy, 'b-', label='Energy')
        axes[2].set_ylabel("Elastic Energy")
        ax2.plot(time_points, torque, 'r--', label='Torque')
        ax2.set_ylabel("Applied Torque")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_title("System Energy and Applied Torque")
        axes[2].grid(True)
        lines1, labels1 = axes[2].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[2].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "nn_vs_fem_comparison.png"))
        plt.close()

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
    parser = argparse.ArgumentParser(description='Dynamic validation for Neural Plates')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='config file path')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt', help='checkpoint path')
    parser.add_argument('--steps', type=int, default=100, help='number of simulation steps (default: 100 as in twisting_beam.py)')
    parser.add_argument('--time', type=float, default=2.4, help='total simulation time (matches twisting_beam.py)')
    parser.add_argument('--torque', type=float, default=1.0e4, help='torque magnitude (matches twisting_beam.py)')
    parser.add_argument('--damping', type=float, default=0.01, help='damping coefficient')
    parser.add_argument('--output', type=str, default='validation_results', help='output directory')
    args = parser.parse_args()
    
    # Setup logger
    setup_logger("dynamic_validator", log_dir=args.output)
    logger = logging.getLogger("dynamic_validator")
    
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

    # Run visualization tests
    try:
        # [existing visualization tests code]
        pass
    except Exception as e:
        logger.error(f"Visualization test failed: {e}")
    
    # Create validator and run simulation with torque
    validator = DynamicValidator(
        routine=routine,
        num_steps=args.steps,
        total_time=args.time,
        torque_magnitude=args.torque,
        torque_axis=[0, 0, 1],  # Default Z-axis torque for twisting
        torque_ramp_time=2.0,   # Ramp up torque over 2.0s (match twisting_beam.py)
        change_time=2.5,        # Torque changes at 2.5s (match twisting_beam.py)
        damping=args.damping,
        output_dir=args.output
    )
    
    # Run simulation
    results = validator.run_simulation()
if __name__ == "__main__":
    main()

