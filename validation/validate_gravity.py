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
    Can apply gravity to test beam bending under its own weight.
    """
    def __init__(self, routine, num_steps=100, total_time=2.4, 
                 gravity_magnitude=9.81, gravity_direction=[0, 0, -1],
                 gravity_ramp_time=0.5, damping=0.01, output_dir="validation_results"):
        """
        Initialize dynamic validator with gravity parameters.
        
        Args:
            routine: Trained Routine object with neural model
            num_steps: Number of simulation steps
            total_time: Total simulation time
            gravity_magnitude: Gravity acceleration (m/s²), default 9.81
            gravity_direction: Direction of gravity [x, y, z], default downward
            gravity_ramp_time: Time to ramp up gravity to full magnitude
            damping: Damping coefficient (0 = no damping, 1 = critically damped)
            output_dir: Directory to save results
        """
        self.routine = routine
        self.total_time = total_time
        self.num_steps = num_steps
        self.dt = total_time / num_steps  # Calculate dt from num_steps
        
        # Gravity parameters
        self.gravity_magnitude = gravity_magnitude
        self.gravity_direction = np.array(gravity_direction) / np.linalg.norm(gravity_direction)  # Normalize
        self.gravity_ramp_time = gravity_ramp_time
        
        # Material parameters
        self.rho = 1000.0  # Density (kg/m³)
        
        # Store original parameters too
        self.damping = damping
        self.output_dir = output_dir
        self.device = routine.device
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger("gravity_validation")
        
        # Get mass matrix (M) from routine
        self.M = routine.M
        
        # Initialize PyTorch tensor state variables (rename to avoid confusion)
        self.z_current = torch.zeros(routine.latent_dim, device=self.device, dtype=torch.float64)
        self.z_prev = torch.zeros(routine.latent_dim, device=self.device, dtype=torch.float64)
        self.u_current_torch = torch.zeros(routine.V.dofmap.index_map.size_global * routine.domain.geometry.dim, 
                                    device=self.device, dtype=torch.float64)
        self.u_prev_torch = torch.zeros_like(self.u_current_torch)
        
        # Initialize FEniCS function state variables
        self.V = routine.V
        self.u = fem.Function(self.V, name="Displacement")
        self.u_prev = fem.Function(self.V, name="Displacement_prev")
        self.u_dot = fem.Function(self.V, name="Velocity")
        self.u_dot_prev = fem.Function(self.V, name="Velocity_prev")
        self.u_ddot = fem.Function(self.V, name="Acceleration")
            
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

            
    def compute_gravity_term(self, u, t, use_reference=True):
        """Compute the gravity potential energy using mass matrix for consistency with FEM."""
        # Skip if no gravity
        if self.gravity_magnitude == 0:
            return torch.tensor(0.0, device=self.device, dtype=torch.float64)
            
        # Scale magnitude based on ramp time
        current_magnitude = self.gravity_magnitude
        if t < self.gravity_ramp_time:
            current_magnitude *= (t / self.gravity_ramp_time)
        
        # Create gravity direction tensor (column vector)
        gravity_dir = torch.tensor(self.gravity_direction, device=self.device, dtype=torch.float64)
        
        # Reshape u to get nodal displacements
        u_reshaped = u.reshape(-1, 3)
        
        # Get original coordinates of the mesh if needed
        if not use_reference:
            coords = torch.tensor(self.routine.domain.geometry.x, device=self.device, dtype=torch.float64)
        
        # Create a load vector representing M·g
        # For each DOF, determine its direction component along gravity
        n_nodes = u_reshaped.shape[0]
        gravity_load = torch.zeros_like(u)
        
        for i in range(n_nodes):
            # Set entries for x, y, z components at this node
            base_idx = i * 3
            gravity_load[base_idx:base_idx+3] = gravity_dir * current_magnitude
        
        # Apply the mass matrix to the gravity load vector
        # M·g gives the consistent nodal forces due to gravity
        gravity_load_petsc = PETSc.Vec().createWithArray(gravity_load.cpu().numpy())
        result_petsc = gravity_load_petsc.duplicate()
        self.M.mult(gravity_load_petsc, result_petsc)  # This gives M·g
        weighted_load = torch.tensor(result_petsc.getArray(), device=self.device, dtype=torch.float64)
        
        if use_reference:
            # For reference calculation, only use displacement part
            gravity_potential = -torch.sum(weighted_load * u)
        else:
            # For absolute calculation, include the initial positions
            absolute_positions = torch.zeros_like(u)
            for i in range(n_nodes):
                base_idx = i * 3
                absolute_positions[base_idx:base_idx+3] = coords[i] + u_reshaped[i]
            gravity_potential = -torch.sum(weighted_load * absolute_positions)
        
        return gravity_potential
                
    def objective_function(self, z, u_current, u_prev, dt, alpha=1.0, t=0.0):
        """
        Objective function to minimize with gravity potential energy included.
        Using Newmark-beta consistent time integration to match FEM solver.
        """
        # Newmark-beta parameters - must match those in run_fem_step
        beta = 0.25
        gamma = 0.5
        
        # Get displacement from latent vector
        l = (self.routine.linear_modes @ z.unsqueeze(1)).squeeze(1)
        y = self.routine.model(z)
        u_next = l + y
        
        # Calculate acceleration from Newmark formula:
        # u_ddot = (1/(β*dt²))*(u_next - u_current - dt*u_dot) - (1/(2β)-1)*u_ddot_current
        # Where:
        # u_dot = (u_current - u_prev)/dt (approximation for current velocity)
        # u_ddot_current = 0 (simplification, or we could estimate from previous steps)
        
        # Estimate current velocity
        u_dot_current = (u_current - u_prev) / dt  # Approximate current velocity

        # Replicate FEM solver's acceleration computation exactly:
        u_ddot_next = (1.0 / (beta * dt * dt)) * (u_next - u_current) - \
                    (1.0 / (beta * dt)) * u_dot_current - \
                    ((1/(2*beta)) - 1) * torch.zeros_like(u_next)  # Last term can be zero initially
        
        # Create M*u_ddot - this represents the inertial forces
        M_petsc_vec = PETSc.Vec().createWithArray(u_ddot_next.cpu().detach().numpy())
        result_petsc_vec = M_petsc_vec.duplicate()
        self.M.mult(M_petsc_vec, result_petsc_vec)
        M_u_ddot = torch.tensor(result_petsc_vec.getArray(), device=self.device, dtype=torch.float64)
        
        # Inertial term: ρ*(u_ddot)*v (same as in FEM weak form)
        temporal = torch.sum(u_ddot_next * M_u_ddot)
        
        # Elastic strain energy term
        elastic_energy = self.compute_energy(z)
        
        # Gravity potential energy
        gravity_energy = self.compute_gravity_term(u_next, t)
        
        # Damping term (Rayleigh damping, like in FEM):
        # C = αM + βK, but we'll just use mass proportional damping (α only)
        # using the current velocity estimate
        damping_coef = self.damping  # Use the same damping as in FEM
        
        # Apply damping: C*u_dot ≈ damping_coef * M * u_dot
        M_velocity_vec = PETSc.Vec().createWithArray(u_dot_current.cpu().detach().numpy())
        result_v_petsc_vec = M_velocity_vec.duplicate()
        self.M.mult(M_velocity_vec, result_v_petsc_vec)
        M_u_dot = torch.tensor(result_v_petsc_vec.getArray(), device=self.device, dtype=torch.float64)
        damping_term = damping_coef * torch.sum(u_dot_current * M_u_dot)
        
        # Total energy is the sum of components
        total_energy = elastic_energy + gravity_energy
        
        # Total objective combines dynamic terms and energy
        objective = temporal + damping_term +  1e9 * total_energy
        
        # Store component values for logging
        self.loss_components = {
            'temporal': temporal.item(),
            'damping': damping_term.item(),
            'elastic_energy': elastic_energy.item(),
            'gravity_energy': gravity_energy.item(),
            'total_energy': total_energy.item(),
            'total': objective.item()
        }
        
        return objective

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
        
        # Material parameters
        E = 10000
        nu = 0.35
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
        psi = 0.5 * mu * (Ic - 3.0 - 2.0 * ufl.ln(J)) + 0.25 * lmbda * (J**2 - 1.0 - 2.0 * ufl.ln(J))
        
        # Integrate over domain to get total energy
        energy_form = psi * ufl.dx(metadata={"quadrature_degree": 2})
        
        # Compute the energy using dolfinx's form assembly
        energy = fem.assemble_scalar(fem.form(energy_form))
        
        return energy
    
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
            topology, cell_types, x = plot.vtk_mesh(self.routine.V)
            self.grid = pyvista.UnstructuredGrid(topology, cell_types, x)
            self.x = x  # Store x coordinates for later use
            
            # Create a plotter with 2x1 layout for vertical stacking (like validate_stretch.py)
            self.plotter = pyvista.Plotter(shape=(1, 2), window_size=[1920, 1080])
            gif_path = os.path.join(self.output_dir, "beam_deformation_gravity.gif")
            self.plotter.open_gif(gif_path)
            
            # Calculate the center of the beam
            center = [(self.x_min + self.x_max) / 2, self.y_center, self.z_center]

            # Set camera for both subplots
            for i in range(2):
                self.plotter.subplot(0, i)  # Vertical stacking (i, 0)
                
                # Position camera to see the entire beam from the side
                self.plotter.camera_position = [
                    (center[0], center[1] - self.beam_length*2, center[2]),  # Camera position
                    center,  # Focal point at center of beam
                    (0, 0, 1)  # Up direction
                ]
                
                # Use a better zoom level to see the whole beam
                self.plotter.zoom_camera(0.7)
            
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
            self.plotter.subplot(0, 1)
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
        z_extrapolated = self.z_current #+ (self.z_current - self.z_prev)
        
        # Start optimization from extrapolated z to maintain momentum
        z = z_extrapolated.clone().detach().requires_grad_(True)
        
        # Use L-BFGS for optimization
        optimizer = torch.optim.LBFGS([z], 
                                    lr=1, 
                                    max_iter=20,
                                    history_size=50,
                                    line_search_fn="strong_wolfe")
        
        # Weight for energy term
        alpha = 1e8  # Lower energy weight to match validate_twist.py
        
        initial_loss = None
        final_loss = None
        initial_components = None
        self.logger.info(f"Starting z optimization at t={current_time:.3f}s, at step {num_iters} of {self.num_steps}")
        
        # Track whether each iteration has had its first evaluation
        first_evals = [False] * num_iters
        
        # Optimize
        for i in range(num_iters):
            def closure():
                optimizer.zero_grad()
                # Pass current time to the objective function for gravity calculation
                loss = self.objective_function(z, u_current, u_prev, dt, alpha, current_time)
                
                # Only log on the FIRST evaluation of each iteration
                nonlocal first_evals, initial_components
                if not first_evals[i]:
                    with torch.no_grad():
                        # First convert z to displacement vector before computing gravity
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
                            self.logger.info(f"  Elastic Energy: {components['elastic_energy']:.3e}")
                            self.logger.info(f"  Gravity Energy: {components['gravity_energy']:.3e}")
                            self.logger.info(f"  Total Energy: {components['total_energy']:.3e}")
                            self.logger.info(f"  Total: {components['total']:.3e}")
                    
                    # Mark this iteration as having had its first evaluation
                    first_evals[i] = True
                    
                loss.backward()
                return loss
                
            loss = optimizer.step(closure)
            
            # Print progress every iteration
            if i % 3 == 0 or i == num_iters-1:  # Log every 3rd iteration and the last one
                self.logger.info(f"  z opt iter {i}/{num_iters}, loss: {loss.item():.6e}")
        # Final logging after optimization
        with torch.no_grad():
            final_loss = self.objective_function(z, u_current, u_prev, dt, alpha, current_time).item()
            final_components = self.loss_components
            
            self.logger.info(f"Final components at t={current_time:.3f}s: ")
            self.logger.info(f"  Temporal: {final_components['temporal']:.3e} (initial: {initial_components['temporal']:.3e})")
            self.logger.info(f"  Elastic Energy: {final_components['elastic_energy']:.3e} (initial: {initial_components['elastic_energy']:.3e})")
            self.logger.info(f"  Gravity Energy: {final_components['gravity_energy']:.3e} (initial: {initial_components['gravity_energy']:.3e})")
            self.logger.info(f"  Total Energy: {final_components['total_energy']:.3e} (initial: {initial_components['total_energy']:.3e})")
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
        
    def run_fem_step(self, t):
        """Run one step of the FEM simulation with gravity force"""
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
        E = 10000
        nu = 0.35
        mu = E / (2 * (1 + nu))
        lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
        
        # Create the weak form
        F = self._create_weak_form(self.u, self.u_prev, self.u_dot_prev, self.u_ddot, v, t, 
                                self.dt, self.routine.domain, mu, lmbda, 
                                self.gravity_magnitude, self.gravity_direction, beta, gamma)
        
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
        
    def _create_weak_form(self, u, u_prev, u_dot_prev, u_ddot, v, t, dt, domain, mu, lmbda, load_magnitude, load_direction, beta, gamma):
        """Create the weak form for the dynamic simulation with gravity"""
        # Kinematics - use the same approach as validate_twist.py
        d = len(u)
        I = ufl.Identity(d)
        F = I + ufl.grad(u)  
        C = F.T * F
        Ic = ufl.tr(C)
        J = ufl.det(F)
        
        # Neo-Hookean strain energy density (incompressible)
        psi = (mu/2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda/2) * (ufl.ln(J))**2
        
        # Define internal work directly from strain energy
        internal_work = psi * ufl.dx
        
        # External work from gravity
        x = ufl.SpatialCoordinate(domain)
        
        # Scale magnitude based on ramp time
        current_magnitude = load_magnitude
        if t < self.gravity_ramp_time:
            current_magnitude *= (t / self.gravity_ramp_time)
        
        # Create load direction vector - use ufl.as_vector for UFL compatibility
        load_dir = ufl.as_vector(load_direction)
        
        # External work due to gravity - directly use the force on the test function
        external_work = ufl.dot(current_magnitude * load_dir, v) * ufl.dx
        
        # Newmark-beta for dynamics
        inertia = ufl.dot(self.rho * u_ddot, v) * ufl.dx
        
        # Return the complete weak form - use ufl.derivative to handle internal_work differentiation
        return ufl.derivative(internal_work, u, v) - external_work + inertia
        

    def plot_energy_vs_gravity(self, save_path=None):
        """
        Create plots comparing neural network energy and FEM energy against time and gravity.
        """
        try:
            import matplotlib.pyplot as plt
            
            # Default save location if not specified
            if save_path is None:
                save_path = os.path.join(self.output_dir, "energy_gravity_comparison.png")
            
            # Collect data for plotting
            time_points = []
            gravity_values = []
            
            # Calculate time points and gravity values
            if hasattr(self, 'fem_energy_history') and self.fem_energy_history:
                num_steps = len(self.fem_energy_history)
                self.logger.info(f"Found {num_steps} FEM energy entries")
                
                for i in range(num_steps):
                    t = (i+1) * self.dt
                    time_points.append(t)
                    
                    # Calculate applied gravity at this time
                    gravity = self.gravity_magnitude
                    if t < self.gravity_ramp_time:
                        gravity *= (t / self.gravity_ramp_time)
                    gravity_values.append(gravity)
            
            # Check if we have energy histories directly available
            self.logger.info(f"Using stored energy histories directly")
            
            # Use only valid entries from both histories
            valid_steps = min(len(self.nn_energy_history), len(self.fem_energy_history))
            nn_energy = self.nn_energy_history[:valid_steps]
            fem_energy = self.fem_energy_history[:valid_steps]
            
            # Ensure time points match the energy data available
            time_points = time_points[:valid_steps]
            gravity_values = gravity_values[:valid_steps]
            
            self.logger.info(f"Plotting energy comparison with {valid_steps} data points")
            self.logger.info(f"NN energy range: [{min(nn_energy):.4e}, {max(nn_energy):.4e}]")
            self.logger.info(f"FEM energy range: [{min(fem_energy):.4e}, {max(fem_energy):.4e}]")
        
            
            # Create figure with three plots
            fig = plt.figure(figsize=(18, 10))
            
            # 1. Plot Energy vs Time (MAIN PLOT)
            ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)  # Make this span both columns
            ax1.plot(time_points, nn_energy, 'b-', linewidth=2, label='Neural Network Energy')
            ax1.plot(time_points, fem_energy, 'r--', linewidth=2, label='FEM Energy')
            ax1.set_ylabel('Energy', fontsize=12)
            ax1.set_xlabel('Time (s)', fontsize=12)
            ax1.set_title('Energy vs Time', fontsize=14)
            ax1.grid(True)
            ax1.legend(loc='upper left', fontsize=12)
            
            # Add gravity as second y-axis on Energy vs Time plot
            ax1_twin = ax1.twinx()
            ax1_twin.plot(time_points, gravity_values, 'g-.', linewidth=1.5, label='Applied Gravity')
            ax1_twin.set_ylabel('Gravity (m/s²)', fontsize=12)
            ax1_twin.legend(loc='upper right', fontsize=12)
            
            # 2. Plot Energy vs Gravity (smaller subplot)
            ax2 = plt.subplot2grid((2, 2), (1, 0))
            ax2.plot(gravity_values, nn_energy, 'bo-', label='Neural Network Energy')
            ax2.plot(gravity_values, fem_energy, 'ro--', label='FEM Energy')
            ax2.set_xlabel('Applied Gravity (m/s²)', fontsize=12)
            ax2.set_ylabel('Energy', fontsize=12)
            ax2.set_title('Energy vs Applied Gravity', fontsize=14)
            ax2.grid(True)
            ax2.legend(fontsize=10)
            
            # 3. Add Energy Difference plot (smaller subplot)
            ax3 = plt.subplot2grid((2, 2), (1, 1))
            energy_diff = np.array(nn_energy) - np.array(fem_energy)
            energy_ratio = np.array(nn_energy) / np.array(fem_energy)
            ax3.plot(time_points, energy_diff, 'k-', label='Energy Difference (NN-FEM)')
            ax3.set_xlabel('Time (s)', fontsize=12)
            ax3.set_ylabel('Energy Difference', fontsize=12)
            ax3.set_title('Energy Difference Between Models', fontsize=14)
            ax3.grid(True)
            
            # Add energy ratio as second y-axis
            ax3_twin = ax3.twinx()
            ax3_twin.plot(time_points, energy_ratio, 'm--', label='Energy Ratio (NN/FEM)')
            ax3_twin.set_ylabel('Energy Ratio', fontsize=12)
            ax3_twin.set_ylim([0, 2])  # Reasonable range for ratio
            
            # Add combined legend
            lines1, labels1 = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3_twin.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
            # Create separate Energy vs Time plot for the final visualization
            energy_time_path = os.path.join(self.output_dir, "energy_vs_time.png")
            plt.figure(figsize=(10, 6))
            plt.plot(time_points, nn_energy, 'b-', linewidth=3, label='Neural Network Energy')
            plt.plot(time_points, fem_energy, 'r--', linewidth=3, label='FEM Energy')
            plt.xlabel('Time (s)', fontsize=14)
            plt.ylabel('Energy', fontsize=14)
            plt.title('Energy vs Time Comparison', fontsize=16)
            plt.grid(True)
            plt.legend(fontsize=12)
            plt.tight_layout()
            plt.savefig(energy_time_path)
            plt.close()
            
            self.logger.info(f"Energy plots saved to {save_path} and {energy_time_path}")
            
            # Modify the plotter function to use the dedicated Energy vs Time plot
            def add_energy_time_plot_to_plotter(plotter=None):
                """Adds the energy vs time plot to a PyVista plotter"""
                if plotter is None and hasattr(self, 'plotter') and self.plotter is not None:
                    plotter = self.plotter
                
                if plotter is not None:
                    try:
                        # Use the dedicated energy vs time plot
                        import pyvista
                        chart_actor = plotter.add_background_image(energy_time_path)
                        return chart_actor
                    except Exception as e:
                        self.logger.error(f"Failed to add energy-time plot to plotter: {e}")
                        return None
                return None
            
            # Store the function as an instance method
            self.add_energy_time_plot_to_plotter = add_energy_time_plot_to_plotter
            
            # Use this method in the final visualization
            if hasattr(self, 'plotter') and self.plotter is not None:
                    # [Final visualization code would use add_energy_time_plot_to_plotter]
                    
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to create energy plots: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
        

        # No changes needed outside the run_simulation method itself for this fix.
    # Make sure you have also addressed the objective function weighting issue separately.

    def run_simulation(self):
        """Run the full dynamic simulation with corrected time-stepping logic"""
        self.logger.info("Starting dynamic simulation with gravity application...")
        
        # --- Variables Initialization ---
        # Keep original structure for history lists
        z_history = []
        u_nn_history = []  # Neural network predictions stored step-by-step
        # FEM solutions populated by run_fem_step into self.fem_solution_history
        displacement_norms_nn = []
        displacement_norms_fem = [] # Calculated post-loop from FEM history
        energy_history_nn = [] # NN Elastic energy
        gravity_history = []   # Applied gravity magnitude history
        time_points = []
        error_history = []     # Error between NN(t) and FEM(t)
        self.u_history = []  # Initialize/reset u_history


        # self.nn_energy_history is used by plot_energy_vs_gravity, ensure it's initialized
        if not hasattr(self, 'nn_energy_history'):
             self.nn_energy_history = []
        self.nn_energy_history.clear() # Clear from previous runs if any

        if not hasattr(self, 'fem_energy_history'):
            self.fem_energy_history = []
        self.fem_energy_history.clear() # Clear from previous runs if any

        # --- Simulation Execution Block ---
        try: # Wrap simulation logic for robust finally block execution
            # We need states at t=0 and t=dt to predict t=2*dt.
            self.logger.info("Setting up initial states using FEM...")
            self.fem_solution_history = [] # Ensure FEM history is clean

            # State at t=0 (Assume zero initial displacement/velocity)
            t = 0.0
            u0_torch = torch.zeros_like(self.u_current_torch)
            z0_torch = torch.zeros_like(self.z_current)
            self.fem_solution_history.append(u0_torch.cpu().numpy().copy()) # FEM history starts at t=0
            # Store initial state in NN history lists
            time_points.append(t)
            z_history.append(z0_torch.cpu().numpy())
            u_nn_history.append(u0_torch.clone())
            displacement_norms_nn.append(0.0)
            energy_nn_t0 = self.compute_energy(z0_torch).item()
            energy_history_nn.append(energy_nn_t0)
            self.nn_energy_history.append(energy_nn_t0) # Also for energy plot
            gravity_history.append(0.0) # Gravity at t=0

            # State at t=dt (Compute using one FEM step)
            t = self.dt
            if not self.run_fem_step(t): # run_fem_step appends to self.fem_solution_history and self.fem_energy_history
                 self.logger.error(f"FEM solver failed during initialization at t={t:.4f}s. Aborting.")
                 raise RuntimeError(f"FEM initialization failed at t={t:.4f}s") # Use exception to trigger finally
            u1_torch = torch.tensor(self.fem_solution_history[-1], device=self.device, dtype=torch.float64)
            z1_torch = self.find_best_latent_vector(u1_torch) # Project FEM state to latent space

            # Store state at t=dt in NN history lists
            time_points.append(t)
            z_history.append(z1_torch.cpu().numpy())
            u_nn_history.append(u1_torch.clone()) # Use FEM result for NN history at t=dt
            displacement_norms_nn.append(torch.norm(u1_torch).item())
            energy_nn_t1 = self.compute_energy(z1_torch).item()
            energy_history_nn.append(energy_nn_t1)
            self.nn_energy_history.append(energy_nn_t1) # Also for energy plot
            current_gravity = self.gravity_magnitude * min(t / self.gravity_ramp_time, 1.0) if self.gravity_ramp_time > 0 else self.gravity_magnitude
            gravity_history.append(current_gravity)

            # --- Initialize NN State Variables for Prediction Loop ---
            # To predict step 2 (t=2*dt), we need state 0 (t=0) and state 1 (t=dt)
            self.u_prev_torch = u0_torch.clone() # State at t=0
            self.u_current_torch = u1_torch.clone() # State at t=dt
            self.z_prev = z0_torch.clone()
            self.z_current = z1_torch.clone()

            self.logger.info(f"Initialization complete using FEM states at t=0 and t={self.dt:.4f}s.")
            self.logger.info(f"NN predictor ready. u_prev=u(0), u_current=u({self.dt:.4f})")

            # Visualization setup
            last_viz_time = 0.0
            viz_interval = self.total_time / 50 # Aim for ~50 frames (adjust as needed)

            # --- Main Simulation Loop (Predicting t=2*dt onwards) ---
            # The loop predicts states for time steps 2, 3, ..., num_steps
            for step in range(2, self.num_steps + 1):
                t = step * self.dt # The time we are *predicting for* in this iteration
                self.logger.info(f"--- Step {step}/{self.num_steps}: Predicting for t={t:.4f}s ---")

                # Calculate target gravity magnitude for this time step
                current_gravity = self.gravity_magnitude * min(t / self.gravity_ramp_time, 1.0) if self.gravity_ramp_time > 0 else self.gravity_magnitude

                # 1. Predict NN state u(t) using u(t-dt) and u(t-2*dt)
                # self.u_current_torch holds u(t-dt) from the *previous* iteration's update
                # self.u_prev_torch holds u(t-2*dt) from the *previous* iteration's update
                z_predicted, u_nn_predicted = self.find_optimal_z(
                    self.u_current_torch,  # Pass state at t-dt
                    self.u_prev_torch,     # Pass state at t-2*dt
                    self.dt,
                    t                      # Pass target time t for objective function
                )

                # --- Store results for the current time step t ---
                time_points.append(t)
                z_history.append(z_predicted.cpu().numpy())
                u_nn_history.append(u_nn_predicted.clone()) # Store the NN prediction for time t
                u_nn_history.append(u_nn_predicted.clone())
                self.u_history = u_nn_history  # Keep u_history synced with u_nn_history

                displacement_norms_nn.append(torch.norm(u_nn_predicted).item())
                energy_nn = self.compute_energy(z_predicted).item()
                energy_history_nn.append(energy_nn)
                self.nn_energy_history.append(energy_nn) # Add to second list too
                gravity_history.append(current_gravity)

                # 2. Run FEM simulation for the same time step t for comparison
                # This appends the FEM solution for time t to self.fem_solution_history
                # and FEM energy to self.fem_energy_history
                fem_success = self.run_fem_step(t)
                if not fem_success:
                    self.logger.warning(f"FEM solver failed at t={t:.4f}s. Storing NaN for FEM results.")
                    u_fem_current_np = None # Use None to indicate failure
                    # Append NaN to FEM energy history if needed (run_fem_step might handle this)
                    if len(self.fem_energy_history) < len(self.fem_solution_history):
                         self.fem_energy_history.append(float('nan'))

                else:
                    # Get the just-computed FEM solution
                    u_fem_current_np = self.fem_solution_history[-1]

                # 3. Compare NN prediction u_nn_predicted with FEM result u_fem_current_np
                if u_fem_current_np is not None:
                    u_fem_current_torch = torch.tensor(u_fem_current_np, device=self.device, dtype=torch.float64)
                    fem_norm = torch.norm(u_fem_current_torch)
                    error = torch.norm(u_nn_predicted - u_fem_current_torch) / fem_norm if fem_norm > 1e-12 else torch.norm(u_nn_predicted - u_fem_current_torch)
                    self.logger.info(f"Relative error between NN and FEM at t={t:.4f}s: {error.item():.6f}")
                else:
                    error = torch.tensor(float('nan')) # Error is NaN if FEM failed

                error_history.append(error.item()) # error_history corresponds to times t=2*dt, 3*dt,...

                # 4. CRITICAL STATE UPDATE for NEXT iteration:
                # Prepare inputs for predicting step t+dt: Need u(t) and u(t-dt)
                self.u_prev_torch = self.u_current_torch.clone() # u_prev becomes state from t-dt
                self.u_current_torch = u_nn_predicted.clone()    # u_current becomes state from t (the one just predicted)
                self.z_prev = self.z_current.clone()
                self.z_current = z_predicted.clone()


                # Logging (matches original frequency and content)
                if step % 5 == 0 or step == self.num_steps:
                    self.logger.info(f"Step {step}/{self.num_steps} completed, Time: {t:.2f}s, NN Energy: {energy_nn:.4e}")
                    nn_max_disp = torch.max(torch.norm(u_nn_predicted.reshape(-1, 3), dim=1)).item()
                    self.logger.info(f"  NN max displacement: {nn_max_disp:.4e}")
                    if u_fem_current_np is not None:
                         fem_max_disp = np.max(np.linalg.norm(u_fem_current_np.reshape(-1, 3), axis=1))
                         self.logger.info(f"  FEM max displacement: {fem_max_disp:.4e}")
                         self.logger.info(f"  Rel. Error: {error.item():.6f}")
                    else:
                         self.logger.info(f"  FEM max displacement: NaN (solver failed)")
                         self.logger.info(f"  Rel. Error: NaN")


                # Visualization (uses state just computed for time t)
                # Compares NN prediction `u_nn_predicted` with FEM solution `self.fem_solution_history[-1]`
                if t - last_viz_time >= viz_interval or step == self.num_steps:
                    # The index `step` corresponds to time `t = step * dt` in the 0-based history lists
                    self.visualize_step(step, t, z_predicted, u_nn_predicted, current_gravity)
                    last_viz_time = t


            # --- Simulation Loop Finished ---
            self.logger.info("Simulation loop completed.")

            # --- Post-Processing (within try block) ---

            # Calculate FEM displacement norms from the collected history
            displacement_norms_fem = [np.linalg.norm(u_fem) if u_fem is not None else float('nan')
                                    for u_fem in self.fem_solution_history]
            if len(displacement_norms_fem) != len(time_points):
                 self.logger.warning(f"Length mismatch: FEM norms ({len(displacement_norms_fem)}) vs Time ({len(time_points)})")
                 # Pad FEM norms if FEM failed early? Or adjust plotting. Let's pad.
                 if len(displacement_norms_fem) < len(time_points):
                     displacement_norms_fem.extend([float('nan')] * (len(time_points) - len(displacement_norms_fem)))


            # Create error plot
            plt.figure(figsize=(10, 6))
            # error_history corresponds to time_points[2:] (times t=2*dt to num_steps*dt)
            plot_times_error = time_points[2:]
            if len(error_history) > 0 and len(plot_times_error) == len(error_history):
                plt.plot(plot_times_error, error_history, 'o-')
                plt.axhline(y=0.05, color='r', linestyle='--', label='5% error threshold')
                plt.xlabel('Time (s)')
                plt.ylabel('Relative Error')
                plt.title('Neural Network vs FEM Relative Error Over Time')
                plt.grid(True)
                plt.legend()
                plt.savefig(os.path.join(self.output_dir, "nn_vs_fem_error.png"))
                self.logger.info(f"Error plot saved to {os.path.join(self.output_dir, 'nn_vs_fem_error.png')}")
            else:
                 self.logger.warning(f"Could not generate error plot. Mismatch: Time points for error ({len(plot_times_error)}), Errors recorded ({len(error_history)}).")
            plt.close() # Close plot figure to free memory


            # Visualize final summary plots (e.g., displacement norms, energy)
            # Ensure visualize_results handles potential NaN values if FEM failed
            self.visualize_results(time_points, displacement_norms_nn,
                                    energy_history_nn, gravity_history) # Pass NN energy

            # Prepare final results dictionary
            results = {
                'time': time_points,
                'z': z_history,
                'displacement_norm_nn': displacement_norms_nn,
                'displacement_norm_fem': displacement_norms_fem,
                'energy_nn': energy_history_nn, # NN elastic energy
                'fem_energy_history': self.fem_energy_history, # FEM energy from run_fem_step
                'error': error_history # Note: length is num_steps-1
            }

            self.logger.info("Successfully prepared results dictionary.")
            return results # Return results on success

        except Exception as e:
            self.logger.error(f"Simulation failed due to an exception: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

            # Attempt to save partial results before exiting try block
            self.logger.warning("Attempting to save partial results...")
            try:
                 # Recalculate FEM norms from whatever history exists
                 displacement_norms_fem_partial = [np.linalg.norm(u_fem) if u_fem is not None else float('nan')
                                                 for u_fem in self.fem_solution_history]
                 # Pad FEM norms if needed
                 if len(displacement_norms_fem_partial) < len(time_points):
                     displacement_norms_fem_partial.extend([float('nan')] * (len(time_points) - len(displacement_norms_fem_partial)))

                 # Make sure all lists passed to visualize_results have the same length
                 min_len = min(len(time_points), len(displacement_norms_nn), len(energy_history_nn), len(gravity_history))
                 self.visualize_results(time_points[:min_len], displacement_norms_nn[:min_len],
                                        energy_history_nn[:min_len], gravity_history[:min_len])
                 self.logger.info("Partial summary plots saved.")
            except Exception as plot_e:
                self.logger.error(f"Failed to save partial summary plots during error handling: {plot_e}")

            # Return None to indicate failure
            return None

        finally:
            # This block executes whether the try block succeeded or failed
            self.logger.info("Executing finally block: Creating final energy plots and closing visualization...")

            # Ensure plot_energy_vs_gravity uses the correct, potentially partial, histories
            # It relies on self.nn_energy_history and self.fem_energy_history
            self.plot_energy_vs_gravity() # Assumes this function handles potentially mismatched lengths

            # Safely close the GIF plotter and add a final frame
            if hasattr(self, 'plotter') and self.plotter is not None:
                try:
                    # Determine final message based on whether 'results' was successfully created
                    sim_outcome_known = 'results' in locals() or 'results' in globals()
                    final_frame_text = "Simulation Complete" if sim_outcome_known and results is not None else "Simulation Failed"
                    self.logger.info(f"Adding final plotter frame with text: '{final_frame_text}'")

                    # Add a few final frames to make the end state visible in the GIF
                    for _ in range(10): # Add 10 identical final frames
                        self.plotter.clear()

                        # Left subplot: Final Status Text
                        self.plotter.subplot(0, 0)
                        self.plotter.add_text(final_frame_text, position="upper_edge", font_size=20, color='white')
                        # Optionally add the final deformed mesh (if available)
                        if u_nn_history: # Check if any NN history exists
                             try:
                                 u_final_nn = u_nn_history[-1]
                                 # Create mesh with deformation for NN solution (similar to visualize_step)
                                 u_quadratic = fem.Function(self.routine.V)
                                 u_quadratic.x.array[:] = u_final_nn.cpu().numpy()
                                 self.u_linear.interpolate(u_quadratic)
                                 u_linear_np = self.u_linear.x.array
                                 nn_grid = self.grid.copy()
                                 nn_grid.point_data["displacement"] = u_linear_np.reshape((-1, 3))
                                 nn_warped = nn_grid.warp_by_vector("displacement", factor=1.0)
                                 self.plotter.add_mesh(nn_warped, color='lightblue', show_edges=True, opacity=0.7)
                             except Exception as mesh_e:
                                 self.logger.warning(f"Could not add final NN mesh to plotter: {mesh_e}")
                        else:
                             # Show undeformed grid if no results
                             if hasattr(self, 'grid'): self.plotter.add_mesh(self.grid, style='wireframe', color='grey', opacity=0.3)


                        # Right subplot: Energy Plot Background
                        self.plotter.subplot(0, 1)
                        try:
                            # Use the method that adds the pre-generated energy plot image
                            if hasattr(self, 'add_energy_time_plot_to_plotter'):
                                self.add_energy_time_plot_to_plotter(plotter=self.plotter)
                            else:
                                self.logger.warning("add_energy_time_plot_to_plotter method not found.")
                                self.plotter.add_text("Energy plot unavailable", position="upper_edge", color="white")
                        except Exception as plot_add_e:
                            self.logger.error(f"Could not add energy plot image to final frame: {plot_add_e}")
                            self.plotter.add_text("Energy plot error", position="upper_edge", color="red")

                        # Write the combined frame to the GIF
                        self.plotter.write_frame()

                    # Close the plotter (which also finalizes the GIF)
                    self.plotter.close()
                    self.logger.info("GIF plotter closed and file finalized.")

                except Exception as close_e:
                    # Log error during cleanup but don't prevent program exit
                    self.logger.error(f"Error during plotter closing/final frame writing in finally block: {close_e}")
            else:
                 self.logger.info("Plotter was not initialized or already closed. Skipping final frame.")
            # --- End of Finally Block ---


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
    parser = argparse.ArgumentParser(description='Dynamic validation with gravity for Neural Plates')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='config file path')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt', help='checkpoint path')
    parser.add_argument('--steps', type=int, default=100, help='number of simulation steps')
    parser.add_argument('--time', type=float, default=1.2, help='total simulation time')
    parser.add_argument('--gravity', type=float, default=9.81, help='gravity magnitude (m/s²)')
    parser.add_argument('--damping', type=float, default=0.01, help='damping coefficient')
    parser.add_argument('--output', type=str, default='validation_results', help='output directory')
    args = parser.parse_args()
    
    # Setup logger
    setup_logger("gravity_validation", log_dir=args.output)
    logger = logging.getLogger("gravity_validation")
    
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

    # Create validator and run simulation with gravity
    validator = DynamicValidator(
        routine=routine,
        num_steps=args.steps,
        total_time=args.time,
        gravity_magnitude=args.gravity,
        gravity_direction=[0, 0, -1],  # Default direction downward
        gravity_ramp_time=0.2,         # Ramp up gravity over 0.5s
        damping=args.damping,
        output_dir=args.output
    )
    
    # Run simulation
    results = validator.run_simulation()

if __name__ == "__main__":
    main()

