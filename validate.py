import os
import torch
import numpy as np
import argparse
from train import Routine, load_config, setup_logger
from dolfinx import mesh, fem, plot, io
import ufl
import pyvista
import matplotlib.pyplot as plt
from mpi4py import MPI
from petsc4py import PETSc
import time
import logging


class DynamicValidator:
    """
    Validates neural plates model with dynamic simulation using finite difference method.
    Minimizes z_{n+1} using: norm{u(z_{n+1}) - 2u_n + u_{n-1}}_M + E(z_{n+1})
    where u(z) is the displacement field corresponding to latent vector z.
    """
    def __init__(self, routine, dt=0.01, total_time=10.0, gravity_change_interval=100,
                 gravity_values=None, damping=0.05, output_dir="validation_results"):
        """
        Initialize dynamic validator.
        
        Args:
            routine: Trained Routine object with neural model
            dt: Time step size
            total_time: Total simulation time
            gravity_change_interval: Number of steps between gravity changes
            gravity_values: List of gravity vectors to cycle through [g_x, g_y, g_z]
            damping: Damping coefficient (0 = no damping, 1 = critically damped)
            output_dir: Directory to save results
        """
        self.routine = routine
        self.dt = dt
        self.total_time = total_time
        self.num_steps = int(total_time / dt)
        self.gravity_change_interval = gravity_change_interval
        
        # Default gravity values if none provided (rotate gravity vector)
        if gravity_values is None:
            g_mag = 0.5 # Standard gravity magnitude
            self.gravity_values = [
                [0, 0, -g_mag],    # Normal gravity (downward)
                [g_mag, 0, -g_mag], # 45 degrees in x-z plane
                [0, g_mag, -g_mag], # 45 degrees in y-z plane
                [-g_mag, 0, -g_mag] # -45 degrees in x-z plane
            ]
        else:
            self.gravity_values = gravity_values
            
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
        
        # Initialize visualization
        self.setup_visualization()
        
    def setup_visualization(self):
        """Set up visualization utilities"""
        # Convert DOLFINx mesh to PyVista format
        topology, cell_types, x = plot.vtk_mesh(self.routine.domain)
        self.grid = pyvista.UnstructuredGrid(topology, cell_types, x)
        
        # Create a linear function space for visualization
        self.V_viz = fem.functionspace(self.routine.domain, ("CG", 1, (3,)))
        self.u_linear = fem.Function(self.V_viz)
        
    def compute_energy(self, z):
        """Compute the elastic energy for given latent vector z"""
        with torch.no_grad():
            # Get linear and non-linear displacements
            l = (self.routine.linear_modes @ z.unsqueeze(1)).squeeze(1)
            y = self.routine.model(z)
            u_total = l + y
            
            # Compute elastic energy
            energy = self.routine.energy_calculator(u_total)
            return energy
            
    def compute_temporal_consistency(self, z, u_current, u_prev, dt):
        """
        Compute the temporal consistency term: ||u(z) - 2u_current + u_prev||^2_M
        Where u(z) is the displacement field corresponding to latent vector z.
        
        Args:
            z: The next latent vector z_{n+1} we're optimizing
            u_current: Current displacement field u_n
            u_prev: Previous displacement field u_{n-1}
            dt: Time step
            
        Returns:
            Weighted norm of the finite difference acceleration approximation
        """
        # Convert z to displacement field - maintain gradients
        l = (self.routine.linear_modes @ z.unsqueeze(1)).squeeze(1)
        y = self.routine.model(z)
        u_next = l + y
        
        # We need a differentiable version that avoids PETSc operations
        if not hasattr(self, 'M_torch'):
            # Do this once and cache it - use dense approach for simplicity
            try:
                self.logger.info("Converting mass matrix to PyTorch tensor...")
                M_dense = self.M.convert('dense')
                M_array = M_dense.getDenseArray()
                self.M_torch = torch.tensor(M_array, device=self.device, dtype=torch.float64)
                self.logger.info(f"Mass matrix converted, shape: {self.M_torch.shape}")
            except Exception as e:
                # Fallback to diagonal mass matrix if conversion fails
                self.logger.warning(f"Failed to convert mass matrix: {e}")
                self.logger.warning("Using diagonal mass approximation instead")
                
                # Extract diagonal of mass matrix (lumped mass approximation)
                self.M_torch = torch.eye(self.M.size[0], device=self.device, dtype=torch.float64)
                self.logger.info("Using identity matrix as fallback")
        
        # Now compute M*u for each vector using torch operations
        M_u_next = torch.matmul(self.M_torch, u_next)
        M_u_current = torch.matmul(self.M_torch, u_current)
        M_u_prev = torch.matmul(self.M_torch, u_prev)
        
        # Compute acceleration term: M*u_next - 2*M*u_current + M*u_prev
        acc_vec = M_u_next - 2.0 * M_u_current + M_u_prev
        
        # Compute squared norm: ||acc_vec||^2
        consistency = torch.sum(acc_vec * acc_vec) / (dt * dt)
        
        return consistency
            
    def update_body_force(self, step):
        """Update body force based on current step"""
        gravity_index = (step // self.gravity_change_interval) % len(self.gravity_values)
        current_gravity = self.gravity_values[gravity_index]
        self.logger.info(f"Step {step}: Updating gravity to {current_gravity}")
        return current_gravity
        
    def objective_function(self, z, u_current, u_prev, dt, alpha=1.0, gravity=None):
        """
        Objective function to minimize: 
        ||u(z) - 2u_current + u_prev||^2_M/dt² + alpha*E(z) + gravity_potential + damping_term
        """
        # Temporal consistency term
        temporal = self.compute_temporal_consistency(z, u_current, u_prev, dt)
        
        # Energy term
        energy = self.compute_energy(z)
        
        # Get displacement for gravity and damping calculations
        l = (self.routine.linear_modes @ z.unsqueeze(1)).squeeze(1)
        y = self.routine.model(z)
        u_next = l + y
        
        # Gravity potential energy
        gravity_potential = 0.0
        if gravity is not None:
            g_vec = torch.tensor(gravity, device=self.device, dtype=torch.float64)
            # Apply gravity to each node (treat each DOF as a unit mass)
            # We use negative sign as gravity potential decreases with height
            gravity_potential = -torch.sum(u_next.reshape(-1, 3) @ g_vec)
        
        # Damping term (if needed)
        damping_term = 0.0
        if self.damping > 0:
            # Approximate velocity: (u_next - u_current)/dt
            velocity = (u_next - u_current) / dt
            # Simple damping: c*||v||^2
            damping_term = self.damping * torch.sum(velocity * velocity)
        
        # Total objective - INCLUDE ALL TERMS
        objective = temporal + alpha * energy + gravity_potential + damping_term
        
        return objective
    
    def find_optimal_z(self, u_current, u_prev, dt, gravity, num_iters=50):
        """Find optimal latent vector z that minimizes the objective function"""
        # Start optimization from current z
        z = self.z_current.clone().detach().requires_grad_(True)
        
        # Use L-BFGS for optimization
        optimizer = torch.optim.LBFGS([z], 
                                    lr=0.1, 
                                    max_iter=20,
                                    line_search_fn='strong_wolfe')
        
        # Weight for energy term
        alpha = 1.0
        
        # Optimize
        for i in range(num_iters):
            def closure():
                optimizer.zero_grad()
                # Pass gravity to the objective function
                loss = self.objective_function(z, u_current, u_prev, dt, alpha, gravity)
                loss.backward()
                return loss
                
            loss = optimizer.step(closure)
        
        # Get optimal displacement
        with torch.no_grad():
            l = (self.routine.linear_modes @ z.unsqueeze(1)).squeeze(1)
            y = self.routine.model(z)
            u_opt = l + y
            
        return z.detach(), u_opt.detach()
    
    def run_simulation(self):
        """Run the full dynamic simulation"""
        self.logger.info("Starting dynamic simulation...")
        
        # Initialize arrays for storing results
        z_history = []
        displacement_norms = []
        energy_history = []
        gravity_history = []
        time_points = []
        
        # Initialize with FEniCSx for accurate dynamics
        initial_gravity = self.gravity_values[0]
        (self.z_prev, self.u_prev), (self.z_current, self.u_current) = self.initialize_with_fenics(initial_gravity)
        
        # Store initial state (t = dt, since FEniCSx gave us u_0 and u_dt)
        z_history.append(self.z_prev.cpu().numpy())  # t = 0
        z_history.append(self.z_current.cpu().numpy())  # t = dt
        
        displacement_norms.append(np.linalg.norm(self.u_prev.cpu().numpy()))  # t = 0
        displacement_norms.append(np.linalg.norm(self.u_current.cpu().numpy()))  # t = dt
        
        energy_history.append(self.compute_energy(self.z_prev).item())  # t = 0
        energy_history.append(self.compute_energy(self.z_current).item())  # t = dt
        
        gravity_history.append(initial_gravity)  # t = 0
        gravity_history.append(initial_gravity)  # t = dt
        
        time_points.append(0.0)  # t = 0
        time_points.append(self.dt)  # t = dt
        
        # Start from step 2 (t = 2*dt) since we already have t=0 and t=dt
        time_offset = 2
        
        # Create plotter for visualization
        plotter = None
        last_viz_time = 0.0  # Start visualizing immediately
        viz_interval = 0.1  # Visualize every 0.1 seconds of simulation time
        
        # Main simulation loop - start from step 2
        for step in range(time_offset, self.num_steps + 1):
            t = step * self.dt
            
            # Update gravity if needed
            current_gravity = self.update_body_force(step)
            
            # Find optimal z
            self.z_next, self.u_next = self.find_optimal_z(
                self.u_current, self.u_prev, self.dt, current_gravity
            )
            
            # Compute energy
            energy = self.compute_energy(self.z_next).item()
            
            # Store results
            z_history.append(self.z_next.cpu().numpy())
            displacement_norms.append(np.linalg.norm(self.u_next.cpu().numpy()))
            energy_history.append(energy)
            gravity_history.append(current_gravity)
            time_points.append(t)
            
            # Log progress
            if step % 10 == 0:
                self.logger.info(f"Step {step}/{self.num_steps}, Time: {t:.2f}s, Energy: {energy:.4e}")
            
            # Visualize if needed
            if t - last_viz_time >= viz_interval:
                self.visualize_step(step, t, self.z_next, self.u_next)
                last_viz_time = t
            
            # Update state for next step
            self.u_prev = self.u_current.clone()
            self.u_current = self.u_next.clone()
            self.z_prev = self.z_current.clone()
            self.z_current = self.z_next.clone()
            
        # Visualize final results
        self.visualize_results(time_points, displacement_norms, energy_history, gravity_history)
        
        return {
            'time': time_points,
            'z': z_history,
            'displacement_norm': displacement_norms,
            'energy': energy_history,
            'gravity': gravity_history
        }
    


    # Add this method to your DynamicValidator class:
    def initialize_with_fenics(self, gravity):
        """
        Initialize simulation using FEniCSx to compute first steps.
        Uses homogeneous Neumann boundary conditions (free-floating body).
        """
        self.logger.info("Initializing simulation with FEniCSx...")
        
        # Get access to the domain and function spaces from routine
        domain = self.routine.domain
        V = self.routine.V
        
        # 1. First solve static equilibrium under gravity
        u_static = fem.Function(V)
        
        # Define the weak form for static equilibrium with gravity
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Material parameters
        E = self.routine.E
        nu = self.routine.nu
        mu = E / (2.0 * (1.0 + nu))
        lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        
        # Strain and stress
        def epsilon(u):
            return ufl.sym(ufl.grad(u))
            
        def sigma(u):
            return lambda_ * ufl.tr(epsilon(u)) * ufl.Identity(3) + 2.0 * mu * epsilon(u)
        
        # Gravity force
        f = fem.Constant(domain, (gravity[0], gravity[1], gravity[2]))
        
        # Weak form: elastic energy + gravity with homogeneous Neumann boundaries
        a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
        L = ufl.inner(f, v) * ufl.dx
        
        # No boundary conditions for homogeneous Neumann (free floating body)
        # However, to handle the singularity, we need to constrain rigid body motions
        # This can be done by fixing a single point or using a nullspace approach
        
        # Define rigid body modes
        # Translation modes
        rigid_modes = []
        for i in range(3):
            mode = fem.Function(V)
            mode_array = np.zeros_like(mode.x.array)
            for j in range(len(mode_array) // 3):
                mode_array[3*j + i] = 1.0  # Unit displacement in direction i
            mode.x.array[:] = mode_array
            rigid_modes.append(mode)
        
        # Rotation modes (around x, y, z axes)
        # These are approximations for small rotations
        coords = domain.geometry.x
        for i in range(3):
            mode = fem.Function(V)
            mode_array = np.zeros_like(mode.x.array)
            for j in range(len(coords)):
                # Cross product of position with axis of rotation
                if i == 0:  # Rotation around x-axis: (0, z, -y)
                    mode_array[3*j + 1] = coords[j, 2]
                    mode_array[3*j + 2] = -coords[j, 1]
                elif i == 1:  # Rotation around y-axis: (-z, 0, x)
                    mode_array[3*j + 0] = -coords[j, 2]
                    mode_array[3*j + 2] = coords[j, 0]
                else:  # Rotation around z-axis: (y, -x, 0)
                    mode_array[3*j + 0] = coords[j, 1]
                    mode_array[3*j + 1] = -coords[j, 0]
            mode.x.array[:] = mode_array
            rigid_modes.append(mode)
        
        # Create nullspace for PETSc
        null_vec = fem.petsc.create_vector(fem.form(L))
        nullspace_basis = [null_vec.copy() for i in range(6)]
        for i, rigid_mode in enumerate(rigid_modes):
            nullspace_basis[i].array[:] = rigid_mode.x.array[:]
            nullspace_basis[i].normalize()

        nullspace = PETSc.NullSpace().create(constant=False, vectors=nullspace_basis)
        
        # Create and assemble matrix - 
        A = fem.petsc.create_matrix(fem.form(a))
        fem.petsc.assemble_matrix(A, fem.form(a))
        A.assemble()  # Critical step that was missing
        
        # Set nullspace after assembly
        A.setNullSpace(nullspace)
        
        # Create and assemble vector
        b = fem.petsc.create_vector(fem.form(L))
        fem.petsc.assemble_vector(b, fem.form(L))
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
        # Project b onto the range (remove nullspace components)
        nullspace.remove(b)
        
        # Set up solver
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        solver.setType("preonly")
        pc = solver.getPC()
        pc.setType("lu")
        
        # Solve system
        x = fem.petsc.create_vector(fem.form(L))
        solver.solve(b, x)
        
        # Copy solution to function
        u_static.x.array[:] = x.array[:]
        
        self.logger.info("Static equilibrium solved with FEniCSx")
        
        # 2. Now compute velocity using a single dynamic step
        # Create velocity function
        v_half = fem.Function(V)  # Velocity at t = dt/2 (midpoint)
        
        # Mass matrix
        density = 10.0  # Material density (adjust as needed)
        u_dot = ufl.TrialFunction(V)
        v_test = ufl.TestFunction(V)
        m = density * ufl.inner(u_dot, v_test) * ufl.dx
        
        # Mass matrix in PETSc format
        M = fem.petsc.create_matrix(fem.form(m))
        fem.petsc.assemble_matrix(M, fem.form(m))
        M.assemble()
        
        # Set same nullspace for mass matrix as for stiffness matrix
        M.setNullSpace(nullspace)
        
        # Create vector for initial acceleration (gravity)
        b = fem.petsc.create_vector(fem.form(L))
        with b.localForm() as b_local:
            b_local.set(0.0)
        fem.petsc.assemble_vector(b, fem.form(L))
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
        # Remove nullspace components from right-hand side
        nullspace.remove(b)
        
        # Initial velocity (half step): v(dt/2) = 0 + dt/2 * a(0)
        # a(0) = M^-1 * b
        acc = b.copy()
        ksp = PETSc.KSP().create(domain.comm)
        ksp.setOperators(M)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        
        # Add this: Use regularization for M if needed
        # if pc.getType() == "lu":
        #     pc.setFactorShiftType(PETSc.Mat.ShiftType.NONZERO)
        #     pc.setFactorShiftAmount(1e-12)
        
        ksp.solve(b, acc)
            
        # v(dt/2) = dt/2 * a(0)
        v_half.x.array[:] = (self.dt/2) * acc.getArray()
        
        # 3. Compute u(dt) = u(0) + dt * v(dt/2)
        u_next = fem.Function(V)
        u_next.x.array[:] = u_static.x.array + self.dt * v_half.x.array
        
        self.logger.info("FEniCSx initialization complete")
        
        # Convert to torch tensors and continue with rest of method as before...
        u_static_np = u_static.x.array
        u_next_np = u_next.x.array
        
        # Convert to torch tensors
        u_0 = torch.tensor(u_static_np, device=self.device, dtype=torch.float64)
        u_1 = torch.tensor(u_next_np, device=self.device, dtype=torch.float64)
        
        # Project these into latent space
        z_0 = self.routine.compute_modal_coordinates(
            u_0.cpu().numpy(), 
            self.routine.linear_modes.cpu().numpy(), 
            self.M
        )
        z_0 = torch.tensor(z_0, device=self.device, dtype=torch.float64)
        
        z_1 = self.routine.compute_modal_coordinates(
            u_1.cpu().numpy(), 
            self.routine.linear_modes.cpu().numpy(), 
            self.M
        )
        z_1 = torch.tensor(z_1, device=self.device, dtype=torch.float64)
        
        # Set initial states
        self.u_prev = u_0
        self.u_current = u_1
        self.z_prev = z_0
        self.z_current = z_1
        
        return (z_0, u_0), (z_1, u_1)
    
    def visualize_step(self, step, t, z, u):
        """Visualize current simulation state"""
        # Create a function in the original function space
        u_quadratic = fem.Function(self.routine.V)
        u_quadratic.x.array[:] = u.cpu().numpy()
        
        # Interpolate to the visualization space
        self.u_linear.interpolate(u_quadratic)
        u_linear_np = self.u_linear.x.array
        
        # Create mesh with deformation
        local_grid = self.grid.copy()
        local_grid.point_data["displacement"] = u_linear_np.reshape((-1, 3))
        local_grid["displacement_magnitude"] = np.linalg.norm(u_linear_np.reshape((-1, 3)), axis=1)
        
        # Warp the mesh by the displacement
        warped = local_grid.warp_by_vector("displacement", factor=1.0)
        
        # Save visualization
        file_path = os.path.join(self.output_dir, f"step_{step:06d}.png")
        
        # Create plotter
        plotter = pyvista.Plotter(off_screen=True, window_size=[1024, 768])
        plotter.add_mesh(warped, scalars="displacement_magnitude", cmap="viridis", show_edges=True)
        plotter.add_title(f"Time: {t:.3f}s, Step: {step}")
        plotter.view_isometric()
        plotter.add_scalar_bar("Displacement Magnitude")
        plotter.screenshot(file_path)
        plotter.close()
        
    def visualize_results(self, time_points, displacement_norms, energy_history, gravity_history):
        """Create summary plots of simulation results"""
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Plot displacement norm over time
        axes[0].plot(time_points, displacement_norms)
        axes[0].set_ylabel("Displacement Norm")
        axes[0].set_title("Displacement Norm Over Time")
        axes[0].grid(True)
        
        # Plot energy over time
        axes[1].plot(time_points, energy_history)
        axes[1].set_ylabel("Elastic Energy")
        axes[1].set_title("System Energy Over Time")
        axes[1].grid(True)
        
        # Plot gravity changes
        gravity_norms = [np.linalg.norm(g) for g in gravity_history]
        axes[2].plot(time_points, gravity_norms)
        # Add vertical lines at gravity change points
        for i in range(1, len(gravity_history)):
            if not np.array_equal(gravity_history[i], gravity_history[i-1]):
                axes[2].axvline(x=time_points[i], color='r', linestyle='--', alpha=0.5)
        
        axes[2].set_ylabel("Gravity Magnitude")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_title("Gravity Changes")
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "simulation_summary.png"))
        plt.close()
        
    def find_static_equilibrium(self, gravity, num_iters=100):
        """Find static equilibrium under given gravity"""
        # Start with zero latent vector
        z = torch.zeros(self.routine.latent_dim, requires_grad=True, device=self.device)
        
        # Use L-BFGS for optimization
        optimizer = torch.optim.LBFGS([z], 
                                     lr=0.1, 
                                     max_iter=20,
                                     line_search_fn='strong_wolfe')
        
        # Optimize to find static equilibrium
        for i in range(num_iters):
            def closure():
                optimizer.zero_grad()
                
                # Get displacement field
                l = (self.routine.linear_modes @ z.unsqueeze(1)).squeeze(1)
                y = self.routine.model(z)
                u = l + y
                
                # Compute elastic energy
                energy = self.routine.energy_calculator(u)
                
                # Compute gravitational potential energy
                # Simplified gravity potential: Sum(m_i * g * h_i)
                # where h_i is displacement in gravity direction
                g_vec = torch.tensor(gravity, device=self.device, dtype=torch.float64)
                potential = 0.0
                
                # Need to apply gravity to each node based on mass
                # This is a simplified approximation
                potential = -torch.sum(u.reshape(-1, 3) @ g_vec)
                
                # Total energy is elastic + potential
                total_energy = energy + potential
                
                total_energy.backward()
                return total_energy
                
            loss = optimizer.step(closure)
        
        # Get final displacement
        with torch.no_grad():
            l = (self.routine.linear_modes @ z.unsqueeze(1)).squeeze(1)
            y = self.routine.model(z)
            u = l + y
            
        return z.detach(), u.detach()

def create_video_from_images(image_folder, output_file, fps=30):
    """Create a video from simulation step images"""
    try:
        import cv2
        import glob
        
        # Get all png files
        image_files = sorted(glob.glob(os.path.join(image_folder, "step_*.png")))
        
        if not image_files:
            print("No image files found!")
            return False
        
        # Get dimensions from first image
        img = cv2.imread(image_files[0])
        height, width, layers = img.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        # Add each image to video
        for image_file in image_files:
            img = cv2.imread(image_file)
            video.write(img)
            
        # Release resources
        cv2.destroyAllWindows()
        video.release()
        
        print(f"Video created at {output_file}")
        return True
    
    except ImportError:
        print("OpenCV not installed. Cannot create video.")
        return False

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Dynamic validation for Neural Plates')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='config file path')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt', help='checkpoint path')
    parser.add_argument('--dt', type=float, default=0.01, help='time step size')
    parser.add_argument('--time', type=float, default=1.0, help='total simulation time')
    parser.add_argument('--interval', type=int, default=100, help='gravity change interval (steps)')
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
    
    # Set up gravity values
    g_mag = 0.5  # Standard gravity magnitude
    gravity_values = [
        [0, 0, -g_mag],         # Normal gravity (downward)
        [0.5*g_mag, 0, -g_mag], # Tilted in x-z plane
        [0, 0.5*g_mag, -g_mag], # Tilted in y-z plane
        [-0.5*g_mag, 0, -g_mag] # Tilted in negative x-z plane
    ]
    
    # Create validator and run simulation
    validator = DynamicValidator(
        routine=routine,
        dt=args.dt,
        total_time=args.time,
        gravity_change_interval=args.interval,
        gravity_values=gravity_values,
        damping=args.damping,
        output_dir=args.output
    )
    
    logger.info("Starting dynamic simulation")
    start_time = time.time()
    results = validator.run_simulation()
    end_time = time.time()
    
    logger.info(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Create video from simulation frames
    logger.info("Creating video from simulation frames")
    video_path = os.path.join(args.output, "simulation.mp4")
    create_video_from_images(args.output, video_path, fps=30)
    
    logger.info(f"Results saved to {args.output}")
    logger.info("Validation complete")

if __name__ == "__main__":
    main()