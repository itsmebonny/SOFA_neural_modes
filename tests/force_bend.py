import dolfinx
import ufl
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import time
import pyvista
from dolfinx import plot, fem, default_scalar_type
import dolfinx.fem.petsc # For assembly functions
# Removed nls import as we solve linear systems now
from dolfinx.io import gmshio

# --- Simulation Parameters ---
L = 10.0
r = 0.5
E = 1.0e5
nu = 0.4
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
rho = 1000.0
t_end = 1.0 # Reduced time for potentially smaller stable dt
num_steps = 10000 # Increased steps for potentially smaller stable dt
dt = t_end / num_steps



# --- Random Force Parameters ---
box_min_x = L * 0.6
box_max_x = L 
box_min_y = -r 
box_max_y = r 
box_min_z = -r 
box_max_z = r 
box_coords_min = np.array([box_min_x, box_min_y, box_min_z])
box_coords_max = np.array([box_max_x, box_max_y, box_max_z])
random_force_density_magnitude = 5.0e3 # Adjust as needed
force_direction = np.array([0.0, 0.0, 1.0])
np.random.seed(42)

print("--- Simulation Setup ---")
print(f"Young's modulus: {E}")
print(f"Poisson's ratio: {nu}")
print(f"Density: {rho}")
print(f"Total simulation time: {t_end}")
print(f"Number of time steps: {num_steps}")
print(f"Time step size: {dt} (Check CFL condition!)") # Warning about stability
print("\n--- Random Force Parameters ---")
print(f"Force Box Min Coords: {box_coords_min}")
print(f"Force Box Max Coords: {box_coords_max}")
print(f"Max Random Force Density Magnitude: {random_force_density_magnitude}")
print(f"Force Direction: {force_direction}")
print("-" * 25)


# --- Mesh ---
mesh_file = "mesh/beam_732.msh"
try:
    mesh, cell_tags, facet_tags = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, 0, gdim=3)
except FileNotFoundError:
    print(f"Error: Mesh file '{mesh_file}' not found.")
    exit()
except Exception as e:
    print(f"Error loading mesh: {e}")
    exit()

x_coords = mesh.geometry.x
x_min_mesh, x_max_mesh = np.min(x_coords[:, 0]), np.max(x_coords[:, 0])
y_min_mesh, y_max_mesh = np.min(x_coords[:, 1]), np.max(x_coords[:, 1])
z_min_mesh, z_max_mesh = np.min(x_coords[:, 2]), np.max(x_coords[:, 2])
print(f"Actual beam length from mesh: {x_max_mesh - x_min_mesh:.4f}")
print("-" * 25)

# Add after the mesh loading section
# --- Compute CFL Condition for Stable Time Step ---
def compute_stable_timestep(mesh, E, nu, rho):
    """Compute the stable time step based on CFL condition for elastic waves."""
    # Calculate Lamé parameters (already have these, but calculating from E, nu for clarity)
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    
    # Compute longitudinal wave speed (p-wave)
    c_p = np.sqrt((lmbda + 2*mu) / rho)
    
    # Estimate minimum element size (characteristic length)
    # For a more accurate estimate, we'd compute cell diameters
    # This is a conservative approximation
    x_range = x_max_mesh - x_min_mesh
    y_range = y_max_mesh - y_min_mesh
    z_range = z_max_mesh - z_min_mesh
    
    # Get cell count from mesh
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    
    # Estimate average element volume
    domain_volume = x_range * y_range * z_range
    avg_cell_volume = domain_volume / num_cells
    
    # Estimate characteristic length (h) as cube root of volume
    h_est = avg_cell_volume ** (1/3)
    
    # For tetrahedral elements, a more conservative factor is applied
    h = h_est * 0.5  # Conservative factor
    
    # Compute critical time step
    dt_crit = h / c_p
    
    return dt_crit, c_p, h

# Compute the stable time step
dt_crit, wave_speed, mesh_size = compute_stable_timestep(mesh, E, nu, rho)

# Print CFL information
print("\n--- CFL Stability Information ---")
print(f"Estimated characteristic mesh size: {mesh_size:.6f} m")
print(f"Elastic wave speed: {wave_speed:.2f} m/s")
print(f"Critical time step (dt_crit): {dt_crit:.6f} s")
print(f"Current time step (dt): {dt:.6f} s")
if dt > dt_crit:
    print(f"WARNING: Current time step is {dt/dt_crit:.2f}x larger than critical value!")
    print("Simulation may be unstable. Consider reducing dt or using implicit integration.")
else:
    print(f"Time step is {dt_crit/dt:.2f}x smaller than critical value (stable).")
    input("Press Enter to continue...")
print("-" * 25)


# --- Function Spaces and Functions ---
V = dolfinx.fem.functionspace(mesh, ("CG", 1, (mesh.geometry.dim,)))

u = dolfinx.fem.Function(V, name="Displacement_n+1") # u at n+1 (unknown in step)
u_curr = dolfinx.fem.Function(V, name="Displacement_n")  # u at n (current)
u_prev = dolfinx.fem.Function(V, name="Displacement_n-1")# u at n-1 (previous)
# We don't explicitly need velocity/acceleration functions for the central diff update
v = ufl.TestFunction(V)                                # Test function
u_trial = ufl.TrialFunction(V) # Trial function for LHS matrix

f_random = dolfinx.fem.Function(V, name="RandomForce") # Force density field
a0 = dolfinx.fem.Function(V, name="InitialAcceleration") # For first step

# --- Identify Degrees of Freedom (DoFs) within the Force Box ---
dof_coordinates = V.tabulate_dof_coordinates()
dofs_in_box_mask = np.logical_and.reduce((
    dof_coordinates[:, 0] >= box_coords_min[0], dof_coordinates[:, 0] <= box_coords_max[0],
    dof_coordinates[:, 1] >= box_coords_min[1], dof_coordinates[:, 1] <= box_coords_max[1],
    dof_coordinates[:, 2] >= box_coords_min[2], dof_coordinates[:, 2] <= box_coords_max[2]
))
dofs_in_box_indices = np.where(dofs_in_box_mask)[0]

force_component_index = np.where(np.abs(force_direction) > 1e-6)[0]
if len(force_component_index) == 1:
    force_component_index = force_component_index[0]
    print(f"Applying force only to component {force_component_index} ('{['x','y','z'][force_component_index]}')")
    component_mask = (dofs_in_box_indices % V.dofmap.bs) == force_component_index
    force_dof_indices = dofs_in_box_indices[component_mask]
    print(f"Found {len(force_dof_indices)} DoFs in the force box for the specified component.")
else:
    print("Applying force to all components of nodes within the box.")
    force_dof_indices = dofs_in_box_indices
    print(f"Found {len(force_dof_indices)} DoFs in the force box.")

if len(force_dof_indices) == 0:
    print("Warning: No degrees of freedom found within the specified force box.")


# --- Constitutive Model (Neo-Hookean - Stress calculation) ---
# We need the 1st Piola-Kirchhoff stress P = F*S
# S = mu*(I - C^{-1}) + lambda*ln(J)*C^{-1} (for compressible Neo-Hooke)
# P = mu*(F - F^{-T}) + lambda*ln(J)*F^{-T}
def pk1_stress(u, mu, lmbda):
    d = mesh.geometry.dim
    I = ufl.Identity(d)
    F = I + ufl.grad(u)
    C = F.T * F
    J = ufl.det(F)
    # Add safeguards for extreme compression
    J_safe = ufl.conditional(ufl.gt(J, 0.01), J, 0.01)  # Prevent collapse
    log_J = ufl.ln(J_safe)
    C_inv = ufl.inv(C)
    S = mu * (I - C_inv) + lmbda * log_J * C_inv
    P = F * S
    return P

# --- Weak Form Components ---
# Internal force contribution to RHS vector (using u_curr = u at time n)
P_curr = pk1_stress(u_curr, mu, lmbda)
internal_force_form = ufl.inner(P_curr, ufl.grad(v)) * ufl.dx

# External force contribution to RHS vector
external_force_form = ufl.dot(f_random, v) * ufl.dx

# Mass matrix contribution (LHS)
mass_form_lhs = rho * ufl.dot(u_trial, v) * ufl.dx

# Previous steps contribution to RHS vector
# M * (2u^n - u^{n-1}) -> integral[ rho * (2u_curr - u_prev) * v ] dx
mass_contrib_rhs_form = rho * ufl.dot(2.0 * u_curr - u_prev, v) * ufl.dx

# Combined RHS form (scaled by dt^2 later)
# L = dt^2 * ( F_ext - K_int ) + M*(2u^n - u^{n-1})
# Note: K_int is represented by assembly of internal_force_form
# We assemble components separately for clarity

# --- Boundary Conditions ---
def fixed_boundary(x):
    return np.isclose(x[0], x_min_mesh)
fixed_dofs = dolfinx.fem.locate_dofs_geometrical(V, fixed_boundary)
u_D_zeros = dolfinx.fem.Function(V) # Function for BC value (0)
u_D_zeros.x.array[:] = 0.0
# u_D_zeros = np.zeros(mesh.geometry.dim, dtype=default_scalar_type) # Old way?
bc = dolfinx.fem.dirichletbc(u_D_zeros, fixed_dofs)
# --- Assemble Mass Matrix (Constant in time) ---
print("Assembling mass matrix...")
A = dolfinx.fem.petsc.assemble_matrix(fem.form(mass_form_lhs), bcs=[bc])
A.assemble()
print("Mass matrix assembly done.")

# --- Setup Linear Solver ---
solver = PETSc.KSP().create(mesh.comm)
solver.setOperators(A) # Set matrix A for the solver
solver.setType(PETSc.KSP.Type.CG) # Conjugate Gradient (good for SPD matrices like M)
solver.getPC().setType(PETSc.PC.Type.JACOBI) # Simple preconditioner (Jacobi)
# Other options: SOR, GAMG, HYPRE (if installed)
# solver.setType(PETSc.KSP.Type.PREONLY) # Use with direct solver PC
# solver.getPC().setType(PETSc.PC.Type.LU) # If MUMPS/SuperLU is available
solver.setTolerances(rtol=1e-8, atol=1e-10) # Tolerances for linear solve


# --- Initialization ---
t = 0.0
# Set initial conditions u(0) = 0, v(0) = 0
u_curr.x.array[:] = 0.0
u_prev.x.array[:] = 0.0
# v0 = dolfinx.fem.Function(V) # Initial velocity (needed for first step)
# v0.x.array[:] = 0.0

# --- Calculate Initial Acceleration (for first step) ---
print("Calculating initial acceleration...")
# Need F_external(0) - K_internal(u(0))
# K_internal(u(0)=0) is 0 for NeoHookean assuming stress-free state at u=0
# So, M * a(0) = F_external(0)

# Apply initial random force (t=0)
f_random.x.array[:] = 0.0
if len(force_dof_indices) > 0:
    random_values = random_force_density_magnitude * (2 * np.random.rand(len(force_dof_indices)) - 1)
    force_vector_component = random_values * force_direction[force_component_index]
    f_random.x.array[force_dof_indices] = force_vector_component
f_random.x.scatter_forward()

# Assemble initial external force vector F_external(0)
L0_form = fem.form(external_force_form)
b_ext0 = dolfinx.fem.petsc.create_vector(L0_form) # Create vector matching form
dolfinx.fem.petsc.assemble_vector(b_ext0, L0_form)
# Apply BCs to RHS vector for initial acceleration solve (zero acceleration at fixed boundary)
dolfinx.fem.petsc.apply_lifting(b_ext0, [fem.form(mass_form_lhs)], [[bc]])
b_ext0.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.petsc.set_bc(b_ext0, [bc])

# Solve M * a0 = b_ext0
solver.solve(b_ext0, a0.x.petsc_vec)
a0.x.scatter_forward() # Sync result
print("Initial acceleration calculation done.")


# --- Calculate u^1 (Displacement at first time step) ---
# Using Taylor expansion: u^1 = u^0 + dt*v^0 + (dt^2/2)*a^0
# Since u^0 = 0 and v^0 = 0, u^1 = (dt^2/2)*a^0
u.x.array[:] = 0.5 * (dt**2) * a0.x.array

# Apply boundary conditions to u^1
dolfinx.fem.petsc.set_bc(u.x.petsc_vec, [bc])
u.x.scatter_forward()

# Update history for the main loop: u_prev is u^0, u_curr is u^1
# u_prev already holds u^0 (zeros)
u_curr.x.array[:] = u.x.array # u_curr now holds u^1


# Add this function to your code, perhaps near the other visualization code
def add_force_glyphs(plotter, dof_coordinates, force_dof_indices, force_component_index, 
                     force_values, force_direction, scale_factor=0.5):
    """
    Add glyphs representing forces to the visualization.
    
    Parameters:
        plotter: PyVista plotter object
        dof_coordinates: Array of DOF coordinates
        force_dof_indices: Indices of DOFs where forces are applied
        force_component_index: Index of force component (0=x, 1=y, 2=z)
        force_values: Values of forces at each DOF
        force_direction: Direction vector of forces
        scale_factor: Scaling factor for arrow size
    """
    if len(force_dof_indices) == 0:
        return
    
    # Get block size (typically 3 for 3D)
    block_size = 3
    
    # Extract coordinates of points where forces are applied
    # For each DOF index, determine the corresponding node index
    node_indices = force_dof_indices // block_size
    unique_nodes = np.unique(node_indices)
    
    # Get unique node coordinates
    force_points = dof_coordinates[unique_nodes * block_size].reshape(-1, 3)
    
    # Create force vectors (all in the same direction, scaled by magnitude)
    # Use the average force value for visualization clarity
    avg_force = np.mean(np.abs(force_values)) if len(force_values) > 0 else 1.0
    force_vectors = np.tile(force_direction * avg_force * scale_factor, (len(unique_nodes), 1))
    
    # Create a PolyData object for the force points
    point_cloud = pyvista.PolyData(force_points)
    point_cloud["vectors"] = force_vectors
    
    # Create glyphs (arrows)
    arrows = point_cloud.glyph(
        orient="vectors",
        scale=False,  # We've already scaled the vectors
        factor=1.0,
        geom=pyvista.Arrow(shaft_radius=0.05, tip_radius=0.15, tip_length=0.25)
    )
    
    # Add arrows to the plot with a distinct color
    plotter.add_mesh(arrows, color="red", label="Applied Forces")
    
    # Add a legend
    plotter.add_legend()




# --- Visualization Setup ---
topology, cell_types, x_initial = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, x_initial)
plotter = pyvista.Plotter(window_size=[800, 600])
plotter.open_gif("beam_random_force_cd.gif")
plotter.camera.zoom(1.5)




# Add initial state (t=0) to visualization
grid.points = x_initial + u_prev.x.array.reshape(x_initial.shape[0], 3) # u_prev = u(0)
u_magnitude = np.linalg.norm(u_prev.x.array.reshape(-1, 3), axis=1)
grid.point_data["Displacement"] = u_magnitude
plotter.add_mesh(grid.copy(), scalars="Displacement", show_edges=True, cmap="viridis", scalar_bar_args={'title': 'Displacement'})
plotter.title = f"Beam Deformation (Central Diff) - Time: {0.00:.2f}s"
plotter.write_frame()

grid.points = x_initial + u_prev.x.array.reshape(x_initial.shape[0], 3)
u_magnitude = np.linalg.norm(u_prev.x.array.reshape(-1, 3), axis=1)
grid.point_data["Displacement"] = u_magnitude
plotter.add_mesh(grid.copy(), scalars="Displacement", show_edges=True, cmap="viridis", 
                 scalar_bar_args={'title': 'Displacement'})

# Add initial force glyphs
initial_force_values = f_random.x.array[force_dof_indices]
add_force_glyphs(plotter, dof_coordinates, force_dof_indices, force_component_index,
                 initial_force_values, force_direction, 
                 scale_factor=L*0.05/random_force_density_magnitude)

plotter.title = f"Beam Deformation (Central Diff) - Time: {0.00:.2f}s"
plotter.write_frame()


# --- Time-Stepping Loop (Starts from n=1 to calculate u^2) ---
print("\n--- Starting Simulation (Central Difference) ---")
# Allocate vectors for RHS assembly
L_int_form = fem.form(internal_force_form)
L_ext_form = fem.form(external_force_form)
L_mass_rhs_form = fem.form(mass_contrib_rhs_form)
b_int = dolfinx.fem.petsc.create_vector(L_int_form)
b_ext = dolfinx.fem.petsc.create_vector(L_ext_form)
b_mass_rhs = dolfinx.fem.petsc.create_vector(L_mass_rhs_form)
b_rhs = dolfinx.fem.petsc.create_vector(L_int_form) # Combined RHS vector

t = dt # Current time is dt (we calculated u^1)
for n in range(1, num_steps): # Loop computes u^2, u^3, ...
    t += dt
    print(f"\nTime step {n+1}/{num_steps}, t = {t:.4f}")

    # --- Update Random Force ---
    f_random.x.array[:] = 0.0
    if len(force_dof_indices) > 0:
        random_values = random_force_density_magnitude * (2 * np.random.rand(len(force_dof_indices)) - 1)
        force_vector_component = random_values * force_direction[force_component_index]
        f_random.x.array[force_dof_indices] = force_vector_component
    f_random.x.scatter_forward()

    # --- Assemble RHS Components ---
    # Internal force: K_internal(u^n) = K_internal(u_curr)
    with b_int.localForm() as loc:
        loc.set(0) # Zero out local form
    dolfinx.fem.petsc.assemble_vector(b_int, L_int_form)
    b_int.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    # External force: F_external^n
    with b_ext.localForm() as loc:
        loc.set(0)
    dolfinx.fem.petsc.assemble_vector(b_ext, L_ext_form)
    b_ext.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    # Mass contribution: M * (2u^n - u^{n-1})
    with b_mass_rhs.localForm() as loc:
        loc.set(0)
    dolfinx.fem.petsc.assemble_vector(b_mass_rhs, L_mass_rhs_form)
    b_mass_rhs.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    # --- Combine RHS Vector ---
    # b = dt^2 * (b_ext - b_int) + b_mass_rhs
    b_rhs.zeroEntries() # Clear combined vector
    b_rhs.axpy(dt**2, b_ext)       # Add dt^2 * b_ext
    b_rhs.axpy(-(dt**2), b_int)    # Subtract dt^2 * b_int
    b_rhs.axpy(1.0, b_mass_rhs)    # Add b_mass_rhs

    # --- Apply Boundary Conditions to RHS ---
    # Apply lifting and set BC value for M * u^{n+1} = b
    dolfinx.fem.petsc.apply_lifting(b_rhs, [fem.form(mass_form_lhs)], [[bc]])
    b_rhs.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b_rhs, [bc])

    # --- Solve Linear System M * u^{n+1} = b ---
    try:
        start_solve = time.time()
        solver.solve(b_rhs, u.x.petsc_vec) # Solve for u^{n+1} (stored in u)
        u.x.scatter_forward() # Sync results across processes
        end_solve = time.time()
        print(f"  Linear solve iterations: {solver.getIterationNumber()} (Solve time: {end_solve - start_solve:.2f}s)")

    except Exception as e:
        print(f"!!! Solver failed at t={t:.4f} with exception: {e} !!!")
        # Possible reasons: Instability (dt too large), singular matrix (shouldn't happen for M), numerical issues
        break

    # --- Update Displacement History ---
    u_prev.x.array[:] = u_curr.x.array # u^{n-1} <- u^n
    u_curr.x.array[:] = u.x.array      # u^n <- u^{n+1}

    # --- Post-processing and Visualization ---
    u_magnitude = np.linalg.norm(u_curr.x.array.reshape(-1, 3), axis=1)
    max_displacement = np.max(u_magnitude) if u_magnitude.size > 0 else 0.0
    print(f"  Max displacement: {max_displacement:.4e}")

    # Stability check (optional but recommended)
    if max_displacement > 1e2 * L : # Arbitrary large displacement check
        print(f"!!! Instability suspected: Max displacement {max_displacement:.2e} > {1e2*L:.2e} at t={t:.4f} !!!")
        print("!!! Consider reducing dt. !!!")
        break

    # Update visualization periodically
 # Update visualization periodically
    if n % 20 == 0 or n == num_steps - 1:
        warped = grid.copy()
        warped.points = x_initial + u_curr.x.array.reshape(x_initial.shape[0], 3)
        warped.point_data["Displacement"] = u_magnitude

        plotter.clear()
        plotter.add_mesh(warped, scalars="Displacement", show_edges=True, cmap="viridis", 
                        scalar_bar_args={'title': 'Displacement'})
        
        # Add force visualization
        current_force_values = f_random.x.array[force_dof_indices]
        add_force_glyphs(plotter, dof_coordinates, force_dof_indices, force_component_index,
                        current_force_values, force_direction, 
                        scale_factor=L*0.05/random_force_density_magnitude)
        
        plotter.title = f"Beam Deformation (Central Diff) - Time: {t:.2f}s"
        plotter.write_frame()

# --- Finalization ---
plotter.close()
solver.destroy() # Clean up PETSc solver object

print("\n--- Simulation Finished ---")
print(f"Results saved to beam_random_force_cd.gif")