import dolfinx
import ufl
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import time
import pyvista
from dolfinx import plot, fem
from dolfinx.io import XDMFFile, VTKFile

import dolfinx.fem.petsc
import dolfinx.nls.petsc

# Simulation parameters
L = 10.0  # Length of the beam
r = 0.5   # Radius of the beam
E = 1.0e5  # Young's modulus
nu = 0.4  # Poisson's ratio
mu = E / (2.0 * (1.0 + nu))  # Shear modulus
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))  # First Lamé parameter
rho = 1000.0  # Density
load_magnitude = 1.0e4  # Magnitude of the stretching force
t_end = 10  # Total simulation time
num_steps = 1000
dt = t_end / num_steps  # Time step size
change_time = 2.5 # Time at which the load changes

#pretty print the parameters
print(f"Beam length: {L}")
print(f"Beam radius: {r}")
print(f"Young's modulus: {E}")
print(f"Poisson's ratio: {nu}")
print(f"Force magnitude: {load_magnitude}")
print(f"Total simulation time: {t_end}")
print(f"Number of time steps: {num_steps}")
print(f"Time step size: {dt}")
print(f"Time at which load changes: {change_time}")


# Create a 3D mesh for the beam
# mesh = dolfinx.mesh.create_box(
#     MPI.COMM_WORLD,
#     [np.array([0.0, -r, -r]), np.array([L, r, r])],
#     [20, 5, 5],
#     cell_type=dolfinx.mesh.CellType.hexahedron
# )

mesh_file ="mesh/beam_732.msh"
# Load mesh from file
from dolfinx.io import gmshio
mesh, cell_tags, facet_tags = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, 0, gdim=3)

# Calculate the actual center line of the mesh
x_coords = mesh.geometry.x
x_min, x_max = np.min(x_coords[:, 0]), np.max(x_coords[:, 0])
y_min, y_max = np.min(x_coords[:, 1]), np.max(x_coords[:, 1])
z_min, z_max = np.min(x_coords[:, 2]), np.max(x_coords[:, 2])
y_center = (y_min + y_max) / 2
z_center = (z_min + z_max) / 2

# Print the mesh info and center line
print(f"Mesh bounds: X: [{x_min:.4f}, {x_max:.4f}], Y: [{y_min:.4f}, {y_max:.4f}], Z: [{z_min:.4f}, {z_max:.4f}]")
print(f"Mesh center line at y={y_center:.4f}, z={z_center:.4f}")
print(f"Actual beam length: {x_max - x_min:.4f}")

# Use the actual beam length for calculations
beam_length = x_max - x_min

# Create function space (vector field for displacements)
V = dolfinx.fem.functionspace(mesh, ("CG", 1, (3,)))

# Define functions
u = dolfinx.fem.Function(V, name="Displacement")
u_prev = dolfinx.fem.Function(V, name="Displacement_prev")  # Previous time step
v = ufl.TestFunction(V)
u_dot = dolfinx.fem.Function(V, name="Velocity")
u_dot_prev = dolfinx.fem.Function(V, name="Velocity_prev")
u_ddot = dolfinx.fem.Function(V, name="Acceleration")

# Define the Neo-Hookean constitutive model
def neo_hookean_model(u):
    # Kinematics
    d = len(u)
    I = ufl.Identity(d)
    F = I + ufl.grad(u)              # Deformation gradient
    C = F.T * F                      # Right Cauchy-Green tensor
    Ic = ufl.tr(C)                   # First invariant
    J = ufl.det(F)                   # Volume ratio
    
    # Neo-Hookean strain energy density (incompressible)
    psi = (mu/2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda/2) * (ufl.ln(J))**2
    
    return psi

# Define the force application function - adjusted for actual center line
def apply_force(x, t):
    # Apply force in the x-direction
    force_dir = ufl.as_vector([1.0, 0.0, 0.0])
    
    # Magnitude that varies with time
    magnitude = load_magnitude
    
    # Ramp up force for the first 2 seconds
    if t < 2.0:
        magnitude *= (t / 2.0)  # Scale magnitude from 0 to 1 over 2 seconds
    
    if t >= change_time:
        magnitude *= 0.5  # Reduce after change_time
    
    return magnitude * force_dir

# Define the weak form using Neo-Hookean material and Newmark-beta with adjusted center
def weak_form(u, u_prev, u_dot_prev, u_ddot, v, t, dt, mesh, mu, lmbda, load_magnitude, L, beta, gamma):
    # Kinematics
    d = len(u)
    I = ufl.Identity(d)
    F = I + ufl.grad(u)              # Deformation gradient
    C = F.T * F                      # Right Cauchy-Green tensor
    Ic = ufl.tr(C)                   # First invariant
    J = ufl.det(F)                   # Volume ratio
    
    # Neo-Hookean strain energy density (incompressible)
    psi = (mu/2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda/2) * (ufl.ln(J))**2
    
    # Internal elastic energy
    internal_work = psi * ufl.dx
    
    # External work from force - applied at the free end
    x = ufl.SpatialCoordinate(mesh)
    
    # Define the force direction and magnitude
    force = apply_force(x, t)
    
    # Define the boundary where the force is applied (x = L)
    def end(x):
        return np.isclose(x[0], x_max)
    
    # Create a facet integration measure for the end boundary
    from dolfinx import mesh as dmesh
    end_facets = dmesh.locate_entities_boundary(mesh, mesh.topology.dim-1, end)
    from dolfinx.mesh import meshtags
    mt = meshtags(mesh, mesh.topology.dim-1, end_facets, np.ones(len(end_facets), dtype=np.int32))
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt, subdomain_id=1)
    
    # External work due to the force
    external_work = ufl.dot(force, v) * ds
    
    # Newmark-beta for dynamics
    inertia = ufl.dot(rho * u_ddot, v) * ufl.dx
    
    return ufl.derivative(internal_work, u, v) - external_work + inertia


# Boundary condition: fixed at the minimum x-value (instead of assuming x=0)
def fixed_boundary(x):
    return np.isclose(x[0], x_min)

# Create the boundary condition - fix all directions at fixed end
boundary_dofs = dolfinx.fem.locate_dofs_geometrical(V, fixed_boundary)
u_D = np.zeros(3, dtype=np.float64)  # 3D zero displacement
bc = dolfinx.fem.dirichletbc(u_D, boundary_dofs, V)

# Set up visualization
pyvista.set_jupyter_backend("static")
topology, cell_types, x = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, x)

# Setup plotter
plotter = pyvista.Plotter()
plotter.open_gif("beam_deformation.gif")

plotter.view_xz()

# Calculate the center of the beam
center = [(x_min + x_max) / 2, y_center, z_center]

# Set camera position to view the beam laterally and centered
plotter.camera_position = [
    (center[0], center[1] - beam_length*2, center[2]),  # Camera position
    center,  # Focal point (center of beam)
    (0, 0, 1)  # Up direction
]


beta = 0.25
gamma = 0.5

# Nonlinear solver parameters
solver_parameters = {
    "nonlinear_solver": "newton",
    "newton_solver": {
        "maximum_iterations": 50,
        "relative_tolerance": 1e-8,
        "linear_solver": "mumps"
    }
}

# Time-stepping loop
t = 0.0
for n in range(num_steps):
    print(f"\nTime step {n+1}/{num_steps}, t = {t+dt:.4f}")
    t += dt
    
    # Newmark-beta update
    u_ddot_expr = fem.Expression(u_ddot, V.element.interpolation_points())
    u_ddot.interpolate(u_ddot_expr)
    
    u_expr = fem.Expression(u, V.element.interpolation_points())
    u.interpolate(u_expr)
    
    u_dot_expr = fem.Expression(u_dot, V.element.interpolation_points())
    u_dot.interpolate(u_dot_expr)
    
    u_ddot.x.array[:] = (1 / (beta * dt**2)) * (u.x.array - u_prev.x.array) - (1 / (beta * dt)) * u_dot_prev.x.array - (1/(2*beta)-1) * u_ddot.x.array
    u_dot.x.array[:] = u_dot_prev.x.array + (1 - gamma) * dt * u_ddot.x.array + gamma * dt * u_ddot.x.array
    
    # Define the nonlinear problem
    try:
        F = weak_form(u, u_prev, u_dot_prev, u_ddot, v, t, dt, mesh, mu, lmbda, load_magnitude, beam_length, beta, gamma)
        J = ufl.derivative(F, u)  # Jacobian
        problem = dolfinx.fem.petsc.NonlinearProblem(F, u, [bc], J=J)
    
        # Create Newton solver
        solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
        solver.convergence_criterion = "residual"
        solver.rtol = 1e-8
        solver.max_it = 50
    
        # Solve the nonlinear problem
        n_its, converged = solver.solve(u)
        assert converged, "Newton solver did not converge"
        print(f"  Newton iterations: {n_its}")
    except Exception as e:
        print(f"Solver failed: {e}")
        break
    
    # Calculate maximum displacement
    u_values = u.x.array.reshape(-1, 3)
    max_displacement = np.max(np.linalg.norm(u_values, axis=1))
    print(f"  Max displacement: {max_displacement:.4f}")
    
    # Update previous solution
    u_prev.x.array[:] = u.x.array
    u_dot_prev.x.array[:] = u_dot.x.array
    
    # Visualize
    if n % 2 == 0:  # Update every other step to improve performance
        warped = grid.copy()
        warped.points[:] = x + u.x.array.reshape(-1, 3)
        
        # Calculate displacement magnitude for coloring
        displacement_magnitude = np.linalg.norm(u.x.array.reshape(-1, 3), axis=1)
        warped.point_data["Displacement"] = displacement_magnitude
        
        warped_surface = warped.extract_surface()
        
        plotter.clear()
        plotter.add_mesh(warped_surface, scalars="Displacement", show_edges=True)
        plotter.add_title(f"Time: {t:.2f}s, Max disp: {max_displacement:.4f}")
        
        # Add a color bar
        plotter.add_scalar_bar(title="Displacement", interactive=True)
        
        plotter.write_frame()
        

# Close visualization and files
plotter.close()

print("Simulation complete. Results saved to beam_deformation.gif")