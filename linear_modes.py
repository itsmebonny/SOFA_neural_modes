import numpy as np
import pyvista
from dolfinx import mesh, fem, io, plot
from dolfinx.fem import form, assemble_matrix
from dolfinx.io import gmshio
from ufl import TrialFunction, TestFunction, inner, dx, grad, sym, Identity, div
from mpi4py import MPI
from petsc4py import PETSc
#import gmsh # No longer needed
import sys
from dolfinx.fem.petsc import assemble_matrix as assemble_matrix_petsc
from slepc4py import SLEPc
from dolfinx.nls.petsc import NewtonSolver

# --------------------
# Parameters
# --------------------
# Mesh options
use_gmsh = True  # Toggle between gmsh (.msh) and box mesh
mesh_file = "mesh/beam_615.msh"  # Path to .msh file if use_gmsh is True
# Material properties

E, nu = 1e6, 0.4  # Example values for soft tissue
rho = 1000

# Lame's constants
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

# Box mesh parameters (used if use_gmsh is False)
l_x, l_y, l_z = 1.0, 1.0, 0.01  # Domain dimensions
n_x, n_y, n_z = 20, 20, 2  # Number of elements

# --------------------
# Geometry
# --------------------
def create_fenicsx_mesh(l_x, l_y, l_z, n_x, n_y, n_z):
    # Use dolfinx.mesh.create_box directly
    domain = mesh.create_box(MPI.COMM_WORLD,
                           [np.array([0.0, 0.0, 0.0]), np.array([l_x, l_y, l_z])],
                           [n_x, n_y, n_z],
                           cell_type=mesh.CellType.hexahedron)
    return domain

# Create or load mesh
domain = None
cell_tags = None
facet_tags = None

if use_gmsh:
    print(f"Reading mesh from {mesh_file}")
    try:
        domain, cell_tags, facet_tags = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, 0, gdim=3,)
        print("Mesh loaded from file")
    except Exception as e:
        print(f"Error loading mesh: {e}")
        print("Falling back to box mesh")
        domain = None
else:
    print("Creating box mesh")
    domain = create_fenicsx_mesh(l_x, l_y, l_z, n_x, n_y, n_z)
    print("Box mesh created")

if domain is None:
    print("No valid mesh loaded. Exiting.")
    sys.exit(1)

# Get domain extents
x_coords = domain.geometry.x
x_min = x_coords[:, 0].min()
x_max = x_coords[:, 0].max()
y_min = x_coords[:, 1].min()
y_max = x_coords[:, 1].max()
z_min = x_coords[:, 2].min()
z_max = x_coords[:, 2].max()
print(f"Domain extents: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}], z=[{z_min}, {z_max}]")

# --------------------
# Function spaces
# --------------------
V = fem.functionspace(domain, ("CG", 1, (3,)))
u_tr = TrialFunction(V)
u_test = TestFunction(V)
print("Function space created")

# Define Dirichlet boundary condition
def fixed_boundary(x):
    return np.isclose(x[0], x_min)

fixed_dofs = fem.locate_dofs_geometrical(V, fixed_boundary)
bc = fem.dirichletbc(np.zeros(3), fixed_dofs, V)

# --------------------
# Functions
# --------------------
def epsilon(u):
    return sym(grad(u))

def sigma(u):
    return lmbda * div(u) * Identity(3) + 2 * mu * epsilon(u)

# --------------------
# Forms & matrices
# --------------------
a_form = inner(sigma(u_tr), epsilon(u_test)) * dx
m_form = rho * inner(u_tr, u_test) * dx

print("Assembling A matrix")
A = assemble_matrix_petsc(form(a_form))#, bcs=[bc])
A.assemble()
print("Assembling M matrix")
M = assemble_matrix_petsc(form(m_form))#, bcs=[bc])
M.assemble()
print("Matrices assembled")

# --------------------
# Eigen-solver
# --------------------
print("Creating eigensolver")
eigensolver = SLEPc.EPS().create(domain.comm)
eigensolver.setOperators(A, M)

N_eig = 3 # number of eigenvalues to match FEniCS example


# Match FEniCS settings
eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)  # Changed to target magnitude
eigensolver.setTarget(0.0)  # Set target to 0.0 for smallest eigenvalues
st = eigensolver.getST()
st.setType(SLEPc.ST.Type.SINVERT)  # 'shift-and-invert'
st.setShift(100.0)  # 'spectral_shift': 100.0

eigensolver.setDimensions(N_eig)  # Set number of eigenvalues to compute
eigensolver.setFromOptions()

print("Solving eigenvalue problem")
eigensolver.solve()
print("Eigenvalue problem solved")

# --------------------
# Post-process
# --------------------
# Eigenfrequencies and visualization
num_modes = N_eig
eigenvalues = []
eigenvectors = []

print("Extracting eigenvalues and eigenvectors")
# Get number of converged eigenvalues
nconv = eigensolver.getConverged()
print(f"Number of converged eigenvalues: {nconv}")

for i in range(min(num_modes, nconv)):
    # Get i-th eigenvalue and eigenvector
    eigenvalue = eigensolver.getEigenvalue(i)
    eigenvalues.append(eigenvalue)
    frequency = np.sqrt(np.real(eigenvalue)) / (2 * np.pi)  # Convert to Hz
    print(f"Eigenvalue {i+1}: {eigenvalue}, Frequency: {frequency:.2f} Hz")

    # Extract eigenvector
    vr = A.createVecRight()

    # Get the eigenvector
    eigensolver.getEigenvector(i, vr)
    eigenvectors.append(vr)

print("Eigenvalues and eigenvectors extracted")

# Stack eigenvectors as columns in a matrix for modal analysis
modal_matrix = np.column_stack([vr.array for vr in eigenvectors])
shape = modal_matrix.shape
print(f"Modal matrix shape: {shape}")

# Function to compute modal coordinates for a given displacement field
def compute_modal_coordinates(u, modal_matrix, M):
    """
    Compute modal coordinates using mass orthonormalization
    Args:
        u: displacement field as numpy array
        modal_matrix: matrix containing eigenvectors as columns
        M: mass matrix
    Returns:
        q: modal coordinates
    """
    # Convert to PETSc vector
    u_vec = PETSc.Vec().createWithArray(u)
    
    # Initialize vector for modal coordinates
    q = np.zeros(modal_matrix.shape[1])
    
    # Compute modal coordinates using mass orthonormalization
    for i in range(modal_matrix.shape[1]):
        phi_i = PETSc.Vec().createWithArray(modal_matrix[:, i])
        Mphi = M.createVecLeft()
        M.mult(phi_i, Mphi)
        q[i] = u_vec.dot(Mphi) / phi_i.dot(Mphi)
    
    return q

# Example: Compute modal coordinates for a given displacement
def analyze_displacement(u_h, modal_matrix, M):
    """
    Analyze a displacement field in terms of modal coordinates
    Args:
        u_h: DOLFINx function containing displacement
        modal_matrix: matrix containing eigenvectors as columns
        M: mass matrix
    """
    # Get displacement array
    u = u_h.x.array
    
    # Compute modal coordinates
    q = compute_modal_coordinates(u, modal_matrix, M)
    
    # Print contribution of each mode
    print("\nModal analysis results:")
    for i, qi in enumerate(q):
        freq = np.sqrt(np.real(eigenvalues[i])) / (2 * np.pi)
        print(f"Mode {i+1} (f = {freq:.2f} Hz): {qi:.3e}")
    
    return q

# Visualization with PyVista
def visualize_eigenmodes(domain, V, eigenvectors, num_modes=3):
    # Create first order function space for visualization
    V_viz = fem.functionspace(domain, ("CG", 1, (3,)))
    
    topology, cell_types, geometry = plot.vtk_mesh(domain)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    for mode in range(num_modes):
        # First get the eigenmode in the original space
        eigenvector = eigenvectors[mode]
        eigenfunction = fem.Function(V)
        eigenfunction.x.array[:] = np.real(eigenvector.array)
        
        # Project onto visualization space
        eigenfunction_viz = fem.Function(V_viz)
        eigenfunction_viz.interpolate(eigenfunction)
        
        # Add displacement field as point data
        displacement = eigenfunction_viz.x.array.reshape(-1, 3)
        grid.point_data["displacement"] = displacement
        
        # Warp the mesh using the added displacement field
        warped = grid.warp_by_vector("displacement", factor=1.5)

        # Calculate frequency from eigenvalue
        frequency = np.sqrt(np.real(eigenvalues[mode])) / (2 * np.pi)  # Convert to Hz

        # Plot the warped mesh using PyVista
        plotter = pyvista.Plotter()
        plotter.add_mesh(warped, scalars=np.linalg.norm(displacement, axis=1), 
                cmap="viridis", show_edges=True)
        plotter.add_title(f"Mode {mode + 1}, f = {frequency:.5f} Hz")
        plotter.show()

print("Visualizing eigenmodes")
visualize_eigenmodes(domain, V, eigenvectors, num_modes=N_eig)
print("Eigenmodes visualized")



# --------------------
# Static force problem: Cantilever beam
# --------------------
print("\nSolving cantilever beam problem...")

# First make sure we've defined the strain and stress correctly
def epsilon(u):
    return sym(grad(u))

def sigma(u):
    return lmbda * div(u) * Identity(3) + 2 * mu * epsilon(u)

# Create function space and functions
u = fem.Function(V, name="Displacement")
u_tr = TrialFunction(V)  # Need trial function for bilinear form
v = TestFunction(V)

# Reset the solution
u.x.array[:] = 0.0

# Get domain extents
x_coords = domain.geometry.x
x_min, x_max = x_coords[:, 0].min(), x_coords[:, 0].max()
y_min, y_max = x_coords[:, 1].min(), x_coords[:, 1].max()
z_min, z_max = x_coords[:, 2].min(), x_coords[:, 2].max()
print(f"Domain extents: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}], z=[{z_min}, {z_max}]")

# Then use these in your boundary definition:
def fixed_boundary(x):
    return np.isclose(x[0], x_min, atol=1e-4)

# Create boundary condition
fixed_dofs = fem.locate_dofs_geometrical(V, fixed_boundary)
bc = fem.dirichletbc(np.zeros(3), fixed_dofs, V)

# Create a point load at the free end
force_magnitude = 5  # Increased force for more visible displacement

# Create a distributed load on the free end
def free_end(x):
    return np.isclose(x[0], x_min, atol=1e-4)

# Create a force function that applies a point load
class PointLoad:
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, x):
        values = np.zeros((3, x.shape[1]))
        # Find points close to the center of the free end
        free_end_pts = np.isclose(x[0], x_max, atol=1e-4)
        center_y = (y_max + y_min) / 2
        center_z = z_max
        
        # Find the point closest to the center of the free end
        if np.any(free_end_pts):
            y_dists = np.abs(x[1, free_end_pts] - center_y)
            z_dists = np.abs(x[2, free_end_pts] - center_z)
            dists = y_dists + z_dists
            closest_idx = np.argmin(dists)
            
            # Apply force to all points on free end
            values[2, free_end_pts] = -self.magnitude / np.sum(free_end_pts)
        
        return values

# Create the force function
force_function = PointLoad(force_magnitude)
force = fem.Function(V)
force.interpolate(force_function)

# Define the linear variational problem (use trial function for bilinear form)
a = inner(sigma(u_tr), epsilon(v)) * dx  # Changed u to u_tr here
L = inner(force, v) * dx

# Create the linear problem
problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

# Solve the linear problem directly
print("Solving linear elasticity problem...")
u.x.array[:] = problem.solve().x.array  # Store result in our function
print("Solution complete")

# Print max displacement
u_max = np.max(np.abs(u.x.array))
print(f"Maximum displacement: {u_max:.6e}")

# Analyze modal participation
print("\nAnalyzing displacement in terms of modal coordinates...")
q = analyze_displacement(u, modal_matrix, M)

# Visualize the static displacement
def visualize_displacement(domain, u):
    V_viz = fem.functionspace(domain, ("CG", 1, (3,)))
    u_viz = fem.Function(V_viz)
    u_viz.interpolate(u)
    
    topology, cell_types, geometry = plot.vtk_mesh(domain)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    
    displacement = u_viz.x.array.reshape(-1, 3)
    grid.point_data["displacement"] = displacement
    warped = grid.warp_by_vector("displacement", factor=100)
    
    plotter = pyvista.Plotter()
    plotter.add_mesh(warped, scalars=np.linalg.norm(displacement, axis=1),
                    cmap="viridis", show_edges=True)
    plotter.add_title("Static Displacement from Point Force")
    plotter.show()

# Visualize the displacement
print("\nVisualizing static displacement...")
visualize_displacement(domain, u)
print("Visualization complete.")

# Also visualize the mode that has the highest participation
highest_mode = np.argmax(np.abs(q))
print(f"\nVisualizing highest participation mode ({highest_mode + 1})...")
visualize_eigenmodes(domain, V, [eigenvectors[highest_mode]], num_modes=1)


# --------------------
# Static force problem: 4-sided loading
# --------------------
print("\nSolving 4-sided loading problem...")

# Create function space and functions
u = fem.Function(V, name="Displacement")
u_tr = TrialFunction(V)  # Need trial function for bilinear form
v = TestFunction(V)

# Reset the solution
u.x.array[:] = 0.0

# Get domain extents
x_coords = domain.geometry.x
x_min, x_max = x_coords[:, 0].min(), x_coords[:, 0].max()
y_min, y_max = x_coords[:, 1].min(), x_coords[:, 1].max()
z_min, z_max = x_coords[:, 2].min(), x_coords[:, 2].max()
print(f"Domain extents: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}], z=[{z_min}, {z_max}]")

# Need a boundary condition to prevent rigid body motion
# Use the same boundary condition as the eigenvalue problem
def fixed_boundary(x):
    return np.isclose(x[0], x_min, atol=1e-4)

# Create boundary condition
fixed_dofs = fem.locate_dofs_geometrical(V, fixed_boundary)
bc = fem.dirichletbc(np.zeros(3), fixed_dofs, V)

# Force magnitude for each side
force_magnitudes = [35, -12, -15, 8]  # List of forces for [x_min, x_max, y_min, y_max]

# Create a force function that applies different forces to all 4 sides
class FourSideLoad:
    def __init__(self, magnitudes):
        # magnitudes should be a list/array with 4 values for [x_min, x_max, y_min, y_max]
        self.magnitudes = magnitudes
    def __call__(self, x):
        values = np.zeros((3, x.shape[1]))
        
        # Find points on each side
        x_min_pts = np.isclose(x[0], x_min, atol=1e-4)
        x_max_pts = np.isclose(x[0], x_max, atol=1e-4)
        y_min_pts = np.isclose(x[1], y_min, atol=1e-4)
        y_max_pts = np.isclose(x[1], y_max, atol=1e-4)
        
        # Count points on each side
        n_x_min = np.sum(x_min_pts)
        n_x_max = np.sum(x_max_pts)
        n_y_min = np.sum(y_min_pts)
        n_y_max = np.sum(y_max_pts)
        
        # Apply forces to each side (if points exist)
        if n_x_min > 0:
            values[2, x_min_pts] = self.magnitudes[0] / n_x_min  # x_min side
            
        if n_x_max > 0:
            values[2, x_max_pts] = self.magnitudes[1] / n_x_max  # x_max side
            
        if n_y_min > 0:
            values[2, y_min_pts] = self.magnitudes[2] / n_y_min  # y_min side
            
        if n_y_max > 0:
            values[2, y_max_pts] = self.magnitudes[3] / n_y_max  # y_max side
            
        return values

# Create the force function
force_function = FourSideLoad(force_magnitudes)
force = fem.Function(V)
force.interpolate(force_function)

# Define the linear variational problem
a = inner(sigma(u_tr), epsilon(v)) * dx
L = inner(force, v) * dx

# Create the linear problem
problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

# Solve the linear problem directly
print("Solving linear elasticity problem...")
u.x.array[:] = problem.solve().x.array
print("Solution complete")

# Print max displacement
u_max = np.max(np.abs(u.x.array))
print(f"Maximum displacement: {u_max:.6e}")

# Analyze modal participation
print("\nAnalyzing displacement in terms of modal coordinates...")
q = analyze_displacement(u, modal_matrix, M)

# Visualize the displacement with increased scaling
def visualize_displacement(domain, u, scale=100):
    V_viz = fem.functionspace(domain, ("CG", 1, (3,)))
    u_viz = fem.Function(V_viz)
    u_viz.interpolate(u)
    
    topology, cell_types, geometry = plot.vtk_mesh(domain)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    
    displacement = u_viz.x.array.reshape(-1, 3)
    grid.point_data["displacement"] = displacement
    warped = grid.warp_by_vector("displacement", factor=scale)
    
    plotter = pyvista.Plotter()
    plotter.add_mesh(warped, scalars=np.linalg.norm(displacement, axis=1),
                    cmap="viridis", show_edges=True)
    plotter.add_title("Static Displacement from 4-Side Loading")
    plotter.show()

# Visualize the displacement
print("\nVisualizing static displacement...")
visualize_displacement(domain, u, scale=1000)  # Increased scale for visibility
print("Visualization complete.")

# Also visualize the mode that has the highest participation
highest_mode = np.argmax(np.abs(q))
print(f"\nVisualizing highest participation mode ({highest_mode + 1})...")
visualize_eigenmodes(domain, V, [eigenvectors[highest_mode]], num_modes=1)