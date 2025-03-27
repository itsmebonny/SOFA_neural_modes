import numpy as np
import pyvista
from dolfinx import mesh, fem, io, plot
from dolfinx.fem import form, Function
from dolfinx.io import gmshio
from ufl import TrialFunction, TestFunction, inner, dx, grad, sym, Identity, div, dot, tr, Constant, ln, det, derivative, action
from mpi4py import MPI
from petsc4py import PETSc
import sys
from dolfinx.fem.petsc import assemble_matrix as assemble_matrix_petsc
from dolfinx.fem.petsc import assemble_vector as assemble_vector_petsc

from slepc4py import SLEPc

import scipy.sparse as sp


# --------------------
# Parameters
# --------------------
# Mesh options
use_gmsh = True  # Toggle between gmsh (.msh) and box mesh
mesh_file = "mesh/beam_732.msh"  # Path to .msh file if use_gmsh is True
# Material properties

E, nu = 10000, 0.35  # Example values for soft tissue
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
        domain, cell_tags, facet_tags = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, 0, gdim=3)
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
u_ = TrialFunction(V)
du = TestFunction(V)
print("Function space created")
tol = 1e-10
# Define Dirichlet boundary condition
def fixed_boundary(x):
    return np.isclose(x[0], x_min)

fixed_dofs = fem.locate_dofs_geometrical(V, fixed_boundary)
bc = fem.dirichletbc(np.zeros(3), fixed_dofs, V)

# --------------------
# Define strain and stress operators as neo-Hookean material model
# --------------------
def eps(v):
    return sym(grad(v))
    
#define linearized neo-hookean sigma
def sigma(v):
    dim = domain.geometry.dim
    return 2.0 * mu * eps(v) + lmbda * tr(eps(v)) * Identity(dim)

# --------------------
# Define the variational forms (matching cantilever_modal.py)
# --------------------
# Stiffness form
k_form = inner(sigma(du), eps(u_)) * dx

# Mass form
m_form = rho * dot(du, u_) * dx


# Number of eigenvalues to compute
N_eig = 5


# --------------------
# Define the variational forms (matching cantilever_modal.py)
# --------------------
# Stiffness form
k_form = inner(sigma(du), eps(u_)) * dx

# Mass form with reduced quadrature (matching SOFA's approach)
m_form = rho * dot(du, u_) * dx(metadata={"quadrature_degree": 1})


# Number of eigenvalues to compute
N_eig = 12


# --------------------
# Assemble system matrices (using SOFA-compatible approach)
# --------------------
print("Assembling stiffness matrix without BCs")
# First assemble WITHOUT boundary conditions
A = assemble_matrix_petsc(form(k_form))  # No BCs passed
A.assemble()

print("Assembling mass matrix without BCs")
M_consistent = assemble_matrix_petsc(form(m_form))
M_consistent.assemble()

# For analysis, convert to scipy for matching SOFA's approach
def petsc_to_scipy(petsc_mat):
    ai, aj, av = petsc_mat.getValuesCSR()
    return sp.csr_matrix((av, aj, ai), shape=petsc_mat.getSize())

# Convert to scipy for manually applying boundary conditions
A_scipy = petsc_to_scipy(A)
M_scipy_consistent = petsc_to_scipy(M_consistent)

print(f"Mass matrix shape: {M_scipy_consistent.shape}")
print(f"Stiffness matrix shape: {A_scipy.shape}")
print(f"Number of fixed DOFs: {len(fixed_dofs)}")

# Manually apply boundary conditions (SOFA style)
print("Manually applying boundary conditions to match SOFA")
for dof in fixed_dofs:
    # For each coordinate (x,y,z)
    for d in range(3):
        # Get the global DOF index
        global_dof = dof * 3 + d
        
        # Zero out row and column in stiffness matrix
        A_scipy[global_dof, :] = 0
        A_scipy[:, global_dof] = 0
        A_scipy[global_dof, global_dof] = 1.0
        
        # Zero out row and column in mass matrix
        M_scipy_consistent[global_dof, :] = 0
        M_scipy_consistent[:, global_dof] = 0

print("Boundary conditions applied manually")


def compute_hybrid_mass_matrix(M_consistent, lumping_ratio=0.4):
    """Create a hybrid mass matrix blending lumped and consistent approaches"""
    # Create a fully lumped mass matrix (diagonal only)
    M_lumped = sp.lil_matrix(M_consistent.shape)
    for i in range(M_consistent.shape[0]):
        M_lumped[i, i] = M_consistent[i, :].sum()
    
    # Convert to CSR for efficient operations
    M_lumped = M_lumped.tocsr()
    
    # Blend the matrices
    M_hybrid = M_lumped * lumping_ratio + M_consistent * (1 - lumping_ratio)
    
    # Preserve total mass
    total_mass_consistent = np.sum(M_consistent.data)
    total_mass_hybrid = np.sum(M_hybrid.data)
    M_hybrid = M_hybrid * (total_mass_consistent / total_mass_hybrid)
    
    return M_hybrid 

# Convert to scipy for processing
M_scipy_consistent = petsc_to_scipy(M_consistent)
print(f"Mass matrix shape: {M_scipy_consistent.shape}")


# Compute hybrid mass matrix
M_hybrid = compute_hybrid_mass_matrix(M_scipy_consistent, lumping_ratio=0.4)
print(f"Hybrid mass matrix diagonal percentage: {np.sum(M_hybrid.diagonal()) / np.sum(M_hybrid.data) * 100:.2f}%")

# Convert back to PETSc for eigensolver
A_petsc = PETSc.Mat().createAIJ(size=A_scipy.shape, 
                               csr=(A_scipy.indptr, A_scipy.indices, A_scipy.data))
M_hybrid_petsc = PETSc.Mat().createAIJ(size=M_hybrid.shape, 
                                      csr=(M_hybrid.indptr, M_hybrid.indices, M_hybrid.data))



print("Creating main eigensolver")
 # Set up SLEPc eigensolver
eigensolver = SLEPc.EPS().create()
eigensolver.setOperators(A_petsc, M_hybrid_petsc)
eigensolver.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
eigensolver.setTarget(0.0)

st = eigensolver.getST()
st.setType(SLEPc.ST.Type.SINVERT)
st.setShift(0.0)
            
# Set dimensions and solve
# Set explicit dimensions to increase chances of convergence
eigensolver.setDimensions(N_eig*2, PETSc.DECIDE)  # Request more to ensure we get enough
eigensolver.setFromOptions()

# Increase max iterations and tolerance
eigensolver.setTolerances(tol=1e-6, max_it=1000)


# Print solver settings for debugging
print(f"Solver type: {eigensolver.getType()}")
print(f"Target value: {eigensolver.getTarget()}")
print(f"Dimensions requested: {eigensolver.getDimensions()[0]}")
print(f"Convergence test: {eigensolver.getConvergenceTest()}")



print("Solving eigenvalue problem...")
eigensolver.solve()
print("Eigenvalue problem solved")

# --------------------
# Extract results
# --------------------
eigenvalues = []
eigenvectors = []

print("Extracting eigenvalues and eigenvectors")
nconv = eigensolver.getConverged()
print(f"Number of converged eigenvalues: {nconv}")
N_eig = min(N_eig, nconv)

for i in range(min(N_eig, nconv)):
    eigenvalue = eigensolver.getEigenvalue(i)
    eigenvalues.append(eigenvalue)
    frequency = np.sqrt(np.real(eigenvalue)) / (2 * np.pi)  # Convert to Hz
    print(f"Eigenvalue {i+1}: {eigenvalue}, Frequency: {frequency:.2f} Hz")

    # Extract eigenvector
    vr = A.createVecRight()
    eigensolver.getEigenvector(i, vr)
    eigenvectors.append(vr)

print("Eigenvalues and eigenvectors extracted")

# Stack eigenvectors as columns in a matrix for modal analysis
modal_matrix = np.column_stack([vr.array for vr in eigenvectors])
shape = modal_matrix.shape
print(f"Modal matrix shape: {shape}")

# --------------------
# Modal participation factors calculation - FIXED VERSION
# --------------------
print("\nCalculating modal participation factors for Y-direction displacement")
u = fem.Function(V)

# Instead of using lambda function interpolation, set values directly
# Create an array of the right shape and fill it with y-direction unit vectors
u_array = np.zeros(V.dofmap.index_map.size_local * 3)
u_array = u_array.reshape(-1, 3)
u_array[:, 1] = 1.0  # Set y component to 1.0
u_array = u_array.flatten()

# Set the values directly
u.x.array[:] = u_array[:len(u.x.array)]

# Calculate total mass
total_mass = fem.assemble_scalar(form(rho * dx(domain=domain)))
print(f"Total mass of the structure: {total_mass:.6e}")

combined_mass = 0
for i, vr in enumerate(eigenvectors):
    # Convert eigenmode to Function
    xi = fem.Function(V)
    xi.x.array[:] = vr.array
    
    # Calculate modal participation factor
    qi_form = rho * dot(xi, u) * dx
    qi = fem.assemble_scalar(form(qi_form))
    
    # Calculate modal mass
    mi_form = rho * dot(xi, xi) * dx
    mi = fem.assemble_scalar(form(mi_form))
    
    # Calculate effective mass
    meff_i = qi**2 / mi if mi > 0 else 0.0
    
    print("-" * 50)
    print(f"Mode {i+1}:")
    print(f"  Modal participation factor: {qi:.2e}")
    print(f"  Modal mass: {mi:.4f}")
    print(f"  Effective mass: {meff_i:.2e}")
    print(f"  Relative contribution: {100 * meff_i / total_mass:.2f}%")
    
    combined_mass += meff_i

print(f"\nTotal relative mass of the first {N_eig} modes: {100 * combined_mass / total_mass:.2f}%")
# --------------------
# Visualization with PyVista
# --------------------
def visualize_eigenmodes(domain, V, eigenvectors, eigenvalues, num_modes=5):
    # Create function space for visualization
    V_viz = fem.functionspace(domain, ("CG", 1, (3,)))
    
    topology, cell_types, geometry = plot.vtk_mesh(domain)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    for mode in range(min(num_modes, len(eigenvectors))):
        # Convert eigenvector to Function
        eigenfunction = fem.Function(V)
        eigenfunction.x.array[:] = np.real(eigenvectors[mode].array)
        
        # Project onto visualization space
        eigenfunction_viz = fem.Function(V_viz)
        eigenfunction_viz.interpolate(eigenfunction)
        
        # Add displacement field as point data
        displacement = eigenfunction_viz.x.array.reshape(-1, 3)
        grid.point_data["displacement"] = displacement
        
        # Scale factor based on maximum displacement
        max_disp = np.max(np.linalg.norm(displacement, axis=1))
        scale = 0.1 / max(max_disp, 1e-10)
        
        # Warp the mesh using the added displacement field
        warped = grid.warp_by_vector("displacement", factor=0.1)

        # Calculate frequency from eigenvalue
        frequency = np.sqrt(np.real(eigenvalues[mode])) / (2 * np.pi)  # Convert to Hz

        # Plot the warped mesh using PyVista
        plotter = pyvista.Plotter()
        plotter.add_mesh(warped, scalars=np.linalg.norm(displacement, axis=1), 
                cmap="viridis", show_edges=True)
        plotter.add_title(f"Mode {mode + 1}, f = {frequency:.5f} Hz")
        plotter.show_axes_all()
        plotter.show()

print("Visualizing eigenmodes")
visualize_eigenmodes(domain, V, eigenvectors, eigenvalues, num_modes=N_eig)
print("Eigenmodes visualized")

# --------------------
# Visualization of a mode given a latent vector
# --------------------
def visualize_from_latent(domain, V, eigenvectors, latent_vector, scale=0.1, title="Deformation from Latent Vector"):
    """
    Visualize the linear deformation corresponding to a given latent vector.
    
    Args:
        domain: The mesh domain
        V: Function space
        eigenvectors: List of eigenvectors
        latent_vector: Vector of weights for each mode (numpy array or list)
        scale: Scaling factor for visualization
        title: Plot title
    """
    # Validate input
    if len(latent_vector) > len(eigenvectors):
        print(f"Warning: Latent vector has {len(latent_vector)} components, but only {len(eigenvectors)} eigenmodes available")
        latent_vector = latent_vector[:len(eigenvectors)]
    
    # Create function for combined deformation
    combined_deformation = fem.Function(V, name="Combined Mode")
    
    # Initialize with zeros
    combined_deformation.x.array[:] = 0.0
    
    # Combine eigenmodes according to latent vector
    for i, (vec, weight) in enumerate(zip(eigenvectors, latent_vector)):
        combined_deformation.x.array[:] += weight * vec.array
        print(f"Added mode {i+1} with weight {weight:.4f}")
    
    # Create visualization function space
    V_viz = fem.functionspace(domain, ("CG", 1, (3,)))
    
    # Project onto visualization space
    deformation_viz = fem.Function(V_viz)
    deformation_viz.interpolate(combined_deformation)
    
    # Create PyVista grid
    topology, cell_types, geometry = plot.vtk_mesh(domain)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    
    # Add displacement field as point data
    displacement = deformation_viz.x.array.reshape(-1, 3)
    grid.point_data["displacement"] = displacement
    
    # Scale factor based on maximum displacement
    max_disp = np.max(np.linalg.norm(displacement, axis=1))
    if max_disp > 1e-10:
        adaptive_scale = scale / max_disp
    else:
        adaptive_scale = scale
        
    print(f"Maximum displacement: {max_disp:.4f}")
    
    # Warp the mesh using the added displacement field
    warped = grid.warp_by_vector("displacement", factor=1)

    # Plot the warped mesh using PyVista
    plotter = pyvista.Plotter()
    plotter.add_mesh(warped, scalars=np.linalg.norm(displacement, axis=1), 
            cmap="viridis", show_edges=True)
    plotter.add_mesh(grid, style="wireframe", color="black", opacity=0.3)  # Original mesh
    plotter.add_title(title)
    plotter.show_axes_all()
    plotter.show()
    
    return combined_deformation

# Define a latent vector to visualize a specific mode
latent_vector = np.random.randn(N_eig) * 5



print(f"Latent vector: {latent_vector}")

print("Visualizing deformation from latent vector")
visualize_from_latent(domain, V, eigenvectors, latent_vector, scale=0.1, title="Deformation from Latent Vector")
print("Deformation visualized")

def compute_neohookean_energy(domain, V, displacement, mu=None, lmbda=None):
    """
    Compute the Neo-Hookean strain energy for a given displacement field.
    
    Args:
        domain: FEniCSx mesh domain
        V: Function space for the displacement field
        displacement: Displacement field as FEniCSx Function or numpy array
        mu: First Lamé parameter (shear modulus). If None, uses global value.
        lmbda: Second Lamé parameter. If None, uses global value.
        
    Returns:
        total_energy: Total strain energy (scalar)
        energy_density: Function representing strain energy density field
    """
    # Use global material parameters if not provided
    if mu is None:
        mu = globals()['mu']
    if lmbda is None:
        lmbda = globals()['lmbda']
    
    # Convert displacement to FEniCSx Function if numpy array
    if isinstance(displacement, np.ndarray):
        u = fem.Function(V)
        u.x.array[:] = displacement
    else:
        u = displacement
    
    # Define Neo-Hookean energy density
    dim = 3  # 3D problem
    F = Identity(dim) + grad(u)      # Deformation gradient
    C = F.T * F                      # Right Cauchy-Green tensor
    Ic = tr(C)                       # First invariant
    J = det(F)                       # Volume ratio
    
    # Neo-Hookean strain energy density (volumetric + isochoric parts)
    psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2
    
    # Create UFL form for the total energy
    energy_form = psi * dx
    
    # Assemble to get the total energy
    total_energy = fem.assemble_scalar(form(energy_form))
    
    # Create a function to visualize energy density
    energy_density_space = fem.functionspace(domain, ("CG", 1))
    energy_density = fem.Function(energy_density_space, name="Energy Density")
    
    # Project the energy density onto the scalar function space
    # In newer FEniCSx versions, we need a direct projection approach
    # Create projection problem: find energy_density such that inner(v, energy_density) = inner(v, psi) for all test functions v
    v = TestFunction(energy_density_space)
    u_trial = TrialFunction(energy_density_space)
    
    a = inner(v, u_trial) * dx
    L = inner(v, psi) * dx
    
    # Solve the projection problem
    problem = fem.petsc.LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "cg"})
    energy_density = problem.solve()
    
    return total_energy, energy_density

def visualize_energy_density(domain, energy_density, title="Neo-Hookean Energy Density"):
    """
    Visualize the energy density field on the mesh.
    
    Args:
        domain: FEniCSx mesh domain
        energy_density: Energy density function
        title: Plot title
    """
    # Create PyVista grid
    topology, cell_types, geometry = plot.vtk_mesh(domain)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    
    # Add energy density as point data
    energy_values = energy_density.x.array
    grid.point_data["energy_density"] = energy_values
    
    # Plot the mesh with energy density
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, scalars="energy_density", cmap="plasma", show_edges=True)
    plotter.add_title(title)
    plotter.show_axes_all()
    plotter.add_scalar_bar(title="Energy Density")
    plotter.show()

# Example usage
print("\nTesting Neo-Hookean energy calculation...")
# Use the latent vector visualization to generate a test displacement
u = visualize_from_latent(domain, V, eigenvectors, latent_vector)

# Compute and visualize energy
total_energy, energy_density = compute_neohookean_energy(domain, V, u)
print(f"Total Neo-Hookean energy: {total_energy:.6e}")
visualize_energy_density(domain, energy_density, title=f"Energy Density (Total: {total_energy:.4e})")