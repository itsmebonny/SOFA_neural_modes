import numpy as np
import pyvista
from dolfinx import mesh, fem, io, plot
from dolfinx.fem import form, Function
from dolfinx.io import gmshio
# Import specific ufl functions needed
from ufl import (TrialFunction, TestFunction, inner, dx, grad, sym, Identity,
                 div, dot, tr, Constant, ln, det, derivative, action, Measure, SpatialCoordinate)


from mpi4py import MPI
from petsc4py import PETSc
import sys
# Use consistent import name for assembly
from dolfinx.fem.petsc import assemble_matrix as assemble_matrix_petsc
from dolfinx.fem.petsc import assemble_vector as assemble_vector_petsc

from slepc4py import SLEPc

# Import scipy sparse, needed for hybrid mass matrix if used
import scipy.sparse as sp

from dolfinx.fem.petsc import NonlinearProblem # Add NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver # Add NewtonSolver
import matplotlib.pyplot as plt # Add matplotlib


# Ensure all necessary imports from the previous block are present
import numpy as np
import pyvista
from dolfinx import mesh, fem, io, plot
from dolfinx.fem import form, Function, Constant # Add Constant
from dolfinx.io import gmshio
# Import specific ufl functions needed
from ufl import (TrialFunction, TestFunction, inner, dx, grad, sym, Identity,
                 div, dot, tr, Constant as ufl_Constant, ln, det, derivative, action, Measure, # Add Measure
                 SpatialCoordinate) # Add SpatialCoordinate
from mpi4py import MPI
from petsc4py import PETSc
import sys
# Use consistent import name for assembly
from dolfinx.fem.petsc import assemble_matrix as assemble_matrix_petsc
from dolfinx.fem.petsc import assemble_vector as assemble_vector_petsc
from slepc4py import SLEPc
# Import scipy sparse, needed for hybrid mass matrix if used
import scipy.sparse as sp

# Import nonlinear solver components
from dolfinx.fem.petsc import NonlinearProblem # Add NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver # Add NewtonSolver
import matplotlib.pyplot as plt # Add matplotlib
import ufl # Import ufl explicitly

# Logging (optional but helpful for debugging solver)
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("NonlinearSolve")

# --------------------
# Parameters
# --------------------
# Mesh options
use_gmsh = True  # Toggle between gmsh (.msh) and box mesh
mesh_file = "mesh/beam_732.msh"  # Path to .msh file if use_gmsh is True
# Material properties

E, nu = 10000, 0.45  # Example values for soft tissue
rho = 1

# Lame's constants
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

# Box mesh parameters (used if use_gmsh is False)
l_x, l_y, l_z = 1.0, 1.0, 0.01  # Domain dimensions
n_x, n_y, n_z = 20, 20, 2  # Number of elements

# Number of eigenvalues to compute
N_eig = 6

# Option to use hybrid mass matrix (set lumping_ratio > 0)
use_hybrid_mass = True # Set to False to use consistent mass matrix
lumping_ratio = 0.4 # Ratio of lumped mass (0=consistent, 1=fully lumped)


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
        # Each process reads the mesh file
        domain, cell_tags, facet_tags = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=3)
        print("Mesh loaded from file.")
    except Exception as e:
        print(f"Error loading mesh: {e}")
        print("Falling back to box mesh")
        domain = None
else:
    print("Creating box mesh")
    domain = create_fenicsx_mesh(l_x, l_y, l_z, n_x, n_y, n_z)
    print("Box mesh created")

if domain is None:
    if MPI.COMM_WORLD.rank == 0:
        print("No valid mesh loaded or created. Exiting.")
    sys.exit(1)

# Get domain extents (on rank 0 for printing)
if MPI.COMM_WORLD.rank == 0:
    x_coords = domain.geometry.x
    x_min = x_coords[:, 0].min()
    x_max = x_coords[:, 0].max()
    y_min = x_coords[:, 1].min()
    y_max = x_coords[:, 1].max()
    z_min = x_coords[:, 2].min()
    z_max = x_coords[:, 2].max()
    print(f"Domain extents: x=[{x_min:.4f}, {x_max:.4f}], y=[{y_min:.4f}, {y_max:.4f}], z=[{z_min:.4f}, {z_max:.4f}]")

# --------------------
# Function spaces
# --------------------
# Use CG 1 for simplicity, ensure degree matches mesh generation if applicable
element_degree = 1
V = fem.functionspace(domain, ("CG", element_degree, (domain.geometry.dim,)))
u_ = TrialFunction(V)
du = TestFunction(V)
print(f"Function space created (Degree {element_degree})")

# Define Dirichlet boundary condition
# Find x_min on this process's portion of the mesh geometry
local_x_coords = domain.geometry.x
local_x_min = np.min(local_x_coords[:, 0]) if local_x_coords.shape[0] > 0 else np.inf
global_x_min = domain.comm.allreduce(local_x_min, op=MPI.MIN)

print(f"Global x_min for BC: {global_x_min:.4f}")
tol = 1e-8 # Tolerance for floating point comparison

def fixed_boundary(x):
    # x is shape (3, num_points)
    return np.isclose(x[0], global_x_min, atol=tol)

# Locate DOFs on the geometry
fixed_dofs_geom = fem.locate_dofs_geometrical(V, fixed_boundary)
# Create BC object with zero displacement
bc_value = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0, 0.0)))
bc = fem.dirichletbc(bc_value, fixed_dofs_geom, V)
print(f"Dirichlet BC defined for x = {global_x_min:.4f}. Num local fixed DOFs: {len(fixed_dofs_geom)}")

# --------------------
# Define strain and stress operators (Linear Elasticity)
# --------------------
def eps(v):
    return sym(grad(v))

def sigma(v):
    # Standard isotropic linear elasticity
    return lmbda * div(v) * Identity(domain.geometry.dim) + 2.0 * mu * eps(v)

# --------------------
# Define the variational forms
# --------------------
print("Defining variational forms...")
# Stiffness form
k_form = inner(sigma(u_), eps(du)) * dx

# Mass form
# Reduced quadrature (degree 1) might match SOFA's lumped mass better if used
# Consistent mass uses default quadrature degree (usually 2*element_degree)
quad_deg_mass = 1 if use_hybrid_mass else element_degree * 2
m_form = rho * dot(u_, du) * dx(metadata={"quadrature_degree": quad_deg_mass})
print(f"Using quadrature degree {quad_deg_mass} for mass matrix assembly.")

# Compile forms
k_form_compiled = form(k_form)
m_form_compiled = form(m_form)

# --------------------
# Assemble system matrices (Corrected Approach)
# --------------------
print("Assembling stiffness matrix A (symmetric, no BCs)...")
# Assemble WITHOUT boundary conditions to preserve symmetry
A = assemble_matrix_petsc(k_form_compiled) # NO bcs=[bc] here!
A.assemble()
print("Stiffness matrix A assembled.")

# Assemble Mass Matrix M (also without BCs during assembly)
print("Assembling mass matrix M (symmetric, no BCs)...")
M = assemble_matrix_petsc(m_form_compiled) # NO bcs
M.assemble()
print("Mass matrix M assembled.")

# --- Optional: Hybrid Mass Matrix Calculation ---
if use_hybrid_mass:
    print(f"Computing hybrid mass matrix with lumping ratio {lumping_ratio}...")
    # Convert M to SciPy CSR for lumping calculations
    ai, aj, av = M.getValuesCSR()
    M_scipy_consistent = sp.csr_matrix((av, aj, ai), shape=M.getSize())

    # Create a fully lumped mass matrix (diagonal only)
    lumped_diag = np.array(M_scipy_consistent.sum(axis=1)).flatten()
    M_scipy_lumped = sp.diags(lumped_diag, format='csr')

    # Blend the matrices
    M_scipy_hybrid = M_scipy_lumped * lumping_ratio + M_scipy_consistent * (1 - lumping_ratio)

    # Preserve total mass (optional but recommended)
    total_mass_consistent = M_scipy_consistent.sum()
    total_mass_hybrid = M_scipy_hybrid.sum()
    if total_mass_hybrid > 1e-15: # Avoid division by zero
         mass_scaling = total_mass_consistent / total_mass_hybrid
         M_scipy_hybrid = M_scipy_hybrid * mass_scaling
         print(f"Hybrid mass matrix scaled by {mass_scaling:.4f} to preserve total mass.")
    else:
         print("Warning: Hybrid mass matrix has near-zero total mass, skipping scaling.")


    # Convert the final hybrid SciPy matrix back to PETSc Mat
    M_hybrid_petsc = PETSc.Mat().createAIJ(size=M_scipy_hybrid.shape,
                                    csr=(M_scipy_hybrid.indptr, M_scipy_hybrid.indices, M_scipy_hybrid.data))
    M_hybrid_petsc.assemble()
    print(f"Hybrid mass matrix created and converted back to PETSc.")
    M_petsc_final = M_hybrid_petsc # Use the hybrid matrix for the solver
else:
    M_petsc_final = M # Use the consistent mass matrix for the solver


# --- Apply Boundary Conditions Symmetrically AFTER Assembly ---
print("Applying boundary conditions symmetrically to A and M...")
# Get the list of DOF indices from the boundary condition object
# Ensure we get the indices correctly associated with the vector space V
constrained_dofs = bc.dof_indices()[0]

if constrained_dofs.size > 0:
    # Use PETSc's MatZeroRowsColumns to enforce BCs while preserving symmetry
    A.zeroRowsColumns(constrained_dofs, diag=1.0)
    # Zero rows/cols in M, setting diag=0.0 seems appropriate for mass
    M_petsc_final.zeroRowsColumns(constrained_dofs, diag=0.0)
    print(f"Applied MatZeroRowsColumns to {constrained_dofs.size} local DOFs.")
else:
    print("No constrained DOFs found on this process, skipping MatZeroRowsColumns.")

# --- Check Matrix Properties (Optional Debugging) ---
# Check symmetry using PETSc's built-in method (more reliable than SciPy checks)
is_A_sym = A.isSymmetric(tol=1e-9)
is_M_sym = M_petsc_final.isSymmetric(tol=1e-9)
print(f"Stiffness matrix A is symmetric (PETSc check): {is_A_sym}")
print(f"Final Mass matrix M is symmetric (PETSc check): {is_M_sym}")
if not is_A_sym or not is_M_sym:
     print("CRITICAL WARNING: Matrices are not symmetric after BC application! Solver assumptions violated.")
     # Consider exiting or adding more debug info here

# Note: Checking positive definiteness directly is computationally expensive.
# For linear elasticity with proper BCs, A should be SPD on the unconstrained subspace.
# M should be SPD if rho > 0 and the mesh is valid.


# --------------------
# Configure SLEPc Eigensolver
# --------------------
print("Configuring SLEPc eigensolver...")
eigensolver = SLEPc.EPS().create(comm=MPI.COMM_WORLD)
eigensolver.setOperators(A, M_petsc_final) # Use the symmetrically modified matrices
eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP) # Generalized Hermitian Eigenvalue Problem (requires symmetric A, M and M>0)

# Target the eigenvalues closest to zero (smallest magnitude)
# For a constrained system, these will be the smallest positive eigenvalues (lowest frequencies)
eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
eigensolver.setTarget(0.0) # Target exactly zero

# Use Shift-and-Invert spectral transformation - efficient for finding eigenvalues near the target
st = eigensolver.getST()
st.setType(SLEPc.ST.Type.SINVERT)
# st.setShift(0.0) # Implicit when target is 0.0

# Solver Settings
num_modes_to_request = N_eig + 5 # Request a few more than needed
eigensolver.setDimensions(num_modes_to_request, PETSc.DECIDE) # Set number of basis vectors
eigensolver.setTolerances(tol=1e-7, max_it=500) # Adjust tolerance and max iterations
eigensolver.setFromOptions() # Allow setting options from command line

# Print solver settings for confirmation
print(f"  Solver type: {eigensolver.getType()}")
print(f"  Problem type: GHEP")
print(f"  Target value: {eigensolver.getTarget():.2e}")
print(f"  Targeting type: TARGET_MAGNITUDE")
print(f"  Spectral Transform: SINVERT")
print(f"  Dimensions requested: {eigensolver.getDimensions()[0]}")
print(f"  Convergence tolerance: {eigensolver.getTolerances()[0]:.2e}")
print(f"  Max iterations: {eigensolver.getTolerances()[1]}")

# --------------------
# Solve the Eigenvalue Problem
# --------------------
print("Solving eigenvalue problem...")
try:
    eigensolver.solve()
    solve_successful = True
except Exception as e:
    print(f"SLEPc solve failed: {e}")
    solve_successful = False
    sys.exit(1) # Exit if solver fails

print("Eigenvalue problem solved.")

# --------------------
# Extract and Process Results
# --------------------
nconv = eigensolver.getConverged()
print(f"Number of converged eigenvalues: {nconv}")

if nconv == 0:
    print("Solver converged zero eigenvalues. Check problem setup, BCs, mesh, solver settings.")
    sys.exit(1)

eigenvalues = []
eigenvectors_petsc = [] # Store PETSc vectors
eigenvectors_np = [] # Store numpy arrays

print("Extracting eigenvalues and eigenvectors...")
# Create prototype PETSc vectors for extracting results
vr, vi = A.createVecs()

actual_modes_found = 0
for i in range(nconv):
    try:
        eigenvalue = eigensolver.getEigenpair(i, vr, vi) # vr = real part, vi = imag part
    except Exception as e:
        print(f"Error getting eigenpair {i}: {e}")
        continue

    # Check if eigenvalue is physically valid (should be > 0 for constrained system)
    # Use a small tolerance to avoid issues with near-zero numerical noise
    lambda_real = eigenvalue.real
    if lambda_real < 1e-9:
        print(f"  Skipping eigenvalue {i+1}: Near-zero or negative real part ({lambda_real:.4e}). Indicates potential RBM or issue.")
        continue

    # Store valid eigenvalue and eigenvector
    eigenvalues.append(lambda_real)
    eigenvectors_petsc.append(vr.copy()) # Important to copy the vector
    eigenvectors_np.append(vr.array.copy()) # Store numpy array version

    frequency_hz = np.sqrt(lambda_real) / (2 * np.pi)
    print(f"  Converged Eigenvalue {actual_modes_found+1}: {lambda_real:.6e} (Frequency: {frequency_hz:.4f} Hz)")
    actual_modes_found += 1

    if actual_modes_found >= N_eig:
        break # Stop once we have enough valid modes

print(f"Successfully extracted {actual_modes_found} valid eigenmodes.")

if actual_modes_found == 0:
    print("ERROR: Failed to extract any valid positive eigenvalues.")
    sys.exit(1)

# Adjust N_eig if fewer modes were found
N_eig = actual_modes_found

# Stack numpy eigenvectors into the modal matrix
modal_matrix = np.column_stack(eigenvectors_np)
print(f"Modal matrix shape: {modal_matrix.shape}")


# --------------------
# Modal participation factors calculation
# --------------------
print("\nCalculating modal participation factors for Y-direction displacement...")
u_rigid = fem.Function(V) # Function to represent rigid motion

# Create an array representing unit displacement in Y
rigid_motion_array = np.zeros((V.dofmap.index_map.size_local, domain.geometry.dim))
rigid_motion_array[:, 1] = 1.0 # Unit displacement in Y
# Set values into the function, handling potential size mismatches if V has ghosts
u_rigid.x.array[:rigid_motion_array.size] = rigid_motion_array.flatten()

# Calculate total mass using the final mass matrix used by the solver
# M_total * vec(1) dot vec(1) is related but difficult. Integrate rho*dx instead.
total_mass_integral = fem.assemble_scalar(form(rho * dx(domain=domain)))
print(f"Total mass computed by integral rho*dx: {total_mass_integral:.6e}")

# Define forms using the Functions
xi = fem.Function(V) # Placeholder for eigenvector function
mi_form = rho * dot(xi, xi) * dx(metadata={"quadrature_degree": quad_deg_mass})
qi_form = rho * dot(xi, u_rigid) * dx(metadata={"quadrature_degree": quad_deg_mass})

combined_effective_mass = 0
for i in range(N_eig):
    # Set eigenvector data into the Function xi
    xi.x.array[:] = eigenvectors_np[i] # Use the numpy array

    # Assemble modal mass and participation factor
    mi = fem.assemble_scalar(form(mi_form))
    qi = fem.assemble_scalar(form(qi_form))

    # Calculate effective mass (handle potential division by zero)
    meff_i = (qi**2) / mi if abs(mi) > 1e-15 else 0.0

    print("-" * 50)
    print(f"Mode {i+1} (Freq: {np.sqrt(eigenvalues[i])/(2*np.pi):.4f} Hz):")
    print(f"  Modal Participation Factor (qi): {qi:.4e}")
    print(f"  Modal Mass (mi): {mi:.4f}")
    print(f"  Effective Mass (Y-dir): {meff_i:.4e}")
    if total_mass_integral > 1e-15:
        print(f"  Relative Contribution (Y-dir): {100 * meff_i / total_mass_integral:.2f}%")
    combined_effective_mass += meff_i

print("-" * 50)
if total_mass_integral > 1e-15:
    print(f"Total Relative Effective Mass (Y-dir) of first {N_eig} modes: {100 * combined_effective_mass / total_mass_integral:.2f}%")
else:
    print("Total mass is near zero, cannot compute relative effective mass.")


# --------------------
# Visualization Functions (remain largely the same)
# --------------------
def visualize_eigenmodes(domain, V, eigenvectors_np, eigenvalues, num_modes_to_plot=5):
    print("Visualizing eigenmodes...")
    topology, cell_types, geometry = plot.vtk_mesh(V.mesh) # Use V.mesh for consistency
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    # Create a single function for interpolation (reuse)
    eigenfunction = fem.Function(V)

    num_modes_to_plot = min(num_modes_to_plot, len(eigenvectors_np))

    for mode_idx in range(num_modes_to_plot):
        # Set the numpy array data into the function
        eigenfunction.x.array[:] = eigenvectors_np[mode_idx]

        # Add displacement field as point data
        # Need to interpolate onto visualization nodes if using higher order elements,
        # but for CG1, node values correspond directly.
        # Ensure correct reshaping based on number of nodes in the grid
        num_points = grid.number_of_points
        displacement = eigenfunction.x.array[:num_points*domain.geometry.dim].reshape(num_points, domain.geometry.dim)
        grid.point_data["displacement"] = displacement
        grid.point_data["magnitude"] = np.linalg.norm(displacement, axis=1)

        # --- Determine reasonable warping ---
        # Heuristic: scale so max displacement is ~10% of characteristic length
        characteristic_length = np.linalg.norm(geometry.max(axis=0) - geometry.min(axis=0))
        max_disp_magnitude = np.max(grid.point_data["magnitude"])
        if max_disp_magnitude > 1e-9:
             warp_factor = 0.05 * characteristic_length / max_disp_magnitude
        else:
             warp_factor = 1.0 # No displacement

        warped = grid.warp_by_vector("displacement", factor=warp_factor)

        frequency = np.sqrt(eigenvalues[mode_idx]) / (2 * np.pi)

        # Plot using PyVista
        plotter = pyvista.Plotter(window_size=[800, 600])
        plotter.add_mesh(warped, scalars="magnitude", cmap="viridis", show_edges=True)
        plotter.add_mesh(grid, style="wireframe", color="grey", opacity=0.2) # Original mesh outline

        plotter.add_title(f"Mode {mode_idx + 1}, f = {frequency:.3f} Hz (Warp: {warp_factor:.1f}x)")
        plotter.show_axes_all()
        plotter.add_scalar_bar(title="Displacement Mag.")
        print(f"  Showing plot for Mode {mode_idx+1}...")
        plotter.show()
        print(f"  Plot closed.")

visualize_eigenmodes(domain, V, eigenvectors_np, eigenvalues, num_modes_to_plot=min(N_eig, nconv))


# --- Functions for latent vector visualization and energy calculation (remain similar) ---
# Note: Pass numpy eigenvector arrays `eigenvectors_np` to these functions

def visualize_from_latent(domain, V, eigenvectors_np_list, latent_vector, title="Deformation from Latent Vector"):
    """Visualize linear combination of modes."""
    if len(latent_vector) > len(eigenvectors_np_list):
        print(f"Warning: Latent vector size ({len(latent_vector)}) > num modes ({len(eigenvectors_np_list)}). Truncating.")
        latent_vector = latent_vector[:len(eigenvectors_np_list)]

    combined_deformation_array = np.zeros_like(eigenvectors_np_list[0])
    for weight, vec_np in zip(latent_vector, eigenvectors_np_list):
        combined_deformation_array += weight * vec_np

    # Create function and set data
    combined_function = fem.Function(V)
    combined_function.x.array[:] = combined_deformation_array

    # Visualization setup
    topology, cell_types, geometry = plot.vtk_mesh(V.mesh)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    num_points = grid.number_of_points
    displacement = combined_function.x.array[:num_points*domain.geometry.dim].reshape(num_points, domain.geometry.dim)
    grid.point_data["displacement"] = displacement
    grid.point_data["magnitude"] = np.linalg.norm(displacement, axis=1)

    characteristic_length = np.linalg.norm(geometry.max(axis=0) - geometry.min(axis=0))
    max_disp_magnitude = np.max(grid.point_data["magnitude"])
    warp_factor = 0.1 * characteristic_length / max_disp_magnitude if max_disp_magnitude > 1e-9 else 1.0
    warped = grid.warp_by_vector("displacement", factor=warp_factor)

    plotter = pyvista.Plotter(window_size=[800, 600])
    plotter.add_mesh(warped, scalars="magnitude", cmap="viridis", show_edges=True)
    plotter.add_mesh(grid, style="wireframe", color="grey", opacity=0.2)
    plotter.add_title(f"{title} (Warp: {warp_factor:.1f}x)")
    plotter.show_axes_all()
    plotter.add_scalar_bar(title="Displacement Mag.")
    plotter.show()
    return combined_function # Return the FEniCS function


def compute_neohookean_energy(domain, V, displacement_func, mu_nh=None, lmbda_nh=None):
    """Compute Neo-Hookean energy."""
    mu_nh = mu_nh if mu_nh is not None else mu
    lmbda_nh = lmbda_nh if lmbda_nh is not None else lmbda

    u = displacement_func # Assume input is already a Function
    I = Identity(domain.geometry.dim)
    F = I + grad(u)
    C = F.T * F
    Ic = tr(C)
    J = det(F)

    # Add small epsilon to log arguments for stability if J can be <= 0
    eps_log = 1e-10
    psi = (mu_nh/2)*(Ic - domain.geometry.dim) - mu_nh*ln(J + eps_log) + (lmbda_nh/2)*(ln(J + eps_log))**2

    energy_form = form(psi * dx)
    total_energy = fem.assemble_scalar(energy_form)

    # Projection for energy density visualization (expensive)
    # Q = fem.functionspace(domain, ("DG", 0)) # Or CG 1
    # trial = TrialFunction(Q)
    # test = TestFunction(Q)
    # a_proj = form(inner(trial, test) * dx)
    # L_proj = form(inner(psi, test) * dx)
    # proj = fem.petsc.LinearProblem(a_proj, L_proj, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    # energy_density_func = proj.solve()
    # energy_density_func.name = "Energy Density"
    # return total_energy, energy_density_func
    print(f"NeoHookean Energy calculation returning total energy only: {total_energy:.4e} (Density projection disabled for performance)")
    return total_energy, None # Disable density projection for now


def visualize_energy_density(domain, V_mesh, energy_density_func, title="Energy Density"):
    """Visualize scalar field."""
    if energy_density_func is None:
        print("No energy density function provided for visualization.")
        return
    topology, cell_types, geometry = plot.vtk_mesh(V_mesh) # Use function space mesh
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    num_points = grid.number_of_points

    # Ensure array slicing is correct for point data assignment
    energy_values = energy_density_func.x.array[:num_points]
    grid.point_data["energy_density"] = energy_values

    plotter = pyvista.Plotter(window_size=[800, 600])
    plotter.add_mesh(grid, scalars="energy_density", cmap="plasma", show_edges=True)
    plotter.add_scalar_bar(title="Energy Density")
    plotter.add_title(title)
    plotter.show_axes_all()
    plotter.show()


# --- Example usage for latent vector and energy ---
if actual_modes_found > 0:
    print("\n--- Visualizing Deformation from Random Latent Vector ---")
    # Use a smaller random vector for visualization stability
    latent_vector_vis = (np.random.rand(N_eig) - 0.5) * 0.1
    print(f"Using latent vector for visualization: {latent_vector_vis[:5]}...")
    combined_deformation_func = visualize_from_latent(domain, V, eigenvectors_np, latent_vector_vis, title="Deformation from Random Latent Vector")

    print("\n--- Computing Neo-Hookean Energy for Combined Deformation ---")
    total_energy_nh, _ = compute_neohookean_energy(domain, V, combined_deformation_func)
    # visualize_energy_density(domain, V.mesh, energy_density_nh) # Optional visualization






if 'eigenvalues' not in globals() or not eigenvalues:
     print("ERROR: Modal analysis did not run or find eigenvalues. Cannot proceed.")
     sys.exit(1)


# ============================================================================
# --- New Block: Neo-Hookean FEM + Modal Projection Test ---
#                (Adapted from FEMDataGenerator Logic)
# ============================================================================
print("\n" + "="*60)
print("--- Starting Neo-Hookean FEM + Modal Projection Test ---")
print("="*60)

# --- 1. Nonlinear Simulation Setup ---

# Reconfirm Material parameters (already defined earlier, just for clarity)
# E, nu, rho, mu, lmbda

# --- Define Helper Function for Nonlinear Solve Step ---

def solve_nonlinear_step(target_magnitude, force_dir_vec, u_initial_guess, fixed_dofs_indices):
    """
    Solves the nonlinear Neo-Hookean problem for a given target force magnitude
    using continuation steps starting from an initial displacement guess.

    Args:
        target_magnitude (float): The final magnitude of the force to apply.
        force_dir_vec (np.ndarray): The unit vector defining force direction.
        u_initial_guess (fem.Function): Function containing the displacement state
                                         to start the continuation from.
        fixed_dofs_indices (np.ndarray): Indices of the fixed DOFs.

    Returns:
        fem.Function: The converged displacement solution.
        bool: True if converged, False otherwise.
    """
    logger.info(f"--- Solving for Target Magnitude: {target_magnitude:.4e} ---")

    # --- Define Functions and Expressions ---
    # Displacement function (start from initial guess)
    u = fem.Function(V, name="Displacement")
    u.x.array[:] = u_initial_guess.x.array[:]

    # Nodal force function (will be updated during continuation)
    f_func = fem.Function(V, name="NodalForces")

    # Test and trial functions
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)

    # Kinematics and energy (Neo-Hookean)
    I = ufl.Identity(domain.geometry.dim)
    F_tensor = I + ufl.grad(u)
    C = F_tensor.T * F_tensor
    Ic = ufl.tr(C)
    J_det = ufl.det(F_tensor) # Renamed to avoid clash with Jacobian matrix J_nl
    # Add small epsilon for stability if J_det can be near zero (though unlikely with continuation)
    eps_log = 1e-10
    psi = (mu/2) * (Ic - 3) - mu * ufl.ln(J_det + eps_log) + (lmbda/2) * (ufl.ln(J_det + eps_log))**2
    internal_energy = psi * ufl.dx

    # External work term using nodal forces
    external_work_expr = ufl.inner(f_func, v) * ufl.dx

    # Residual (weak form) as UFL expression
    residual_expr = ufl.derivative(internal_energy, u, v) - external_work_expr

    # Jacobian (tangent stiffness) as UFL expression
    jacobian_expr = ufl.derivative(residual_expr, u, du)

    # Inside solve_nonlinear_step, repace the force application code with this:

    # --- Calculate Target Nodal Force Vector ---
    f_target_array = np.zeros_like(u.x.array)
    coords = V.tabulate_dof_coordinates()

    # First, identify nodes at the free end (opposite to fixed boundary)
    # Find global x_max
    local_x_coords = domain.geometry.x
    local_x_max = np.max(local_x_coords[:, 0]) if local_x_coords.shape[0] > 0 else -np.inf
    global_x_max = domain.comm.allreduce(local_x_max, op=MPI.MAX)
    print(f"Global x_max for traction BC: {global_x_max:.4f}")

    # Count nodes at x = x_max and apply force only to those
    free_end_nodes = []
    for node_idx in range(len(coords) // domain.geometry.dim):
        base_idx = node_idx * domain.geometry.dim
        node_x = coords[base_idx, 0]  # x-coordinate of this node
        # Check if this node is at x_max
        if np.isclose(node_x, global_x_max, atol=1e-6):
            free_end_nodes.append(node_idx)

    logger.info(f"Found {len(free_end_nodes)} nodes at the free end (x = {global_x_max:.4f})")

    # Early exit if no free end nodes found
    if not free_end_nodes:
        logger.error("No free end nodes found! Cannot apply force.")
        return u, False

    # Calculate force per free-end node (total force distributed evenly)
    if len(free_end_nodes) > 0:
        force_per_node = target_magnitude / len(free_end_nodes) * force_dir_vec
        logger.info(f"Force per free-end node: {force_per_node}")
        
        # Apply force to the free end nodes only
        for node_idx in free_end_nodes:
            dof_start = node_idx * domain.geometry.dim
            for d in range(domain.geometry.dim):
                dof_idx = dof_start + d
                if dof_idx < f_target_array.shape[0]:  # Ensure within bounds
                    f_target_array[dof_idx] = force_per_node[d]
    else:
        logger.warning("No free-end nodes found!")





    # --- Continuation Steps ---
    num_steps = 15 # Number of continuation steps
    initial_scale = 0.01 # Start with a small fraction of the target force

    # Determine the scale of the force currently represented by u_initial_guess
    # This is hard without knowing the previous force. Assume we start continuation
    # relative to the *target* magnitude increase.
    # A simpler approach: Always ramp from 0 to the current target scale.
    current_force_scale = 0.0 # Assume initial guess corresponds to zero force for simplicity now
                              # TODO: A better approach would pass the previous magnitude

    # Scales for continuation from current_force_scale to 1.0 (relative to target_magnitude)
    force_scales = np.linspace(current_force_scale, 1.0, num_steps + 1)[1:] # Start from first step > current
    # Alternative: Logspace if large jumps expected
    # force_scales = np.logspace(np.log10(max(initial_scale, current_force_scale)), np.log10(1.0), num_steps)

    logger.info(f"Continuation scales (relative to target mag): {force_scales}")

    # Compile forms once (if possible, depends on UFL details with f_func updates)
    try:
        residual_form = fem.form(residual_expr)
        jacobian_form = fem.form(jacobian_expr)
        logger.info("UFL forms compiled.")
    except Exception as e:
        logger.error(f"Failed to compile UFL forms: {e}")
        return u, False # Return initial guess, indicate failure

    last_successful_u_array = u_initial_guess.x.array.copy()
    converged_fully = False

    for step, scale in enumerate(force_scales):
        try:
            # Update force function with the scaled target force
            current_f_array = f_target_array * scale
            f_func.x.array[:] = current_f_array
            f_func.x.scatter_forward() # Ensure updates are propagated if needed in parallel

            logger.info(f"Step {step+1}/{num_steps}: scale={scale:.6f}")

            # --- Create Nonlinear Problem and Solver (inside loop for safety) ---
            # If forms don't need recompiling, this could be outside loop
            problem = NonlinearProblem(residual_form, u, bcs=[bc], J=jacobian_form)
            solver = NewtonSolver(MPI.COMM_WORLD, problem)

            # Solver settings (tuned for potentially stiff problems)
            solver.convergence_criterion = "incremental" #"residual"
            solver.rtol = 1e-6
            solver.atol = 1e-8
            solver.max_it = 30 # Increased iterations
            solver.report = True # Get solver details

            # PETSc options (important for convergence)
            ksp = solver.krylov_solver
            opts = PETSc.Options()
            prefix = ksp.getOptionsPrefix()
            opts[f"{prefix}ksp_type"] = "preonly" 
            opts[f"{prefix}pc_type"] = "lu" 
            opts[f"{prefix}pc_factor_mat_solver_type"] = "mumps"


            opts[f"{prefix}snes_linesearch_type"] = "bt"  # backtracking
            opts[f"{prefix}snes_linesearch_damping"] = 0.8
            opts[f"{prefix}snes_linesearch_maxstep"] = 1.0

            ksp.setFromOptions()


            # --- Solve for this step ---
            logger.info(f"Solving nonlinear system for step {step+1}...")
            n_its, converged = solver.solve(u) # Solves and updates u

            if converged:
                logger.info(f"Step {step+1} converged in {n_its} iterations.")
                last_successful_u_array = u.x.array.copy() # Save state
                converged_fully = (step == len(force_scales) - 1) # True only if last step converges

                # Optional: Check displacement magnitude
                disp_norm = np.linalg.norm(u.x.array)
                logger.debug(f"  Displacement norm: {disp_norm:.4e}")
                if disp_norm > 1e6: # Check for excessive displacements
                     logger.warning("Large displacement norm encountered.")
                     # return u, False # Optionally fail early

            else:
                logger.warning(f"Step {step+1} DID NOT CONVERGE after {n_its} iterations.")
                # Restore last successful state and stop continuation
                u.x.array[:] = last_successful_u_array
                return u, False # Return last good state, indicate failure

        except Exception as e:
            logger.error(f"Error during nonlinear solve for scale {scale:.6f}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Restore last successful state and stop
            u.x.array[:] = last_successful_u_array
            return u, False # Return last good state, indicate failure

    # Return the final converged solution (or last successful one if loop broke early)
    return u, converged_fully


# --- 2. Controlled Loading Protocol ---

# Define fixed force direction (unit vector)
np.random.seed(123) # for reproducibility
fixed_force_dir = np.random.randn(domain.geometry.dim)
fixed_force_dir /= np.linalg.norm(fixed_force_dir)
logger.info(f"Using fixed force direction: {fixed_force_dir}")

# Define force magnitudes
# Adjust range/steps based on expected stiffness and nonlinearity
force_magnitudes = np.linspace(0.0, 1000.0, 11) # Example: 0 to 200 load units, 11 steps
logger.info(f"Testing force magnitudes: {force_magnitudes}")

# Store results
nonlinear_solutions_np = []
nonlinear_solutions_petsc = []

# Get fixed DOF indices (needed for applying force correctly)
fixed_dofs_global = bc.dof_indices()[0] # Assuming these are global? Check dolfinx docs/examples
# The solver function expects local indices, but force application needs care.
# Using the bc object directly in the solver is safer.
# For force calculation, let's find fixed nodes and map to DOFs.
fixed_boundary_nodes_mask = np.isclose(V.tabulate_dof_coordinates()[:, 0], V.mesh.geometry.x[:,0].min())
# This mask needs care - it might not perfectly align with DOFs if high order, etc.
# Let's rely on zeroing out forces using local_fixed_dofs inside the solver function.


# --- Solve Loop ---
print("\n--- Solving Nonlinear Problems for Increasing Loads ---")
# Initial guess for the first step is zero displacement
u_current = fem.Function(V)
u_current.x.array[:] = 0.0

for i, mag in enumerate(force_magnitudes):
    if mag == 0.0: # Skip zero magnitude, solution is known (zero)
        logger.info("Magnitude = 0.0, storing zero displacement.")
        zero_vec_np = np.zeros_like(u_current.x.array)
        # Create a PETSc vector using modern FEniCSx approach
        zero_vec_petsc = PETSc.Vec().createWithArray(zero_vec_np)
        nonlinear_solutions_np.append(zero_vec_np)
        nonlinear_solutions_petsc.append(zero_vec_petsc)        
        continue

    # Solve for the current magnitude, starting from the previous solution
    u_solution, converged = solve_nonlinear_step(mag, fixed_force_dir, u_current, fixed_dofs_global)

    if converged:
        logger.info(f"Successfully converged for Magnitude = {mag:.4f}")
        # Store solution
        sol_copy_petsc = PETSc.Vec().createWithArray(u_solution.x.array.copy()) 
        nonlinear_solutions_petsc.append(sol_copy_petsc)
        nonlinear_solutions_np.append(u_solution.x.array.copy()) # Store NumPy array view
        # Update current solution for the next step's initial guess
        u_current.x.array[:] = u_solution.x.array[:]
    else:
        logger.warning(f"Solver failed to converge fully for Magnitude = {mag:.4f}. Stopping load increase.")
        # Store the last computed (potentially partially converged) state? Or NaN/Zeros?
        # Let's store the state returned by the solver (last successful internal step)
        sol_copy_petsc = PETSc.Vec().createWithArray(u_solution.x.array.copy())
        nonlinear_solutions_petsc.append(sol_copy_petsc)
        nonlinear_solutions_np.append(u_solution.x.array.copy())
        # Fill remaining steps with NaN or break
        remaining_steps = len(force_magnitudes) - (i + 1)
        if remaining_steps > 0:
            logger.warning(f"Appending {remaining_steps} NaN results.")
            nan_array = np.full_like(u_current.x.array, np.nan)
            nan_vec_petsc = PETSc.Vec().createWithArray(np.full_like(u_current.x.array, np.nan))
            for _ in range(remaining_steps):
                nonlinear_solutions_np.append(nan_array.copy())
                nonlinear_solutions_petsc.append(nan_vec_petsc.copy())
        break # Stop the loop

print(f"\nStored {len(nonlinear_solutions_np)} nonlinear solutions.")
# Filter out potential NaN results before projection if necessary
valid_indices = [k for k, arr in enumerate(nonlinear_solutions_np) if not np.isnan(arr).any()]
if len(valid_indices) < len(nonlinear_solutions_np):
     logger.warning(f"Found {len(nonlinear_solutions_np) - len(valid_indices)} NaN solutions due to convergence failure.")
     # Only use valid solutions for projection/plotting
     nonlinear_solutions_np = [nonlinear_solutions_np[k] for k in valid_indices]
     nonlinear_solutions_petsc = [nonlinear_solutions_petsc[k] for k in valid_indices]
     force_magnitudes_plot = [force_magnitudes[k] for k in valid_indices]
else:
     force_magnitudes_plot = force_magnitudes


# --- 3. Modal Reconstruction Test ---
# (This part remains largely unchanged, uses the newly generated solutions)
print("\n--- Performing Modal Reconstruction ---")

# Need mass-normalized eigenvectors (assuming this was done previously)
# Check if normalized vectors exist
if 'eigenvectors_petsc_normalized' not in globals():
    print("Mass-normalizing eigenvectors...")
    eigenvectors_petsc_normalized = []
    eigenvectors_np_normalized = []
    # Create temporary vectors needed inside the loop
    M_phi = A.createVecRight() # Vector to store M * phi_i (Use matrix A for creation)

    for i in range(len(eigenvectors_petsc)): # Use the actual number of valid modes found
        phi_i_petsc = eigenvectors_petsc[i] # Get original PETSc eigenvector

        # Calculate modal mass: m_i = phi_i^T * M * phi_i
        M_petsc_final.mult(phi_i_petsc, M_phi) # M_phi = M * phi_i
        m_i = phi_i_petsc.dot(M_phi)

        if m_i < 1e-12: # Avoid division by zero / instability
            logger.warning(f"Modal mass for mode {i+1} is near zero ({m_i:.3e}). Skipping normalization/use.")
            continue

        # Normalize: phi_norm = phi / sqrt(m_i)
        norm_factor = 1.0 / np.sqrt(m_i)
        phi_i_petsc_norm = phi_i_petsc.copy() # Create a new vector for the normalized version
        phi_i_petsc_norm.scale(norm_factor)

        # Store normalized versions
        eigenvectors_petsc_normalized.append(phi_i_petsc_norm)
        eigenvectors_np_normalized.append(phi_i_petsc_norm.array.copy()) # Store NumPy array
else:
     print("Using previously mass-normalized eigenvectors.")


N_modes_normalized = len(eigenvectors_petsc_normalized)
if N_modes_normalized == 0:
    print("ERROR: No modes available after normalization check. Exiting reconstruction.")
    sys.exit(1)
else:
    print(f"Using {N_modes_normalized} mass-normalized modes for projection.")


# Perform projection and reconstruction for each VALID nonlinear solution
reconstruction_errors_Mnorm = []
reconstruction_errors_L2norm = []

# Pre-allocate vectors for efficiency
u_nl_petsc = A.createVecLeft() # PETSc vector for current nonlinear solution
u_recon_petsc = A.createVecLeft() # PETSc vector for reconstructed solution
M_u_nl = A.createVecRight()     # Vector M * u_nl
diff_vec = A.createVecLeft()    # Vector u_nl - u_recon
M_diff = A.createVecRight()     # Vector M * (u_nl - u_recon)

for k, u_nl_array in enumerate(nonlinear_solutions_np): # Loop through valid solutions
    u_nl_petsc = nonlinear_solutions_petsc[k] # Get the corresponding PETSc vector

    # Calculate modal coordinates q_i = phi_i_norm^T * M * u_nl
    modal_coords = np.zeros(N_modes_normalized)
    M_petsc_final.mult(u_nl_petsc, M_u_nl) # M_u_nl = M * u_nl (do this once)
    for i in range(N_modes_normalized):
        phi_i_norm_petsc = eigenvectors_petsc_normalized[i]
        q_i = phi_i_norm_petsc.dot(M_u_nl)
        modal_coords[i] = q_i

    # Reconstruct solution: u_recon = sum(q_i * phi_i_norm)
    u_recon_petsc.zeroEntries() # Reset reconstruction vector
    for i in range(N_modes_normalized):
        u_recon_petsc.axpy(modal_coords[i], eigenvectors_petsc_normalized[i]) # u_recon += q_i * phi_i_norm

    # Calculate error vector: e = u_nl - u_recon
    diff_vec.zeroEntries()
    diff_vec.axpy(1.0, u_nl_petsc)
    diff_vec.axpy(-1.0, u_recon_petsc)

    # Calculate M-norm squared of error: ||e||_M^2 = e^T * M * e
    M_petsc_final.mult(diff_vec, M_diff) # M_diff = M * e
    error_M_norm_sq = diff_vec.dot(M_diff)
    # Calculate M-norm squared of original solution: ||u_nl||_M^2 = u_nl^T * M * u_nl
    unl_M_norm_sq = u_nl_petsc.dot(M_u_nl)

    # Calculate relative M-norm error
    if unl_M_norm_sq > 1e-15:
        relative_error_M = np.sqrt(max(0, error_M_norm_sq) / unl_M_norm_sq) # Ensure non-negative under sqrt
    else:
        relative_error_M = 0.0 if error_M_norm_sq < 1e-15 else 1.0 # Error is 0 if both norms are 0, else 100% error
    reconstruction_errors_Mnorm.append(relative_error_M)

    # Calculate relative L2-norm error: ||u_nl - u_recon||_L2 / ||u_nl||_L2
    error_L2_norm_sq = diff_vec.dot(diff_vec) # ||e||_L2^2
    unl_L2_norm_sq = u_nl_petsc.dot(u_nl_petsc) # ||u_nl||_L2^2

    if unl_L2_norm_sq > 1e-15:
         relative_error_L2 = np.sqrt(max(0, error_L2_norm_sq) / unl_L2_norm_sq) # Ensure non-negative
    else:
         relative_error_L2 = 0.0 if error_L2_norm_sq < 1e-15 else 1.0
    reconstruction_errors_L2norm.append(relative_error_L2)

    print(f"  Load Mag {force_magnitudes_plot[k]:.2f}: Rel L2 Error = {relative_error_L2*100:.2f}%, Rel M-Norm Error = {relative_error_M*100:.2f}%")


# --- 4. Validation Plot ---
# (This part remains unchanged, uses filtered magnitudes if needed)
print("\n--- Plotting Reconstruction Accuracy ---")

if not reconstruction_errors_L2norm:
     print("No valid reconstruction results to plot.")
else:
     plt.figure(figsize=(10, 6))
     plt.plot(force_magnitudes_plot, 100 * (1 - np.array(reconstruction_errors_L2norm)), 'o-', label=f'L2 Norm Accuracy ({N_modes_normalized} modes)')
     plt.plot(force_magnitudes_plot, 100 * (1 - np.array(reconstruction_errors_Mnorm)), 's--', label=f'M-Norm Accuracy ({N_modes_normalized} modes)')

     plt.xlabel("Applied Force Magnitude")
     plt.ylabel("Reconstruction Accuracy (%)")
     plt.title(f"Modal Reconstruction Accuracy vs. Load (Neo-Hookean, {N_modes_normalized} Modes)")
     plt.legend()
     plt.grid(True)
     plt.ylim([min(0, plt.ylim()[0]), 105]) # Adjust y-min if accuracy drops below 0
     plt.show()

print("\n--- Neo-Hookean FEM + Modal Projection Test Finished ---")
print("="*60)


# Optional: Visualize the last valid nonlinear solution and its reconstruction
# (This part remains unchanged but uses the last *valid* solution)
if nonlinear_solutions_np and N_modes_normalized > 0:
    print("\n--- Visualizing Last Valid Nonlinear Step and Reconstruction ---")
    last_valid_idx = len(nonlinear_solutions_np) - 1

    # Visualize Full Nonlinear Solution
    u_nl_viz = Function(V)
    u_nl_viz.x.array[:] = nonlinear_solutions_petsc[last_valid_idx].array[:]
    # Need the visualize_from_latent function from the original script
    # Assuming visualize_from_latent is defined as before:
    if 'visualize_from_latent' in globals():
        visualize_from_latent(domain, V, [u_nl_viz.x.array], [1.0], title=f"Full Nonlinear Solution (Mag={force_magnitudes_plot[last_valid_idx]:.2f})")

        # Visualize Reconstructed Solution (u_recon_petsc holds the last reconstruction result)
        u_recon_viz = Function(V)
        u_recon_viz.x.array[:] = u_recon_petsc.array[:]
        visualize_from_latent(domain, V, [u_recon_viz.x.array], [1.0], title=f"Modal Reconstruction ({N_modes_normalized} modes)")
    else:
        print("Skipping visualization - 'visualize_from_latent' function not found.")


print("\nScript completely finished.")