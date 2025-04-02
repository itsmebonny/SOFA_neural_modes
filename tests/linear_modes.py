import numpy as np
import pyvista
from dolfinx import mesh, fem, io, plot
from dolfinx.fem import form, Function
from dolfinx.io import gmshio
# Import specific ufl functions needed
from ufl import (TrialFunction, TestFunction, inner, dx, grad, sym, Identity,
                 div, dot, tr, Constant, ln, det, derivative, action)
from mpi4py import MPI
from petsc4py import PETSc
import sys
# Use consistent import name for assembly
from dolfinx.fem.petsc import assemble_matrix as assemble_matrix_petsc
from dolfinx.fem.petsc import assemble_vector as assemble_vector_petsc

from slepc4py import SLEPc

# Import scipy sparse, needed for hybrid mass matrix if used
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

# Number of eigenvalues to compute
N_eig = 12

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
    eigenvectors_np.append(vr.array_r.copy()) # Store numpy array version

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
             warp_factor = 0.1 * characteristic_length / max_disp_magnitude
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

visualize_eigenmodes(domain, V, eigenvectors_np, eigenvalues, num_modes_to_plot=min(N_eig, 12))


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

print("\nScript finished.")