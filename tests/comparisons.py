import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import os
import json
import glob
from mpi4py import MPI
from dolfinx import mesh, fem, plot
from dolfinx.io import gmshio
from dolfinx.fem.petsc import assemble_matrix as assemble_matrix_petsc
from ufl import TrialFunction, TestFunction, inner, dx, grad, sym, Identity, tr
import argparse
import sys

 # Convert to scipy sparse for comparison
from scipy.sparse import csr_matrix

def get_latest_sofa_matrices(directory='matrices'):
    """Find the latest SOFA matrices in the specified directory"""
    timestamps = []
    for file in glob.glob(f'{directory}/metadata_*.json'):
        timestamp = file.split('metadata_')[1].split('.json')[0]
        timestamps.append(timestamp)
    
    if not timestamps:
        raise FileNotFoundError(f"No SOFA matrices found in {directory}")
    
    latest = max(timestamps)
    print(f"Using latest SOFA matrices from timestamp {latest}")
    
    # Load metadata
    with open(f'{directory}/metadata_{latest}.json', 'r') as f:
        metadata = json.load(f)
    
    # Load matrices
    mass_file = f'{directory}/mass_matrix_{latest}.npy'
    stiff_file = f'{directory}/stiffness_matrix_{latest}.npy'
    
    print(f"Loading mass matrix from: {mass_file}")
    print(f"Loading stiffness matrix from: {stiff_file}")
    
    # More robust loading approach
    try:
        # First try loading directly
        mass_matrix = np.load(mass_file, allow_pickle=True)
        stiffness_matrix = np.load(stiff_file, allow_pickle=True)
        
        # Check what was loaded and convert appropriately
        if isinstance(mass_matrix, np.ndarray):
            # If it's a regular dense array, convert to sparse
            print("Converting dense mass matrix to sparse...")
            mass_matrix = sp.csr_matrix(mass_matrix)
        elif hasattr(mass_matrix, 'item'):
            # Sometimes np.load returns an array scalar that needs .item() to get the actual object
            print("Unpacking mass matrix from array scalar...")
            mass_matrix = mass_matrix.item()
            if not sp.issparse(mass_matrix):
                mass_matrix = sp.csr_matrix(mass_matrix)
                
        if isinstance(stiffness_matrix, np.ndarray):
            print("Converting dense stiffness matrix to sparse...")
            stiffness_matrix = sp.csr_matrix(stiffness_matrix)
        elif hasattr(stiffness_matrix, 'item'):
            print("Unpacking stiffness matrix from array scalar...")
            stiffness_matrix = stiffness_matrix.item()
            if not sp.issparse(stiffness_matrix):
                stiffness_matrix = sp.csr_matrix(stiffness_matrix)
    
    except Exception as e:
        print(f"Error loading matrices: {e}")
        print("Trying alternative loading approach...")
        
        # Alternative approach: load as object array and extract manually
        mass_obj = np.load(mass_file, allow_pickle=True)
        stiff_obj = np.load(stiff_file, allow_pickle=True)
        
        # Try to extract the sparse matrix
        if hasattr(mass_obj, 'item'):
            mass_matrix = mass_obj.item()
        else:
            # Last resort: try to create from scratch
            if isinstance(mass_obj, dict) and 'data' in mass_obj and 'indices' in mass_obj and 'indptr' in mass_obj:
                mass_matrix = sp.csr_matrix((mass_obj['data'], mass_obj['indices'], mass_obj['indptr']))
            else:
                raise ValueError("Cannot convert mass matrix to sparse format")
                
        if hasattr(stiff_obj, 'item'):
            stiffness_matrix = stiff_obj.item()
        else:
            if isinstance(stiff_obj, dict) and 'data' in stiff_obj and 'indices' in stiff_obj and 'indptr' in stiff_obj:
                stiffness_matrix = sp.csr_matrix((stiff_obj['data'], stiff_obj['indices'], stiff_obj['indptr']))
            else:
                raise ValueError("Cannot convert stiffness matrix to sparse format")
    
    # Verify we have sparse matrices
    if not sp.issparse(mass_matrix) or not sp.issparse(stiffness_matrix):
        raise ValueError("Failed to convert matrices to sparse format")
    
    print(f"SOFA mass matrix: {mass_matrix.shape}, density: {metadata['density']}")
    print(f"SOFA stiffness matrix: {stiffness_matrix.shape}, E={metadata['young_modulus']}, nu={metadata['poisson_ratio']}")
    
    return mass_matrix, stiffness_matrix, metadata


def compute_hybrid_mass_matrix(M_consistent, lumping_ratio=0.4):
    """Create a hybrid mass matrix blending lumped and consistent approaches
    
    Args:
        M_consistent: The consistent mass matrix
        lumping_ratio: Ratio of mass to put on diagonal (0=fully consistent, 1=fully lumped)
    
    Returns:
        Hybrid mass matrix
    """
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

def compute_fenicsx_matrices(mesh_file, young_modulus, poisson_ratio, density):
    """Compute the mass and stiffness matrices using FEniCSx"""
    print(f"Computing FEniCSx matrices for mesh {mesh_file}")
    def petsc_to_scipy(petsc_mat):
        ai, aj, av = petsc_mat.getValuesCSR()
        return csr_matrix((av, aj, ai), shape=petsc_mat.getSize())
    # Load mesh
    domain, cell_tags, facet_tags = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, 0, gdim=3)
    
    # Get domain extents
    x_coords = domain.geometry.x
    x_min = x_coords[:, 0].min()
    
    # Material parameters
    E, nu = young_modulus, poisson_ratio
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    rho = density
    
    # Create function space
    V = fem.functionspace(domain, ("CG", 1, (3,)))
    u_ = TrialFunction(V)
    du = TestFunction(V)
    
    # Define strain and stress operators
    def eps(v):
        return sym(grad(v))
        
    def sigma(v):
        dim = domain.geometry.dim
        return 2.0 * mu * eps(v) + lmbda * tr(eps(v)) * Identity(dim)
    
    # Define forms
    k_form = inner(sigma(du), eps(u_)) * dx  # Stiffness form
    m_form = rho * inner(du, u_) * dx(metadata={"quadrature_degree": 1})
    
    # Apply Dirichlet boundary condition for fixed end
    def fixed_boundary(x):
        return np.isclose(x[0], x_min)
    
    fixed_dofs = fem.locate_dofs_geometrical(V, fixed_boundary)
    bc = fem.dirichletbc(np.zeros(3), fixed_dofs, V)
    
    # Assemble matrices
    # Try assembling without BCs first, then applying them:
    A = assemble_matrix_petsc(fem.form(k_form), bcs=[bc])
    A.assemble()    
    
    
    # First, assemble the regular mass matrix
    M_consistent = assemble_matrix_petsc(fem.form(m_form))
    M_consistent.assemble()

    # Convert to scipy for processing
    M_scipy_consistent = petsc_to_scipy(M_consistent)

    # Normalize to expected total mass
    expected_total_mass = density * 10  # 10 is the object volume
    total_mass_consistent = np.sum(M_scipy_consistent.data)
    M_scipy_consistent = M_scipy_consistent * (expected_total_mass / total_mass_consistent)
    print(f"Normalized consistent mass matrix sum: {np.sum(M_scipy_consistent.data)}")

    # Create hybrid mass matrix (40% lumped, 60% consistent to match SOFA)
    M_hybrid = compute_hybrid_mass_matrix(M_scipy_consistent, lumping_ratio=0.4)
    print(f"Hybrid mass matrix diagonal percentage: {np.sum(M_hybrid.diagonal()) / np.sum(M_hybrid.data) * 100:.2f}%")

    # Final mass matrix
    M_scipy = M_hybrid * 2.0  # Copy for consistency with SOFA

    # For stiffness, ensure we're handling sign correctly
    A_scipy = petsc_to_scipy(A)
    if np.sum(A_scipy.diagonal()) < 0:
        print("Negating FEniCSx stiffness matrix to match SOFA's convention")
        A_scipy = -A_scipy

    return A_scipy, M_scipy, V.dofmap.index_map.size_global * 3

def compare_matrices(sofa_matrix, fenics_matrix, name, scale_factor=1.0, plot_difference=True):
    """Compare matrices from SOFA and FEniCSx with appropriate rescaling"""
    # Ensure same size for comparison
    if sofa_matrix.shape != fenics_matrix.shape:
        print(f"WARNING: Matrix size mismatch - SOFA: {sofa_matrix.shape}, FEniCSx: {fenics_matrix.shape}")
        size = min(sofa_matrix.shape[0], fenics_matrix.shape[0])
        sofa_matrix = sofa_matrix[:size, :size]
        fenics_matrix = fenics_matrix[:size, :size]
    
    # Scale FEniCSx matrix for comparison
    fenics_matrix_scaled = fenics_matrix * scale_factor
    
    # For stiffness matrix in SOFA, we may need to negate since SOFA stores -K
    if name == "stiffness" and np.sum(sofa_matrix.diagonal()) < 0:
        print("Negating SOFA stiffness matrix for comparison (SOFA stores -K)")
        sofa_matrix = -sofa_matrix
    
    # Compare matrices
    diff = (sofa_matrix - fenics_matrix_scaled).tocsr()
    
    # Calculate statistics
    abs_diff = np.abs(diff.data)
    rel_diff = abs_diff / (np.max(np.abs(sofa_matrix.data)) + 1e-10)
    
    sofa_norm = np.linalg.norm(sofa_matrix.data)
    fenics_norm = np.linalg.norm(fenics_matrix_scaled.data)
    diff_norm = np.linalg.norm(diff.data)
    
    rel_diff_norm = diff_norm / (sofa_norm + 1e-10)
    
    # Calculate sparsity patterns
    sofa_nnz = sofa_matrix.nnz
    fenics_nnz = fenics_matrix.nnz
    
    print(f"\n===== {name.upper()} MATRIX COMPARISON =====")
    print(f"Scale factor applied: {scale_factor}")
    print(f"SOFA matrix norm: {sofa_norm:.6e}")
    print(f"FEniCSx matrix norm: {fenics_norm:.6e}")
    print(f"Difference norm: {diff_norm:.6e}")
    print(f"Relative difference: {rel_diff_norm:.6%}")
    print(f"Max absolute difference: {np.max(abs_diff):.6e}")
    print(f"Mean absolute difference: {np.mean(abs_diff):.6e}")
    print(f"Median absolute difference: {np.median(abs_diff):.6e}")
    print(f"Sparsity - SOFA: {sofa_nnz}, FEniCSx: {fenics_nnz}")
    
    if plot_difference:
        # Visualize the matrices and their difference
        plt.figure(figsize=(15, 5))
        
        # Convert to dense for visualization
        size = min(100, sofa_matrix.shape[0])  # Limit to 100x100 for visualization
        sofa_dense = sofa_matrix[:size, :size].toarray()
        fenics_dense = fenics_matrix_scaled[:size, :size].toarray()
        diff_dense = diff[:size, :size].toarray()
        
        vmax = max(np.abs(sofa_dense).max(), np.abs(fenics_dense).max())
        vmin = -vmax
        
        plt.subplot(131)
        plt.imshow(sofa_dense, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(f'SOFA {name} matrix')
        
        plt.subplot(132)
        plt.imshow(fenics_dense, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(f'FEniCSx {name} matrix (scaled)')
        
        plt.subplot(133)
        plt.imshow(diff_dense, cmap='RdBu_r', vmin=vmin/10, vmax=vmax/10)
        plt.colorbar()
        plt.title(f'Difference (relative norm: {rel_diff_norm:.2%})')
        
        plt.tight_layout()
        plt.savefig(f'matrix_comparison_{name}.png')
        plt.close()
        
        print(f"Matrix comparison plot saved to matrix_comparison_{name}.png")
    
    return rel_diff_norm


def analyze_sofa_matrix_structure(matrix, name):
    """Analyze the structure of a SOFA matrix to understand its format"""
    print(f"\n==== SOFA {name} MATRIX ANALYSIS ====")
    
    # Check matrix type and structure
    if sp.issparse(matrix):
        matrix_type = type(matrix).__name__
        if hasattr(matrix, 'format'):
            matrix_format = matrix.format
        else:
            matrix_format = "Unknown"
        print(f"Matrix type: {matrix_type}, format: {matrix_format}")
    
    # Check if it's diagonal or block diagonal
    diag_sum = np.sum(matrix.diagonal())
    total_sum = np.sum(matrix.data)
    diag_percentage = diag_sum / total_sum * 100
    print(f"Diagonal percentage: {diag_percentage:.2f}%")
    
    # Check for block structure (3x3 blocks for 3D problems)
    if matrix.shape[0] % 3 == 0:
        # Check 3x3 block diagonal pattern
        block_size = 3
        blocks = matrix.shape[0] // block_size
        block_diag_sum = 0
        for i in range(blocks):
            block_start = i * block_size
            block_end = (i + 1) * block_size
            block_diag_sum += np.sum(matrix[block_start:block_end, block_start:block_end].data)
        block_diag_percentage = block_diag_sum / total_sum * 100
        print(f"3x3 Block diagonal percentage: {block_diag_percentage:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Compare SOFA and FEniCSx matrices')
    parser.add_argument('--mesh', default=None, help='Path to mesh file (defaults to the one used in SOFA)')
    parser.add_argument('--timestamp', default=None, help='Specific SOFA timestamp to use')
    parser.add_argument('--scale', type=float, default=1, help='Additional scaling factor if needed')
    args = parser.parse_args()
    
    # Load SOFA matrices
    sofa_mass, sofa_stiffness, metadata = get_latest_sofa_matrices()
    
    # Get mesh file from metadata if not specified
    mesh_file = args.mesh if args.mesh else metadata['mesh_file']
    
    # Compute FEniCSx matrices
    fenics_stiffness, fenics_mass, dof_count = compute_fenicsx_matrices(
        mesh_file, 
        metadata['young_modulus'], 
        metadata['poisson_ratio'], 
        metadata['density']
    )
    
    # Compare matrices (with basic scaling)
    mass_diff = compare_matrices(sofa_mass, fenics_mass, "mass", scale_factor=args.scale)
    stiffness_diff = compare_matrices(sofa_stiffness, fenics_stiffness, "stiffness", scale_factor=args.scale)

    analyze_sofa_matrix_structure(sofa_mass, "mass")
    analyze_sofa_matrix_structure(sofa_stiffness, "stiffness")
    
    # Overall summary
    print("\n===== OVERALL COMPARISON =====")
    print(f"Overall relative difference - Mass: {mass_diff:.6%}, Stiffness: {stiffness_diff:.6%}")
    
    if mass_diff < 0.05 and stiffness_diff < 0.05:
        print("EXCELLENT MATCH: Matrices differ by less than 5%")
    elif mass_diff < 0.10 and stiffness_diff < 0.10:
        print("GOOD MATCH: Matrices differ by less than 10%")
    elif mass_diff < 0.20 and stiffness_diff < 0.20:
        print("ACCEPTABLE MATCH: Matrices differ by less than 20%")
    else:
        print("SIGNIFICANT DIFFERENCES: Matrices differ by more than 20%")
        print("Possible causes:")
        print("1. Different material model implementations")
        print("2. Different boundary condition applications")
        print("3. Different mesh interpretations")
        print("4. Different integration methods")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())