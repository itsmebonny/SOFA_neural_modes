import numpy as np
import pyvista
import torch
import yaml
import os
import sys
import logging
import matplotlib.pyplot as plt

# --- FEniCSx/PETSc/SLEPc Imports ---
from dolfinx import mesh, fem, io, plot
from dolfinx.fem import form, Function, Constant
from dolfinx.io import gmshio
from ufl import (TrialFunction, TestFunction, inner, dx, grad, sym, Identity,
                 div, dot, tr, ln, det, derivative, action, Measure, SpatialCoordinate)
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.fem.petsc import assemble_matrix as assemble_matrix_petsc
from dolfinx.fem.petsc import assemble_vector as assemble_vector_petsc
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
# Note: SLEPc is now imported via train.py/Routine if needed there,
# but we won't run the solver here directly.

# --- Import Routine and Network from train.py ---
script_dir = os.path.dirname(__file__)
training_dir = os.path.abspath(os.path.join(script_dir, '..', 'training'))
if training_dir not in sys.path:
    sys.path.insert(0, training_dir)
try:
    from training.train import Routine, ResidualNet # Or Net, depending on your config
    print("Successfully imported Routine and ResidualNet from train.py")
except ImportError as e:
    print(f"Error importing from train.py: {e}")
    print("Please ensure train.py is in the Python path or adjust the import.")
    sys.exit(1)
# --- End Routine/Network Import ---

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("Evaluation") # Changed logger name

# ============================================================================
# --- Load Configuration and Initialize Routine ---
# ============================================================================
print("\n" + "="*60)
print("--- Initializing Routine from Training Configuration ---")
print("="*60)

# Load config file used for training
config_path = os.path.abspath(os.path.join('configs', 'default.yaml'))
if not os.path.exists(config_path):
    print(f"ERROR: Training config file not found at {config_path}")
    sys.exit(1)

try:
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    print(f"Loaded training configuration from {config_path}")
except Exception as e:
    print(f"Error loading config file {config_path}: {e}")
    sys.exit(1)

# Instantiate the Routine class
# This will load mesh, define V, compute modes, matrices (A, M), BCs, etc.
try:
    engine = Routine(cfg)
    print("Routine initialized successfully.")
    # --- Extract necessary objects from Routine ---
    domain = engine.domain
    V = engine.V
    A = engine.A # Stiffness matrix (with BCs applied symmetrically)
    M_petsc_final = engine.M # Final Mass matrix (hybrid or consistent, with BCs applied)
    bc = engine.bc # Boundary condition object
    eigenvalues = engine.eigenvalues # Numpy array of eigenvalues
    # engine.linear_modes is a numpy array (num_dofs x num_modes)
    linear_modes_np = engine.linear_modes.cpu().numpy() # Get numpy array from tensor
    mu = engine.mu
    lmbda = engine.lambda_
    rho = engine.rho # Density might be needed if forms are redefined

    # Get number of modes computed by Routine
    N_eig = linear_modes_np.shape[1]
    print(f"Using {N_eig} linear modes computed by Routine.")

    # Convert numpy eigenvectors back to PETSc vectors for projection/reconstruction
    eigenvectors_petsc = []
    proto_vec = A.createVecLeft() # Create a template vector
    for i in range(N_eig):
        vec_np = linear_modes_np[:, i]
        petsc_vec = PETSc.Vec().createWithArray(vec_np.copy(), comm=MPI.COMM_WORLD)
        # Ensure the PETSc vector has the correct layout/size matching the matrix
        # This might require scattering if running in parallel and Routine gathered modes
        # For now, assume serial or compatible parallel layout from Routine
        eigenvectors_petsc.append(petsc_vec)
    print(f"Converted {len(eigenvectors_petsc)} numpy modes back to PETSc vectors.")

except Exception as e:
    print(f"Error initializing Routine: {e}")
    import traceback
    print(traceback.format_exc())
    sys.exit(1)

# ============================================================================
# --- Use Trained Neural Network from Routine ---
# ============================================================================
print("\n" + "="*60)
print("--- Using Trained Neural Network from Routine ---")
print("="*60)
# Determine device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Get the model directly from the initialized Routine instance
if hasattr(engine, 'model') and engine.model is not None:
    neural_model = engine.model
    # Ensure the model from Routine is on the correct device and in eval mode
    neural_model.to(device)
    neural_model.eval()
    print(f"Using neural model ({neural_model.__class__.__name__}) loaded by Routine.")
    checkpoint_dir = cfg.get('training', {}).get('checkpoint_dir', 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_dir, 'best.pt') # Or specify another checkpoint file

    if os.path.exists(checkpoint_path):
        try:
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=engine.device)
            # Load model state dict carefully
            engine.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Successfully loaded model state from epoch {checkpoint.get('epoch', 'N/A')}")
        except Exception as e_load:
            print(f"ERROR loading checkpoint from {checkpoint_path}: {e_load}")
            print("Proceeding with the initialized (likely untrained) model.")
            # Optionally exit if loading is critical: sys.exit(1)
    else:
        print(f"WARNING: Checkpoint file not found at {checkpoint_path}")
        print("Proceeding with the initialized (likely untrained) model.")

    # Verify latent dimension consistency
    # Assuming the model has an 'input_layer' or similar to check input size
    try:
        # Attempt to get the expected input dimension from the model's first linear layer
        model_latent_dim = next(neural_model.parameters()).shape[1] # Heuristic for first layer input
        if model_latent_dim != cfg['model']['latent_dim']:
             logger.warning(f"Model's expected input dim ({model_latent_dim}) might differ from config ({cfg['model']['latent_dim']})")
        latent_dim_config = cfg['model']['latent_dim'] # Still get from config for reconstruction limit
    except Exception as e:
        logger.warning(f"Could not automatically verify model's latent dim: {e}")
        latent_dim_config = cfg['model']['latent_dim'] # Fallback to config

    # Ensure output dim matches the DOF count from the Routine's function space
    output_dim_model = V.dofmap.index_map.size_global * domain.geometry.dim

else:
    print("ERROR: Routine object ('engine') does not have a 'model' attribute or it is None.")
    print("Ensure Routine initializes and loads the model correctly.")
    sys.exit(1)

# ============================================================================
# --- Neo-Hookean FEM + Modal Projection Test ---
# ============================================================================
print("\n" + "="*60)
print("--- Starting Neo-Hookean FEM + Modal Projection Test ---")
print("="*60)

# --- 1. Nonlinear Simulation Setup ---
# Material parameters (mu, lmbda) are already loaded from engine

# --- Define Helper Function for Nonlinear Solve Step ---
# This function now uses V, domain, mu, lmbda, bc from the 'engine' instance
def solve_nonlinear_step(target_magnitude, force_dir_vec, u_initial_guess):
    """
    Solves the nonlinear Neo-Hookean problem for a given target force magnitude
    using continuation steps starting from an initial displacement guess.
    Uses V, domain, mu, lmbda, bc from the global 'engine' instance.

    Args:
        target_magnitude (float): The final magnitude of the force to apply.
        force_dir_vec (np.ndarray): The unit vector defining force direction.
        u_initial_guess (fem.Function): Function containing the displacement state
                                         to start the continuation from.

    Returns:
        fem.Function: The converged displacement solution.
        bool: True if converged, False otherwise.
    """
    logger.info(f"--- Solving for Target Magnitude: {target_magnitude:.4e} ---")

    # --- Define Functions and Expressions (using engine's V, domain, mu, lmbda) ---
    u = fem.Function(V, name="Displacement")
    u.x.array[:] = u_initial_guess.x.array[:]
    f_func = fem.Function(V, name="NodalForces")
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)
    I = ufl.Identity(domain.geometry.dim)
    F_tensor = I + ufl.grad(u)
    C = F_tensor.T * F_tensor
    Ic = ufl.tr(C)
    J_det = ufl.det(F_tensor)
    eps_log = 1e-10
    psi = (mu/2) * (Ic - 3) - mu * ufl.ln(J_det + eps_log) + (lmbda/2) * (ufl.ln(J_det + eps_log))**2
    internal_energy = psi * ufl.dx
    external_work_expr = ufl.inner(f_func, v) * ufl.dx
    residual_expr = ufl.derivative(internal_energy, u, v) - external_work_expr
    jacobian_expr = ufl.derivative(residual_expr, u, du)

    # --- Calculate Target Nodal Force Vector (apply at x_max) ---
    f_target_array = np.zeros_like(u.x.array)
    coords = V.tabulate_dof_coordinates()
    local_x_coords = domain.geometry.x
    local_x_max = np.max(local_x_coords[:, 0]) if local_x_coords.shape[0] > 0 else -np.inf
    global_x_max = domain.comm.allreduce(local_x_max, op=MPI.MAX)
    logger.info(f"Applying traction force at x = {global_x_max:.4f}")

    free_end_nodes = []
    # Iterate through unique nodes associated with the DOFs
    unique_coords = np.unique(coords[:, :domain.geometry.dim], axis=0)
    node_indices_at_xmax = np.where(np.isclose(unique_coords[:, 0], global_x_max, atol=1e-6))[0]

    if len(node_indices_at_xmax) == 0:
        logger.error("No nodes found at the free end! Cannot apply force.")
        return u, False
    logger.info(f"Found {len(node_indices_at_xmax)} unique nodes at the free end.")

    # Map unique node indices back to DOF indices
    free_end_dofs = []
    for node_idx in node_indices_at_xmax:
        # Find DOFs corresponding to this unique coordinate
        matching_dof_indices = np.where(np.all(np.isclose(coords[:, :domain.geometry.dim], unique_coords[node_idx]), axis=1))[0]
        free_end_dofs.extend(matching_dof_indices)

    # Distribute force over the DOFs at the free end
    # This simple distribution might need refinement depending on element type/order
    num_force_dofs = len(free_end_dofs)
    if num_force_dofs > 0:
        force_per_dof_component = target_magnitude / num_force_dofs # Total force / num DOFs at end
        logger.info(f"Distributing force over {num_force_dofs} DOFs at free end.")
        for dof_idx in free_end_dofs:
             component_index = dof_idx % domain.geometry.dim # 0 for x, 1 for y, 2 for z
             f_target_array[dof_idx] = force_per_dof_component * force_dir_vec[component_index]
    else:
         logger.warning("No DOFs identified at free end for force application!")


    # --- Continuation Steps ---
    num_steps = 15
    force_scales = np.linspace(0.0, 1.0, num_steps + 1)[1:] # Ramp from 0 to 1

    try:
        residual_form = fem.form(residual_expr)
        jacobian_form = fem.form(jacobian_expr)
    except Exception as e:
        logger.error(f"Failed to compile UFL forms: {e}")
        return u, False

    last_successful_u_array = u_initial_guess.x.array.copy()
    converged_fully = False

    for step, scale in enumerate(force_scales):
        try:
            current_f_array = f_target_array * scale
            f_func.x.array[:] = current_f_array
            f_func.x.scatter_forward()

            logger.info(f"Step {step+1}/{num_steps}: scale={scale:.6f}")

            # Use engine's bc object
            problem = NonlinearProblem(residual_form, u, bcs=[bc], J=jacobian_form)
            solver = NewtonSolver(MPI.COMM_WORLD, problem)
            solver.convergence_criterion = "incremental"
            solver.rtol = 1e-6
            solver.atol = 1e-8
            solver.max_it = 30
            solver.report = False # Less verbose for evaluation

            ksp = solver.krylov_solver
            opts = PETSc.Options()
            prefix = ksp.getOptionsPrefix()
            opts[f"{prefix}ksp_type"] = "preonly"
            opts[f"{prefix}pc_type"] = "lu"
            opts[f"{prefix}pc_factor_mat_solver_type"] = "mumps" # Ensure MUMPS is available
            opts[f"{prefix}snes_linesearch_type"] = "bt"
            ksp.setFromOptions()

            logger.info(f"Solving nonlinear system for step {step+1}...")
            n_its, converged = solver.solve(u)

            if converged:
                logger.info(f"Step {step+1} converged in {n_its} iterations.")
                last_successful_u_array = u.x.array.copy()
                converged_fully = (step == len(force_scales) - 1)
            else:
                logger.warning(f"Step {step+1} DID NOT CONVERGE after {n_its} iterations.")
                u.x.array[:] = last_successful_u_array
                return u, False

        except Exception as e:
            logger.error(f"Error during nonlinear solve for scale {scale:.6f}: {e}")
            logger.error(traceback.format_exc())
            u.x.array[:] = last_successful_u_array
            return u, False

    return u, converged_fully


# --- 2. Controlled Loading Protocol ---
# Define fixed force direction (unit vector) - Same as before
np.random.seed(123)
fixed_force_dir = np.random.randn(domain.geometry.dim)
fixed_force_dir /= np.linalg.norm(fixed_force_dir)
logger.info(f"Using fixed force direction: {fixed_force_dir}")

# Define force magnitudes - Same as before
force_magnitudes = np.linspace(0.0, 1000.0, 11)
logger.info(f"Testing force magnitudes: {force_magnitudes}")

# Store results
nonlinear_solutions_np = []
nonlinear_solutions_petsc = []

# --- Solve Loop ---
print("\n--- Solving Nonlinear Problems for Increasing Loads ---")
u_current = fem.Function(V)
u_current.x.array[:] = 0.0

for i, mag in enumerate(force_magnitudes):
    if mag == 0.0:
        logger.info("Magnitude = 0.0, storing zero displacement.")
        zero_vec_np = np.zeros_like(u_current.x.array)
        zero_vec_petsc = PETSc.Vec().createWithArray(zero_vec_np.copy()) # Use copy
        nonlinear_solutions_np.append(zero_vec_np)
        nonlinear_solutions_petsc.append(zero_vec_petsc)
        continue

    # Solve for the current magnitude, starting from the previous solution
    # No need to pass fixed_dofs_indices, bc is used internally
    u_solution, converged = solve_nonlinear_step(mag, fixed_force_dir, u_current)

    if converged:
        logger.info(f"Successfully converged for Magnitude = {mag:.4f}")
        sol_copy_petsc = PETSc.Vec().createWithArray(u_solution.x.array.copy())
        nonlinear_solutions_petsc.append(sol_copy_petsc)
        nonlinear_solutions_np.append(u_solution.x.array.copy())
        u_current.x.array[:] = u_solution.x.array[:]
    else:
        logger.warning(f"Solver failed to converge fully for Magnitude = {mag:.4f}. Stopping load increase.")
        sol_copy_petsc = PETSc.Vec().createWithArray(u_solution.x.array.copy()) # Store last good state
        nonlinear_solutions_petsc.append(sol_copy_petsc)
        nonlinear_solutions_np.append(u_solution.x.array.copy())
        remaining_steps = len(force_magnitudes) - (i + 1)
        if remaining_steps > 0:
            logger.warning(f"Appending {remaining_steps} NaN results.")
            nan_array = np.full_like(u_current.x.array, np.nan)
            nan_vec_petsc = PETSc.Vec().createWithArray(np.full_like(u_current.x.array, np.nan))
            for _ in range(remaining_steps):
                nonlinear_solutions_np.append(nan_array.copy())
                nonlinear_solutions_petsc.append(nan_vec_petsc.copy())
        break

print(f"\nStored {len(nonlinear_solutions_np)} nonlinear solutions.")
valid_indices = [k for k, arr in enumerate(nonlinear_solutions_np) if not np.isnan(arr).any()]
if len(valid_indices) < len(nonlinear_solutions_np):
     logger.warning(f"Found {len(nonlinear_solutions_np) - len(valid_indices)} NaN solutions due to convergence failure.")
     nonlinear_solutions_np = [nonlinear_solutions_np[k] for k in valid_indices]
     nonlinear_solutions_petsc = [nonlinear_solutions_petsc[k] for k in valid_indices]
     force_magnitudes_plot = [force_magnitudes[k] for k in valid_indices]
else:
     force_magnitudes_plot = force_magnitudes


# --- 3. Modal Reconstruction & Energy Comparison ---
print("\n" + "="*60)
print("--- Performing Modal Reconstruction & Energy Comparison ---")
print("="*60)

# --- Mass-Normalize Eigenvectors (using engine.M) ---
print("Mass-normalizing eigenvectors using engine.M...")
eigenvectors_petsc_normalized = []
eigenvectors_np_normalized = []
M_phi = A.createVecRight() # Use A or M_petsc_final for template

for i in range(len(eigenvectors_petsc)): # Use the PETSc vectors created earlier
    phi_i_petsc = eigenvectors_petsc[i]
    M_petsc_final.mult(phi_i_petsc, M_phi) # M_phi = M * phi_i
    m_i = phi_i_petsc.dot(M_phi)

    if m_i < 1e-12:
        logger.warning(f"Modal mass for mode {i+1} is near zero ({m_i:.3e}). Skipping.")
        continue

    norm_factor = 1.0 / np.sqrt(m_i)
    phi_i_petsc_norm = phi_i_petsc.copy()
    phi_i_petsc_norm.scale(norm_factor)
    eigenvectors_petsc_normalized.append(phi_i_petsc_norm)
    eigenvectors_np_normalized.append(phi_i_petsc_norm.array.copy())

N_modes_normalized = len(eigenvectors_petsc_normalized)
if N_modes_normalized == 0:
    print("ERROR: No modes available after normalization check. Exiting reconstruction.")
    sys.exit(1)
else:
    print(f"Using {N_modes_normalized} mass-normalized modes for projection.")

# Ensure latent_dim_recon doesn't exceed available modes or config dim
latent_dim_recon = min(latent_dim_config, N_modes_normalized)
if latent_dim_recon == 0:
     print("ERROR: Latent dimension for reconstruction is zero. Exiting.")
     sys.exit(1)
print(f"Reconstructing using {latent_dim_recon} dimensions.")

# Select the modes and corresponding numpy arrays to use for reconstruction
modes_petsc_recon = eigenvectors_petsc_normalized[:latent_dim_recon]
modes_np_recon = eigenvectors_np_normalized[:latent_dim_recon] # Use normalized numpy arrays
# Create torch tensor from the *normalized* numpy modes for neural input
linear_modes_matrix_th = torch.tensor(np.column_stack(modes_np_recon), device=device, dtype=torch.float64)

# Store results
reconstruction_errors_Mnorm = []
reconstruction_errors_L2norm = []
energy_baseline_nl = []
energy_delta_linear = []
energy_delta_neural = []
energy_relative_delta_linear = []
energy_relative_delta_neural = []

# Pre-allocate FEniCSx Functions for energy calculation (using engine.V)
u_nl_func = Function(V, name="Nonlinear_Solution")
u_recon_linear_func = Function(V, name="Linear_Reconstruction")
u_recon_neural_func = Function(V, name="Neural_Reconstruction")

# Pre-allocate PETSc vectors (using engine.A/M for templates)
# u_nl_petsc = A.createVecLeft() # Defined in loop
u_recon_linear_petsc = A.createVecLeft()
u_recon_neural_petsc = A.createVecLeft()
M_u_nl = A.createVecRight()
diff_vec_linear = A.createVecLeft()
diff_vec_neural = A.createVecLeft()
M_diff_linear = A.createVecRight()
M_diff_neural = A.createVecRight()


# --- Reconstruction Loop ---
for k, u_nl_array in enumerate(nonlinear_solutions_np): # Loop through valid solutions
    u_nl_petsc = nonlinear_solutions_petsc[k]

    # --- Calculate Modal Coordinates (using normalized modes) ---
    modal_coords = np.zeros(latent_dim_recon)
    M_petsc_final.mult(u_nl_petsc, M_u_nl) # M_u_nl = M * u_nl
    for i in range(latent_dim_recon):
        phi_i_norm_petsc = modes_petsc_recon[i]
        q_i = phi_i_norm_petsc.dot(M_u_nl)
        modal_coords[i] = q_i

    # --- Linear Reconstruction (using normalized modes) ---
    u_recon_linear_petsc.zeroEntries()
    for i in range(latent_dim_recon):
        u_recon_linear_petsc.axpy(modal_coords[i], modes_petsc_recon[i])

    # --- Neural Reconstruction ---
    z_latent = torch.tensor(modal_coords, device=device, dtype=torch.float64).unsqueeze(0)
    print(f"    Debug: z_latent shape = {z_latent.shape}, dtype = {z_latent.dtype}")
    print(f"    Debug: z_latent = {z_latent}")
    with torch.no_grad():
        y_correction_th = neural_model(z_latent).squeeze(0)

    y_norm = torch.linalg.norm(y_correction_th).item()
    u_lin_norm = u_recon_linear_petsc.norm() # Get norm of linear part
    print(f"    Debug: ||Linear Recon|| = {u_lin_norm:.4e}, ||Neural Correction y|| = {y_norm:.4e}")


    # Check and handle size mismatch (important!)
    if y_correction_th.shape[0] != output_dim_model:
         logger.error(f"Neural network output size {y_correction_th.shape[0]} != DOF count {output_dim_model}")
         # Attempt to pad/truncate - THIS IS RISKY, indicates a setup error
         if y_correction_th.shape[0] < output_dim_model:
             padded_y = torch.zeros(output_dim_model, device=device, dtype=torch.float64)
             padded_y[:y_correction_th.shape[0]] = y_correction_th
             y_correction_th = padded_y
             logger.warning("Padded neural network output.")
         elif y_correction_th.shape[0] > output_dim_model:
             y_correction_th = y_correction_th[:output_dim_model]
             logger.warning("Truncated neural network output.")

    y_correction_np = y_correction_th.cpu().numpy()
    # Create PETSc vector carefully, ensuring compatibility
    y_correction_petsc = PETSc.Vec().createWithArray(y_correction_np.copy()) # Use copy

    # u_neural = u_linear_recon + y
    u_recon_neural_petsc.zeroEntries()
    u_recon_neural_petsc.axpy(1.0, u_recon_linear_petsc)
    u_recon_neural_petsc.axpy(1.0, y_correction_petsc)

    u_neu_norm = u_recon_neural_petsc.norm()
    print(f"    Debug: ||Neural Recon u_neu|| = {u_neu_norm:.4e}")


    # --- Calculate Displacement Errors (L2 and M-norm) ---
    # Linear Error
    diff_vec_linear.zeroEntries(); diff_vec_linear.axpy(1.0, u_nl_petsc); diff_vec_linear.axpy(-1.0, u_recon_linear_petsc)
    error_L2_norm_sq_lin = diff_vec_linear.dot(diff_vec_linear)
    M_petsc_final.mult(diff_vec_linear, M_diff_linear); error_M_norm_sq_lin = diff_vec_linear.dot(M_diff_linear)
    # Neural Error
    diff_vec_neural.zeroEntries(); diff_vec_neural.axpy(1.0, u_nl_petsc); diff_vec_neural.axpy(-1.0, u_recon_neural_petsc)
    error_L2_norm_sq_neu = diff_vec_neural.dot(diff_vec_neural)
    M_petsc_final.mult(diff_vec_neural, M_diff_neural); error_M_norm_sq_neu = diff_vec_neural.dot(M_diff_neural)
    # Norms of original solution
    unl_L2_norm_sq = u_nl_petsc.dot(u_nl_petsc)
    unl_M_norm_sq = u_nl_petsc.dot(M_u_nl)
    # Relative errors
    rel_err_L2_lin = np.sqrt(max(0, error_L2_norm_sq_lin) / unl_L2_norm_sq) if unl_L2_norm_sq > 1e-15 else (0.0 if error_L2_norm_sq_lin < 1e-15 else 1.0)
    rel_err_M_lin = np.sqrt(max(0, error_M_norm_sq_lin) / unl_M_norm_sq) if unl_M_norm_sq > 1e-15 else (0.0 if error_M_norm_sq_lin < 1e-15 else 1.0)
    rel_err_L2_neu = np.sqrt(max(0, error_L2_norm_sq_neu) / unl_L2_norm_sq) if unl_L2_norm_sq > 1e-15 else (0.0 if error_L2_norm_sq_neu < 1e-15 else 1.0)
    rel_err_M_neu = np.sqrt(max(0, error_M_norm_sq_neu) / unl_M_norm_sq) if unl_M_norm_sq > 1e-15 else (0.0 if error_M_norm_sq_neu < 1e-15 else 1.0)

    reconstruction_errors_L2norm.append((rel_err_L2_lin, rel_err_L2_neu))
    reconstruction_errors_Mnorm.append((rel_err_M_lin, rel_err_M_neu))

    # --- Calculate Energies ---
    u_nl_func.x.array[:] = u_nl_petsc.array[:]
    u_recon_linear_func.x.array[:] = u_recon_linear_petsc.array[:]
    u_recon_neural_func.x.array[:] = u_recon_neural_petsc.array[:]

    with torch.no_grad(): # Ensure no gradients are computed here
            # Convert PETSc vectors to torch tensors on the correct device
            u_nl_th = torch.tensor(u_nl_petsc.array, device=engine.device, dtype=torch.float64)
            u_recon_linear_th = torch.tensor(u_recon_linear_petsc.array, device=engine.device, dtype=torch.float64)
            u_recon_neural_th = torch.tensor(u_recon_neural_petsc.array, device=engine.device, dtype=torch.float64)

            # Call the energy calculator from the Routine instance
            # The calculator's forward method computes the energy
            try:
                E_nl = engine.energy_calculator(u_nl_th).item()
            except Exception as e_nl:
                logger.error(f"Error computing energy for NL solution: {e_nl}")
                E_nl = float('nan')

            try:
                E_linear = engine.energy_calculator(u_recon_linear_th).item()
            except Exception as e_lin:
                logger.error(f"Error computing energy for Linear recon: {e_lin}")
                E_linear = float('nan')

            try:
                E_neural = engine.energy_calculator(u_recon_neural_th).item()
                # <<<--- Add J check specifically for the tensor causing NaN --- >>>
                if np.isnan(E_neural):
                    logger.warning("NaN detected in Neural Energy. Re-checking J from PyTorch model if possible.")
                    # If your PyTorch energy model has a method to check J (it might not directly)
                    # you could call it here. Otherwise, rely on the fact that NaN occurred.
                    # Example placeholder:
                    # if hasattr(engine.energy_calculator, 'check_jacobian'):
                    #     engine.energy_calculator.check_jacobian(u_recon_neural_th)

            except Exception as e_neu:
                logger.error(f"Error computing energy for Neural recon: {e_neu}")
                E_neural = float('nan')


    # --- Calculate Energy Deltas ---
    # Use np.abs because energies might be NaN
    delta_E_lin = np.abs(E_linear - E_nl)
    delta_E_neu = np.abs(E_neural - E_nl)
    # Handle potential division by zero or NaN in E_nl
    E_nl_abs = np.abs(E_nl)
    rel_delta_E_lin = delta_E_lin / E_nl_abs if E_nl_abs > 1e-12 else (0.0 if delta_E_lin < 1e-12 else 1.0)
    rel_delta_E_neu = delta_E_neu / E_nl_abs if E_nl_abs > 1e-12 else (0.0 if delta_E_neu < 1e-12 else 1.0)

    # Store results (handle potential NaNs)
    energy_baseline_nl.append(E_nl)
    energy_delta_linear.append(delta_E_lin)
    energy_delta_neural.append(delta_E_neu)
    energy_relative_delta_linear.append(rel_delta_E_lin)
    energy_relative_delta_neural.append(rel_delta_E_neu)

    # Print results for this step (using np.nan_to_num for cleaner printing)
    print(f"  Load Mag {force_magnitudes_plot[k]:.2f}:")
    print(f"    Disp Err (L2) : Lin={rel_err_L2_lin*100:.2f}%, Neu={rel_err_L2_neu*100:.2f}%")
    print(f"    Disp Err (M)  : Lin={rel_err_M_lin*100:.2f}%, Neu={rel_err_M_neu*100:.2f}%")
    print(f"    Energy (Base) : {np.nan_to_num(E_nl):.4e}")
    print(f"    Energy Err(Abs): Lin={np.nan_to_num(delta_E_lin):.4e}, Neu={np.nan_to_num(delta_E_neu):.4e}")
    print(f"    Energy Err(Rel): Lin={np.nan_to_num(rel_delta_E_lin)*100:.2f}%, Neu={np.nan_to_num(rel_delta_E_neu)*100:.2f}%")
# --- End Reconstruction Loop ---


# --- 4. Validation Plots ---
# (This part remains unchanged, uses filtered magnitudes if needed)
print("\n--- Plotting Reconstruction Accuracy ---")

if not reconstruction_errors_L2norm or not energy_delta_linear:
     print("No valid reconstruction results to plot.")
else:
     errors_L2_lin = [err[0] for err in reconstruction_errors_L2norm]
     errors_L2_neu = [err[1] for err in reconstruction_errors_L2norm]
     errors_M_lin = [err[0] for err in reconstruction_errors_Mnorm]
     errors_M_neu = [err[1] for err in reconstruction_errors_Mnorm]

     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

     ax1.plot(force_magnitudes_plot, 100 * np.array(errors_L2_lin), 'o-', label=f'Linear L2 Error ({latent_dim_recon} modes)')
     ax1.plot(force_magnitudes_plot, 100 * np.array(errors_L2_neu), 's-', label=f'Neural L2 Error ({latent_dim_recon} modes)')
     ax1.set_ylabel("Relative Displacement Error (%)")
     ax1.set_title(f"Modal Reconstruction Displacement Error vs. Load ({latent_dim_recon} Modes)")
     ax1.legend()
     ax1.grid(True, linestyle=':')
     ax1.set_yscale('log')

     ax2.plot(force_magnitudes_plot, 100 * np.array(energy_relative_delta_linear), 'o-', label=f'Linear Rel. Energy Error ({latent_dim_recon} modes)')
     ax2.plot(force_magnitudes_plot, 100 * np.array(energy_relative_delta_neural), 's-', label=f'Neural Rel. Energy Error ({latent_dim_recon} modes)')
     ax2.set_ylabel("Relative Energy Error (%)")
     ax2.set_xlabel("Applied Force Magnitude")
     ax2.set_title(f"Modal Reconstruction Energy Error vs. Load ({latent_dim_recon} Modes)")
     ax2.legend(loc='upper left')
     ax2.grid(True, linestyle=':')
     ax2.set_yscale('log')

     plt.tight_layout()
     plt.show()

print("\n--- Reconstruction & Energy Comparison Finished ---")
print("="*60)


# --- 5. Optional: Visualize the last valid nonlinear solution and its reconstruction ---
# (Uses engine.domain, engine.V)

# Define visualization function (adapted from previous version)
def visualize_deformation(grid_pv, u_func_viz, title="Deformation"):
    """Visualize a displacement field using PyVista."""
    local_grid = grid_pv.copy()
    u_np = u_func_viz.x.array
    local_grid.point_data["displacement"] = u_np.reshape((-1, 3))
    local_grid["magnitude"] = np.linalg.norm(u_np.reshape((-1, 3)), axis=1)

    # Simple warp factor
    max_disp = np.max(local_grid["magnitude"])
    warp_factor = 0.1 / max(max_disp, 1e-9) # Aim for ~10% warp of max displacement

    warped = local_grid.warp_by_vector("displacement", factor=warp_factor)

    plotter = pyvista.Plotter(window_size=[800, 600])
    plotter.add_mesh(warped, scalars="magnitude", cmap="viridis", show_edges=True)
    plotter.add_mesh(grid_pv, style="wireframe", color="grey", opacity=0.2)
    plotter.add_title(f"{title} (Warp: {warp_factor:.1f}x)")
    plotter.show_axes_all()
    plotter.add_scalar_bar(title="Displacement Mag.")
    plotter.show()

if nonlinear_solutions_np:
    print("\n--- Visualizing Last Valid Nonlinear Step and Reconstructions ---")
    last_valid_idx = len(nonlinear_solutions_np) - 1

    # Create PyVista grid from engine's domain
    topology_viz, cell_types_viz, x_viz = plot.vtk_mesh(domain)
    grid_viz = pyvista.UnstructuredGrid(topology_viz, cell_types_viz, x_viz)

    # Visualize Full Nonlinear Solution
    u_nl_viz_func = Function(V)
    u_nl_viz_func.x.array[:] = nonlinear_solutions_petsc[last_valid_idx].array[:]
    visualize_deformation(grid_viz, u_nl_viz_func, title=f"Full Nonlinear Solution (Mag={force_magnitudes_plot[last_valid_idx]:.2f})")

    # Visualize Linear Reconstructed Solution
    u_recon_lin_viz_func = Function(V)
    u_recon_lin_viz_func.x.array[:] = u_recon_linear_petsc.array[:] # u_recon_linear_petsc holds the last result
    visualize_deformation(grid_viz, u_recon_lin_viz_func, title=f"Linear Reconstruction ({latent_dim_recon} modes)")

    # Visualize Neural Reconstructed Solution
    u_recon_neu_viz_func = Function(V)
    u_recon_neu_viz_func.x.array[:] = u_recon_neural_petsc.array[:] # u_recon_neural_petsc holds the last result
    visualize_deformation(grid_viz, u_recon_neu_viz_func, title=f"Neural Reconstruction ({latent_dim_recon} modes)")

else:
    print("Skipping final visualization as no valid nonlinear solutions were computed.")


print("\nScript completely finished.")