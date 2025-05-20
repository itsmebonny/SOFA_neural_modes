import Sofa
import SofaRuntime
import numpy as np 
import os
from Sofa import SofaDeformable
from time import process_time, time
import datetime
# add network path to the python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../network'))

import json

import os
import json
import datetime
import numpy as np

from scipy import sparse
from scipy.sparse.linalg import eigsh

# Attempt to import PETSc and SLEPc
try:
    from petsc4py import PETSc
    from slepc4py import SLEPc
    HAS_PETSC_SLEPC = True
    print("PETSc and SLEPc found. Will use for eigenvalue computation.")
except ImportError:
    HAS_PETSC_SLEPC = False
    print("PETSc and/or SLEPc not found. Will use SciPy for eigenvalue computation.")
    print("For potentially better performance and more solver options, consider installing petsc4py and slepc4py.")




class AnimationStepController(Sofa.Core.Controller):
    def __init__(self, node, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.mass = kwargs.get('mass')
        self.fem = kwargs.get('fem')
        self.linear_solver = kwargs.get('linear_solver')
        self.surface_topo = kwargs.get('surface_topo')
        self.MO1 = kwargs.get('MO1')
        self.fixed_box = kwargs.get('fixed_box')
     
        self.key = kwargs.get('key')
        self.iteration = kwargs.get("sample")
        self.start_time = 0
        self.root = node
        self.save = False
        self.l2_error, self.MSE_error = [], []
        self.l2_deformation, self.MSE_deformation = [], []
        self.RMSE_error, self.RMSE_deformation = [], []
        self.directory = kwargs.get('directory')
        # Add material properties to controller
        self.young_modulus = kwargs.get('young_modulus', 5000)
        self.poisson_ratio = kwargs.get('poisson_ratio', 0.25)
        self.density = kwargs.get('density', 10)
        self.volume = kwargs.get('volume', 1)
        self.total_mass = kwargs.get('total_mass', 10)
        self.mesh_filename = kwargs.get('mesh_filename', 'unknown')
        print(f"Using directory: {self.directory}")
        print(f"Material properties: E={self.young_modulus}, nu={self.poisson_ratio}, rho={self.density}")
        
        # Add eigenmodes visualization parameters
        self.show_modes = kwargs.get('show_modes', True)
        self.current_mode_index = -1  # Start with -1 to compute matrices first
        self.mode_animation_step = 0
        self.mode_animation_steps = kwargs.get('steps_per_mode', 100)  # Steps to animate each mode
        self.mode_scale = kwargs.get('mode_scale', 50.0)  # Scaling factor for eigenmodes
        self.num_modes_to_show = kwargs.get('num_modes_to_show', 5)
        self.modes_computed = False
        self.eigenvectors = None
        self.eigenvalues = None
        self.original_positions = None
        self.transition_steps = 20  # Steps to transition between modes
        self.pause_steps = 30  # Steps to pause at maximum amplitude
      


    def onSimulationInitDoneEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the scene graph initialisation.
        """
        self.inputs = []
        self.outputs = []       
        if self.save:
            if not os.path.exists('modal_data'):
                os.mkdir('modal_data')
            # get current time from computer and create a folder with that name
            if not os.path.exists(f'modal_data/{self.directory}'):
                os.makedirs(f'modal_data/{self.directory}')
            print(f"Saving data to modal_data/{self.directory}")
        self.sampled = False

        surface = self.surface_topo
        self.idx_surface = surface.triangles.value.reshape(-1)
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Store the original positions for mode animation
        if self.show_modes and self.MO1:
            self.original_positions = np.copy(self.MO1.position.value)
            print(f"Stored original positions with shape {self.original_positions.shape}")


    def onAnimateBeginEvent(self, event):
        self.start_time = process_time()

    
    def onAnimateEndEvent(self, event):
        
        
        # Compute matrices and eigenmodes on the first step
        if self.current_mode_index == -1:
            self.computeMatricesAndModes()
            self.current_mode_index = 0
            self.mode_animation_step = 0
            return
            

        
        # Display the current mode if modes were computed successfully
        if self.show_modes and self.modes_computed and self.eigenvectors is not None:
            self.displayMode(self.current_mode_index)
            self.current_mode_index = (self.current_mode_index + 1) % self.num_modes_to_show
            self.mode_animation_step += 1
        
        self.end_time = process_time()
            
    
    def computeMatricesAndModes(self):
        """Compute mass and stiffness matrices, then solve for eigenmodes"""
        print("Computing mass and stiffness matrices...")
        self.mass_matrix = self.mass.assembleMMatrix()
        self.stiffness_matrix = self.fem.assembleKMatrix()

        global HAS_PETSC_SLEPC
        
        print(f"Mass matrix shape: {self.mass_matrix.shape}")
        print(f"Stiffness matrix shape: {self.stiffness_matrix.shape}")
        
        # Get fixed DOFs from BoxROI
        fixed_indices = self.fixed_box.indices.value
        print(f"Number of fixed points: {len(fixed_indices)}")
        
        # Convert point indices to DOF indices (each point has 3 DOFs - x, y, z)
        fixed_dofs = []
        for idx in fixed_indices:
            fixed_dofs.extend([3*idx, 3*idx+1, 3*idx+2])  # x, y, z components
        
        fixed_dofs = np.array(fixed_dofs)
        print(f"Number of fixed DOFs: {len(fixed_dofs)}")
        
        # Convert matrices to CSR format for efficient modification
        if not isinstance(self.stiffness_matrix, sparse.csr_matrix):
            stiff_matrix = sparse.csr_matrix(self.stiffness_matrix)
        else:
            stiff_matrix = self.stiffness_matrix
            
        if not isinstance(self.mass_matrix, sparse.csr_matrix):
            mass_matrix = sparse.csr_matrix(self.mass_matrix)
        else:
            mass_matrix = self.mass_matrix
        
        # Get size of the system
        n = stiff_matrix.shape[0]

        
        
        # Convert to LIL format first for efficient modification
        stiff_matrix = stiff_matrix.tolil()
        mass_matrix = mass_matrix.tolil()

        # Apply constraints
        for dof in fixed_dofs:
            stiff_matrix[dof, :] = 0
            stiff_matrix[:, dof] = 0
            stiff_matrix[dof, dof] = 1.0
            
            mass_matrix[dof, :] = 0
            mass_matrix[:, dof] = 0
            # No need to set mass diagonal

        fixed_dofs = np.array(fixed_dofs)
        print(f"Number of fixed DOFs: {len(fixed_dofs)}")


        # Convert back to CSR for efficient computation
        stiff_matrix = stiff_matrix.tocsr()
        mass_matrix = mass_matrix.tocsr()
        stiff_matrix = -stiff_matrix  # Negate the stiffness matrix
        self.stiffness_matrix = stiff_matrix
        self.mass_matrix = mass_matrix
        
        print("Boundary conditions applied to matrices")

        # Create directory for matrices and mesh data
        matrices_dir = 'matrices'
        os.makedirs(matrices_dir, exist_ok=True)
        output_subdir = os.path.join(matrices_dir, self.timestamp)
        os.makedirs(output_subdir, exist_ok=True)


        # --- Save Fixed DOFs ---
        fixed_dofs_path = os.path.join(output_subdir, f'fixed_dofs_{self.timestamp}.npy')
        try:
            np.save(fixed_dofs_path, fixed_dofs)
            print(f"Fixed DOFs saved to {fixed_dofs_path}")
            fixed_dofs_saved = True
        except Exception as e:
            print(f"Error saving fixed DOFs: {e}")
            fixed_dofs_saved = False

        # --- Save Matrices ---
        # Use scipy.sparse.save_npz for sparse matrices
        try:
            sparse.save_npz(os.path.join(output_subdir, f'mass_matrix_{self.timestamp}.npz'), self.mass_matrix)
            sparse.save_npz(os.path.join(output_subdir, f'stiffness_matrix_{self.timestamp}.npz'), self.stiffness_matrix)
            print("Sparse matrices saved in .npz format.")
        except Exception as e:
            print(f"Error saving sparse matrices: {e}. Trying numpy save...")
            # Fallback to numpy save if sparse fails (might lose sparsity)
            np.save(os.path.join(output_subdir, f'mass_matrix_{self.timestamp}.npy'), self.mass_matrix)
            np.save(os.path.join(output_subdir, f'stiffness_matrix_{self.timestamp}.npy'), self.stiffness_matrix)

        # --- Save Mesh Data ---
        try:
            coordinates = self.MO1.rest_position.value
            # Assuming topology container is named 'triangleTopo' and holds tetrahedra
            # Adjust 'triangleTopo' if your topology container has a different name
            # Adjust '.tetrahedra' if it holds different element types (e.g., .hexahedra)
            elements = self.surface_topo.tetrahedra.value

            np.save(os.path.join(output_subdir, f'coordinates_{self.timestamp}.npy'), coordinates)
            np.save(os.path.join(output_subdir, f'elements_{self.timestamp}.npy'), elements)
            print(f"Mesh data saved: Coordinates shape {coordinates.shape}, Elements shape {elements.shape}")
            mesh_saved = True
        except AttributeError as e:
             print(f"Error accessing mesh data (check component names 'MO1', 'surface_topo', '.tetrahedra'): {e}")
             mesh_saved = False
        except Exception as e:
             print(f"An unexpected error occurred while saving mesh data: {e}")
             mesh_saved = False


        # --- Save Metadata ---
        metadata = {
            'timestamp': self.timestamp,
            'mesh_file_source': self.mesh_filename, # Original source mesh
            'young_modulus': self.young_modulus,
            'poisson_ratio': self.poisson_ratio,
            'density': self.density,
            'num_dofs': self.mass_matrix.shape[0],
            'matrix_format': 'npz' if 'sparse' in locals() and 'save_npz' in sparse.__dict__ else 'npy',
            'coordinates_file': f'coordinates_{self.timestamp}.npy' if mesh_saved else None,
            'elements_file': f'elements_{self.timestamp}.npy' if mesh_saved else None,
            'fixed_dofs_file': f'fixed_dofs_{self.timestamp}.npy' if fixed_dofs_saved else None, 
            'eigenvalues_file': f'eigenvalues_{self.timestamp}.npy',
            'eigenvectors_file': f'eigenvectors_{self.timestamp}.npy',
            'frequencies_file': f'frequencies_{self.timestamp}.npy'
        }

        metadata_path = os.path.join(output_subdir, f'metadata_{self.timestamp}.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Matrices, mesh data, and metadata saved to {output_subdir}")

        
        # Now compute eigenmodes
        if HAS_PETSC_SLEPC:
            print(f"Computing {self.num_modes_to_show} eigenmodes using SLEPc...")
            try:
                # Convert sparse matrices to PETSc format
                # from petsc4py import PETSc # Already imported at the top
                # from slepc4py import SLEPc # Already imported at the top
                
                # Convert to CSR format first if needed
                if not isinstance(self.stiffness_matrix, sparse.csr_matrix):
                    K_csr = self.stiffness_matrix.tocsr()
                    M_csr = self.mass_matrix.tocsr() 
                else:
                    K_csr = self.stiffness_matrix
                    M_csr = self.mass_matrix
                
                # Create PETSc matrices
                n = K_csr.shape[0]
                K_petsc = PETSc.Mat().createAIJ(size=(n, n), 
                                            csr=(K_csr.indptr, K_csr.indices, K_csr.data))
                M_petsc = PETSc.Mat().createAIJ(size=(n, n), 
                                            csr=(M_csr.indptr, M_csr.indices, M_csr.data))
                K_petsc.assemble()
                M_petsc.assemble()
                
                # Set up SLEPc eigensolver
                eigensolver = SLEPc.EPS().create()
                eigensolver.setOperators(K_petsc, M_petsc)
                eigensolver.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
                eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP) # Generalized Hermitian Eigenvalue Problem Kx = lambda Mx
                eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE) # Smallest magnitude eigenvalues
                eigensolver.setTarget(0.0) # Target eigenvalue is 0

                st = eigensolver.getST()
                st.setType(SLEPc.ST.Type.SINVERT) # Spectral Transformation: Shift-and-Invert
                st.setShift(0.0) # Shift for SINVERT
                            
                # Set dimensions and solve
                eigensolver.setDimensions(nev=self.num_modes_to_show + 6) # Request a few more modes for stability
                eigensolver.setFromOptions()
                
                print("Solving eigenvalue problem with SLEPc...")
                eigensolver.solve()
                
                # Get number of converged eigenvalues
                nconv = eigensolver.getConverged()
                print(f"Number of converged eigenvalues (SLEPc): {nconv}")
                
                if nconv > 0:
                    # Extract eigenvalues and eigenvectors
                    eigenvalues_list = []
                    eigenvectors_list = []
                    
                    for i in range(nconv):
                        eigenvalue = eigensolver.getEigenvalue(i).real
                        # SLEPc might return very small negative eigenvalues for GHEP due to numerical precision,
                        # especially for rigid body modes if not fully constrained.
                        # We are interested in positive eigenvalues for physical modes.
                        # if eigenvalue > 1e-9: # Filter out potentially non-physical or zero eigenvalues
                        eigenvalues_list.append(eigenvalue)
                        vr = K_petsc.createVecRight()
                        eigensolver.getEigenvector(i, vr)
                        eigenvectors_list.append(vr.getArray().copy()) # Ensure it's a copy
                        
                    if not eigenvalues_list:
                        print("Warning: No positive eigenvalues found with SLEPc. Check constraints and problem setup.")
                        self.modes_computed = False
                        return

                    # Convert to numpy arrays
                    self.eigenvalues = np.array(eigenvalues_list)
                    self.eigenvectors = np.array(eigenvectors_list).T # Transpose to have modes as columns
                    
                    # Sort eigenvalues (smallest magnitude first) and corresponding eigenvectors
                    idx = np.abs(self.eigenvalues).argsort()
                    self.eigenvalues = self.eigenvalues[idx]
                    self.eigenvectors = self.eigenvectors[:, idx]

                    # Select the requested number of modes (typically smallest non-zero)
                    # Often, the first few modes might be rigid body modes (eigenvalue close to 0)
                    # We usually want the smallest *positive* eigenvalues for elastic modes.
                    # For now, let's take the smallest `num_modes_to_show` by magnitude.
                    if len(self.eigenvalues) > self.num_modes_to_show:
                        self.eigenvalues = self.eigenvalues[:self.num_modes_to_show]
                        self.eigenvectors = self.eigenvectors[:, :self.num_modes_to_show]
                    
                    # Calculate natural frequencies
                    self.frequencies = np.sqrt(np.abs(self.eigenvalues)) / (2 * np.pi)

                    print("Checking eigenvector norms before saving (SLEPc):")
                    for i in range(min(5, self.eigenvectors.shape[1])):
                        mode_norm = np.linalg.norm(self.eigenvectors[:, i])
                        print(f"  Norm of Mode {i}: {mode_norm:.4e}")


                    #save the eigenvalues and eigenvector as the matrices 
                    np.save(f'{matrices_dir}/{self.timestamp}/eigenvalues_{self.timestamp}.npy', self.eigenvalues)
                    np.save(f'{matrices_dir}/{self.timestamp}/eigenvectors_{self.timestamp}.npy', self.eigenvectors)
                    np.save(f'{matrices_dir}/{self.timestamp}/frequencies_{self.timestamp}.npy', self.frequencies)
                    
                    print("Eigenmode computation successful with SLEPc!")
                    for i in range(min(nconv, len(self.eigenvalues))):
                        print(f"Mode {i+1}: λ = {self.eigenvalues[i]:.6e}, f = {self.frequencies[i]:.4f} Hz")
                        
                    # [Rest of your code for saving eigenmodes]
                    self.modes_computed = True
                    
                else:
                    print("No eigenvalues converged!")
                    self.modes_computed = False
                
            except Exception as e:
                print(f"Error computing eigenmodes with SLEPc: {e}")
                import traceback
                traceback.print_exc()
                self.modes_computed = False
                print("Falling back to SciPy due to SLEPc error.")
                HAS_PETSC_SLEPC = False # Force SciPy fallback

        if not HAS_PETSC_SLEPC: # If PETSc/SLEPc not available or failed
            print(f"Computing {self.num_modes_to_show} eigenmodes using SciPy eigsh...")
            try:
                # Ensure matrices are in CSR format for SciPy
                if not isinstance(self.stiffness_matrix, sparse.csr_matrix):
                    K_csr = self.stiffness_matrix.tocsr()
                else:
                    K_csr = self.stiffness_matrix
                if not isinstance(self.mass_matrix, sparse.csr_matrix):
                    M_csr = self.mass_matrix.tocsr()
                else:
                    M_csr = self.mass_matrix

                # SciPy's eigsh solves Kx = lambda Mx.
                # It expects K to be symmetric and M to be symmetric positive-definite.
                # We are looking for smallest magnitude eigenvalues, so we use sigma=0.
                # Note: K_csr is -K from SOFA, so we use -K_csr to get positive eigenvalues for Kx = lambda Mx
                
                # We need to solve K*v = lambda*M*v. Our K_csr is already -K_sofa.
                # So we solve (-K_sofa)*v = lambda*M*v.
                # eigsh finds eigenvalues of A relative to M. Here A = -K_sofa.
                # We want smallest eigenvalues, so sigma=0 is appropriate.
                # 'LM' means largest magnitude. For sigma=0, this means eigenvalues closest to 0.
                
                # Number of eigenvalues to compute. Add a few extra for stability and to discard zero modes.
                num_to_compute_scipy = self.num_modes_to_show + 6 
                
                # Ensure K_csr is symmetric. It should be from FEM.
                # Ensure M_csr is symmetric and positive definite.
                
                # We use K_csr directly (which is -Stiffness_SOFA)
                # eigsh solves A * x = lambda * M * x
                # Here, A = K_csr (our -Stiffness_SOFA)
                # We want smallest eigenvalues of K_SOFA, which are largest eigenvalues of -K_SOFA (if all are negative)
                # or eigenvalues of -K_SOFA closest to zero.
                
                # Let's use the original SOFA stiffness (positive definite) and find smallest eigenvalues
                # K_orig_sofa = -self.stiffness_matrix # self.stiffness_matrix is already -K_sofa
                
                # We want to solve K_sofa u = omega^2 M u
                # Our self.stiffness_matrix is -K_sofa.
                # So we solve -self.stiffness_matrix u = omega^2 M u
                # Or, self.stiffness_matrix u = -omega^2 M u
                # eigsh solves A u = lambda M u.
                # If A = self.stiffness_matrix, then lambda = -omega^2. We want lambda closest to 0 from negative side.
                
                # Let's use K_actual = -self.stiffness_matrix (which is the actual positive definite K)
                K_actual_csr = -K_csr 
                
                eigenvalues_scipy, eigenvectors_scipy = eigsh(
                    A=K_actual_csr, 
                    k=num_to_compute_scipy, 
                    M=M_csr, 
                    sigma=1e-6, # Look for eigenvalues near 0 (but not exactly 0 to avoid issues with singular K if unconstrained)
                    which='LM', # Largest Magnitude when sigma is used means closest to sigma.
                    v0=None, # Initial guess vector
                    ncv=None, # Number of Lanczos vectors
                    maxiter=None, # Max iterations
                    tol=1e-9, # Tolerance
                    return_eigenvectors=True
                )

                # Sort eigenvalues (smallest first) and corresponding eigenvectors
                idx = eigenvalues_scipy.argsort()
                self.eigenvalues = eigenvalues_scipy[idx]
                self.eigenvectors = eigenvectors_scipy[:, idx]

                # Filter out very small or negative eigenvalues if they are due to rigid body modes or numerical noise
                # Keep only the smallest positive eigenvalues
                positive_eigenvalues_mask = self.eigenvalues > 1e-9 # Threshold for "positive"
                self.eigenvalues = self.eigenvalues[positive_eigenvalues_mask]
                self.eigenvectors = self.eigenvectors[:, positive_eigenvalues_mask]
                
                if len(self.eigenvalues) > self.num_modes_to_show:
                    self.eigenvalues = self.eigenvalues[:self.num_modes_to_show]
                    self.eigenvectors = self.eigenvectors[:, :self.num_modes_to_show]
                elif len(self.eigenvalues) == 0:
                    print("SciPy eigsh did not find any positive eigenvalues. Check constraints and problem setup.")
                    self.modes_computed = False
                    return


                self.frequencies = np.sqrt(self.eigenvalues) / (2 * np.pi) # Eigenvalues from eigsh are omega^2

                print("Checking eigenvector norms before saving (SciPy):")
                for i in range(min(5, self.eigenvectors.shape[1])):
                    mode_norm = np.linalg.norm(self.eigenvectors[:, i])
                    print(f"  Norm of Mode {i}: {mode_norm:.4e}")


                #save the eigenvalues and eigenvector as the matrices 
                np.save(f'{matrices_dir}/{self.timestamp}/eigenvalues_{self.timestamp}.npy', self.eigenvalues)
                np.save(f'{matrices_dir}/{self.timestamp}/eigenvectors_{self.timestamp}.npy', self.eigenvectors)
                np.save(f'{matrices_dir}/{self.timestamp}/frequencies_{self.timestamp}.npy', self.frequencies)
                
                print("Eigenmode computation successful with SLEPc!")
                for i in range(min(nconv, len(self.eigenvalues))):
                    print(f"Mode {i+1}: λ = {self.eigenvalues[i]:.6e}, f = {self.frequencies[i]:.4f} Hz")
                    
                # [Rest of your code for saving eigenmodes]
                self.modes_computed = True
                
   
                    
            except Exception as e:
                print(f"Error computing eigenmodes with SLEPc: {e}")
                import traceback
                traceback.print_exc()
                self.modes_computed = False



    def displayMode(self, mode_index):
        """Directly display an eigenmode with proper scaling"""
        if not self.modes_computed or self.eigenvectors is None:
            print("No modes computed yet")
            return
        
        # Get number of nodes in the mesh
        num_nodes = self.MO1.rest_position.shape[0]
        
        # Safety check
        if mode_index >= self.eigenvectors.shape[1]:
            mode_index = 0
        
        # Get the current eigenmode
        current_mode = self.eigenvectors[:, mode_index]
        
        # Reshape the mode for easier processing
        mode_reshaped = current_mode.reshape(-1, 3)
        
        # Calculate model characteristic size (diagonal of bounding box)
        bbox_min = np.min(self.MO1.rest_position.value, axis=0)
        bbox_max = np.max(self.MO1.rest_position.value, axis=0)
        model_size = np.linalg.norm(bbox_max - bbox_min)
        
        # Normalize the mode by its maximum displacement
        max_displacement = np.max(np.linalg.norm(mode_reshaped, axis=1))
        if max_displacement > 1e-10:  # Avoid division by zero
            normalized_mode = current_mode / max_displacement
        else:
            normalized_mode = current_mode
        
        # Calculate proper scaling factor (10% of model size)
        scale_factor = 0.1 * model_size
        
        # Apply the scaling
        displacement = scale_factor * normalized_mode
        
        # Reshape and apply the displacement
        if len(displacement) == 3 * num_nodes:
            displacement_reshaped = displacement.reshape(num_nodes, 3)
        else:
            print(f"Warning: Mode shape ({len(displacement)}) doesn't match expected size ({3 * num_nodes}).")
            if len(displacement) > 3 * num_nodes:
                displacement_reshaped = displacement[:3 * num_nodes].reshape(num_nodes, 3)
            else:
                padded = np.zeros(3 * num_nodes)
                padded[:len(displacement)] = displacement
                displacement_reshaped = padded.reshape(num_nodes, 3)
        
        # Update the mechanical object positions
        with self.MO1.position.writeable() as pos:
            pos[:] = self.MO1.rest_position + displacement_reshaped
        
        # Print mode information
        print(f"Displaying mode {mode_index + 1}/{self.eigenvectors.shape[1]}")
        print(f"Eigenvalue: {self.eigenvalues[mode_index]:.6e}, Frequency: {self.frequencies[mode_index]:.4f} Hz")
        print(f"Scale factor: {scale_factor:.4f} (based on model size: {model_size:.4f})")
            
        
    

    def close(self):
        print("Closing simulation")


def createScene(rootNode, config=None, directory=None, sample=0, key=(0, 0, 0), *args, **kwargs):
    """
    Create SOFA scene with parameters from a YAML config file
    
    Args:
        rootNode: SOFA root node
        config: Dict with configuration parameters from YAML file (or None to use defaults)
        directory: Output directory
        sample: Sample index
        key: Key tuple (x, y, r) for the simulation
    """
    # Handle default config if not provided
    if config is None:
        config = {
            'physics': {'gravity': [0, 0, 0], 'dt': 0.01},
            'material': {'youngs_modulus': 5000, 'poissons_ratio': 0.25, 'density': 10},
            'mesh': {'filename': 'mesh/beam_615.msh'},
            'constraints': {'fixed_box': [-0.01, -0.01, -0.02, 1.01, 0.01, 0.02]}
        }
    
    # Set basic simulation parameters
    rootNode.dt = config['physics'].get('dt', 0.01)
    rootNode.gravity = config['physics'].get('gravity', [0, 0, 0])
    rootNode.name = 'root'

    # Add required plugins
    required_plugins = [
        'MultiThreading',
        'Sofa.Component.Constraint.Projective',
        'Sofa.Component.Engine.Select',
        'Sofa.Component.LinearSolver.Iterative',
        'Sofa.Component.LinearSolver.Direct',
        'Sofa.Component.Mass',
        'Sofa.Component.IO.Mesh',
        'Sofa.Component.Mapping.Linear', 
        'Sofa.Component.MechanicalLoad',
        'Sofa.Component.ODESolver.Backward',
        'Sofa.Component.SolidMechanics.FEM.Elastic',
        'Sofa.Component.StateContainer',
        'Sofa.Component.Topology.Container.Dynamic',
        'Sofa.Component.Topology.Container.Grid',
        'Sofa.Component.Visual',
        'Sofa.GL.Component.Rendering3D',
        'SofaMatrix'
    ]
    
    for plugin in required_plugins:
        rootNode.addObject('RequiredPlugin', name=plugin)

    # Add basic scene components
    rootNode.addObject('DefaultAnimationLoop')
    rootNode.addObject('DefaultVisualManagerLoop') 
    rootNode.addObject('VisualStyle', displayFlags="showBehaviorModels showCollisionModels")

    # Get material properties from config
    young_modulus = config['material'].get('youngs_modulus', 5000)
    poisson_ratio = config['material'].get('poissons_ratio', 0.25)
    density = config['material'].get('density', 10)
    volume = config['material'].get('volume', 1)
    num_modes_to_show = config['model'].get('latent_dim', 5)
    total_mass = density * volume
    print(f"Using E={young_modulus}, nu={poisson_ratio}, rho={density}, V={volume}, M={total_mass}")

    # Calculate Lamé parameters
    mu = young_modulus / (2 * (1 + poisson_ratio))
    lam = young_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    print(f"Using mu={mu}, lambda={lam}")
    mu_lam_str = f"{mu} {lam}"

    # Get mesh filename from config
    mesh_filename = config['mesh'].get('filename', 'mesh/beam_615.msh')

    # Create high resolution solution node
    exactSolution = rootNode.addChild('HighResSolution2D', activated=True)
    exactSolution.addObject('MeshGmshLoader', name='grid', filename=mesh_filename)
    surface_topo = exactSolution.addObject('TetrahedronSetTopologyContainer', name='triangleTopo', src='@grid')
    MO1 = exactSolution.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@grid')
    
    # Add system components
    mass = exactSolution.addObject('MeshMatrixMass', totalMass=total_mass, name="SparseMass", topology="@triangleTopo")
    
    # Get solver parameters from config
    rayleighStiffness = config['physics'].get('rayleigh_stiffness', 0.1)
    rayleighMass = config['physics'].get('rayleigh_mass', 0.1)
    
    solver = exactSolution.addObject('EulerImplicitSolver', name="ODEsolver", 
                                   rayleighStiffness=rayleighStiffness, 
                                   rayleighMass=rayleighMass)
    
    linear_solver = exactSolution.addObject('CGLinearSolver', 
                                          template="CompressedRowSparseMatrixMat3x3d",
                                          iterations=config['physics'].get('solver_iterations', 1000), 
                                          tolerance=config['physics'].get('solver_tolerance', 1e-10), 
                                          threshold=config['physics'].get('solver_threshold', 1e-10), 
                                          warmStart=True)
    
    # fem = exactSolution.addObject('TetrahedronHyperelasticityFEMForceField',
    #                             name="FEM", 
    #                             materialName="NeoHookean", 
    #                             ParameterSet=mu_lam_str)

    fem = exactSolution.addObject('TetrahedronFEMForceField', # Store reference
                           name="LinearFEM",
                           youngModulus=young_modulus,
                           poissonRatio=poisson_ratio,
                           method="small") 

                            
    
    # Get constraint box from config
    fixed_box_coords = config['constraints'].get('fixed_box', [-0.01, -0.01, -0.02, 1.01, 0.01, 0.02])
    fixed_box = exactSolution.addObject('BoxROI', 
                                      name='ROI',
                                      box=" ".join(str(x) for x in fixed_box_coords), 
                                      drawBoxes=True)
    
    exactSolution.addObject('FixedConstraint', indices="@ROI.indices")
    
    # Add visual model
    visual = exactSolution.addChild("visual")
    visual.addObject('OglModel', src='@../DOFs', color='0 1 0 1')
    visual.addObject('BarycentricMapping', input='@../DOFs', output='@./')



    # Create and add controller with all components
    controller = AnimationStepController(rootNode, 
                                       mass=mass, 
                                       fem=fem,
                                       linear_solver=linear_solver,
                                       surface_topo=surface_topo,
                                       MO1=MO1, 
                                       fixed_box=fixed_box,
                                       directory=directory, 
                                       sample=sample,
                                       key=key, 
                                       young_modulus=young_modulus,
                                       poisson_ratio=poisson_ratio,
                                       density=density,
                                       volume=volume,
                                       total_mass=total_mass,
                                       mesh_filename=mesh_filename,
                                       num_modes_to_show=num_modes_to_show,
                                       **kwargs)
    rootNode.addObject(controller)

    return rootNode, controller

if __name__ == "__main__":
    import Sofa.Gui
    from tqdm import tqdm
    import yaml
    import argparse
    
    # Add argument parser
    parser = argparse.ArgumentParser(description='SOFA Matrix Creation and Modal Analysis')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--gui', action='store_true', help='Enable GUI mode')
    parser.add_argument('--steps', type=int, default=2, help='Number of steps to run in headless mode')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {args.config}")
    except Exception as e:
        print(f"Error loading config from {args.config}: {str(e)}")
        print("Using default configuration")
        config = None
    
    # Required plugins
    required_plugins = [
        'MultiThreading',
        'Sofa.Component.Constraint.Projective',
        'Sofa.Component.Engine.Select',
        'Sofa.Component.LinearSolver.Iterative',
        'Sofa.Component.LinearSolver.Direct',
        'Sofa.Component.Mass',
        'Sofa.Component.IO.Mesh',
        'Sofa.Component.Mapping.Linear', 
        'Sofa.Component.MechanicalLoad',
        'Sofa.Component.ODESolver.Backward',
        'Sofa.Component.SolidMechanics.FEM.Elastic',
        'Sofa.Component.StateContainer',
        'Sofa.Component.Topology.Container.Dynamic',
        'Sofa.Component.Topology.Container.Grid',
        'Sofa.Component.Visual',
        'SofaMatrix'
    ]

    # Import all required plugins
    # for plugin in required_plugins:
    #     SofaRuntime.importPlugin(plugin)

    # Create simulation directory
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    output_dir = os.path.join('modal_data', timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Setup and run simulation
    root = Sofa.Core.Node("root")
    rootNode, controller = createScene(
        root,
        config=config,
        directory=timestamp,
        sample=0,
        key=(0, 0, 0)  # default key values
    )

    # Initialize simulation
    Sofa.Simulation.initRoot(root)
    controller.save = True

    if args.gui:
        Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
        Sofa.Gui.GUIManager.createGUI(root, __file__)
        Sofa.Gui.GUIManager.SetDimension(800, 600)
        Sofa.Gui.GUIManager.MainLoop(root)
        Sofa.Gui.GUIManager.closeGUI()
    else:
        for _ in tqdm(range(args.steps), desc="Simulation progress"):
            Sofa.Simulation.animate(root, root.dt.value)

    controller.close()

   