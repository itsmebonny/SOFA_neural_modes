import Sofa
import SofaRuntime
import numpy as np 
import os
from Sofa import SofaDeformable
from time import process_time, time
import datetime
from sklearn.preprocessing import MinMaxScaler
# add network path to the python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../network'))

import json

import os
import json
import datetime
import numpy as np

import glob
import traceback
from scipy import sparse
from scipy.sparse.linalg import eigsh



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

        # Convert back to CSR for efficient computation
        stiff_matrix = stiff_matrix.tocsr()
        mass_matrix = mass_matrix.tocsr()
        stiff_matrix = -stiff_matrix  # Negate the stiffness matrix
        self.stiffness_matrix = stiff_matrix
        self.mass_matrix = mass_matrix
        
        print("Boundary conditions applied to matrices")
        
        # Create directory for matrices
        matrices_dir = 'matrices'
        os.makedirs(matrices_dir, exist_ok=True)
        matrices_dir = 'matrices'
        os.makedirs(matrices_dir, exist_ok=True)
        if not os.path.exists(f'{matrices_dir}/{self.timestamp}'):
            os.makedirs(f'{matrices_dir}/{self.timestamp}')
        
        # Save matrices
        np.save(f'{matrices_dir}/mass_matrix_{self.timestamp}.npy', self.mass_matrix)
        np.save(f'{matrices_dir}/stiffness_matrix_{self.timestamp}.npy', self.stiffness_matrix)
        
        # Save metadata
        metadata = {
            'timestamp': self.timestamp,
            'mesh_file': self.mesh_filename,
            'young_modulus': self.young_modulus,
            'poisson_ratio': self.poisson_ratio,
            'density': self.density,
            'size': self.mass_matrix.shape[0]
        }
        
        with open(f'{matrices_dir}/metadata_{self.timestamp}.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Matrices saved to {matrices_dir} with timestamp {self.timestamp}")
        
        # Now compute eigenmodes directly
        print(f"Computing {self.num_modes_to_show} eigenmodes using SLEPc...")
        try:
            # Convert sparse matrices to PETSc format
            from petsc4py import PETSc
            from slepc4py import SLEPc
            
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
            eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
            eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
            eigensolver.setTarget(0.0)

            st = eigensolver.getST()
            st.setType(SLEPc.ST.Type.SINVERT)
            st.setShift(0.0)
                        
            # Set dimensions and solve
            eigensolver.setDimensions(nev=self.num_modes_to_show)
            eigensolver.setFromOptions()
            
            print("Solving eigenvalue problem...")
            eigensolver.solve()
            
            # Get number of converged eigenvalues
            nconv = eigensolver.getConverged()
            print(f"Number of converged eigenvalues: {nconv}")
            
            if nconv > 0:
                # Extract eigenvalues and eigenvectors
                eigenvalues = []
                eigenvectors = np.zeros((n, nconv))
                
                for i in range(min(nconv, self.num_modes_to_show)):
                    eigenvalue = eigensolver.getEigenvalue(i).real
                    eigenvalues.append(eigenvalue)
                    
                    # Extract eigenvector
                    vr = K_petsc.createVecRight()
                    eigensolver.getEigenvector(i, vr)
                    eigenvectors[:, i] = vr.getArray()
                    
                # Convert to numpy arrays
                self.eigenvalues = np.array(eigenvalues)
                self.eigenvectors = eigenvectors
                
                # Sort eigenvalues (smallest first)
                idx = self.eigenvalues.argsort()
                self.eigenvalues = self.eigenvalues[idx]
                self.eigenvectors = self.eigenvectors[:, idx]
                
                # Calculate natural frequencies
                self.frequencies = np.sqrt(np.abs(self.eigenvalues)) / (2 * np.pi)

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
    
    fem = exactSolution.addObject('TetrahedronHyperelasticityFEMForceField',
                                name="FEM", 
                                materialName="NeoHookean", 
                                ParameterSet=mu_lam_str)
                            
    
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
        "Sofa.GL.Component.Rendering3D",
        "Sofa.GL.Component.Shader",
        "Sofa.Component.StateContainer",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.IO.Mesh",
        "Sofa.Component.MechanicalLoad",
        "Sofa.Component.Engine.Select",
        "Sofa.Component.SolidMechanics.FEM.Elastic",
        "MultiThreading",
        "SofaMatrix",
        "Sofa.Component.SolidMechanics.FEM.HyperElastic"
    ]

    # Import all required plugins
    for plugin in required_plugins:
        SofaRuntime.importPlugin(plugin)

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
    Sofa.Simulation.init(root)
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

   