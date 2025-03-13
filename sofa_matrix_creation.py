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

import matplotlib.pyplot as plt


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
        self.mesh_filename = kwargs.get('mesh_filename', 'unknown')
        print(f"Using directory: {self.directory}")
        print(f"Material properties: E={self.young_modulus}, nu={self.poisson_ratio}, rho={self.density}")
        
      
        


    def onSimulationInitDoneEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the scene graph initialisation.
        """
        self.inputs = []
        self.outputs = []       
        if self.save:
            if not os.path.exists('modal_data'):
                os.mkdir('modal_data')
            # get current time from computer format yyyy-mm-dd-hh-mm-ss and create a folder with that name
            if not os.path.exists(f'modal_data/{self.directory}'):
                os.makedirs(f'modal_data/{self.directory}')
            print(f"Saving data to modal_data/{self.directory}")
        self.sampled = False

        surface = self.surface_topo

        self.idx_surface = surface.triangles.value.reshape(-1)

        self.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')






    def onAnimateBeginEvent(self, event):
    
    
        self.start_time = process_time()
       



    def onAnimateEndEvent(self, event):
        self.end_time = process_time()
        
        # Get matrices from SOFA
        self.mass_matrix = self.mass.assembleMMatrix()
        self.stiffness_matrix = self.fem.assembleKMatrix()
        
        print(f"Mass matrix shape: {self.mass_matrix.shape}")
        print(f"Stiffness matrix shape: {self.stiffness_matrix.shape}")

        print(f"Matrices type: {type(self.mass_matrix)}")

        
        # Create standard directory for matrices
        matrices_dir = 'matrices'
        os.makedirs(matrices_dir, exist_ok=True)
        
        # Generate timestamp for unique identification
        
        
        # Save matrices to standard location
        np.save(f'{matrices_dir}/mass_matrix_{self.timestamp}.npy', self.mass_matrix)
        np.save(f'{matrices_dir}/stiffness_matrix_{self.timestamp}.npy', self.stiffness_matrix)
        
        # Save metadata about the mesh and material properties using controller's stored values
        metadata = {
            'timestamp': self.timestamp,
            'mesh_file': self.mesh_filename,
            'young_modulus': self.young_modulus,
            'poisson_ratio': self.poisson_ratio,
            'density': self.density,
            'size': self.mass_matrix.shape[0]
        }
        
        with open(f'{matrices_dir}/metadata_{timestamp}.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Matrices saved to {matrices_dir} with timestamp {self.timestamp}")
            
        

    
    
    def close(self):
        print("Closing simulation")

def compute_modal_analysis(matrices_path=None, timestamp=None, num_modes=20, 
                          visualize=True, save_modes=True, output_dir=None, 
                          mesh_file=None, visualize_3d=False):
    """
    Compute generalized eigenvalue problem for modal analysis using SciPy sparse matrix operations
    """
    print("Computing modal analysis...")
    
    # Set default matrices path
    if matrices_path is None:
        matrices_path = 'matrices'
    
    # Find the latest matrices if timestamp not specified
    if timestamp is None:
        mass_files = glob.glob(f'{matrices_path}/mass_matrix_*.npy')
        if not mass_files:
            raise FileNotFoundError(f"No mass matrices found in {matrices_path}")
            
        # Get latest file by timestamp
        latest_file = max(mass_files, key=os.path.getctime)
        timestamp = latest_file.split('mass_matrix_')[1].split('.npy')[0]
        print(f"Using latest matrices with timestamp {timestamp}")
    
    # Load matrices
    mass_matrix_file = f'{matrices_path}/mass_matrix_{timestamp}.npy'
    stiff_matrix_file = f'{matrices_path}/stiffness_matrix_{timestamp}.npy'
    
    if not os.path.exists(mass_matrix_file) or not os.path.exists(stiff_matrix_file):
        raise FileNotFoundError(f"Matrix files for timestamp {timestamp} not found")
    
    print(f"Loading matrices from {mass_matrix_file} and {stiff_matrix_file}")
    
    # Load matrices
    try:
        mass_matrix_obj = np.load(mass_matrix_file, allow_pickle=True)
        stiff_matrix_obj = np.load(stiff_matrix_file, allow_pickle=True)
        
        # Handle 0-dimensional array (scalar container)
        if isinstance(mass_matrix_obj, np.ndarray) and mass_matrix_obj.ndim == 0:
            print("Detected 0-dimensional array. Extracting contained object...")
            mass_matrix_obj = mass_matrix_obj.item()
            stiff_matrix_obj = stiff_matrix_obj.item()
        
        # Ensure matrices are in CSR format for efficiency
        if not isinstance(mass_matrix_obj, sparse.csr_matrix):
            if hasattr(mass_matrix_obj, 'tocsr'):
                mass_matrix = mass_matrix_obj.tocsr()
                stiff_matrix = stiff_matrix_obj.tocsr()
                print("Converted matrices to CSR format")
            else:
                raise TypeError(f"Cannot convert {type(mass_matrix_obj)} to CSR format")
        else:
            mass_matrix = mass_matrix_obj
            stiff_matrix = stiff_matrix_obj
        
        print(f"Matrix shapes: Mass {mass_matrix.shape}, Stiffness {stiff_matrix.shape}")
        
        # Check if matrices are symmetric (they should be for physical correctness)
        is_mass_symmetric = ((mass_matrix - mass_matrix.T).data ** 2).sum() < 1e-10
        is_stiff_symmetric = ((stiff_matrix - stiff_matrix.T).data ** 2).sum() < 1e-10
        
        if not is_mass_symmetric:
            print("Warning: Mass matrix is not symmetric. Symmetrizing...")
            mass_matrix = 0.5 * (mass_matrix + mass_matrix.T)
        
        if not is_stiff_symmetric:
            print("Warning: Stiffness matrix is not symmetric. Symmetrizing...")
            stiff_matrix = 0.5 * (stiff_matrix + stiff_matrix.T)
        
        # Solve the generalized eigenvalue problem: K φ = λ M φ
        print(f"Solving generalized eigenvalue problem for {num_modes} modes...")
        try:
            eigenvalues, eigenvectors = eigsh(stiff_matrix, k=num_modes, M=mass_matrix, 
                                            sigma=1e-5, which='LM', maxiter=10000, tol=1e-3)
            
            # Sort by eigenvalue (smallest first)
            idx = eigenvalues.argsort()
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Calculate natural frequencies in Hz
            frequencies = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)
            
            print("Modal analysis completed successfully!")
            
            # Print first few eigenvalues and frequencies
            print("\nEigenvalues (λ) and Natural Frequencies (Hz):")
            for i in range(min(10, num_modes)):
                print(f"Mode {i+1}: λ = {eigenvalues[i]:.6e}, f = {frequencies[i]:.4f} Hz")
            
            # Save results if requested
            if save_modes:
                # Use the same output directory as the matrices
                output_dir = matrices_path  # Save in the matrices directory
                
                # Create a subdirectory with the timestamp
                output_dir = os.path.join(output_dir, timestamp)
                os.makedirs(output_dir, exist_ok=True)
                
                # Save eigenvalues and eigenvectors
                np.save(f'{output_dir}/eigenvalues_{timestamp}.npy', eigenvalues)
                np.save(f'{output_dir}/eigenvectors_{timestamp}.npy', eigenvectors)
                np.save(f'{output_dir}/frequencies_{timestamp}.npy', frequencies)
                
                # Save a summary in JSON format
                summary = {
                    'timestamp': timestamp,
                    'num_modes': num_modes,
                    'eigenvalues': eigenvalues.tolist(),
                    'frequencies': frequencies.tolist(),
                }
                
                with open(f'{output_dir}/modal_summary_{timestamp}.json', 'w') as f:
                    json.dump(summary, f, indent=2)
                
                print(f"Modal analysis results saved to {output_dir}")
            
            # Visualize if requested
            if visualize_3d and mesh_file:
                visualize_linear_modes_vedo(
                    eigenvectors=eigenvectors,
                    mesh_file=mesh_file,
                    num_modes=min(10, num_modes),  # Limit to 10 modes for visualization
                    output_dir=output_dir,
                    timestamp=timestamp
                )
            
            return eigenvalues, eigenvectors, frequencies
            
        except Exception as e:
            print(f"Error solving eigenvalue problem: {e}")
            print(traceback.format_exc())
            return None, None, None
            
    except Exception as e:
        print(f"Error loading matrices: {e}")
        print(traceback.format_exc())
        return None, None, None

def visualize_linear_modes_vedo(eigenvectors, mesh_file=None, num_modes=5, scale=None, output_dir=None, timestamp=None):
    """
    Visualize linear modes (eigenvectors) using vedo
    """
    try:
        from vedo import Mesh, Plotter, load, Points, vector
        import numpy as np
        import traceback
    except ImportError:
        print("Required packages not found. Install with:")
        print("pip install vedo")
        return None
    
    print("\nVisualizing linear modes using vedo...")
    
    # Figure out the eigenvector structure
    ev_shape = eigenvectors.shape
    print(f"Eigenvector matrix shape: {ev_shape}")
    
    # For FEM with 3 DOFs per node, the number of nodes should be ev_shape[0]/3
    n_nodes = ev_shape[0] // 3
    print(f"Detected {n_nodes} nodes with 3 DOFs each")
    
    # Reshape eigenvectors to organize by nodes (each node has 3 DOFs)
    # This assumes eigenvectors are stored as [x1,y1,z1,x2,y2,z2,...] for each mode
    modes_by_node = []
    for mode_idx in range(min(num_modes, ev_shape[1])):
        mode_vec = eigenvectors[:, mode_idx]
        # Reshape to have 3 DOFs per row
        mode_reshaped = np.zeros((n_nodes, 3))
        for i in range(3):  # x, y, z components
            mode_reshaped[:, i] = mode_vec[i::3]
        modes_by_node.append(mode_reshaped)
    
    print(f"Reshaped mode vectors to {len(modes_by_node)} modes of shape {modes_by_node[0].shape}")
    
    # Create base points - either from mesh or create a grid
    if mesh_file is None:
        print("No mesh file provided, creating uniform grid for visualization")
        # Create a uniform grid as base points
        grid_size = int(np.cbrt(n_nodes))
        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)
        z = np.linspace(0, 1, grid_size)
        xx, yy, zz = np.meshgrid(x, y, z)
        base_points = np.vstack([xx.flatten(), yy.flatten(), zz.flatten()]).T
        # Truncate or pad if necessary
        if base_points.shape[0] < n_nodes:
            # Pad with zeros
            base_points = np.vstack([base_points, np.zeros((n_nodes - base_points.shape[0], 3))])
        elif base_points.shape[0] > n_nodes:
            # Truncate
            base_points = base_points[:n_nodes, :]
    else:
        print(f"Loading mesh from {mesh_file}")
        try:
            mesh = load(mesh_file)
            if mesh is None:
                raise ValueError("Mesh loaded as None")
            base_points = mesh.points()
            print(f"Loaded mesh with {len(base_points)} points")
            # Check if the number of points matches eigenvector data
            if len(base_points) != n_nodes:
                print(f"Warning: Mesh has {len(base_points)} points but eigenvectors suggest {n_nodes} nodes.")
                print("Using eigenvector structure for visualization...")
                # Create evenly spaced points in a line for visualization
                base_points = np.zeros((n_nodes, 3))
                base_points[:, 0] = np.linspace(0, 1, n_nodes)
        except Exception as e:
            print(f"Error loading mesh with vedo: {e}")
            print(traceback.format_exc())
            print("Creating uniform grid for visualization...")
            # Create a uniform grid for visualization
            size = int(np.cbrt(n_nodes)) + 1  # Cubic root of n_nodes, rounded up
            x = np.linspace(0, 1, size)
            y = np.linspace(0, 1, size)
            z = np.linspace(0, 1, size)
            xx, yy, zz = np.meshgrid(x, y, z)
            base_points = np.vstack([xx.flatten(), yy.flatten(), zz.flatten()]).T[:n_nodes]
    
    print(f"Base points shape: {base_points.shape}")
    
    # Use fixed scale or auto-compute
    if scale is None:
        # Auto-compute scale based on mesh size and eigenvectors magnitude
        mesh_size = np.max(np.ptp(base_points, axis=0))
        max_mode_disp = 0
        for mode_data in modes_by_node:
            mode_max = np.max(np.abs(mode_data))
            max_mode_disp = max(max_mode_disp, mode_max)
        
        if max_mode_disp > 0:
            scale = 0.2 * mesh_size / max_mode_disp
        else:
            scale = 1.0
    
    print(f"Using displacement scale factor: {scale}")
    
    # Create a plotter with explicit sizing
    plotter = Plotter(shape=(min(num_modes, len(modes_by_node)), 3), interactive=True)
    
    # Visualize each mode
    for mode_idx, mode_data in enumerate(modes_by_node[:num_modes]):
        # Generate displacements at different scales
        scales = [-1.0, 0, 1.0]  # Show negative, zero and positive displacement
        
        for j, factor in enumerate(scales):
            # Scale the mode shape
            scaled_displacement = mode_data * scale * factor
            
            # Create new displaced points
            displaced_points = base_points + scaled_displacement
            
            # Create a new Points object with the displaced points
            local_mesh = Points(displaced_points, r=5)
            
            # Compute displacement magnitude for coloring
            displacement_magnitude = np.linalg.norm(scaled_displacement, axis=1)
            local_mesh.pointdata["displacement_magnitude"] = displacement_magnitude
            
            # Add to plot with colormap
            plotter.at([mode_idx, j]).show(
                local_mesh.cmap("jet", "displacement_magnitude"),
                title=f"Mode {mode_idx+1} ({'+' if factor > 0 else '-' if factor < 0 else '0'})"
            )
    
    # Save to file if requested
    if output_dir and timestamp:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/mode_shapes_{timestamp}.png"
        plotter.screenshot(filename)
        print(f"Mode shape visualization saved to {filename}")
    
    # Show the plot
    plotter.show()
    
    return plotter



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
    total_mass = density * volume
    print(f"Using E={young_modulus}, nu={poisson_ratio}, rho={total_mass}")

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
                                       density=total_mass,
                                       mesh_filename=mesh_filename,
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

    compute_modes = True
    mesh_file = config['mesh'].get('filename', 'mesh/beam_615.msh') if config else None


    # Compute modal analysis if requested
    if compute_modes:
        # Get parameters from config if not provided as arguments
        matrices_config = config['matrices']
        matrices_path = matrices_config.get('matrices_path', 'matrices')
        timestamp = matrices_config.get('timestamp', None)
        num_modes = config['model'].get('latent_dim', 20)
        visualize = matrices_config.get('visualize', False)
        save_modes = matrices_config.get('save_modes', True)

        
        print(f"\nRunning modal analysis with {num_modes} modes...")
        
        # Use the timestamp from the controller to ensure we use the just-created matrices
        if timestamp is None and hasattr(controller, 'timestamp'):
            timestamp = controller.timestamp
            
        visualize_3d = False
        
        eigenvalues, eigenvectors, frequencies = compute_modal_analysis(
            matrices_path=matrices_path,
            timestamp=timestamp,
            num_modes=num_modes,
            visualize=visualize,
            save_modes=save_modes,
            output_dir=output_dir,
            mesh_file=mesh_file,
            visualize_3d=visualize_3d
        )

        print(f"Eigenvalues: {eigenvalues.shape}, Eigenvectors: {eigenvectors.shape}, Frequencies: {frequencies.shape}")
    else:
        print("\nSkipping modal analysis. Use --compute-modes to perform analysis.")