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
        self.mesh_filename = kwargs.get('mesh_filename', 'unknown')
        print(f"Using directory: {self.directory}")
        print(f"Material properties: E={self.young_modulus}, nu={self.poisson_ratio}, rho={self.density}")
        
        # Add eigenmodes visualization parameters
        self.show_modes = kwargs.get('show_modes', True)
        self.current_mode_index = -1  # Start with -1 to compute matrices first
        self.mode_animation_step = 0
        self.mode_animation_steps = kwargs.get('steps_per_mode', 100)  # Steps to animate each mode
        self.mode_scale = kwargs.get('mode_scale', 50.0)  # Scaling factor for eigenmodes
        self.num_modes_to_show = kwargs.get('num_modes_to_show', 10)
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
            
        steps_per_mode = 5  # You can adjust this value or make it a parameter
    
        if self.mode_animation_step >= steps_per_mode:
            self.mode_animation_step = 0
            self.current_mode_index = (self.current_mode_index + 1) % self.num_modes_to_show
            print(f"Switching to mode {self.current_mode_index + 1}/{self.num_modes_to_show}")
        
        # Display the current mode if modes were computed successfully
        if self.show_modes and self.modes_computed and self.eigenvectors is not None:
            self.displayMode(self.current_mode_index)
        
        self.end_time = process_time()
            
    
    def computeMatricesAndModes(self):
        """Compute mass and stiffness matrices, then solve for eigenmodes"""
        print("Computing mass and stiffness matrices...")
        self.mass_matrix = self.mass.assembleMMatrix()
        self.stiffness_matrix = self.fem.assembleKMatrix()
        
        print(f"Mass matrix shape: {self.mass_matrix.shape}")
        print(f"Stiffness matrix shape: {self.stiffness_matrix.shape}")
        
        # Create directory for matrices
        matrices_dir = 'matrices'
        os.makedirs(matrices_dir, exist_ok=True)
        
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
        print(f"Computing {self.num_modes_to_show} eigenmodes...")
        try:
            # Ensure matrices are in CSR format
            if not isinstance(self.mass_matrix, sparse.csr_matrix):
                if hasattr(self.mass_matrix, 'tocsr'):
                    mass_matrix = self.mass_matrix.tocsr()
                    stiff_matrix = self.stiffness_matrix.tocsr()
                else:
                    mass_matrix = self.mass_matrix
                    stiff_matrix = self.stiffness_matrix
            else:
                mass_matrix = self.mass_matrix
                stiff_matrix = self.stiffness_matrix
            
            # Make sure matrices are symmetric
            mass_matrix = 0.5 * (mass_matrix + mass_matrix.transpose())
            stiff_matrix = 0.5 * (stiff_matrix + stiff_matrix.transpose())
            
            # Solve eigenvalue problem
            eigenvalues, eigenvectors = eigsh(
                stiff_matrix, 
                k=self.num_modes_to_show, 
                M=mass_matrix,
                sigma=1e-5, 
                which='LM',
                maxiter=10000,
                tol=1e-3
            )
            
            # Sort by eigenvalue (smallest first)
            idx = eigenvalues.argsort()
            self.eigenvalues = eigenvalues[idx]
            self.eigenvectors = eigenvectors[:, idx]
            
            # Calculate natural frequencies
            self.frequencies = np.sqrt(np.abs(self.eigenvalues)) / (2 * np.pi)
            
            print("Eigenmode computation successful!")
            for i in range(min(5, len(self.eigenvalues))):
                print(f"Mode {i+1}: λ = {self.eigenvalues[i]:.6e}, f = {self.frequencies[i]:.4f} Hz")
            
            # Save the eigenmodes
            output_dir = os.path.join(matrices_dir, self.timestamp)
            os.makedirs(output_dir, exist_ok=True)
            np.save(f'{output_dir}/eigenvalues_{self.timestamp}.npy', self.eigenvalues)
            np.save(f'{output_dir}/eigenvectors_{self.timestamp}.npy', self.eigenvectors)
            np.save(f'{output_dir}/frequencies_{self.timestamp}.npy', self.frequencies)
            
            self.modes_computed = True
            print(f"Eigenmodes saved to {output_dir}")
            
        except Exception as e:
            print(f"Error computing eigenmodes: {e}")
            import traceback
            traceback.print_exc()
            self.modes_computed = False

    def displayMode(self, mode_index):
        """Directly display an eigenmode without animation"""
        if not self.modes_computed or self.eigenvectors is None:
            print("No modes computed yet")
            return
        
        if mode_index >= self.num_modes_to_show:
            mode_index = 0
        
        # Get the current eigenmode
        current_mode = self.eigenvectors[:, mode_index]
        
        # Calculate appropriate scaling based on eigenvalue
        mode_specific_scale = self.mode_scale / np.sqrt(np.abs(self.eigenvalues[mode_index]))
        
        # Apply the eigenmode displacement to the mechanical object
        displacement = mode_specific_scale * current_mode
        
        # Update the mechanical object positions
        with self.MO1.position.writeable() as pos:
            pos[:] = displacement.reshape((-1, 3))
        
        # Print mode information
        print(f"Displaying mode {mode_index + 1}/{self.num_modes_to_show}")
        print(f"Eigenvalue: {self.eigenvalues[mode_index]:.6e}, Frequency: {self.frequencies[mode_index]:.4f} Hz")
        print(f"Press '+' to see next mode, '-' to see previous mode")
        
    
    def animateCurrentMode(self):
        """Animate the current eigenmode on the mechanical object"""
        if not self.modes_computed or self.eigenvectors is None:
            return
        
        if self.current_mode_index >= self.num_modes_to_show:
            self.current_mode_index = 0  # Loop back to first mode
            print("Looping back to first eigenmode")
        
        # Get the current eigenmode
        current_mode = self.eigenvectors[:, self.current_mode_index]
        
        # Calculate animation factor - sine wave with transition period
        if self.mode_animation_step < self.transition_steps:
            # Transition in
            factor = np.sin(np.pi * self.mode_animation_step / (2 * self.transition_steps))
        elif self.mode_animation_step < self.transition_steps + self.pause_steps:
            # Hold at maximum
            factor = 1.0
        elif self.mode_animation_step < 2 * self.transition_steps + self.pause_steps:
            # Transition back to zero
            t = self.mode_animation_step - (self.transition_steps + self.pause_steps)
            factor = np.cos(np.pi * t / (2 * self.transition_steps))
        elif self.mode_animation_step < 2 * self.transition_steps + 2 * self.pause_steps:
            # Hold at zero
            factor = 0.0
        elif self.mode_animation_step < 3 * self.transition_steps + 2 * self.pause_steps:
            # Transition to negative
            t = self.mode_animation_step - (2 * self.transition_steps + 2 * self.pause_steps)
            factor = -np.sin(np.pi * t / (2 * self.transition_steps))
        elif self.mode_animation_step < 3 * self.transition_steps + 3 * self.pause_steps:
            # Hold at negative maximum
            factor = -1.0
        elif self.mode_animation_step < 4 * self.transition_steps + 3 * self.pause_steps:
            # Transition back to zero
            t = self.mode_animation_step - (3 * self.transition_steps + 3 * self.pause_steps)
            factor = -np.cos(np.pi * t / (2 * self.transition_steps))
        else:
            factor = 0.0
            
        # Calculate the scale based on eigenvalue - higher eigenmodes need more scaling
        mode_specific_scale = self.mode_scale / np.sqrt(np.abs(self.eigenvalues[self.current_mode_index]))
        
        # Apply the eigenmode displacement to the mechanical object
        n_nodes = len(self.original_positions)
        n_dofs_per_node = 3  # Assuming 3D model with x,y,z coordinates
        
        # Create a new array for the positions
        new_positions = np.copy(self.original_positions)
        
        # Apply displacements
        for i in range(n_nodes):
            for j in range(n_dofs_per_node):
                dof_index = i * n_dofs_per_node + j
                if dof_index < len(current_mode):
                    new_positions[i][j] += factor * mode_specific_scale * current_mode[dof_index]
        
        # Update the mechanical object positions - FIXED HERE
        with self.MO1.position.writeable() as pos:
            pos[:] = new_positions  # Use slice assignment instead of .assign()
        
        # Update simulation step and mode index
        self.mode_animation_step += 1
        total_steps_per_mode = 4 * self.transition_steps + 4 * self.pause_steps
        
        # Move to the next mode when animation cycle completes
        if self.mode_animation_step >= total_steps_per_mode:
            print(f"Moving to next eigenmode: {self.current_mode_index + 1}")
            self.current_mode_index += 1
            self.mode_animation_step = 0
            
            # Display mode frequency
            if self.current_mode_index < len(self.frequencies):
                print(f"Mode {self.current_mode_index}: Frequency = {self.frequencies[self.current_mode_index-1]:.4f} Hz")


        

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

   