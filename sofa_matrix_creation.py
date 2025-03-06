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
    
    #     self.bad_sample = False
    #     # reset positions
    #     #self.MO1.position.value = self.MO1.rest_position.value
   
    #     self.z = np.random.uniform(-1, 1)
    #     self.phi = np.random.uniform(0, 2*np.pi)
    #     self.versor = np.array([np.sqrt(1 - self.z**2) * np.cos(self.phi), np.sqrt(1 - self.z**2) * np.sin(self.phi), self.z])
    #     self.magnitude = np.random.uniform(190, 200)
    #     self.externalForce = self.magnitude * self.versor
      


    #     #save the matrices in a file
    #     if self.save:
    #         np.save(f'modal_data/{self.directory}/mass_matrix.npy', self.mass_matrix)
    #         np.save(f'modal_data/{self.directory}/stiffness_matrix.npy', self.stiffness_matrix)

    #    # Define random box
    #     side = np.random.randint(1, 6)
    #     if side == 1:
    #         x_min = np.random.uniform(2, 8.0)
    #         x_max = x_min + 2
    #         y_min = np.random.uniform(0, 3.0)
    #         y_max = y_min + 2
    #         z_min = -0.1
    #         z_max = 0.1
    #     elif side == 2:
    #         x_min = np.random.uniform(2, 8.0)
    #         x_max = x_min + 2
    #         y_min = np.random.uniform(-1, 3.0)
    #         y_max = y_min + 2
    #         z_min = 0.39
    #         z_max = 0.41
    #     elif side == 3:
    #         x_min = np.random.uniform(2, 8.0)
    #         x_max = x_min + 2
    #         y_min = -0.1
    #         y_max = 0.1
    #         z_min = np.random.uniform(0, 0.5)
    #         z_max = z_min + 0.5
    #     elif side == 4:
    #         x_min = np.random.uniform(2, 8.0)
    #         x_max = x_min + 2
    #         y_min = 4.9
    #         y_max = 5.1
    #         z_min = np.random.uniform(0, 0.2)
    #         z_max = z_min + 0.2
    #     elif side == 5:
    #         x_min = 9.99
    #         x_max = 10.01
    #         y_min = np.random.uniform(0, 3.0)
    #         y_max = y_min + 2
    #         z_min = np.random.uniform(0, 0.2)
    #         z_max = z_min + 0.2
    #     elif side == 7:
    #         x_min = 9.9
    #         x_max = 10.1
    #         y_min = -0.1
    #         y_max = 5.1
    #         z_min = -0.1
    #         z_max = 0.5

        


        
    #     #print(f"==================== Intersected squares: {intersect_count}  with magnitude {self.magnitude}====================")

    #     bbox = [x_min, y_min, z_min, x_max, y_max, z_max]


    #     # Get the intersection with the surface
    #     indices = list(self.cff_box.indices.value)
    #     indices = list(set(indices).intersection(set(self.idx_surface)))


    #     # Get the intersection with the fixed box

    #     indices_fixed = list(self.fixed_box.indices.value)
    #     indices_fixed = list(set(indices_fixed).intersection(set(self.idx_surface)))

    #     # self.exactSolution.removeObject(self.cff_box)
    #     # self.cff_box = self.exactSolution.addObject('BoxROI', name='ROI2', box=bbox, drawBoxes=True)
    #     # self.cff_box.init()

    
    #     # if self.iteration == 0:
    #     #     self.exactSolution.removeObject(self.cff)
    #     #     self.cff = self.exactSolution.addObject('ConstantForceField', indices=indices, totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1",  rayleighStiffness=0.1)
    #     #     self.cff.init()



    #     x, y, r = self.key
    #     #x, y, r are the coordinates of the center of the circle and the radius of the circle, find the indices of the nodes that are inside the circle
    #     indices_circle = np.where((self.MO1.rest_position.value[:, 0] - x)**2 + (self.MO1.rest_position.value[:, 1] - y)**2 <= r**2)[0]

    #     #find the indices of the nodes that are inside the bounding box


    #     indices_forces = np.where((self.MO1.rest_position.value[:, 0] >= x_min) & (self.MO1.rest_position.value[:, 0] <= x_max) & (self.MO1.rest_position.value[:, 1] >= y_min) & (self.MO1.rest_position.value[:, 1] <= y_max) & (self.MO1.rest_position.value[:, 2] >= z_min) & (self.MO1.rest_position.value[:, 2] <= z_max))[0]

        

    #     self.bounding_box = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max, "z_min": z_min, "z_max": z_max}
    #     self.versor_rounded = [round(i, 4) for i in self.versor]
    #     self.versor_rounded = list(self.versor_rounded)
    #     self.force_info = {"magnitude": round(self.magnitude, 4), "versor": self.versor_rounded}
    #     self.indices_Dir = list(indices_fixed)
    #     self.indices_hole = list(indices_circle)
    #     self.indices_forces = list(indices_forces)
        
        

    #     for i in range(len(indices_fixed)):
    #         self.indices_Dir[i] = int(indices_fixed[i])
    #     for i in range(len(indices_circle)):
    #         self.indices_hole[i] = int(indices_circle[i])
    #     for i in range(len(indices_forces)):
    #         self.indices_forces[i] = int(indices_forces[i])

   
      

    #     #print(f"Bounding box: [{x_min}, {y_min}, {z_min}, {x_max}, {y_max}, {z_max}]")
    #     #print(f"Side: {side}")
    #     if indices_forces.size == 0:
    #         #print("Empty intersection")
    #         self.bad_sample = True
        self.start_time = process_time()
       



    def onAnimateEndEvent(self, event):
        self.end_time = process_time()
        
        # Get matrices from SOFA
        self.mass_matrix = self.mass.assembleMMatrix()
        self.stiffness_matrix = self.fem.assembleKMatrix()
        
        print(f"Mass matrix shape: {self.mass_matrix.shape}")
        print(f"Stiffness matrix shape: {self.stiffness_matrix.shape}")
        
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
            
        

    def compute_displacement(self, mechanical_object):
        # Compute the displacement between the high and low resolution solutions
        U = mechanical_object.position.value.copy() - mechanical_object.rest_position.value.copy()
        return U
    
    def compute_velocity(self, mechanical_object):
        # Compute the velocity of the high resolution solution
        return mechanical_object.velocity.value.copy()
    

    def compute_rest_position(self, mechanical_object):
        # Compute the position of the high resolution solution
        return mechanical_object.rest_position.value.copy()
    
    def compute_position(self, mechanical_object):
        # Compute the position of the high resolution solution
        return mechanical_object.position.value.copy()
    
    def compute_edges(self, grid):
        # create a matrix with the edges of the grid and their length
        edges = grid.edges.value.copy()
        positions = grid.position.value.copy()
        matrix = np.zeros((len(edges)*2, 3))
        for i, edge in enumerate(edges):
            # account for the fact the edges must be bidirectional
            matrix[len(edges) + i, 0] = edge[1]
            matrix[len(edges) + i, 1] = edge[0]
            matrix[len(edges) + i, 2] = np.linalg.norm(positions[edge[0]] - positions[edge[1]])
            matrix[i, 0] = edge[0]
            matrix[i, 1] = edge[1]
            matrix[i, 2] = np.linalg.norm(positions[edge[0]] - positions[edge[1]])
        return matrix
    
    def generate_versors(self, n_versors):
            # Initialize an empty list to store the versors
            a = 4*np.pi/n_versors
            d = np.sqrt(a)
            M_theta = int(np.round(np.pi/d))
            d_theta = np.pi/M_theta
            d_phi = a/d_theta
            versors = []
            for m in range(M_theta):
                theta = np.pi*(m + 0.5)/M_theta
                M_phi = int(np.round(2*np.pi*np.sin(theta)/d_phi))
                for n in range(M_phi):
                    phi = 2*np.pi*n/M_phi
                    versors.append([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
            return np.array(versors)
    
    def close(self):
        print("Closing simulation")


def createScene(rootNode, filename, directory, sample, key, *args, **kwargs):
    rootNode.dt = 0.01
    rootNode.gravity = [0, 0, 0]
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

    # Material properties
    young_modulus = 5000
    poisson_ratio = 0.25

    mu = young_modulus / (2 * (1 + poisson_ratio))
    lam = young_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    print(f"Using mu={mu}, lambda={lam}")
    mu_lam_str = f"{mu} {lam}"

    # Create high resolution solution node
    exactSolution = rootNode.addChild('HighResSolution2D', activated=True)
    exactSolution.addObject('MeshGmshLoader', name='grid', filename=filename)
    surface_topo = exactSolution.addObject('TetrahedronSetTopologyContainer', name='triangleTopo', src='@grid')
    MO1 = exactSolution.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@grid')
    
    # Add system components
    mass = exactSolution.addObject('MeshMatrixMass', totalMass=10, name="SparseMass", topology="@triangleTopo")
    solver = exactSolution.addObject('EulerImplicitSolver', name="ODEsolver", rayleighStiffness=0.1, rayleighMass=0.1)
    linear_solver = exactSolution.addObject('CGLinearSolver', 
                                          template="CompressedRowSparseMatrixMat3x3d",
                                          iterations=1000, tolerance=1e-10, threshold=1e-10, warmStart=True)
    
    fem = exactSolution.addObject('TetrahedronHyperelasticityFEMForceField',
                                name="FEM", materialName="StVenantKirchhoff", ParameterSet=mu_lam_str)
    
    # Add constraints and forces
    fixed_box = exactSolution.addObject('BoxROI', name='ROI',
                                      box="-0.01 -0.01 -0.02 1.01 0.01 0.02", drawBoxes=True)
    exactSolution.addObject('FixedConstraint', indices="@ROI.indices")
    
    # cff_box = exactSolution.addObject('BoxROI', name='ROI2',
    #                                 box="9.9 -0.1 -0.1 10.1 5.1 0.5", drawBoxes=True)
    # cff = exactSolution.addObject('ConstantForceField', indices="@ROI2.indices",
    #                             totalForce=[0, 0, 0], showArrowSize=0.1,
    #                             showColor="0.2 0.2 0.8 1")

    # Add visual model
    visual = exactSolution.addChild("visual")
    visual.addObject('OglModel', src='@../DOFs', color='0 1 0 1')
    visual.addObject('BarycentricMapping', input='@../DOFs', output='@./')

    # Create and add controller with all components
    controller = AnimationStepController(rootNode, mass=mass, fem=fem,
                                  linear_solver=linear_solver,
                                  surface_topo=surface_topo,
                                  MO1=MO1, fixed_box=fixed_box,
                                  directory=directory, sample=sample,
                                  key=key, 
                                  young_modulus=young_modulus,
                                  poisson_ratio=poisson_ratio,
                                  density=10,  # Using the value from MeshMatrixMass
                                  mesh_filename=filename,
                                  **kwargs)
    rootNode.addObject(controller)

    return rootNode, controller


if __name__ == "__main__":
    import Sofa.Gui
    from tqdm import tqdm
    
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

    ### MODIFIABLE PARAMETERS
    filename = 'beam_615'
    filename = f"mesh/{filename}.msh"
    USE_GUI = True
    steps = 10


    # Setup and run simulation
    root = Sofa.Core.Node("root")
    rootNode, controller = createScene(
        root,
        filename = filename,
        directory=timestamp,
        sample=0,
        key=(0, 0, 0)  # default key values
    )

    # Initialize simulation
    Sofa.Simulation.init(root)
    controller.save = True

    if USE_GUI:
        Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
        Sofa.Gui.GUIManager.createGUI(root, __file__)
        Sofa.Gui.GUIManager.SetDimension(800, 600)
        Sofa.Gui.GUIManager.MainLoop(root)
        Sofa.Gui.GUIManager.closeGUI()
    else:
        for _ in tqdm(range(steps), desc="Simulation progress"):
            Sofa.Simulation.animate(root, root.dt.value)

    controller.close()


