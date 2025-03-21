import gmsh
import sys
import os
import math

# Create a rectangle in the grid defined by the points (x_min, y_min) and (x_max, y_max)


def create_beam(x_min, y_min, x_max, y_max, z_min, z_max, lc):
    # Create the points
    p1 = gmsh.model.geo.addPoint(x_min, y_min, z_min, lc)
    p2 = gmsh.model.geo.addPoint(x_max, y_min, z_min, lc)
    p3 = gmsh.model.geo.addPoint(x_max, y_max, z_min, lc)
    p4 = gmsh.model.geo.addPoint(x_min, y_max, z_min, lc)
    p5 = gmsh.model.geo.addPoint(x_min, y_min, z_max, lc)
    p6 = gmsh.model.geo.addPoint(x_max, y_min, z_max, lc)
    p7 = gmsh.model.geo.addPoint(x_max, y_max, z_max, lc)
    p8 = gmsh.model.geo.addPoint(x_min, y_max, z_max, lc)

    # Create the lines
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    l5 = gmsh.model.geo.addLine(p5, p6)
    l6 = gmsh.model.geo.addLine(p6, p7)
    l7 = gmsh.model.geo.addLine(p7, p8)
    l8 = gmsh.model.geo.addLine(p8, p5)
    l9 = gmsh.model.geo.addLine(p1, p5)
    l10 = gmsh.model.geo.addLine(p2, p6)
    l11 = gmsh.model.geo.addLine(p3, p7)
    l12 = gmsh.model.geo.addLine(p4, p8)

    # Create the curve loop
    cl1 = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    cl2 = gmsh.model.geo.addCurveLoop([l5, l6, l7, l8])
    cl3 = gmsh.model.geo.addCurveLoop([-l9, l1, l10, -l5])
    cl4 = gmsh.model.geo.addCurveLoop([-l3, l11, l7, -l12])
    cl5 = gmsh.model.geo.addCurveLoop([l10, l6, -l11, -l2])
    cl6 = gmsh.model.geo.addCurveLoop([l9, -l8, -l12, l4])

    # Create the surface
    s1 = gmsh.model.geo.addPlaneSurface([cl1])
    s2 = gmsh.model.geo.addPlaneSurface([cl2])
    s3 = gmsh.model.geo.addPlaneSurface([cl3])
    s4 = gmsh.model.geo.addPlaneSurface([cl4])
    s5 = gmsh.model.geo.addPlaneSurface([cl5])
    s6 = gmsh.model.geo.addPlaneSurface([cl6])

    # Create volume
    v = gmsh.model.geo.addSurfaceLoop([s1, s2, s3, s4, s5, s6])
    v = gmsh.model.geo.addVolume([v])

    gmsh.model.geo.synchronize()

    # Add physical groups with meaningful names
    # Volume (tag=1)
    gmsh.model.addPhysicalGroup(3, [v], tag=1, name="volume")

    # Bottom surface (tag=2)
    gmsh.model.addPhysicalGroup(2, [s1], tag=2, name="bottom")

    # Top surface (tag=3)
    gmsh.model.addPhysicalGroup(2, [s2], tag=3, name="top")

    # Front surface (tag=4)
    gmsh.model.addPhysicalGroup(2, [s3], tag=4, name="front")

    # Back surface (tag=5)
    gmsh.model.addPhysicalGroup(2, [s4], tag=5, name="back")

    # Right surface (tag=6)
    gmsh.model.addPhysicalGroup(2, [s5], tag=6, name="right")

    # Left surface (tag=7)
    gmsh.model.addPhysicalGroup(2, [s6], tag=7, name="left")

    # generate the mesh
    gmsh.model.mesh.generate(3)

    # compute the number of nodes
    nodes = gmsh.model.mesh.getNodes()
    nb_nodes = len(nodes[0])

    # Save the mesh
    gmsh.write(f"mesh/beam_{nb_nodes}.msh")

    # Launch the GUI to see the results
    gmsh.fltk.run()


def create_plate_with_hole(length=10, width=5, thickness=1,
                           hole_radius=1, hole_x=5, hole_y=2.5, lc=0.5):
    """Create 3D plate with circular hole using OpenCASCADE"""

    # Create rectangle
    rect = gmsh.model.occ.addRectangle(0, 0, 0, length, width)

    # Create circular hole
    circle = gmsh.model.occ.addDisk(hole_x, hole_y, 0, hole_radius, hole_radius)

    # Cut hole from rectangle
    plate = gmsh.model.occ.cut([(2, rect)], [(2, circle)])
    base_surface = plate[0][0][1]  # Get the resulting surface tag

    # Extrude to create volume
    volume = gmsh.model.occ.extrude([(2, base_surface)], 0, 0, thickness)

    # Synchronize the model
    gmsh.model.occ.synchronize()
    
    # Set mesh size using the lc parameter
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

    # Add physical groups
    # Volume (tag=1)
    gmsh.model.addPhysicalGroup(3, [volume[1][1]], tag=1, name="volume")

    # Bottom surface (tag=2)
    gmsh.model.addPhysicalGroup(2, [base_surface], tag=2, name="bottom")

    # Top surface (tag=3)
    top_surface = volume[0][1]
    gmsh.model.addPhysicalGroup(2, [top_surface], tag=3, name="top")

    # Get side surfaces
    side_surfaces = [ent[1] for ent in volume if ent[0] == 2 and
                     ent[1] != base_surface and ent[1] != top_surface]

    # Left boundary (x=0) (tag=4)
    left_surfaces = []
    for surf in side_surfaces:
        com = gmsh.model.occ.getCenterOfMass(2, surf)
        if abs(com[0]) < 1e-3:  # x ≈ 0
            left_surfaces.append(surf)
    gmsh.model.addPhysicalGroup(2, left_surfaces, tag=4, name="left")

    # Right boundary (x=length) (tag=5)
    right_surfaces = []
    for surf in side_surfaces:
        com = gmsh.model.occ.getCenterOfMass(2, surf)
        if abs(com[0] - length) < 1e-3:  # x ≈ length
            right_surfaces.append(surf)
    gmsh.model.addPhysicalGroup(2, right_surfaces, tag=5, name="right")

    # Hole surface (tag=6)
    hole_surfaces = []
    for surf in side_surfaces:
        com = gmsh.model.occ.getCenterOfMass(2, surf)
        if ((com[0] - hole_x)**2 + (com[1] - hole_y)**2) < (hole_radius + 1e-3)**2:
            hole_surfaces.append(surf)
    gmsh.model.addPhysicalGroup(2, hole_surfaces, tag=6, name="hole")

    # Generate mesh
    gmsh.model.mesh.generate(3)

    # Save mesh
    nodes = gmsh.model.mesh.getNodes()
    nb_nodes = len(nodes[0])
    filename = f"plate_x{hole_x}_y{hole_y}_r{hole_radius}_n{nb_nodes}.msh"
    directory = "mesh/"
    gmsh.write(directory + filename)

    # Launch GUI
    gmsh.fltk.run()


def stl_to_volume_mesh(stl_file, output_file=None, mesh_size=0.1, gui=False):
    """
    Convert an STL surface mesh to a volumetric mesh using GMSH, ensuring compatibility with DOLFINx.

    Args:
        stl_file (str): Path to input STL file
        output_file (str, optional): Path to output MSH file. If None, derived from input path.
        mesh_size (float, optional): Target mesh element size. Default is 0.1.
        gui (bool, optional): Whether to show the GMSH GUI. Default is False.

    Returns:
        str: Path to the created MSH file
    """
    import gmsh
    import os
    import sys
    import numpy as np

    # Check if file exists
    if not os.path.exists(stl_file):
        print(f"Error: STL file not found: {stl_file}")
        return None

    # Generate output filename if not provided
    if output_file is None:
        dir_name = os.path.dirname(stl_file)
        base_name = os.path.splitext(os.path.basename(stl_file))[0]
        output_file = os.path.join(dir_name, f"{base_name}.msh")

    # Initialize GMSH
    # gmsh.initialize() # Initialize GMSH outside the function
    gmsh.option.setNumber("General.Terminal", 1)  # Enable terminal output
    gmsh.model.add("VolumetricModel")

    print(f"Converting {stl_file} to volumetric mesh...")

    # Method 1: Try using OpenCASCADE (more robust)
    try:
        # Set mesh size
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size/2)

        # Load the STL file
        print("Loading STL file using OpenCASCADE...")
        shapes = gmsh.model.occ.importShapes(stl_file)

        # Try to create a volume
        print("Creating volume from STL surface...")

        # First, ensure the surface orientation is consistent
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.classifySurfaces(np.pi/2, True, True)
        gmsh.model.mesh.createGeometry()

        # Get all surfaces
        surfaces = gmsh.model.getEntities(2)

        # Create a surface loop and volume
        print(f"Found {len(surfaces)} surfaces, creating volume...")
        surface_loop = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfaces])
        volume = gmsh.model.geo.addVolume([surface_loop])

        # Synchronize the model
        gmsh.model.geo.synchronize()

        # Add physical group for the volume
        gmsh.model.addPhysicalGroup(3, [volume], tag=1, name="volume")

        print("Successfully created volume model")

    except Exception as e:
        print(f"OpenCASCADE method failed: {e}")
        print("Trying alternative approach...")

        # Method 2: Direct STL import and meshing
        try:
            # Reset the model
            gmsh.model.remove()
            gmsh.model.add("VolumetricModel")

            # Import STL directly
            gmsh.merge(stl_file)

            # Create volume using the healing procedure
            gmsh.option.setNumber("Mesh.StlOneSolidPerSurface", 0)
            gmsh.option.setNumber("Mesh.StlRemoveDuplicateTriangles", 1)
            gmsh.option.setNumber("Mesh.StlAngularDeflection", 0.5)
            gmsh.option.setNumber("Mesh.StlLinearDeflection", 0.01)

            # Create 3D mesh directly
            print("Creating 3D mesh directly...")
            gmsh.model.mesh.createTopology()

            # Get all surfaces
            surfaces = gmsh.model.getEntities(2)
            if surfaces:
                print(f"Found {len(surfaces)} surfaces")

                # Create a surface loop
                try:
                    gmsh.model.geo.synchronize()
                    surface_loop = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfaces])
                    volume = gmsh.model.geo.addVolume([surface_loop])
                    gmsh.model.geo.synchronize()
                except Exception as e:
                    print(f"Could not create explicit volume: {e}")
                    print("Will try to mesh the domain directly")
            else:
                print("No surfaces found, will try to mesh directly")

        except Exception as e:
            print(f"Alternative approach also failed: {e}")
            print("Continuing with basic meshing operation")

    # Generate the volumetric mesh
    print("Generating volume mesh...")
    try:
        # Set more robust meshing options
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Delaunay
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.5)

        gmsh.model.mesh.generate(3)

        # Add physical tag for the volume to ensure DOLFINx can read it
        volumes = gmsh.model.getEntities(dim=3)
        if volumes:
            volume_tags = [v[1] for v in volumes]
            gmsh.model.addPhysicalGroup(3, volume_tags, tag=1)
            gmsh.model.setPhysicalName(3, 1, "Volume")  # Set name for DOLFINx

        # Get all surfaces
        surfaces = gmsh.model.getEntities(dim=2)

        # Add physical groups for surfaces
        if surfaces:
            for surf in surfaces:
                surface_tag = surf[1]
                gmsh.model.addPhysicalGroup(2, [surface_tag], tag=surface_tag)
                gmsh.model.setPhysicalName(2, surface_tag, f"Surface_{surface_tag}")
        else:
            print("Warning: No surfaces found in the mesh.")

        # Save the mesh
        gmsh.write(output_file)
        print(f"Mesh saved to: {output_file}")

        # Show GUI if requested
        if gui:
            gmsh.fltk.run()

        return output_file

    except Exception as e:
        print(f"Mesh generation failed: {e}")
        import traceback
        print(traceback.format_exc())
        # gmsh.finalize() # REMOVE THIS LINE
        return None

    finally:
        # Clean up
        if not gui:
            pass  # gmsh.finalize() # REMOVE THIS LINE
        else:
            gmsh.fltk.run()  # Keep GUI open if requested


def generate_cat(lc=0.1):
    """Generates a stylized 3D cat using GMSH."""

    # Head
    head_radius = 1.0
    head_center = [0, 0, 0]
    head = gmsh.model.occ.addSphere(head_center[0], head_center[1], head_center[2], head_radius)

    # Ears
    ear_height = 0.7
    ear_width = 0.5
    ear_depth = 0.2
    ear_distance = 0.7  # Distance from the center of the head

    # Left ear
    left_ear_center = [head_center[0] - ear_distance, head_center[1] + head_radius, head_center[2]]
    left_ear = gmsh.model.occ.addCone(left_ear_center[0], left_ear_center[1], left_ear_center[2],
                                       0, 0, ear_height, ear_width, ear_depth)

    # Right ear
    right_ear_center = [head_center[0] + ear_distance, head_center[1] + head_radius, head_center[2]]
    right_ear = gmsh.model.occ.addCone(right_ear_center[0], right_ear_center[1], right_ear_center[2],
                                        0, 0, ear_height, ear_width, ear_depth)

    # Body
    body_height = 2.0
    body_radius = 1.2
    body_center = [head_center[0], head_center[1] - (head_radius + body_height / 2), head_center[2]]
    body = gmsh.model.occ.addCylinder(body_center[0], body_center[1], body_center[2],
                                        0, body_height, 0, body_radius)

    # Tail
    tail_length = 1.5
    tail_radius = 0.3
    tail_center = [head_center[0], body_center[1] - body_height / 2, head_center[2]]
    tail_direction = [0, -1, 0.5]  # Pointing downwards and slightly back
    tail = gmsh.model.occ.addCylinder(tail_center[0], tail_center[1], tail_center[2],
                                        tail_direction[0], tail_direction[1], tail_direction[2], tail_radius)
    gmsh.model.occ.rotate([(3, tail)], head_center[0], head_center[1], head_center[2],
                           1, 0, 0, math.pi / 6)  # Rotate tail slightly upwards

    # Combine parts
    parts = [(3, head), (3, left_ear), (3, right_ear), (3, body), (3, tail)]
    union = gmsh.model.occ.fuse(parts[0], parts[1:], removeObject=True, removeTool=True)

    # Synchronize
    gmsh.model.occ.synchronize()

    # Add physical groups
    volume_tag = 1
    gmsh.model.addPhysicalGroup(3, [volume_tag], tag=1, name="cat_volume")

    # Generate mesh
    gmsh.model.mesh.generate(3)

    # Save mesh
    nodes = gmsh.model.mesh.getNodes()
    nb_nodes = len(nodes[0])
    filename = f"mesh/cat_n{nb_nodes}.msh"
    gmsh.write(filename)

    # Launch GUI
    gmsh.fltk.run()

def create_plate_shell(length=1.0, width=1.0, lc=0.05):
    """Create 2D shell mesh (mid-surface) for a plate"""
    gmsh.model.add("plate_shell")
    
    # Create rectangle representing mid-surface
    rect = gmsh.model.occ.addRectangle(0, 0, 0, length, width)
    
    # Synchronize the model
    gmsh.model.occ.synchronize()
    
    # Set mesh size
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)
    
    # Add physical groups
    gmsh.model.addPhysicalGroup(2, [rect], tag=1, name="surface")
    
    # Add boundary physical groups
    boundary = gmsh.model.getBoundary([(2, rect)])
    left_points = [p for p in boundary if abs(gmsh.model.occ.getCenterOfMass(1, abs(p[1]))[0]) < 1e-6]
    right_points = [p for p in boundary if abs(gmsh.model.occ.getCenterOfMass(1, abs(p[1]))[0] - length) < 1e-6]
    
    if left_points:
        gmsh.model.addPhysicalGroup(1, [abs(p[1]) for p in left_points], tag=2, name="left")
    if right_points:
        gmsh.model.addPhysicalGroup(1, [abs(p[1]) for p in right_points], tag=3, name="right")
    
    # Generate mesh
    gmsh.model.mesh.generate(2)  # 2D mesh
    
    # Save mesh
    nodes = gmsh.model.mesh.getNodes()
    nb_nodes = len(nodes[0])
    filename = f"mesh/plate_shell_n{nb_nodes}.msh"
    gmsh.write(filename)
    
    # Launch GUI
    gmsh.fltk.run()


if __name__ == '__main__':
    import sys
    import os

    mesh_type = "beam"

    if mesh_type == "beam":
        gmsh.initialize(sys.argv)
        gmsh.model.add("beam")
        create_beam(0, 0, 10, 1, 0, 1, 0.25)
        gmsh.finalize()
    elif mesh_type == "plate":
        n_simulations = 1
        length = 1
        width = 1
        thickness = 0.01
        hole_x = 0.7
        hole_y = 0.3
        hole_radius = 0.2
    
        # check if the hole is not too close to the edge

        gmsh.initialize(sys.argv)
        gmsh.model.add("plate")
        create_plate_with_hole(length, width, thickness, hole_radius, hole_x, hole_y, 0.07)
        gmsh.finalize()
    elif mesh_type == "stl":
        object_folder = "mesh/objects"

        # Initialize GMSH
        gmsh.initialize(sys.argv)

        # Loop through all STL files in the object folder
        for root, _, files in os.walk(object_folder):
            for file in files:
                if file.endswith(".stl"):
                    stl_file = os.path.join(root, file)

                    # Generate output filename
                    dir_name = os.path.dirname(stl_file)
                    base_name = os.path.splitext(os.path.basename(stl_file))[0]
                    output_file = os.path.join(dir_name, f"{base_name}.msh")

                    print(f"Attempting to convert {stl_file} to {output_file}")
                    result = stl_to_volume_mesh(stl_file, output_file, mesh_size=0.01, gui=False)  # Reduced mesh_size
                    if result is None:
                        print(f"Conversion of {stl_file} failed.")
                    else:
                        print(f"Conversion of {stl_file} complete.")
                    print("\n")

        print("All STL conversions complete.")
        gmsh.finalize()
    elif mesh_type == "cat":
        gmsh.initialize(sys.argv)
        gmsh.model.add("cat")
        generate_cat()
        gmsh.finalize()

    elif mesh_type == "stl_one":


        stl_file = "mesh/armadillo.stl"
        if not os.path.exists(stl_file):
            print(f"Error: STL file not found: {stl_file}")
        else:
            gmsh.initialize(sys.argv)
            # Generate output filename
            dir_name = os.path.dirname(stl_file)
            base_name = os.path.splitext(os.path.basename(stl_file))[0]
            output_file = os.path.join(dir_name, f"{base_name}.msh")

            print(f"Attempting to convert {stl_file} to {output_file}")
            result = stl_to_volume_mesh(stl_file, output_file, mesh_size=1, gui=True)  # Reduced mesh_size
            if result is None:
                print(f"Conversion of {stl_file} failed.")
            else:
                print(f"Conversion of {stl_file} complete.")
            gmsh.finalize()
    elif mesh_type == "shell":
        gmsh.initialize(sys.argv)
        gmsh.model.add("plate_shell")
        create_plate_shell()
        gmsh.finalize()