import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def parse_obj(file_path):
    vertices = []
    faces = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':  # Vertex
                vertices.append([float(p) for p in parts[1:]])
            elif parts[0] == 'f':  # Face
                # Extract vertex indices, assuming faces are triangular or quad
                face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                faces.append(face)

    return vertices, faces

def render_obj(vertices, faces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

   # Convert face indices to vertex coordinates
    poly3d = [[vertices[vert_id] for vert_id in face] for face in faces]

    # Add collection to plot
    ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.5))

    # Set plot parameters
    x, y, z = zip(*vertices)
    ax.set_xlim([min(x), max(x)])
    ax.set_ylim([min(y), max(y)])
    ax.set_zlim([min(z), max(z)])
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    plt.show()

# File path to your .obj file
obj_file = "rubiks_cube.obj"
vertices, faces = parse_obj(obj_file)
render_obj(vertices, faces)
