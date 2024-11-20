import ray
from ImLite import *
from utils import *
import importlib

class ExampleSceneDef(object):
    def __init__(self, camera, scene, lights):
        self.camera = camera;
        self.scene = scene;
        self.lights = lights;

    def render(self, output_path=None, output_shape=None, gamma_correct=True, srgb_whitepoint=None):
        importlib.reload(ray)
        if(output_shape is None):
            output_shape=[128,128];
        if(srgb_whitepoint is None):
            srgb_whitepoint = 1.0;
        pix = ray.render_image(self.camera, self.scene, self.lights, output_shape[1], output_shape[0]);
        im = None;
        if(gamma_correct):
            cam_img_ui8 = to_srgb8(pix / srgb_whitepoint)
            im = Image(pixels=cam_img_ui8);
        else:
            im = im = Image(pixels=pix);
        if(output_path is None):
            return im;
        else:
            im.writeToFile(output_path);


def TwoSpheresExample():
    importlib.reload(ray)
    tan = ray.Material(vec([0.7, 0.7, 0.4]), 0.6)
    gray = ray.Material(vec([0.2, 0.2, 0.2]))

    scene = ray.Scene([
        ray.Sphere(vec([0, 0, 0]), 0.5, tan),
        ray.Sphere(vec([0, -40, 0]), 39.5, gray),
    ])

    lights = [
        ray.PointLight(vec([12, 10, 5]), vec([300, 300, 300])),
        ray.AmbientLight(0.1),
    ]
    camera = ray.Camera(vec([3, 1.7, 5]), target=vec([0, 0, 0]), vfov=25, aspect=16 / 9)
    return ExampleSceneDef(camera=camera, scene=scene, lights=lights);


def ThreeSpheresExample():
    importlib.reload(ray)
    tan = ray.Material(vec([0.4, 0.4, 0.2]), k_s=0.3, p=90, k_m=0.3)
    blue = ray.Material(vec([0.2, 0.2, 0.5]), k_m=0.5)
    gray = ray.Material(vec([0.2, 0.2, 0.2]), k_m=0.4)

    scene = ray.Scene([
        ray.Sphere(vec([-0.7, 0, 0]), 0.5, tan),
        ray.Sphere(vec([0.7, 0, 0]), 0.5, blue),
        ray.Sphere(vec([0, -40, 0]), 39.5, gray),
    ])

    lights = [
        ray.PointLight(vec([12, 10, 5]), vec([300, 300, 300])),
        ray.AmbientLight(0.1),
    ]

    camera = ray.Camera(vec([3, 1.2, 5]), target=vec([0, -0.4, 0]), vfov=24, aspect=16 / 9)
    return ExampleSceneDef(camera=camera, scene=scene, lights=lights);


def CubeExample():
    importlib.reload(ray)
    tan = ray.Material(vec([0.7, 0.7, 0.4]), 0.6)
    gray = ray.Material(vec([0.2, 0.2, 0.2]))

    # Read the triangle mesh for a 2x2x2 cube, and scale it down to 1x1x1 to fit the scene.
    vs_list = 0.5 * read_obj_triangles(open("cube.obj"))

    scene = ray.Scene([
                      # Make a big sphere for the floor
                      ray.Sphere(vec([0, -40, 0]), 39.5, gray),
                  ] + [
                      # Make triangle objects from the vertex coordinates
                      ray.Triangle(vs, tan) for vs in vs_list
                  ])

    lights = [
        ray.PointLight(vec([12, 10, 5]), vec([300, 300, 300])),
        ray.AmbientLight(0.1),
    ]

    camera = ray.Camera(vec([3, 1.7, 5]), target=vec([0, 0, 0]), vfov=25, aspect=16 / 9)
    return ExampleSceneDef(camera=camera, scene=scene, lights=lights);

def OrthoFriendlyExample(sphere_radius = 0.25):
    gray = ray.Material(vec([0.5, 0.5, 0.5]))

    # One small sphere centered at z=-0.5
    scene = ray.Scene([
        ray.Sphere(vec([0, 0, -0.5]), sphere_radius, gray),
    ])

    lights = [
        ray.AmbientLight(0.5),
    ]
    camera = ray.Camera(vec([0,0,0]), target=vec([0, 0, -0.5]), vfov=90, aspect=1)
    return ExampleSceneDef(camera=camera, scene=scene, lights=lights);

def CornellBoxExample():
    importlib.reload(ray)
    red = ray.Material(vec([0.7, 0.1, 0.1]), k_m=1)
    green = ray.Material(vec([0.1, 0.7, 0.1]), k_m=1)
    white = ray.Material(vec([0.7, 0.7, 0.7]), k_m=1)
    blue = ray.Material(vec([0.1, 0.1, 0.7]), k_m=1)
    gray = ray.Material(vec([0.2, 0.2, 0.2]), k_m=1)
    # tan = ray.Material(vec([0.7, 0.7, 0.4]), 0.6)
    tan = red

    scene = ray.Scene([
        ray.Triangle([vec([0, 0, 0]), vec([1, 0, 0]), vec([0, 1, 0])], white),
        ray.Triangle([vec([1, 1, 0]), vec([0, 1, 0]), vec([1, 0, 0])], white),
        ray.Triangle([vec([0, 0, 0]), vec([0, 0, 1]), vec([1, 0, 0])], red),
        ray.Triangle([vec([1, 0, 1]), vec([1, 0, 0]), vec([0, 0, 1])], red),
        ray.Triangle([vec([0, 0, 0]), vec([0, 1, 0]), vec([0, 0, 1])], green),
        ray.Triangle([vec([0, 1, 1]), vec([0, 0, 1]), vec([0, 1, 0])], green),
        ray.Triangle([vec([1, 0, 0]), vec([1, 0, 1]), vec([1, 1, 0])], blue),
        ray.Triangle([vec([1, 1, 1]), vec([1, 1, 0]), vec([1, 0, 1])], blue),
        ray.Triangle([vec([0, 1, 0]), vec([1, 1, 0]), vec([0, 1, 1])], gray),
        ray.Triangle([vec([1, 1, 1]), vec([0, 1, 1]), vec([1, 1, 0])], gray),
        # make back wall
        ray.Triangle([vec([0, 0, 1]), vec([1, 0, 1]), vec([0, 1, 1])], white),
        ray.Triangle([vec([1, 1, 1]), vec([0, 1, 1]), vec([1, 0, 1])], white),
        ray.Sphere(vec([0.3, 0.3, 0.3]), 0.2, tan),
    ], bg_color=vec([0,0,0]))

    lights = [
        ray.PointLight(vec([0.5, 0.5, 0.5]), .1),
        # make a ceiling light
        # ray.PointLight(vec([0.5, 0.5, 0.5]), vec([1, 1, 1])),
        # Add another light source
        # ray.PointLight(vec([1.5, 0.1, 0.5]), vec([1, 1, 1])),
        # ray.PointLight(vec([-0.5, 0.5, 0.5]), vec([1, 1, 1])),
        ray.AmbientLight(0.1),
        # ray.PointLight(vec([0.5, 2, 0.5]), vec([2, 2, 2])),
    ]

    camera = ray.Camera(vec([0.75, 0.5, 1]), target=vec([0.5, 0.5, 0.5]), vfov=70, aspect=16 / 9)
    return ExampleSceneDef(camera=camera, scene=scene, lights=lights);

def RubiksCubeExample():
    importlib.reload(ray)
    tan = ray.Material(vec([0.7, 0.7, 0.4]), 0.6)
    green = ray.Material(vec([0.0, 1.0, 0.0]), 0.6)
    red = ray.Material(vec([1.0, 0.0, 0.0]), 0.6)
    blue = ray.Material(vec([0.0, 0.0, 1.0]), 0.6)
    white = ray.Material(vec([1.0, 1.0, 1.0]), 0.6)
    yellow = ray.Material(vec([1.0, 1.0, 0.0]), 0.6)
    orange = ray.Material(vec([1.0, 0.5, 0.0]), 0.6)
    gray = ray.Material(vec([0.2, 0.2, 0.2]), 0.6)

    # Read the triangle mesh for a 2x2x2 cube, and scale it down to 1x1x1 to fit the scene.
    vs_list = 0.5 * read_obj_triangles(open("new_rubiks.obj"))

    scene = ray.Scene([
                      # Make a big sphere for the floor
                      ray.Sphere(vec([0, -40, 0]), 39.5, gray),
                  ] + [
                      # Make triangle objects from the vertex coordinates
                      ray.Triangle(vs, gray) for vs in vs_list
                  ])

    lights = [
        ray.PointLight(vec([12, 10, 5]), vec([300, 300, 300])),
        ray.AmbientLight(0.1),
    ]

    camera = ray.Camera(vec([10, 10, 10]), target=vec([0, 0, 0]), vfov=25, aspect=16 / 9)
    return ExampleSceneDef(camera=camera, scene=scene, lights=lights);

def TorusExample():
    importlib.reload(ray)
    tan = ray.Material(vec([0.7, 0.7, 0.4]), k_s=0.3, p=90, k_m=0.3)
    gold = ray.Material(vec([0.9, 0.9, 0.5]), k_s=0.6, p=90, k_m=0.6)
    gray = ray.Material(vec([0.2, 0.2, 0.2]), k_m=0.4)
    blue = ray.Material(vec([0.2, 0.2, 0.5]), k_m=0.5)
    #glass = ray.Material(vec([0, 0, 0]), k_m=0.4, k_a=0, k_s=0, is_glass=True, ref_idx=1.1)
    
    scene = ray.Scene([
        ray.Torus(vec([-0.3, 0.7, -0.3]), 0.8, 0.2, blue, vec([np.pi/3,0,np.pi/6]))
    ])

    lights = [
        ray.PointLight(vec([12, 10, 5]), vec([300, 300, 300])),
        ray.AmbientLight(0.1),
    ]

    camera = ray.Camera(vec([6, 2, 2]), target=vec([0, 0, 0]), vfov=25, aspect=16 / 9)

    return ExampleSceneDef(camera=camera, scene=scene, lights=lights)

def EllipsoidExample():
    importlib.reload(ray)

    tan = ray.Material(vec([0.7, 0.7, 0.4]), k_s=0.3, p=90, k_m=0.3)

    scene = ray.Scene([
        ray.Ellipsoid(vec([3, 4.3, 4]), vec([0.75, 0.9, 0.75]), tan)
    ])
    lights = [
        ray.PointLight(vec([15, 10, 5]), vec([500, 500, 500])),
        ray.AmbientLight(1),
    ]

    camera = ray.Camera(vec([5, 5, 0]), target=vec([3.5, 0, 7]), vfov=90, aspect= 1)

    return ExampleSceneDef(camera=camera, scene=scene, lights=lights)

def CylinderExample():
    importlib.reload(ray)

    tan = ray.Material(vec([0.7, 0.7, 0.4]), k_s=0.3, p=90, k_m=0.3)

    scene = ray.Scene([
        ray.Cyclinder(vec([2, 2, 2]), vec([0, 1, 0]), 0.75, 1.5, tan)
    ])
    lights = [
        ray.PointLight(vec([15, 10, 5]), vec([500, 500, 500])),
        ray.AmbientLight(1),
    ]

    camera = ray.Camera(vec([5, 5, 0]), target=vec([3.5, 0, 7]), vfov=90, aspect= 1)

    return ExampleSceneDef(camera=camera, scene=scene, lights=lights)

def LebronJamesExample():
    importlib.reload(ray)
    tan = ray.Material(vec([0.7, 0.7, 0.4]))

    # rot_mat_45 = np.array([[np.sqrt(2)/2, -np.sqrt(2)/2, 0],
    #                        [np.sqrt(2)/2, np.sqrt(2)/2, 0],
    #                        [0, 0, 1]])

    theta = np.deg2rad(45)
    rot_mat_45 = np.array([[np.cos(theta), -np.sin(theta), 0],
                           [np.sin(theta), np.cos(theta), 0],
                           [0, 0, 1]])
    rot_mat_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                          [0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]])
    # rot_mat_y = rot_mat_y @ rot_mat_45
    rot_mat_y = np.eye(3)
    # organize a cube
    scene = ray.Scene([
        # ray.SquareTexture([
        #     vec(rot_mat_y @ [0, 0, 0]),
        #     vec(rot_mat_y @ [1, 0, 0]),
        #     vec(rot_mat_y @ [0, 1, 0]),
        #     vec(rot_mat_y @ [1, 1, 0]),
        # ], tan, texture="lebron.png"),
        # ray.SquareTexture([
        #     vec(rot_mat_y @ [1, 0, 0]),
        #     vec(rot_mat_y @ [1, 0, -1]),
        #     vec(rot_mat_y @ [1, 1, 0]),
        #     vec(rot_mat_y @ [1, 1, -1]),
        # ], tan, texture="lebron2.png"),
        # ray.SquareTexture([
        #     vec(rot_mat_y @ [0, 1, 0]),
        #     vec(rot_mat_y @ [1, 1, 0]),
        #     vec(rot_mat_y @ [0, 1, -1]),
        #     vec(rot_mat_y @ [1, 1, -1]),
        # ], tan, texture="lebron3.png"),
        # ray.SquareTexture([
        #     vec(rot_mat_y @ [1, 0, -1]),
        #     vec(rot_mat_y @ [0, 0, -1]),
        #     vec(rot_mat_y @ [1, 1, -1]),
        #     vec(rot_mat_y @ [0, 1, -1]),
        # ], tan, texture="lebron4.png"),
        ray.SquareTexture([
            vec(rot_mat_y @ [0, 0, -1]),
            vec(rot_mat_y @ [0, 0, 0]),
            vec(rot_mat_y @ [0, 1, -1]),
            vec(rot_mat_y @ [0, 1, 0]),
        ], tan, texture="lebron5.png"),
    ] ,bg_color=vec([0, 0, 0]),
    )

    lights = [
        ray.AmbientLight(1),
    ]

    # camera = ray.Camera(vec([.5, .5, .75]), target=vec([0.5, 0.5, -0.5]), vfov=90, aspect=1)
    # Create line from center to the corner of the cube
    line = vec([0, 1, -1]) - vec([0.5, 0.5, -0.5])
    line = line / np.linalg.norm(line)
    origin = line * 1.5 + vec([0.5, 0.5, -0.5])
    camera = ray.Camera(origin, target=vec([0.5, 0.5, -0.5]), vfov=60, aspect=1)
    return ExampleSceneDef(camera=camera, scene=scene, lights=lights);
