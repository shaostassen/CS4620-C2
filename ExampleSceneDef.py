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

def CrownExample():
    importlib.reload(ray)
    
    # color 
    tan = ray.Material(vec([0.7, 0.7, 0.4]))
    blue = ray.Material(vec([0.2, 0.2, 0.5]), k_m=0.5)


    # cylinder parameter
    height = 0.4 
    radius = 0.5 
    ep = 0

    # Ellipsoid parameter
    x = 0.07
    y = 0.25
    z = 0.07

    e_height = 0.9
    # make the crown
    scene = ray.Scene([
        ray.Cylinder(vec([0.5, 0.5, -0.5]), vec([0, 1, 0]), radius, height, tan),
        ray.Ellipsoid(vec([0.5+radius-ep, e_height, -0.5]), vec([x, y, z]), blue),
        ray.Ellipsoid(vec([0.5-radius+ep, e_height, -0.5]), vec([x, y, z]), blue),
        ray.Ellipsoid(vec([0.5+radius/2, e_height, -0.5-radius-ep]), vec([x, y, z]), blue),
        ray.Ellipsoid(vec([0.5+radius/2, e_height, -0.5+radius-ep]), vec([x, y, z]), blue),
        ray.Ellipsoid(vec([0.5-radius/2, e_height, -0.5-radius-ep]), vec([x, y, z]), blue),
        ray.Ellipsoid(vec([0.5-radius/2, e_height, -0.5+radius-ep]), vec([x, y, z]), blue),
        
    ], bg_color=vec([0, 0, 0]),)

    lights = [
        ray.AmbientLight(1),
    ]
    origin = vec([0.5, 1.5, 2.5])
    #origin = vec([0, 3, 2])
    camera = ray.Camera(origin, target=vec([0.5, 0.5, -0.5]), vfov=60, aspect=1)
    return ExampleSceneDef(camera=camera, scene=scene, lights=lights)

def CylinderExample():
    importlib.reload(ray)

    tan = ray.Material(vec([0.7, 0.7, 0.4]), k_s=0.3, p=90, k_m=0.3)

    scene = ray.Scene([
        ray.Cylinder(vec([2, 2, 2]), vec([0, 1, 0]), 0.75, 1.5, tan)
    ])
    lights = [
        ray.PointLight(vec([15, 10, 5]), vec([500, 500, 500])),
        ray.AmbientLight(1),
    ]

    camera = ray.Camera(vec([5, 5, 0]), target=vec([3.5, 0, 7]), vfov=90, aspect= 1)

    return ExampleSceneDef(camera=camera, scene=scene, lights=lights)

def GemExample():
    importlib.reload(ray)
    red = ray.Material(vec([0.7, 0.1, 0.1]), k_m=1)
    green = ray.Material(vec([0.1, 0.7, 0.1]), k_m=1)
    blue = ray.Material(vec([0.1, 0.1, 0.7]), k_m=1)
    gold = ray.Material(vec([0.9, 0.9, 0.5]), k_m=1)
    pink = ray.Material(vec([0.9, 0.5, 0.9]), k_m=1)
    orange = ray.Material(vec([0.9, 0.5, 0.1]), k_m=1)

    base = 0.08
    gem_height = 0.3
    x = 0.5
    y = 0
    z = -0.5

    offset = 0.6

    scene = ray.Scene([
        ray.Ellipsoid(vec([x-2.5*offset, y, z]), vec([gem_height, base, base]), red),
        ray.Ellipsoid(vec([x-1.5*offset, y, z-0.2]), vec([gem_height, base, base]), green),
        ray.Ellipsoid(vec([x-offset, y, z]), vec([gem_height, base, base]), blue),
        ray.Ellipsoid(vec([x+offset, y, z]), vec([gem_height, base, base]), gold),
        ray.Ellipsoid(vec([x+1.5*offset, y, z-0.2]), vec([gem_height, base, base]), pink),
        ray.Ellipsoid(vec([x+2.5*offset, y, z]), vec([gem_height, base, base]), orange),
    ])

    lights = [
        ray.PointLight(vec([12, 10, 5]), vec([300, 300, 300])),
        ray.AmbientLight(0.1),
    ]
    origin = vec([0.5, 1.5, 2.5])
    camera = ray.Camera(origin, target=vec([0.5, 0.5, 0.5]), vfov=60, aspect=1)
    return ExampleSceneDef(camera=camera, scene=scene, lights=lights);

def LebronJamesExample():
    importlib.reload(ray)
    tan = ray.Material(vec([0.7, 0.7, 0.4]), p=90, k_m=0.3)

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
        ray.SquareTexture([
            vec(rot_mat_y @ [0, 0, 0]),
            vec(rot_mat_y @ [1, 0, 0]),
            vec(rot_mat_y @ [0, 1, 0]),
            vec(rot_mat_y @ [1, 1, 0]),
        ], tan, texture="lebron.png", reference_dir=vec([0, 0, 1])),
        ray.SquareTexture([
            vec(rot_mat_y @ [1, 0, 0]),
            vec(rot_mat_y @ [1, 0, -1]),
            vec(rot_mat_y @ [1, 1, 0]),
            vec(rot_mat_y @ [1, 1, -1]),
        ], tan, texture="lebron2.png", reference_dir=vec([1, 0, 0])),
        ray.SquareTexture([
            vec(rot_mat_y @ [0, 1, 0]),
            vec(rot_mat_y @ [1, 1, 0]),
            vec(rot_mat_y @ [0, 1, -1]),
            vec(rot_mat_y @ [1, 1, -1]),
        ], tan, texture="lebron3.png", reference_dir=vec([0, 1, 0])),
        ray.SquareTexture([
            vec(rot_mat_y @ [1, 0, -1]),
            vec(rot_mat_y @ [0, 0, -1]),
            vec(rot_mat_y @ [1, 1, -1]),
            vec(rot_mat_y @ [0, 1, -1]),
        ], tan, texture="lebron4.png", reference_dir=vec([0, 0, -1])),
        ray.SquareTexture([
            vec(rot_mat_y @ [0, 0, -1]),
            vec(rot_mat_y @ [0, 0, 0]),
            vec(rot_mat_y @ [0, 1, -1]),
            vec(rot_mat_y @ [0, 1, 0]),
        ], tan, texture="lebron5.png", reference_dir=vec([-1, 0, 0])),
    ] ,bg_color=vec([0, 0, 0]),
    )

    lights = [
        # ray.AmbientLight(.05),
        ray.PointLight(vec([-0.5, 1.5, -1.5]), vec([10, 10, 10])),
    ]

    # camera = ray.Camera(vec([.5, .5, .75]), target=vec([0.5, 0.5, -0.5]), vfov=90, aspect=1)
    # Create line from center to the corner of the cube
    line = vec([0, 1, -1]) - vec([0.5, 0.5, -0.5])
    line = line / np.linalg.norm(line)
    origin = line * 1.5 + vec([0.5, 0.5, -0.5])
    camera = ray.Camera(origin, target=vec([0.5, 0.5, -0.5]), vfov=60, aspect=1)
    return ExampleSceneDef(camera=camera, scene=scene, lights=lights);

def CSGExample():
    importlib.reload(ray)
    tan = ray.Material(vec([0.7, 0.7, 0.4]), k_s=0.3, p=90, k_m=0.3)
    blue = ray.Material(vec([0.2, 0.2, 0.5]), k_m=0.5)
    gold = ray.Material(vec([0.9, 0.9, 0.2]), k_m=0.3, k_s=0.9)
    blue_1 = ray.Material(vec([0.1, 0.1, 0.7]), k_m=0.7, k_s=0.3)

    obj1 = ray.Sphere(vec([0, 0, 0]), 2, tan)
    obj2 = ray.Sphere(vec([0, 0, -1]), 2, blue)

    # ray.Cylinder(vec([0.5, 1.25, -0.5]), vec([0, 1, 0]), radius, height, gold),
    # ray.Ellipsoid(vec([0.5+radius-ep, e_height, -0.5]), vec([x_1, y_1, z_1]), blue_1),
    # ray.Ellipsoid(vec([0.5-radius+ep, e_height, -0.5]), vec([x_1, y_1, z_1]), blue_1),
    # ray.Ellipsoid(vec([0.5+radius/2, e_height, -0.5-radius-ep]), vec([x_1, y_1, z_1]), blue_1),
    # ray.Ellipsoid(vec([0.5+radius/2, e_height, -0.5+radius-ep]), vec([x_1,  y_1, z_1]), blue_1),
    # ray.Ellipsoid(vec([0.5-radius/2, e_height, -0.5-radius-ep]), vec([x_1,  y_1, z_1]), blue_1),
    # ray.Ellipsoid(vec([0.5-radius/2, e_height, -0.5+radius-ep]), vec([x_1,  y_1, z_1]), blue_1),
    # # gems
    # ray.Ellipsoid(vec([x-2.5*offset, y, z]), vec([gem_height, base, base]), blue_1),
    # ray.Ellipsoid(vec([x-1.5*offset, y, z+0.6]), vec([gem_height, base, base]), green),
    # ray.Ellipsoid(vec([x, y, z+1.0]), vec([gem_height, base, base]), blue_1),
    # # ray.Ellipsoid(vec([x+offset, y, z+0.6]), vec([gem_height, base, base]), gold),
    # ray.Ellipsoid(vec([x+1.5*offset, y, z+0.6]), vec([gem_height, base, base]), pink),
    # ray.Ellipsoid(vec([x+2.5*offset, y, z]), vec([gem_height, base, base]), orange),

    # cylinder parameter
    height = 0.4 
    radius = 0.5 
    ep = 0

    # Ellipsoid parameter
    x_1 = 0.07
    y_1 = 0.25
    z_1 = 0.07

    e_height = 0.4 + 1.25

    crown = ray.Cylinder(vec([0.5, 1.25, -0.5]), vec([0, 1, 0]), radius, height, gold)
    jewel1 = ray.Ellipsoid(vec([0.5+radius-ep, e_height, -0.5]), vec([x_1, y_1, z_1]), blue_1)
    jewel2 = ray.Ellipsoid(vec([0.5-radius+ep, e_height, -0.5]), vec([x_1, y_1, z_1]), blue_1)
    jewel3 = ray.Ellipsoid(vec([0.5+radius/2, e_height, -0.5-radius-ep]), vec([x_1, y_1, z_1]), blue_1)
    jewel4 = ray.Ellipsoid(vec([0.5+radius/2, e_height, -0.5+radius-ep]), vec([x_1,  y_1, z_1]), blue_1)
    jewel5 = ray.Ellipsoid(vec([0.5-radius/2, e_height, -0.5-radius-ep]), vec([x_1,  y_1, z_1]), blue_1)
    jewel6 = ray.Ellipsoid(vec([0.5-radius/2, e_height, -0.5+radius-ep]), vec([x_1,  y_1, z_1]), blue_1)
    jewel7 = ray.Ellipsoid(vec([0.5, e_height-.2, -0.5+radius-ep]), vec([x_1,  .1, z_1]), blue_1)

    jewel_combined = ray.ShapeCSG(jewel1, jewel2, "union")
    jewel_combined = ray.ShapeCSG(jewel_combined, jewel3, "union")
    jewel_combined = ray.ShapeCSG(jewel_combined, jewel4, "union")
    jewel_combined = ray.ShapeCSG(jewel_combined, jewel5, "union")
    jewel_combined = ray.ShapeCSG(jewel_combined, jewel6, "union")
    # jewel_combined = ray.ShapeCSG(jewel_combined, jewel7, "union")
    crown = ray.ShapeCSG(crown, jewel_combined, "union")
    crown = ray.ShapeCSG(crown, jewel7, "subtraction")




    scene = ray.Scene([
        crown,
        # ray.ShapeCSG(obj1, obj2, "subtraction")
    ], bg_color=vec([0, 0, 0]),)
    lights = [
        # ray.PointLight(vec([12, 10, 5]), vec([300, 300, 300])),
        ray.AmbientLight(1),
    ]

    side_origin = vec([1, 1, 1])
    side_origin = vec([0.5, 0.5, 1])

    camera = ray.Camera(eye=side_origin, target=vec([0.5,1,-0.5]), vfov=90, aspect=16 / 9)
    return ExampleSceneDef(camera=camera, scene=scene, lights=lights);


def LebronCrownExample():
    importlib.reload(ray)

    theta = np.deg2rad(45)
    rot_mat_45 = np.array([[np.cos(theta), -np.sin(theta), 0],
                           [np.sin(theta), np.cos(theta), 0],
                           [0, 0, 1]])
    rot_mat_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                          [0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]])
    # rot_mat_y = rot_mat_y @ rot_mat_45
    rot_mat_y = np.eye(3)

    # color 
    tan = ray.Material(vec([0.7, 0.7, 0.4]))
    blue = ray.Material(vec([0.2, 0.2, 0.5]), k_m=0.5)


    # cylinder parameter
    height = 0.4 
    radius = 0.5 
    ep = 0

    # Ellipsoid parameter
    x_1 = 0.07
    y_1 = 0.25
    z_1 = 0.07

    e_height = 0.4 + 1.25
    red = ray.Material(vec([0.7, 0.1, 0.1]), k_m=0.7, k_s=0.3)
    green = ray.Material(vec([0.1, 0.7, 0.1]), k_m=0.7, k_s=0.3)
    blue_1 = ray.Material(vec([0.1, 0.1, 0.7]), k_m=0.7, k_s=0.3)
    gold = ray.Material(vec([0.9, 0.9, 0.2]), k_m=0.3, k_s=0.9)
    pink = ray.Material(vec([0.9, 0.5, 0.9]), k_m=0.7, k_s=0.3)
    orange = ray.Material(vec([0.9, 0.5, 0.1]), k_m=0.7, k_s=0.3)
    cyan = ray.Material(vec([0.1, 0.1, 0.7]))
    gold2 = ray.Material(vec([1, 1, 0.1]), k_m=0.5, k_s=0.8)

    gray = ray.Material(vec([0.1, 0.1, 0.1]), k_m=0.4, k_s=0.3, p=90, k_a=0.1)

    base = 0.08
    gem_height = 0.3
    x = 0.5
    y = 0
    z = -0.5

    offset = 0.6

    # organize a cube
    scene = ray.Scene([
        ray.SquareTexture([
            vec(rot_mat_y @ [0, 0, 0]),
            vec(rot_mat_y @ [1, 0, 0]),
            vec(rot_mat_y @ [0, 1, 0]),
            vec(rot_mat_y @ [1, 1, 0]),
        ], tan, texture="lebron.png", reference_dir=vec([0, 0, 1])),
        ray.SquareTexture([
            vec(rot_mat_y @ [1, 0, 0]),
            vec(rot_mat_y @ [1, 0, -1]),
            vec(rot_mat_y @ [1, 1, 0]),
            vec(rot_mat_y @ [1, 1, -1]),
        ], tan, texture="lebron2.png", reference_dir=vec([1, 0, 0])),
        ray.SquareTexture([
            vec(rot_mat_y @ [0, 1, 0]),
            vec(rot_mat_y @ [1, 1, 0]),
            vec(rot_mat_y @ [0, 1, -1]),
            vec(rot_mat_y @ [1, 1, -1]),
        ], tan, texture="lebron3.png", reference_dir=vec([0, 1, 0])),
        ray.SquareTexture([
            vec(rot_mat_y @ [1, 0, -1]),
            vec(rot_mat_y @ [0, 0, -1]),
            vec(rot_mat_y @ [1, 1, -1]),
            vec(rot_mat_y @ [0, 1, -1]),
        ], tan, texture="lebron4.png", reference_dir=vec([0, 0, -1])),
        ray.SquareTexture([
            vec(rot_mat_y @ [0, 0, -1]),
            vec(rot_mat_y @ [0, 0, 0]),
            vec(rot_mat_y @ [0, 1, -1]),
            vec(rot_mat_y @ [0, 1, 0]),
        ], tan, texture="lebron5.png", reference_dir=vec([-1, 0, 0])),
        ray.SphereTexture(
            vec([0.5, 0, 1.25]), 0.5, cyan, texture="basketball1.png"
        ),
        ray.Torus(
            vec([0.5, 0, -0.5]), 0.8, 0.1, tan, vec([np.pi/2,0,0])),
        ray.Cylinder(vec([0.5, 1.25, -0.5]), vec([0, 1, 0]), radius, height, gold),
        ray.Ellipsoid(vec([0.5+radius-ep, e_height, -0.5]), vec([x_1, y_1, z_1]), blue_1),
        ray.Ellipsoid(vec([0.5-radius+ep, e_height, -0.5]), vec([x_1, y_1, z_1]), blue_1),
        ray.Ellipsoid(vec([0.5+radius/2, e_height, -0.5-radius-ep]), vec([x_1, y_1, z_1]), blue_1),
        ray.Ellipsoid(vec([0.5+radius/2, e_height, -0.5+radius-ep]), vec([x_1,  y_1, z_1]), blue_1),
        ray.Ellipsoid(vec([0.5-radius/2, e_height, -0.5-radius-ep]), vec([x_1,  y_1, z_1]), blue_1),
        ray.Ellipsoid(vec([0.5-radius/2, e_height, -0.5+radius-ep]), vec([x_1,  y_1, z_1]), blue_1),
        # gems
        ray.Ellipsoid(vec([x-2.5*offset, y, z]), vec([gem_height, base, base]), blue_1),
        ray.Ellipsoid(vec([x-1.5*offset, y, z+0.6]), vec([gem_height, base, base]), green),
        ray.Ellipsoid(vec([x, y, z+1.0]), vec([gem_height, base, base]), blue_1),
        # ray.Ellipsoid(vec([x+offset, y, z+0.6]), vec([gem_height, base, base]), gold),
        ray.Ellipsoid(vec([x+1.5*offset, y, z+0.6]), vec([gem_height, base, base]), pink),
        ray.Ellipsoid(vec([x+2.5*offset, y, z]), vec([gem_height, base, base]), orange),
        ray.Sphere(vec([0, -40, 0]), 39.5, gray),
        ray.ShapeCSG(ray.Cone(vec([-5, -1, 1]), vec([0, 1, 0]), 3, 0.3, gold2), ray.Sphere(vec([-5, 1.8, 1]), 0.5, gold2), "union"),
        ray.ShapeCSG(ray.Cone(vec([-2, -1, 1]), vec([0, 1, 0]), 3, 0.3, gold2), ray.Sphere(vec([-2, 1.8, 1]), 0.5, gold2), "union"),
        ray.ShapeCSG(ray.Cone(vec([2, -1, 1]), vec([0, 1, 0]), 3, 0.3, gold2), ray.Sphere(vec([2, 1.8, 1]), 0.5, gold2), "union"),
        ray.ShapeCSG(ray.Cone(vec([5, -1, 1]), vec([0, 1, 0]), 3, 0.3, gold2), ray.Sphere(vec([5, 1.8, 1]), 0.5, gold2), "union"),
        
    ] ,bg_color=vec([0, 0, 0]),
    )

    lights = [
        ray.AmbientLight(1),
        ray.PointLight(vec([0.5, 5, -0.5]), 10*vec([.9, .9, .9])),
        ray.PointLight(vec([0.5, 5, 5]), 30*vec([.9, .9, .2])),
        # Add red back light
        ray.PointLight(vec([0.5, 0.5, -3]), 0.5*vec([10, 1, 1])),
        # Add gold back light right
        ray.PointLight(vec([3, 0.5, -0.5]), 0.5*vec([10, 10, 1])),
        # Add green back light left
        # ray.PointLight(vec([-2, 0.5, -0.5]), 0.2*vec([1, 10, 1])),
    ]

    # camera = ray.Camera(vec([.5, .5, .75]), target=vec([0.5, 0.5, -0.5]), vfov=90, aspect=1)
    # Create line from center to the corner of the cube
    origin = vec([0.5, 1.5, 2.5])
    #origin = vec([0, 5, 0])
    # line = vec([0, 2, 0]) - vec([0, 0, 0])
    # line = line / np.linalg.norm(line)
    # origin = line * 5 + vec([0, 0, 0])
    #origin = 2*vec([0, 2, .1])
    #camera = ray.Camera(origin, target=vec([0, 0, 0]), vfov=90, aspect=1)
    
    corner = vec([0, 1, 0])
    target = vec([0.5, 0.25, -0.5])
    corner_dir = corner - target
    corner_dir = corner_dir / np.linalg.norm(corner_dir)
    origin = corner_dir * 4 + target + vec([0, -0.95, 0])
    
    
    camera = ray.Camera(eye=origin, target=target, vfov=60, aspect=16/9)
    return ExampleSceneDef(camera=camera, scene=scene, lights=lights);

def ConeExample():
    importlib.reload(ray)
    tan = ray.Material(vec([0.7, 0.7, 0.4]), k_s=0.3, p=90, k_m=0.3)
    blue = ray.Material(vec([0.2, 0.2, 0.5]), k_m=0.5)
    gray = ray.Material(vec([0.2, 0.2, 0.2]), k_m=0.4)

    scene = ray.Scene([
        ray.Cone(vec([0, 7, 0]), vec([0, -1, 0]), 4, 0.3, tan),
        #ray.Sphere(vec([0, -40, 0]), 39.5, gray),
    ])

    lights = [
        ray.AmbientLight(0.8),
    ]

    camera = ray.Camera(vec([0, 0, 10]), target=vec([0, 5, 0]), vfov=60, aspect=16 / 9)
    return ExampleSceneDef(camera=camera, scene=scene, lights=lights);