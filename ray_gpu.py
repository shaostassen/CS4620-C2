import numpy as np

from utils import *

"""
Core implementation of the ray tracer.  This module contains the classes (Sphere, Mesh, etc.)
that define the contents of scenes, as well as classes (Ray, Hit) and functions (shade) used in 
the rendering algorithm, and the main entry point `render_image`.

In the documentation of these classes, we indicate the expected types of arguments with a
colon, and use the convention that just writing a tuple means that the expected type is a
NumPy array of that shape.  Implementations can assume these types are preconditions that
are met, and if they fail for other type inputs it's an error of the caller.  (This might 
not be the best way to handle such validation in industrial-strength code but we are adopting
this rule to keep things simple and efficient.)
"""


class Ray:

    def __init__(self, origin, direction, start=0., end=np.inf):
        """Create a ray with the given origin and direction.

        Parameters:
          origin : (3,) -- the start point of the ray, a 3D point
          direction : (3,) -- the direction of the ray, a 3D vector (not necessarily normalized)
          start, end : float -- the minimum and maximum t values for intersections
        """
        # Convert these vectors to double to help ensure intersection
        # computations will be done in double precision
        self.origin = np.array(origin, np.float64)
        self.direction = np.array(direction, np.float64)
        self.start = start
        self.end = end


class Material:

    def __init__(self, k_d, k_s=0., p=20., k_m=0., k_a=None, flag=None):
        """Create a new material with the given parameters.

        Parameters:
          k_d : (3,) -- the diffuse coefficient
          k_s : (3,) or float -- the specular coefficient
          p : float -- the specular exponent
          k_m : (3,) or float -- the mirror reflection coefficient
          k_a : (3,) -- the ambient coefficient (defaults to match diffuse color)
        """
        self.k_d = k_d
        self.k_s = k_s
        self.p = p
        self.k_m = k_m
        self.k_a = k_a if k_a is not None else k_d
        self.flag = flag


class Hit:

    def __init__(self, t, point=None, normal=None, material=None):
        """Create a Hit with the given data.

        Parameters:
          t : float -- the t value of the intersection along the ray
          point : (3,) -- the 3D point where the intersection happens
          normal : (3,) -- the 3D outward-facing unit normal to the surface at the hit point
          material : (Material) -- the material of the surface
        """
        self.t = t
        self.point = point
        self.normal = normal
        self.material = material

# Value to represent absence of an intersection
no_hit = Hit(np.inf)


class Sphere:

    def __init__(self, center, radius, material):
        """Create a sphere with the given center and radius.

        Parameters:
          center : (3,) -- a 3D point specifying the sphere's center
          radius : float -- a Python float specifying the sphere's radius
          material : Material -- the material of the surface
        """
        self.center = center
        self.radius = radius
        self.material = material

    def intersect(self, ray):
      """Computes the first (smallest t) intersection between a ray and this sphere.

      Parameters:
        ray : Ray -- the ray to intersect with the sphere
      Return:
        Hit -- the hit data
      """
      # TODO A4 implement this function
      a_term = np.dot(ray.direction, ray.direction)
      b_term = 2 * np.dot(ray.direction, ray.origin - self.center)
      c_term = np.dot(ray.origin - self.center, ray.origin - self.center) - self.radius*self.radius
      discriminant = b_term*b_term - 4*a_term*c_term

      if discriminant < 0: return no_hit
      
      discriminant = np.sqrt(discriminant, dtype=np.float64)
      pos_t = (-b_term + discriminant) / (2*a_term)
      neg_t = (-b_term - discriminant) / (2*a_term)

      if pos_t < ray.start: pos_t = np.inf
      if neg_t < ray.start: neg_t = np.inf

      if pos_t == np.inf and neg_t == np.inf: return no_hit

      t = min(pos_t, neg_t)
      if ray.start < t < ray.end:
          point = ray.origin + t*ray.direction
          normal = (point - self.center) / self.radius
          return Hit(t, point, normal, self.material)
      return no_hit


class Triangle:

    def __init__(self, vs, material):
        """Create a triangle from the given vertices.

        Parameters:
          vs (3,3) -- an arry of 3 3D points that are the vertices (CCW order)
          material : Material -- the material of the surface
        """
        self.vs = vs
        self.material = material

    def intersect(self, ray):
        """Computes the intersection between a ray and this triangle, if it exists.

        Parameters:
          ray : Ray -- the ray to intersect with the triangle
        Return:
          Hit -- the hit data
        """
        # TODO A4 implement this function
        v0, v1, v2 = self.vs
        e1 = v1 - v0
        e2 = v2 - v0
        h = np.cross(ray.direction, e2)
        a = np.dot(e1, h)

        # Ray is parallel to the triangle
        if abs(a) < 1e-6: return no_hit

        f = 1.0 / a
        s = ray.origin - v0
        u = f * np.dot(s, h)
        if u < 0.0 or u > 1.0: return no_hit

        q = np.cross(s, e1)
        v = f * np.dot(ray.direction, q)
        if v < 0.0 or u + v > 1.0: return no_hit

        t = f * np.dot(e2, q)
        if t > ray.start and t < ray.end:
            hit_point = ray.origin + t * ray.direction
            normal = normalize(np.cross(e1, e2))  # Normal is perpendicular to the triangle
            return Hit(t, hit_point, normal, self.material)
        return no_hit

class Camera:

    def __init__(self, eye=vec([0,0,0]), target=vec([0,0,-1]), up=vec([0,1,0]), 
                 vfov=90.0, aspect=1.0):
        """Create a camera with given viewing parameters.

        Parameters:
          eye : (3,) -- the camera's location, aka viewpoint (a 3D point)
          target : (3,) -- where the camera is looking: a 3D point that appears centered in the view
          up : (3,) -- the camera's orientation: a 3D vector that appears straight up in the view
          vfov : float -- the full vertical field of view in degrees
          aspect : float -- the aspect ratio of the camera's view (ratio of width to height)
        """
        self.eye = eye
        self.aspect = aspect
        self.target = target
        self.vfov = vfov
        self.f = 1.0
        
        def normalize_float64(v): return (v.astype(np.float64) / np.linalg.norm(v)).astype(np.float64)

        # Define the camera coordinate system
        self.forward = normalize_float64(self.eye - self.target)
        self.right = normalize_float64(np.cross(up, self.forward))
        self.up = np.cross(self.forward, self.right)
        self.M = np.array([
            [self.right[0], self.up[0], self.forward[0], self.eye[0]],
            [self.right[1], self.up[1], self.forward[1], self.eye[1]],
            [self.right[2], self.up[2], self.forward[2], self.eye[2]],
            [0, 0, 0, 1]], dtype=np.float64)

        # Define image plane dimensions
        self.fov_scale = np.tan(np.deg2rad(self.vfov/2))
        self.height = 2 * self.f * self.fov_scale
        self.width = self.aspect * self.height

        # Define texture transformation matrix
        self.texture_transform = np.array([
            [self.width, 0, -self.width/2],
            [0, -self.height, self.height/2],
            [0, 0, 1]
        ])
        # TODO A4 implement this constructor to store whatever you need for ray generation

    def generate_ray(self, img_point):
        """Compute the ray corresponding to a point in the image.

        Parameters:
          img_point : (2,) -- a 2D point in [0,1] x [0,1], where (0,0) is the upper left
                      corner of the image and (1,1) is the lower right.
        Return:
          Ray -- The ray corresponding to that image location (not necessarily normalized)
        """
        # TODO A4 implement this function
        
        # Transform the texture point to image plane coordinates
        img_point = np.array([img_point[0], img_point[1], 1], dtype=np.float64)
        image_plane_coords = self.texture_transform @ img_point

        camera_coords = np.array([image_plane_coords[0], image_plane_coords[1], -self.f, 0], dtype=np.float64)
        direction_homo = self.M @ camera_coords
        return Ray(self.eye, direction_homo[:3])


class PointLight:

    def __init__(self, position, intensity):
        """Create a point light at given position and with given intensity

        Parameters:
          position : (3,) -- 3D point giving the light source location in scene
          intensity : (3,) or float -- RGB or scalar intensity of the source
        """
        self.position = position
        self.intensity = intensity

    def illuminate(self, ray, hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        # TODO A4 implement this function
        def normalize_float64(v): return (v.astype(np.float64) / np.linalg.norm(v)).astype(np.float64)
        
        ep = 1e-6
        hit_point = hit.point + ep * hit.normal
        light_dir = normalize_float64(self.position - hit_point)
        light_dist = np.linalg.norm(self.position - hit_point)
        
        light_ray = Ray(hit_point, light_dir)
        light_hit = scene.intersect(light_ray)

        if light_hit.t < np.inf and light_hit.t < light_dist: return vec([0, 0, 0])

        fallout_factor = 1 / (light_dist * light_dist)
        skidding_factor = max(np.dot(hit.normal, light_dir), 0)

        bisector = normalize_float64(light_dir - normalize_float64(ray.direction))
        specular_factor = np.dot(hit.normal, bisector) ** hit.material.p

        return self.intensity * fallout_factor * skidding_factor * (hit.material.k_d + hit.material.k_s * specular_factor)
    

class AmbientLight:

    def __init__(self, intensity):
        """Create an ambient light of given intensity

        Parameters:
          intensity (3,) or float: the intensity of the ambient light
        """
        self.intensity = intensity

    def illuminate(self, ray, hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        # TODO A4 implement this function
        return self.intensity * hit.material.k_a


class Scene:

    def __init__(self, surfs, bg_color=vec([0.2,0.3,0.5])):
        """Create a scene containing the given objects.

        Parameters:
          surfs : [Sphere, Triangle] -- list of the surfaces in the scene
          bg_color : (3,) -- RGB color that is seen where no objects appear
        """
        self.surfs = surfs
        self.bg_color = bg_color

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and the scene.

        Parameters:
          ray : Ray -- the ray to intersect with the scene
        Return:
          Hit -- the hit data
        """
        # TODO A4 implement this function
        first_t = np.inf
        first_hit = no_hit

        for surf in self.surfs:
            hit = surf.intersect(ray)
            if hit and hit.t < first_t:
                first_t = hit.t
                first_hit = hit

        return first_hit


MAX_DEPTH = 4

def shade(ray, hit, scene, lights, depth=0):
    """Compute shading for a ray-surface intersection.

    Parameters:
      ray : Ray -- the ray that hit the surface
      hit : Hit -- the hit data
      scene : Scene -- the scene
      lights : [PointLight or AmbientLight] -- the lights
      depth : int -- the recursion depth so far
    Return:
      (3,) -- the color seen along this ray
    When mirror reflection is being computed, recursion will only proceed to a depth
    of MAX_DEPTH, with zero contribution beyond that depth.
    """
    # TODO A4 implement this function
    if hit.t == np.inf: return scene.bg_color

    total_light = np.array([0, 0, 0], dtype=np.float64)
    for light in lights: total_light += light.illuminate(ray, hit, scene)

    can_reflect = np.any(hit.material.k_m > 0) # Check if the material can reflect light
    if depth < MAX_DEPTH and can_reflect:
      ep  = 1e-6
      hit_point = hit.point + ep * hit.normal

      reflect_dir = 2 * np.dot(hit.normal, -ray.direction) * hit.normal + ray.direction
      reflect_ray = Ray(hit_point, reflect_dir)
      
      reflect_color = shade(reflect_ray, scene.intersect(reflect_ray), scene, lights, depth + 1)
      total_light += hit.material.k_m * reflect_color

    return np.clip(total_light, 0, 1, dtype=np.float64)

def render_image(camera, scene, lights, nx, ny):
  """Render a ray traced image.

  Parameters:
    camera : Camera -- the camera defining the view
    scene : Scene -- the scene to be rendered
    lights : Lights -- the lights illuminating the scene
    nx, ny : int -- the dimensions of the rendered image
  Returns:
    (ny, nx, 3) float32 -- the RGB image
  """
  # TODO A4 implement this function
  output_image = np.zeros((ny, nx, 3), np.float32)
  for i in range(ny):
    for j in range(nx):
      x = (j + 0.5) / nx
      y = (i + 0.5) / ny
      
      ray = camera.generate_ray((x, y))

      # Step 1
      # hit = scene.surfs[0].intersect(ray)
      # if hit.t < np.inf: output_image[i, j] = np.array([1, 1, 1])

      # Step 2-7
      hit = scene.intersect(ray)
      output_image[i, j] = shade(ray, hit, scene, lights).astype(np.float32)
  return output_image


from numba import cuda
import math
@cuda.jit(device=True)
def sphere_intersect(sphere, ray):
    a_term = np.dot(ray.direction, ray.direction)
    b_term = 2 * np.dot(ray.direction, ray.origin - sphere.center)
    c_term = np.dot(ray.origin - sphere.center, ray.origin - sphere.center) - sphere.radius*sphere.radius
    discriminant = b_term*b_term - 4*a_term*c_term

    if discriminant < 0: return no_hit
    
    discriminant = np.sqrt(discriminant, dtype=np.float64)
    pos_t = (-b_term + discriminant) / (2*a_term)
    neg_t = (-b_term - discriminant) / (2*a_term)

    if pos_t < ray.start: pos_t = np.inf
    if neg_t < ray.start: neg_t = np.inf

    if pos_t == np.inf and neg_t == np.inf: return no_hit

    t = min(pos_t, neg_t)
    if ray.start < t < ray.end:
        point = ray.origin + t*ray.direction
        normal = (point - sphere.center) / sphere.radius
        return Hit(t, point, normal, sphere.material)
    return no_hit

@cuda.jit
def trace_pixel_gpu(nx, ny, cam_texture_transform, cam_M, cam_eye, cam_f, spheres_center, spheres_radius, inf, output):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    if i < ny and j < nx: 
        x = (j + 0.5) / nx
        y = (i + 0.5) / ny

        # ray = camera.generate_ray((x, y))
        image_place_x = cam_texture_transform[0, 0] * x + cam_texture_transform[0, 1] * y + cam_texture_transform[0, 2]
        image_place_y = cam_texture_transform[1, 0] * x + cam_texture_transform[1, 1] * y + cam_texture_transform[1, 2]
        # image_place_z = cam_texture_transform[2, 0] * x + cam_texture_transform[2, 1] * y + cam_texture_transform[2, 2]

        camera_coords_x = image_place_x
        camera_coords_y = image_place_y
        camera_coords_z = -cam_f
        camera_coords_w = 0

        dir_x = cam_M[0, 0] * camera_coords_x + cam_M[0, 1] * camera_coords_y + cam_M[0, 2] * camera_coords_z + cam_M[0, 3] * camera_coords_w
        dir_y = cam_M[1, 0] * camera_coords_x + cam_M[1, 1] * camera_coords_y + cam_M[1, 2] * camera_coords_z + cam_M[1, 3] * camera_coords_w
        dir_z = cam_M[2, 0] * camera_coords_x + cam_M[2, 1] * camera_coords_y + cam_M[2, 2] * camera_coords_z + cam_M[2, 3] * camera_coords_w
        # dir_w = cam_M[3, 0] * camera_coords_x + cam_M[3, 1] * camera_coords_y + cam_M[3, 2] * camera_coords_z + cam_M[3, 3] * camera_coords_w

        ray_origin_x = cam_eye[0]
        ray_origin_y = cam_eye[1]
        ray_origin_z = cam_eye[2]

        ray_dir_x = dir_x
        ray_dir_y = dir_y
        ray_dir_z = dir_z

        # hit = scene.intersect(ray)
        closest_t = inf
        closest_hit_x = 0
        closest_hit_y = 0
        closest_hit_z = 0
        closest_normal_x = 0
        closest_normal_y = 0
        closest_normal_z = 0

        
        for i in range(len(spheres_center)):
            sphere_center, sphere_radius = spheres_center[i], spheres_radius[i]
            a_term = ray_dir_x * ray_dir_x + ray_dir_y * ray_dir_y + ray_dir_z * ray_dir_z
            b_term = 2 * (ray_dir_x * (ray_origin_x - sphere_center[0]) + ray_dir_y * (ray_origin_y - sphere_center[1]) + ray_dir_z * (ray_origin_z - sphere_center[2]))
            c_term = (ray_origin_x - sphere_center[0]) * (ray_origin_x - sphere_center[0]) + (ray_origin_y - sphere_center[1]) * (ray_origin_y - sphere_center[1]) + (ray_origin_z - sphere_center[2]) * (ray_origin_z - sphere_center[2]) - sphere_radius * sphere_radius

            discriminant = b_term * b_term - 4 * a_term * c_term

            if discriminant < 0: continue
            
            discriminant = math.sqrt(discriminant)

            pos_t, neg_t = (-b_term + discriminant) / (2 * a_term), (-b_term - discriminant) / (2 * a_term)
            if pos_t < 0: pos_t = inf
            if neg_t < 0: neg_t = inf

            if pos_t == inf and neg_t == inf: continue

            t = min(pos_t, neg_t)
            if 0 < t < closest_t:
                closest_t = t
                closest_hit_x = ray_origin_x + t * ray_dir_x
                closest_hit_y = ray_origin_y + t * ray_dir_y
                closest_hit_z = ray_origin_z + t * ray_dir_z
                closest_normal_x = (closest_hit_x - sphere_center[0]) / sphere_radius
                closest_normal_y = (closest_hit_y - sphere_center[1]) / sphere_radius
                closest_normal_z = (closest_hit_z - sphere_center[2]) / sphere_radius

        # output_image[i, j] = shade(ray, hit, scene, lights).astype(np.float32)

