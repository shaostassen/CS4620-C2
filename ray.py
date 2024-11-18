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

    def __init__(self, k_d, k_s=0., p=20., k_m=0., k_a=None):
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
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        discriminant = b*b - 4*a*c

        if discriminant > 0:
            sqrt_discriminant = np.sqrt(discriminant)
            # Check the first root
            t1 = (-b - sqrt_discriminant) / (2.0 * a)
            if ray.start < t1 < ray.end:
                hit_point = ray.origin + t1 * ray.direction
                normal = normalize(hit_point - self.center)
                return Hit(t1, hit_point, normal, self.material)
            
            # Check the second root
            t2 = (-b + sqrt_discriminant) / (2.0 * a)
            if ray.start < t2 < ray.end:
                hit_point = ray.origin + t2 * ray.direction
                normal = normalize(hit_point - self.center)
                return Hit(t2, hit_point, normal, self.material)
        elif discriminant == 0:
            t = -b / (2.0 * a)
            if ray.start < t < ray.end:
                hit_point = ray.origin + t * ray.direction
                normal = normalize(hit_point - self.center)
                return Hit(t, hit_point, normal, self.material)
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
        if abs(a) < 1e-5:
            return no_hit

        f = 1.0 / a
        s = ray.origin - v0
        u = f * np.dot(s, h)
        if u < 0.0 or u > 1.0:
            return no_hit

        q = np.cross(s, e1)
        v = f * np.dot(ray.direction, q)
        if v < 0.0 or u + v > 1.0:
            return no_hit

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
        self.f = None;
        self.M = np.eye(4)

        # self.target = target
        # self.vfov = vfov
        # self.up = up
        # self.aspect = aspect

        #self.f = None; # you should set this to the distance from your center of projection to the image plane
        #self.M = np.eye(4);  # set this to the matrix that transforms your camera's coordinate system to world coordinates
        # TODO A4 implement this constructor to store whatever you need for ray generation
        # self.fov_scale = np.tan(np.deg2rad(vfov) / 2)
        # forward = normalize(target - eye)
        # right = normalize(np.cross(forward, up))
        #up = np.cross(right, forward)

        # self.right = right
        # self.up = up
        # self.forward = forward
        # self.f = np.linalg.norm(target - eye)


        # self.w = (self.eye - self.target) / np.linalg.norm(self.eye - self.target)
        # self.u = np.cross(self.up, self.w) / np.linalg.norm(np.cross(self.up, self.w))
        # self.v = np.cross(self.w, self.u)
        # self.f = 1.0
        # self.h = 2 * self.f * np.tan(np.deg2rad(self.vfov) / 2)
        # self.w_plane = self.aspect * self.h

        # self.W = np.array([[self.w_plane, 0, -self.w_plane /2], [0, -self.h, self.h /2], [0, 0, 1]])

        # self.M = np.eye(4)
        # self.M[0, :3] = self.u
        # self.M[1, :3] = self.v
        # self.M[2, :3] = self.w
        # self.M[:3, 3] = self.eye



    def generate_ray(self, img_point):
        """Compute the ray corresponding to a point in the image.

        Parameters:
          img_point : (2,) -- a 2D point in [0,1] x [0,1], where (0,0) is the upper left
                      corner of the image and (1,1) is the lower right.
        Return:
          Ray -- The ray corresponding to that image location (not necessarily normalized)
        """
        # TODO A4 implement this function
        #aspect_ratio = self.aspect
        #x = (2 * img_point[0] - 1) * self.fov_scale * aspect_ratio
        #y = (1 - 2 * img_point[1]) * self.fov_scale
        #direction = normalize(self.forward + x * self.right + y * self.up)


        # img_point_homogeneous = np.array([img_point[0], img_point[1], 1])
        # image_plane_coords = self.W @ img_point_homogeneous
        # image_plane_point = (self.eye + self.f * (-self.w) + image_plane_coords[0] * self.u + image_plane_coords[1] *self.v)
        # direction = normalize(image_plane_point - self.eye)
        # return Ray(self.eye, direction)


        # return Ray(image_plane_point, direction)
           
        # px = (2 * img_point[0] - 1) * (self.w_plane / 2)
        # py = (1 - 2 * img_point[1]) * (self.h / 2)
        
        # # Calculate the point on the image plane in world coordinates
        # image_plane_point = (self.eye + 
        #                     self.f * (-self.w) + 
        #                     px * self.u + 
        #                     py * self.v)
        
        # # Ray direction (from eye to the image plane point)
        # direction = normalize(image_plane_point - self.eye)
        
        # return Ray(self.eye, direction)
        return Ray(vec([0,0,0]), vec([0,0,-1]))



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
        light_dir = normalize(self.position - hit.point)
        shadow_ray = Ray(hit.point + hit.normal * 1e-3, light_dir)
        shadow_hit = scene.intersect(shadow_ray)
        if shadow_hit.t < np.inf:
            return np.zeros(3)  # shadowed, no illumination
        # Compute the diffuse reflection
        diff = max(np.dot(hit.normal, light_dir), 0)
        return diff * self.intensity * hit.material.k_d


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
        closest_hit = no_hit
        for surf in self.surfs:
            hit = surf.intersect(ray)
            if hit and hit.t < closest_hit.t:
                closest_hit = hit
        return closest_hit


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
    """Compute shading for a ray-surface intersection."""
    color = np.zeros(3)
    
    # Compute the color from ambient light
    color += hit.material.k_a * hit.material.k_d
    
    # Handle point lights and ambient lights
    for light in lights:
        light_color = light.illuminate(ray, hit, scene)
        color += light_color
    
    # Handle mirror reflection
    # if depth < MAX_DEPTH and (isinstance(hit.material.k_m, np.ndarray) and hit.material.k_m.any()) or (isinstance(hit.material.k_m, (int, float)) and hit.material.k_m > 0):
    #     reflection_ray = Ray(hit.point + hit.normal * 1e-3, ray.direction - 2 * np.dot(ray.direction, hit.normal) * hit.normal)
    #     reflection_hit = scene.intersect(reflection_ray)
    #     if reflection_hit:
    #         reflection_color = shade(reflection_ray, reflection_hit, scene, lights, depth + 1)
    #         color += reflection_color

    return np.clip(color, 0.0, 1.0)


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
    output_image = np.zeros((ny, nx, 3), np.float64)
    for i in range(ny):
      for j in range(nx):
        img_point = np.array([(j -0.5)  / nx, (i-0.5) / ny])
        # point = np.array([2 * img_point[0] - 1, 1 - 2 * img_point[1], -1, 0])
        point_x = 2 * img_point[0] - 1
        point_y = 1 - 2 * img_point[1]
        point = np.array([point_x, point_y, 0])
        direction = np.array([0, 0, -1])
        ray = Ray(point, direction)

        intersection = scene.surfs[0].intersect(ray)
        if intersection.t < np.inf:
          output_image[i, j] = np.array([1, 1, 1])
    return output_image
  