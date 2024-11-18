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
        d = ray.direction
        a = np.dot(d, d)
        b = 2.0 * np.dot(oc, d)
        c = np.dot(oc, oc) - self.radius**2
        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            return no_hit
        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2 * a)
        t2 = (-b + sqrt_discriminant) / (2 * a)
        if t1 > ray.start and t1 < ray.end:
            t = t1
        elif t2 > ray.start and t2 < ray.end:
            t = t2
        else:
            return no_hit

        intersection_point = ray.origin + t * d
        normal = (intersection_point - self.center) / self.radius
        return Hit(t=t, point=intersection_point, normal=normal, material=self.material)


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
        a = self.vs[0]
        b = self.vs[1]
        c = self.vs[2]
        m = np.array([a - b, a - c, ray.direction]).T  
        n = a - ray.origin 
        
        try:
            beta, gamma, t = np.linalg.solve(m, n)
        except np.linalg.LinAlgError:
            return no_hit
        
        if t < ray.start or t > ray.end:
            return no_hit
        if beta < 0:
            return no_hit
        if gamma < 0:
            return no_hit
        if beta + gamma > 1:
            return no_hit
        
        intersection_point = ray.origin + t * ray.direction
        normal = np.cross(b - a, c - a)
        normal = normal / np.linalg.norm(normal)
        return Hit(t=t, point=intersection_point, normal=normal, material=self.material)


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
        self.target = target
        self.up = up
        self.vfov = vfov
        self.aspect = aspect
        
        self.w = (self.eye - self.target) / np.linalg.norm(self.eye - self.target)
        self.u = np.cross(self.up, self.w) / np.linalg.norm(np.cross(self.up, self.w))
        self.v = np.cross(self.w, self.u)
        self.f = 1.0
        self.h = 2 * np.tan(np.radians(self.vfov) / 2)
        self.w_plane = self.aspect * self.h

        #W matrix for texture coords -> image plane coords
        self.W = np.array([
            [self.w_plane, 0, -self.w_plane / 2],
            [0, -self.h, self.h / 2],
            [0, 0, 1]
        ])

        #transformation for camera->world coords
        self.M = np.eye(4)
        self.M[0, :3] = self.u
        self.M[1, :3] = self.v
        self.M[2, :3] = self.w
        self.M[:3, 3] = self.eye
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
        img_point_homogeneous = np.array([img_point[0], img_point[1], 1])

        #texture coords to image plane coords
        image_plane_coords = self.W @ img_point_homogeneous

        #position of image plane point in world coords
        image_plane_point = (self.eye + self.f * (-self.w) + image_plane_coords[0] * self.u + image_plane_coords[1] * self.v)

        ray_direction = image_plane_point - self.eye
        return Ray(self.eye, ray_direction)


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
        epsilon = 1e-5
        n = hit.normal

        incoming_light_normalized = (self.position - hit.point) / np.linalg.norm(self.position - hit.point)
        ray_direction_normalized = ray.direction / np.linalg.norm(ray.direction)

        shadow_ray_origin = hit.point + epsilon * n
        shadow_ray_direction = self.position - shadow_ray_origin
        shadow_ray = Ray(shadow_ray_origin, shadow_ray_direction)
        shadow_hit = scene.intersect(shadow_ray)
        distance_to_light = np.linalg.norm(self.position - shadow_ray_origin)
        light_dir_normalized = shadow_ray_direction / distance_to_light

        if shadow_hit != no_hit and shadow_hit.t < distance_to_light:
            return vec([0, 0, 0])

        attenuation = 1 / (distance_to_light ** 2)
        cos_theta = max(0, np.dot(n, light_dir_normalized))

        # specular
        h = (incoming_light_normalized - ray_direction_normalized) / np.linalg.norm(incoming_light_normalized - ray_direction_normalized)
        specular_term = np.dot(n, h) ** hit.material.p
        color_intensity = self.intensity if isinstance(self.intensity, float) else self.intensity

        return color_intensity * attenuation * cos_theta * (hit.material.k_d + hit.material.k_s * specular_term)
        


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
        #print("Ambient light intensity:", self.intensity)
        #print("ambient light returns " + str(hit.material.k_a * self.intensity))
        return hit.material.k_a * self.intensity


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
        min_t = np.inf
        closest_hit = no_hit
        for surf in self.surfs:
            hit = surf.intersect(ray)
            if hit != no_hit and hit.t < min_t:
                min_t = hit.t
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
    if hit == no_hit:
        return scene.bg_color

    total_light_contrib = vec([0, 0, 0])

    for light in lights:
        light_contrib = light.illuminate(ray, hit, scene)
        total_light_contrib += light_contrib

    # if depth < MAX_DEPTH and np.any(hit.material.k_m > 0):
    #     v = -ray.direction
    #     n = hit.normal
    #     epsilon = 1e-5
    #     reflection_ray_origin = hit.point + epsilon * n
    #     reflection_dir = 2 * np.dot(n, v) * n - v
    #     reflection_ray = Ray(reflection_ray_origin, reflection_dir)
    #     reflected_color = shade(reflection_ray, scene.intersect(reflection_ray), scene, lights, depth + 1)
    #     total_light_contrib += hit.material.k_m * reflected_color

    final_color = np.clip(total_light_contrib, 0, 1).astype(np.float32)
    return np.array(final_color, np.float32)

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
    output_image = np.zeros((ny, nx, 3), np.float32)

    for i in range(ny):
        for j in range(nx):
            u = (j + 0.5) / nx  # Offset by .5 for centers of pixels
            v = (i + 0.5) / ny

            ray = camera.generate_ray((u, v))
            hit = scene.intersect(ray)

            output_image[i, j] = shade(ray, hit, scene, lights)

    return np.array(output_image, np.float32)


