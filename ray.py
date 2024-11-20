import numpy as np
import cv2
import json

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

    def __init__(self, origin, direction, start=0., end=np.inf, n=1):
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
        self.n = n
    def serialize(self):
          return {
              "origin": self.origin.tolist(),
              "direction": self.direction.tolist(),
              "start": self.start,
              "end": self.end,
              "n": self.n,
          }

class Material:

    def __init__(self, k_d, k_s=0., p=20., k_m=0., k_a=None, flag=None, transparent=False, n =1):
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
        if flag is None:
            self.flag = None
            self.query_texture = None
        else:
          self.flag = flag[0]
          self.query_texture = flag[1]
        self.transparent = transparent
        self.n = n

    def serialize(self):
      return {
          "k_d": self.k_d.tolist(),
          "k_s": self.k_s.tolist() if isinstance(self.k_s, np.ndarray) else self.k_s,
          "p": self.p,
          "k_m": self.k_m.tolist() if isinstance(self.k_m, np.ndarray) else self.k_m,
          "k_a": self.k_a.tolist(),
          "flag": self.flag,
          "transparent": self.transparent,
          "n": self.n,
      }


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
        self.first_t = t
        self.second_t = t
        # self.first_t = None
        # self.second_t = None
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
          hit = Hit(t, point, normal, self.material)
          hit.first_t = pos_t
          hit.second_t = neg_t

          # return Hit(t, point, normal, self.material)
          return hit
      return no_hit
    
    def serialize(self):
        return {
            "center": self.center.tolist(),
            "radius": self.radius,
            "material": self.material.serialize(),
        }

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
    def serialize(self):
        return {
            "vs": self.vs.tolist(),
            "material": self.material.serialize(),
        }
    

class SquareTexture:
    def __init__(self, vs, material, texture="lebron.png", reference_dir=np.array([-1, 0, 0])):
        self.vs = vs # (4, 3)
        self.material = Material(material.k_d, material.k_s, material.p, material.k_m, material.k_a, ("texture", self.query_texture))
        self.texture = cv2.imread(texture)
        self.texture = cv2.cvtColor(self.texture, cv2.COLOR_BGR2RGB)

        def reorder(points):
            edge1 = points[1] - points[0]
            edge2 = points[2] - points[0]
            
            # Compute the normal vector
            normal = np.cross(edge1, edge2)
            
            # Check the alignment of the normal with the reference direction
            alignment = np.dot(normal, reference_dir)
            
            # If the normal points in the opposite direction, swap v1 and v2
            if alignment < 0: return [points[0], points[2], points[1]]
            return [points[0], points[1], points[2]]
        
        # Reorder the vertices to be in CCW order
        self.triangle1 = Triangle(reorder(vs[:3]), material)
        self.triangle2 = Triangle(reorder([vs[1], vs[2], vs[3]]), material)
        
        # Define plan for texture transformation
        line1 = vs[1] - vs[0]
        line2 = vs[2] - vs[0]
        
        # Characterize the corners using basis vectors
        self.u = line1 / np.linalg.norm(line1)
        self.v = line2 / np.linalg.norm(line2)

        # Solve au + bv - v[i] = 0 for i = 0, 1, 2, 3
        X = np.zeros((4,2))
        self.A = np.array([
            [self.u[0], self.v[0]],
            [self.u[1], self.v[1]],
            [self.u[2], self.v[2]],
        ])
        # Add some epsilon to avoid singular matrix
        for i in range(4):
            B = np.array([vs[i][0], vs[i][1], vs[0][2]]) - np.array([vs[0][0], vs[0][1], vs[0][2]])
            cur_x = np.linalg.lstsq(self.A, B, rcond=None)[0]
            X[i][0] = cur_x[0]
            X[i][1] = cur_x[1]

        # Define the texture transformation matrix 2D
        self.texture_transform = cv2.getPerspectiveTransform(X.astype(np.float32), np.array([[0, 0], [1, 0], [0, 1], [1, 1]]).astype(np.float32))


    def intersect(self, ray):
        hit1 = self.triangle1.intersect(ray)
        hit2 = self.triangle2.intersect(ray)
        if hit1.t == np.inf and hit2.t == np.inf: return no_hit

        real_hit = hit1 if hit1.t < hit2.t else hit2
        return Hit(real_hit.t, real_hit.point, real_hit.normal, self.material)
    def query_texture(self, uv):
        # Characterize uv using basis vectors
        B = np.array([uv[0], uv[1], uv[2]]) - np.array([self.vs[0][0], self.vs[0][1], self.vs[0][2]])
        # X = np.linalg.solve(self.A, B)
        X = np.linalg.lstsq(self.A, B, rcond=None)[0]
        y, x = X[1], X[0]
        x = int(x * self.texture.shape[1])
        y = int(y * self.texture.shape[0])
        y = self.texture.shape[0] - y - 1

        y, x = np.clip([y, x], 0, self.texture.shape[0] - 1)
        return self.texture[y, x]
    def serialize(self):
        return {
            "vs": self.vs.tolist(),
            "material": self.material.serialize(),
            "texture": "embedded",  # Assuming the texture will be encoded separately
        }
        
class SphereTexture:
    def __init__(self, center, radius, material, texture="basketball.png"):
        self.material = Material(material.k_d, material.k_s, material.p, material.k_m, material.k_a, ("texture", self.query_texture))
        self.sphere = Sphere(center, radius, self.material)
        self.texture = cv2.imread(texture)
        self.texture = cv2.cvtColor(self.texture, cv2.COLOR_BGR2RGB)
    def intersect(self, ray):
        # return self.sphere.intersect(ray)
        hit = self.sphere.intersect(ray)
        return Hit(hit.t, hit.point, hit.normal, self.material)
    def query_texture(self, uv):
        # Compute the spherical coordinates
        uv = uv - self.sphere.center
        # phi = np.arccos(uv[2] / self.sphere.radius)
        phi = np.arccos(np.clip(uv[2] / self.sphere.radius, -1, 1))
        theta = np.arctan2(uv[1], uv[0])
        if theta < 0: theta += 2 * np.pi
        
        # Rotate thetas by 45 degrees
        # theta += np.deg2rad(360-75)
        # if theta > 2 * np.pi: theta -= 2 * np.pi
        
        # phi += np.deg2rad(90)
        # if phi > np.pi: phi -= np.pi
        
        # theta += np.pi / 2
        
        # Compute the texture coordinates
        x = int(theta / (2 * np.pi) * self.texture.shape[1])
        y = int((1- (phi / np.pi)) * self.texture.shape[0])

        # y, x = np.clip([y, x], 0, self.texture.shape[0] - 1)
        y = np.clip(y, 0, self.texture.shape[0] - 1)
        x = np.clip(x, 0, self.texture.shape[1] - 1)
        return self.texture[y, x]
        # return vec([.1,.1,.7])*255
        
    def serialize(self):
        return {
            "center": self.sphere.center.tolist(),
            "radius": self.sphere.radius,
            "material": self.sphere.material.serialize(),
            "texture": "embedded",  # Assuming the texture will be encoded separately
        }

class Ellipsoid:
    def __init__(self, center, radii, material):
        """Create an ellipsoid with the given center and radii."""
        self.center = center
        self.radii = radii
        self.material = material
        
    def _transform_ray_to_local_space(self, ray):
        """
        Translate the ray's origin to the ellipsoid's local coordinate system.
        """
        local_origin = ray.origin - self.center
        local_direction = ray.direction
        return local_origin, local_direction
    def _quadratic_coefficients(self, ray_origin, ray_direction):
        """
        Calculate coefficients for the quadratic equation of intersection.
        """
        inv_radii_sq = 1 / np.square(self.radii)
        A = np.sum(inv_radii_sq * np.square(ray_direction))
        B = 2 * np.sum(inv_radii_sq * ray_origin * ray_direction)
        C = np.sum(inv_radii_sq * np.square(ray_origin)) - 1
        return A, B, C
    def intersect(self, ray):
        """
        Compute the intersection of the ellipsoid with a ray.
        """
        # Transform ray to ellipsoid's local space
        local_origin, local_direction = self._transform_ray_to_local_space(ray)
        # Compute quadratic coefficients
        A, B, C = self._quadratic_coefficients(local_origin, local_direction)
        # Solve the quadratic equation
        roots = np.roots([A, B, C])
        real_roots = [t for t in roots if np.isreal(t)]
        real_roots = np.real(real_roots)
        # Find the closest valid intersection point
        t_min = float("inf")
        for t in real_roots:
            if ray.start <= t <= ray.end and t < t_min:
                t_min = t
        if t_min == float("inf"):
            return no_hit
        # Compute intersection details
        intersection_point = ray.origin + t_min * ray.direction
        normal = self.compute_normal(intersection_point)
        return Hit(t_min, intersection_point, normal, self.material)
    def compute_normal(self, point):
        """
        Calculate the normal vector at a given point on the ellipsoid's surface.
        
        Args:
            point (array-like): Point on the ellipsoid surface.
        
        Returns:
            array: Normalized normal vector.
        """
        local_normal = (point - self.center) / np.square(self.radii)
        return local_normal / np.linalg.norm(local_normal)
class Torus:
    def __init__(self, center, big_radius, small_radius, material, euler_angles):
        """Create a torus with the given center and radii.
        Parameters:
          center : (3,) -- a 3D point specifying the torus's center
          r1 : float -- a Python float specifying the torus's minor radius
          r2 : float -- a Python float specifying the torus's major radius
          material : Material -- the material of the surface
          euler_angles : (3,) -- the Euler angles for the torus
        """
        self.center = np.asarray(center, dtype=np.float64)
        self.big_radius = big_radius
        self.small_radius = small_radius
        self.material = material
        self.euler_angles = euler_angles
    def _compute_rotation_matrix(self):
        """Compute combined rotation matrix from the Euler angles."""
        def euler_x(theta):
            return np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
        def euler_y(phi):
            return np.array([[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]])
        def euler_z(psi):
            return np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
        return euler_z(self.euler_angles[2]) @ euler_y(self.euler_angles[1]) @ euler_x(self.euler_angles[0])
        
    def intersect(self, ray):
        """
        Calculate intersection between the torus and a ray.
        
        Args:
            ray (Ray): Ray object containing origin, direction, and bounds.
        
        Returns:
            Hit: Intersection data or no hit.
        """
        euler_angles = self._compute_rotation_matrix()
        inverse_matrix = np.linalg.inv(euler_angles)
        local_origin = inverse_matrix @ (ray.origin - self.center)
        local_direction = normalize(inverse_matrix @ ray.direction)
        ray = Ray(local_origin, local_direction, ray.start, ray.end)
        ray_origin = ray.origin
        ray_direction = ray.direction
        # Calculate the coefficients for the quartic equation
        R, r = self.big_radius, self.small_radius
        origin_product = np.dot(ray_origin, ray_origin)
        direction_product = np.dot(ray_direction, ray_direction)
        origin_direction_product = np.dot(ray_origin, ray_direction)
        R2, r2 = R ** 2, r ** 2
        A = direction_product ** 2
        B = 4 * direction_product * origin_direction_product
        C = 2 * direction_product * (origin_product - R2 - r2) + 4 * R2 * (local_direction[2] ** 2) + 4 * origin_direction_product ** 2
        D = 4 * (origin_product - R2 - r2) * origin_direction_product + 8 * R2 * local_direction[2] * local_origin[2]
        E = (origin_product - R2 - r2) ** 2 - 4 * R2 * (r2 - local_origin[2] ** 2)
        # Solve the equation
        solution = np.roots([A, B, C, D, E])
        real = [np.real(sol) for sol in solution if np.isreal(sol) and ray.start <= sol <= ray.end]
        if not real:
            return no_hit
      
        closest = min(real)
        hit_point = local_origin + closest * local_direction
        projected = hit_point.copy()
        projected[2] = 0
        projected_radius_vector = projected / np.linalg.norm(projected) * self.big_radius
        local_normal = normalize(hit_point - projected_radius_vector)
        world_normal = euler_angles @ local_normal
        world_intersection = euler_angles @ hit_point + self.center
        return Hit(closest, world_intersection, world_normal, self.material)
    
class Cylinder: 
    def __init__(self, base, axis, radius, height, material):
        """
        Initialize a cylinder for ray tracing.
        Args:
            base (array-like): The 3D coordinates of the base center of the cylinder.
            axis (array-like): The normalized direction vector of the cylinder's axis.
            radius (float): The radius of the cylinder.
            height (float): The height of the cylinder.
            material (Material): The material of the cylinder's surface.
        """
        self.base = np.asarray(base, dtype=np.float64)
        self.axis = np.asarray(axis, dtype=np.float64)
        self.radius = radius
        self.height = height
        self.material = material
    def intersect(self, ray):
        """
        Calculate the intersection of a ray with the cylinder.
        Args:
            ray (Ray): The ray to intersect with the cylinder.
        Returns:
            Hit: Intersection details or no hit.
        """
        # Step 1: Compute auxiliary vectors
        dp = ray.origin - self.base
        v = ray.direction - np.dot(ray.direction, self.axis) * self.axis
        w = dp - np.dot(dp, self.axis) * self.axis
        # Step 2: Quadratic coefficients
        a = np.dot(v, v)
        b = 2 * np.dot(v, w)
        c = np.dot(w, w) - self.radius**2
        # Step 3: Solve quadratic equation
        roots = np.roots([a, b, c])
        real_roots = [t for t in roots if np.isreal(t)]
        real_roots = np.real(real_roots)
        # Step 4: Filter valid roots within cylinder bounds
        t_min = float("inf")
        intersection_point = None
        for t in real_roots:
            if ray.start <= t <= ray.end:
                point = ray.origin + t * ray.direction
                height_from_base = np.dot(point - self.base, self.axis)
                # Check if point lies within the cylinder's height
                if 0 <= height_from_base <= self.height and t < t_min:
                    t_min = t
                    intersection_point = point
        if t_min == float("inf"):
            return no_hit
        # Step 5: Compute the surface normal at the intersection
        normal = self.compute_normal(intersection_point)
        return Hit(t_min, intersection_point, normal, self.material)
    def compute_normal(self, point):
        """
        Compute the surface normal at a given point on the cylinder.
        Args:
            point (array-like): The point on the cylinder's surface.
        Returns:
            array: Normalized normal vector.
        """
        # Project the point onto the cylinder axis
        to_point = point - self.base
        projection_length = np.dot(to_point, self.axis)
        axis_projection = self.base + projection_length * self.axis
        # Compute the vector perpendicular to the axis
        normal = point - axis_projection
        return normal / np.linalg.norm(normal)   
    
  # class Crown:
  #   def __init__(self, c_h, c_r, ep, e_x, e_y, e_z, e_h):
  #       self.c_h = c_h
  #       self.c_r = c_r
  #       self.ep = ep
  #       self.e_x = e_x
  #       self.e_y = e_y
  #       self.e_z = e_z
  #       self.e_h = e_h
  #   def intersect(self, ray):


 # CSG operations

class Cone:
    def __init__(self, apex, axis, height, angle, material):
        self.apex = np.array(apex, dtype=np.float64)
        self.axis = np.array(axis) / np.linalg.norm(axis)
        self.height = height
        self.angle = angle
        self.cos2_angle = np.cos(angle) ** 2 
        self.material = material

    def intersect(self, ray):
        ray_direction = np.array(ray.direction) / np.linalg.norm(ray.direction)
        v = self.axis
        co = ray.origin - self.apex
        
        # Quadratic coefficients
        a = np.dot(ray_direction, v)**2 - self.cos2_angle
        b = 2 * (np.dot(ray_direction, v) * np.dot(co, v) - np.dot(ray_direction, co) * self.cos2_angle)
        c = np.dot(co, v)**2 - np.dot(co, co) * self.cos2_angle
        
        # Solve the quadratic equation at^2 + bt + c = 0
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            return no_hit  # No intersection
        
        # Compute the roots
        t1 = (-b - np.sqrt(discriminant)) / (2 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2 * a)
        
        # Validate the intersection points
        t_min = min(t1,t2)
        if t_min > 0:
            intersection = ray.origin + t_min * ray.direction
            height_check = np.dot(intersection - self.apex, v)
            if 0 <= height_check <= self.height:
                normal = self.compute_normal(intersection)
                return Hit(t_min, intersection, normal, self.material)
        
        return no_hit  # No valid intersection
    
    def compute_normal(self, point):
        """
        Compute the surface normal at a given point on the cone.
        Args:
            point (array-like): The point on the cone's surface.
        Returns:
            array: Normalized normal vector.
        """
        # Project the point onto the cone axis
        to_point = point - self.apex
        projection_length = np.dot(to_point, self.axis)
        axis_projection = self.apex + projection_length * self.axis
        # Compute the vector perpendicular to the axis
        normal = point - axis_projection
        return normal / np.linalg.norm(normal)
    
# class Trophy:
    
#     @staticmethod
#     def work(self, mat, origin):
#         surfs = []
#         cy1 = Cylinder(origin, vec([0, 1, 0]), 0.5, 0.2, mat)
#         origin[1] += 0.2 
#         cy2 = Cylinder(origin, vec([0, 1, 0]), 0.3, 0.2, mat)

    

class ShapeBoolean:
    def __init__(self, obj1, obj2, operation):
        self.obj1 = obj1
        self.obj2 = obj2
        self.operation = operation

    def intersect(self, ray):
        if self.operation == "union":
            return self.union(ray)
        elif self.operation == "intersection":
            return self.intersection(ray)
        elif self.operation == "difference":
            return self.difference(ray)
      
    def get_intersection_point(self, t, ray):
        return ray.origin + t * ray.direction
  
    def get_hit_normal(self, t, hit1, hit2):
        # if np.any(t == hit1.first_t) or np.any(t == hit1.second_t):
        #     return hit1.normal
        # elif np.any(t == hit2.first_t) or np.any(t == hit2.second_t):
        #     return hit2.normal
        if t == hit1.first_t or t == hit1.second_t:
            return hit1.normal
        elif t == hit2.first_t or t == hit2.second_t:
            return hit2.normal
        
    def union(self, ray):
        # print("union")
        hit1 = self.obj1.intersect(ray)
        hit2 = self.obj2.intersect(ray)

        min_hit1_t = min(hit1.first_t, hit1.second_t)
        min_hit2_t = min(hit2.first_t, hit2.second_t)

        return hit1 if min_hit1_t < min_hit2_t else hit2
    
    def intersection(self, ray):
        hit1 = self.obj1.intersect(ray)
        hit2 = self.obj2.intersect(ray)
        
        first_t = max(min(hit1.first_t, hit1.second_t), min(hit2.first_t, hit2.second_t))
        second_t = min(max(hit1.first_t, hit1.second_t), max(hit2.first_t, hit2.second_t))

        point = self.get_intersection_point(first_t, ray)
        normal = self.get_hit_normal(first_t, hit1, hit2)
        return (first_t, first_t, second_t, point, normal, self.material)
    
    def difference(self, ray):
        hit1 = self.obj1.intersect(ray)
        hit2 = self.obj2.intersect(ray)
        if hit2.first_t < hit1.first_t and hit1.second_t < hit2.second_t:
            return no_hit
        elif hit2.first_t > hit1.second_t or hit2.second_t < hit1.first_t:
            return hit1
        elif hit1.first_t < hit2.first_t and hit2.first_t < hit1.second_t:
            point = self.get_intersection_point(hit2.first_t, ray)
            normal = self.get_hit_normal(hit2.first_t, hit1, hit2)
            return Hit(first_t=hit1.first_t, second_t=hit2.first_t, point=point, normal=normal, material=self.material)

        elif hit1.first_t < hit2.second_t and hit2.second_t < hit1.second_t:
            point = self.get_intersection_point(hit2.second_t, ray)
            normal = self.get_hit_normal(hit2.second_t, hit1, hit2)
            return Hit(hit2.second_t, hit1.second_t, point, normal, self.material)
        
        else: 
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
    def serialize(self):
        return {
            "eye": self.eye.tolist(),
            "target": self.target.tolist(),
            "up": self.up.tolist(),
            "vfov": self.vfov,
            "aspect": self.aspect,
        }

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

        k_d = hit.material.k_d if hit.material.flag != "texture" else hit.material.query_texture(hit_point) / 255

        # return self.intensity * fallout_factor * skidding_factor * (hit.material.k_d + hit.material.k_s * specular_factor)
        return self.intensity * fallout_factor * skidding_factor * (k_d + hit.material.k_s * specular_factor)
    def serialize(self):
        return {
            "position": self.position.tolist(),
            "intensity": self.intensity.tolist() if isinstance(self.intensity, np.ndarray) else self.intensity,
        }
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
        # return self.intensity * hit.material.k_a
        # print("here")
        if hit.material.flag == "texture":
            uv = hit.point
            texture_color = hit.material.query_texture(uv)
            return texture_color / 255 * self.intensity
        return self.intensity * hit.material.k_a
    def serialize(self):
        return {
            "intensity": self.intensity.tolist() if isinstance(self.intensity, np.ndarray) else self.intensity,
        }


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
    def serialize(self):
        return {
            "surfs": [surf.serialize() for surf in self.surfs],
            "bg_color": self.bg_color.tolist(),
        }


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


# Define function so we can run in parallel
def trace_ray(i, j, scene, lights, camera, nx, ny, depth=MAX_DEPTH):
    x = (j + 0.5) / nx
    y = (i + 0.5) / ny
    ray = camera.generate_ray((x, y))
    hit = scene.intersect(ray)
    return shade(ray, hit, scene, lights, depth=0)

# import multiprocessing
from pathos.multiprocessing import Pool
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
  # output_image = np.zeros((ny, nx, 3), np.float32)
  # for i in range(ny):
  #   for j in range(nx):
  #     x = (j + 0.5) / nx
  #     y = (i + 0.5) / ny
      
  #     ray = camera.generate_ray((x, y))

  #     # Step 1
  #     # hit = scene.surfs[0].intersect(ray)
  #     # if hit.first_t < np.inf: output_image[i, j] = np.array([1, 1, 1])

  #     # Step 2-7
  #     hit = scene.intersect(ray)
  #     output_image[i, j] = shade(ray, hit, scene, lights).astype(np.float32)
  # return output_image

  # Parallelize the rendering process
  pool = Pool()
  args = [(i, j, scene, lights, camera, nx, ny) for i in range(ny) for j in range(nx)]
  output_image = np.array(pool.starmap(trace_ray, args)).reshape(ny, nx, 3)
  pool.close()
  pool.join()
  return output_image

