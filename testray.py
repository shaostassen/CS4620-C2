import unittest
import numpy as np
from ray import *
from utils import normalize, vec

def assert_direction_matches(v, w):
    np.testing.assert_almost_equal(normalize(v), normalize(w))


def flipy_vec(vect):
    v = vec(vect);
    v[1] = 1-v[1];
    return v;

class TestSphereIntersect(unittest.TestCase):

    def confirm_hit(self, sphere, ray):
        # make sure hit is self-consistent, then return it
        hit = sphere.intersect(ray)
        self.assertLess(hit.t, np.inf)
        np.testing.assert_almost_equal(ray.origin + hit.t * ray.direction, hit.point)
        np.testing.assert_almost_equal(normalize(hit.point - sphere.center), hit.normal)
        self.assertAlmostEqual(np.linalg.norm(hit.point - sphere.center), sphere.radius)
        self.assertIs(hit.material, sphere.material)
        return hit

    def test_unitsphere_hits(self):
        unit_sphere = Sphere(np.array([0,0,0]), 1.0, None)
        # dead center hit
        hit = self.confirm_hit(unit_sphere, Ray(vec([2.0,0.0,0.0]), vec([-1.0,0.0,0.0])))
        self.assertAlmostEqual(hit.t, 1.0)
        # dead center with non-unit direction
        hit = self.confirm_hit(unit_sphere, Ray(vec([3.0,0.0,0.0]), vec([-2.0,0.0,0.0])))
        self.assertAlmostEqual(hit.t, 1.0)
        # off center hit
        hit = self.confirm_hit(unit_sphere, Ray(vec([1.0,0.5,0.0]), vec([-1.0,0.0,0.0])))
        self.assertAlmostEqual(hit.t, 1 - np.sin(np.pi/3))
        # center hit from off axis
        hit = self.confirm_hit(unit_sphere, Ray(vec([2.0,3.0,4.0]), vec([-2.0,-3.0,-4.0])))
        self.assertAlmostEqual(hit.t, 1 - 1 / np.sqrt(29))

    def test_unitsphere_misses(self):
        unit_sphere = Sphere(vec([0,0,0]), 1.0, None)
        # on axis miss
        hit = unit_sphere.intersect(Ray(vec([2.0,3.0,0.0]), vec([-1.0,0.0,0.0])))
        self.assertEqual(hit.t, np.inf)

    def test_nonunit_hits(self):
        # all the same as the first case, but scaled by 3 and shifted by (-1, -5, -7)
        sphere = Sphere(vec([-1,-5,-7]), 3.0, None)
        hit = self.confirm_hit(sphere, Ray(vec([5.0,-5.0,-7.0]), vec([-3.0,0.0,0.0])))
        self.assertAlmostEqual(hit.t, 1.0)
        hit = self.confirm_hit(sphere, Ray(vec([8.0,-5.0,-7.0]), vec([-6.0,0.0,0.0])))
        self.assertAlmostEqual(hit.t, 1.0)
        hit = self.confirm_hit(sphere, Ray(vec([2.0,-3.5,-7.0]), vec([-3.0,0.0,0.0])))
        self.assertAlmostEqual(hit.t, 1 - np.sin(np.pi/3))


class TestCamera(unittest.TestCase):

    def test_default_camera(self):
        # A camera located at the origin facing the -z direction
        cam = Camera()
        # Center ray is straight down the axis
        ray = cam.generate_ray(flipy_vec([0.5, 0.5]))
        np.testing.assert_almost_equal(ray.origin, vec([0,0,0]))
        assert_direction_matches(ray.direction, vec([0,0,-1]))
        # FOV is 90 degrees, so corner rays are centered in octants
        ray = cam.generate_ray(flipy_vec([0, 0]))
        assert_direction_matches(ray.direction, vec([-1,-1,-1]))
        ray = cam.generate_ray(flipy_vec([1, 0]))
        assert_direction_matches(ray.direction, vec([ 1,-1,-1]))
        ray = cam.generate_ray(flipy_vec([0, 1]))
        assert_direction_matches(ray.direction, vec([-1, 1,-1]))

    def test_fov(self):
        # A camera with a different fov: rays should be scaled in x and y
        vfov = 60
        cam = Camera(vfov=vfov)
        s = np.tan(vfov/2 * np.pi/180)
        # Center ray is still straight down the axis
        ray = cam.generate_ray(flipy_vec([0.5, 0.5]))
        np.testing.assert_almost_equal(ray.origin, vec([0,0,0]))
        assert_direction_matches(ray.direction, vec([0,0,-1]))

    def test_aspect(self):
        # A camera with a different aspect ratio: rays should be scaled in x
        aspect = 1.5
        cam = Camera(aspect=aspect)
        # Center ray is still straight down the axis
        ray = cam.generate_ray(flipy_vec([0.5, 0.5]))
        np.testing.assert_almost_equal(ray.origin, vec([0,0,0]))
        assert_direction_matches(ray.direction, vec([0,0,-1]))

    def test_square_frame(self):
        # A camera with a frame where up is equal to v
        cam = Camera(eye=vec([1,2,2]), target=vec([1,4,2]), up=vec([0,0,1]))
        # Center ray is straight down the y axis
        ray = cam.generate_ray(flipy_vec([0.5, 0.5]))
        np.testing.assert_almost_equal(ray.origin, vec([1,2,2]))
        assert_direction_matches(ray.direction, vec([0,1,0]))
        # corners are like default camera but (x,y) is (x, z)
        ray = cam.generate_ray(flipy_vec([0, 0]))
        assert_direction_matches(ray.direction, vec([-1, 1,-1]))
        ray = cam.generate_ray(flipy_vec([1, 0]))
        assert_direction_matches(ray.direction, vec([ 1, 1,-1]))

    def test_arbitrary_frame(self):
        # A camera that lines up with nothing in particular
        eye = vec([3,4,5])
        target = vec([6,7,8])
        up = vec([1,2,3])
        vfov = 47
        cam = Camera(eye=eye, target=target, up=up, vfov=vfov)
        # Center ray points towards target
        ray = cam.generate_ray(flipy_vec([0.5, 0.5]))
        np.testing.assert_almost_equal(ray.origin, eye)
        assert_direction_matches(ray.direction, target - eye)


class TestPoinLight(unittest.TestCase):

    def shading_test(self, p, n, v, l, r, I, material, scene):
        # test with shading at p with normal n and view/illum directions v/l
        # r is distance to light, I is intensity
        t = 1.3        # arbitrary value
        d = -2.3 * v   # arbitrary scale
        ray = Ray(p - t*d, d)  # ray consistent with hit
        hit = Hit(t, p, n, material)
        light = PointLight(p + r * normalize(l), I)
        return light.illuminate(ray, hit, scene)

    def test_diffuse(self):
        # light directly overhead, unit distance and intensity
        np.testing.assert_allclose(
            self.shading_test(
                vec([0,0,0]), vec([0,1,0]), vec([1, 1, 0]),  # p, n, v
                vec([0,1,0]), 1, vec([1,1,1]),  # l, r, I
                Material(vec([0.2,0.4,0.6])), Scene([])
            ),
            vec([0.2,0.4,0.6])
        )
        # light at 60 degrees, unit distance and intensity
        np.testing.assert_allclose(
            self.shading_test(
                vec([0,0,0]), vec([0,1,0]), vec([1, 1, 0]),  # p, n, v
                vec([0,1,np.sqrt(3)]), 1, vec([1,1,1]),  # l, r, I
                Material(vec([0.2,0.4,0.6])), Scene([])
            ),
            0.5 * vec([0.2,0.4,0.6])
        )


class TestTriangleIntersect(unittest.TestCase):

    def test_simple(self):
        # A triangle on the xy plane and perpendicular rays
        tri = Triangle(np.array([[0,0,0], [1,0,0], [0,1,0]]), None)
        hit = tri.intersect(Ray(vec([0.3, 0.3, 1]), vec([0, 0, -1])))
        self.assertAlmostEqual(hit.t, 1.)
        np.testing.assert_allclose(hit.point, [0.3, 0.3, 0])
        np.testing.assert_allclose(hit.normal, [0, 0, 1])
        hit = tri.intersect(Ray(vec([-0.3, 0.3, 1]), vec([0, 0, -1])))
        self.assertEqual(hit.t, np.inf)

    def test_transformed(self):
        # The same triangle under a linear xf of positive determinant
        M = np.array([[3,1,4],[1,5,9],[2,6,5]])
        M = np.sign(np.linalg.det(M)) * M  # ensure no reflection
        u = np.array([2,7,1])
        tri = Triangle(np.array([u + M @ [0,0,0], u + M @ [1,0,0], u + M @ [0,1,0]]), None)
        hit = tri.intersect(Ray(u + M @ [0.3, 0.3, 1], M @ [0, 0, -1]))
        self.assertAlmostEqual(hit.t, 1.)
        np.testing.assert_allclose(hit.point, u + M @ [0.3, 0.3, 0])
        assert_direction_matches(hit.normal, np.linalg.inv(M.transpose()) @ [0, 0, 1])
        hit = tri.intersect(Ray(u + M @ [-0.3, 0.3, 1], M @ [0, 0, -1]))
        self.assertEqual(hit.t, np.inf)



if __name__ == '__main__':
    unittest.main()
