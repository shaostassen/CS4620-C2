from utils import *
from ray import *
from cli import render


gray = Material(vec([0.5, 0.5, 0.5]))

# One small sphere centered at z=-0.5
scene = Scene([
    Sphere(vec([0, 0, -0.5]), 0.25, gray),
])

lights = [
    AmbientLight(0.5),
]
camera = Camera(vec([0, 0, 0]), target=vec([0, 0, -0.5]), vfov=90, aspect=1)


render(camera, scene, lights)