import itertools
from chemistry.optimization import optimize_on_sphere_rfo
from chemistry.optimization.delta_strategies import RFO
from chemistry.optimization.stop_strategies import GradNorm, SaddlePointType


def shs(func, dir, r0, dr):
    path = []

    # for itr in itertools.count():
    for itr in range(35):
        path.append(dir)

        print('Iteration {}: r = {}. Sphere optimization started:'.format(itr, r0))
        sphere_path = optimize_on_sphere_rfo(func, r0, dir, RFO(0), GradNorm(1e-7) & SaddlePointType(0))
        print('Sphere optimization returned path of length {}'.format(len(sphere_path)))

        dir = sphere_path[-1]
        r0 += dr

    return path