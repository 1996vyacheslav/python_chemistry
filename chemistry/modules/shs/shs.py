import logging
import itertools
import numpy as np
from math import *
from chemistry.utils import linalg
from chemistry.optimization import optimize_on_sphere_rfo
from chemistry.optimization.delta_strategies import RFO
from chemistry.optimization.stop_strategies import GradNorm, SaddlePointType


def search_for_initial_directions(func, r):
    known_directions = []

    for axis in range(func.n_dims):
        for sign in [-1, 1]:
            dir = sign * linalg.eye(func.n_dims, axis)
            path = optimize_on_sphere_rfo(func, r, dir, RFO(0), GradNorm(1e-8))
            dir = path[-1]

            flag = True
            for known_dir in known_directions:
                dot = known_dir.dot(dir) / np.linalg.norm(known_dir) / np.linalg.norm(dir)
                logging.info('from previous: angle = {}, dot = {}'.format(acos(dot), dot))
                if dot > .99:
                    logging.info('close to previous direction. break')
                    flag = False
                    break
            if flag:
                logging.info('new initial direction: {}'.format(dir))
                known_directions.append(dir)

    return known_directions


def shs(func, dir, r0, dr):
    path = []

    try:
        last_energy = func(dir)
        # for itr in itertools.count():
        for itr in range(35):
            path.append(dir)

            print('\n\nIteration {}: r = {}. Sphere optimization started:'.format(itr, r0))
            sphere_path = optimize_on_sphere_rfo(func, r0, dir, RFO(0), GradNorm(1e-7) & SaddlePointType(0))
            dir = sphere_path[-1]
            print('Sphere optimization returned path of length {}'.format(len(sphere_path)))

            new_energy = func(dir)
            if itr > 3 and new_energy < last_energy:
                print('On iteration {} new energy = {} < {} = last_energy. Break'.format(itr, new_energy, last_energy))
                break

            r0 += dr

        if itr == 35:
            print('Iteration limit ({}) exceeded. Break'.format(35))

    except Exception as exc:
        print(exc)

    return path
