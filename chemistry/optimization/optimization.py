import logging
import itertools
import numpy as np

from chemistry.functions import PolarCoordsWithDirection, GaussianException
from chemistry.utils import linalg


def optimize_structure_rfo(molecule, struct, rfo, stop_strategy):
    path = []
    zero = np.zeros(molecule.n_dims - 6)

    for itr in itertools.count():
        path.append(struct)

        motionless = linalg.get_motionless(molecule, struct)
        value, grad, hess = motionless.value_grad_hess(zero)

        delta = rfo(itr=itr, x=zero, val=value, grad=grad, hess=hess)
        print('\n\nnew iteration\nvalue = {}, grad norm = {}, delta norm = {}'.format(value, np.linalg.norm(grad), np.linalg.norm(delta)))
        print('delta norm = {} [{}]'.format(np.linalg.norm(delta), delta))

        if stop_strategy(itr=iter, x=zero, val=value, grad=grad, hess=hess, delta=delta):
            print('break')
            break

        while True:
            try:
                next_value = motionless(delta)
                expected = grad.dot(delta) + .5 * delta.dot(hess.dot(delta))
                real = next_value - value

                print('delta norm = {} [{}]'.format(np.linalg.norm(delta), delta))
                print('expected = {}, real = {}, d = {}'.format(expected, real, abs(expected - real) / abs(expected)))
                print()
                if abs(expected - real) / abs(expected) < .3:
                    break
            except GaussianException as exc:
                print('exception => decreasing delta')

            delta *= .5

        struct = motionless.transform(delta)

    return path



def optimize_on_sphere(func, r, dir, delta_strategy, stop_strategy):
    """
    Optimizes function on sphere surface with center in zero

    :param func: function to optimize
    :param r: radius of sphere to optimize on
    :param dir: initial direction. Vector of norm r
    :param delta_strategy: iteration delta strategy
    :param stop_strategy: iteration stop strategy
    :return: optimization path of directions
    """

    path = []
    skips1 = []
    skips2 = []

    phi = np.zeros(func.n_dims - 1)

    from chemistry.optimization.delta_strategies import Newton, RFO
    newton = Newton()
    rfo = RFO()

    for itr in itertools.count():
        path.append(dir)

        in_polar = PolarCoordsWithDirection(func, r, dir)
        # value, grad = in_polar.value_grad(phi)
        value, grad, hess = in_polar.value_grad_hess(phi)

        skips1.append(in_polar.transform(newton(grad, hess)))
        skips2.append(in_polar.transform(rfo(grad, hess)))

        delta = delta_strategy(itr=itr, x=phi, val=value, grad=grad)

        print(value, dir, np.linalg.norm(grad), np.linalg.norm(delta))

        if stop_strategy(itr=iter, x=phi, val=value, grad=grad, delta=delta):
            break

        dir = in_polar.transform(delta)

    return path, skips1, skips2


def optimize_on_sphere_rfo(func, r, dir, rfo, stop_strategy, comp_eps=1e-9):
    path = []
    phi = np.zeros(func.n_dims - 1)

    for itr in itertools.count():
        path.append(dir)

        in_polar = PolarCoordsWithDirection(func, r, dir)
        value, grad, hess = in_polar.value_grad_hess(phi)

        import matplotlib.pyplot as plt
        plt.imshow(np.abs(hess), cmap='hot', interpolation='nearest')
        plt.show()

        delta = rfo(itr=itr, x=phi, val=value, grad=grad, hess=hess)
        logging.debug('new iteration\nvalue = {}, grad norm = {}, delta norm = {}\nhess values:\n{}'.format(
            value, np.linalg.norm(grad), np.linalg.norm(delta), linalg.calc_singular_values(hess)
        ))
        logging.debug('new iteration\nvalue = {}, grad norm = {}, delta norm = {}'.format(
            value, np.linalg.norm(grad), np.linalg.norm(delta)
        ))

        if stop_strategy(itr=iter, x=phi, val=value, grad=grad, delta=delta, hess=hess):
            break

        while True:
            next_value = in_polar(delta)

            expected = grad.dot(delta) + .5 * delta.dot(hess.dot(delta))
            real = next_value - value

            d = abs(expected - real) / abs(expected)
            logging.debug('delta norm = {}, expected = {}, real = {}, d = {}'.format(
                np.linalg.norm(delta), expected, real, d
            ))
            if abs(expected - real) < comp_eps or d < .3:
                break
            delta *= .5

        dir = in_polar.transform(delta)

    return path


