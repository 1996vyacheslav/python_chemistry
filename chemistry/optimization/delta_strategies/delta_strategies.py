import numpy as np


class FollowGradient:
    def __init__(self, speed=1.):
        self.speed = speed

    def __call__(self, grad, **kwargs):
        return -self.speed * grad


class Newton:
    def __init__(self):
        pass

    def __call__(self, grad, hess, **kwargs):
        return -np.linalg.inv(hess).dot(grad)


class RFO:
    def __init__(self, eig_number):
        self.eig_number = eig_number

    def __call__(self, grad, hess, **kwargs):
        grad = np.expand_dims(grad, -1)
        matr = np.block([[hess, grad], [grad.T, 0]])

        w, v = np.linalg.eigh(matr)

        return (v[:-1, self.eig_number] / v[-1, self.eig_number]).ravel()
