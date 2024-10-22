import numpy as np
from utils.base import Function

A = 1 
B = 10

A_rat = 10

A_ack = 20
B_ack = 0.2
C_ack = 2 * np.pi

RosenBrock = Function (
        func = lambda x : (A-x[0])**2  + B*(x[1] - x[0] **2) **2, 
        grad_func = lambda x : np.array ([
            -2* (A-x[0]) - 4*B*x[0]*(x[1] - x[0]**2),
            2*B* (x[1] - x[0]**2)
            ]),
        name = "rosenbrock"
        )

Rastrigin = Function(
    func=lambda xy: A_rat + xy[0]**2 + xy[1]**2 - A_rat * (np.cos(2 * np.pi * xy[0]) + np.cos(2 * np.pi * xy[1])),
    grad_func=lambda xy: np.array([
        2 * xy[0] + 2 * A_rat * np.pi * np.sin(2 * np.pi * xy[0]),
        2 * xy[1] + 2 * A_rat * np.pi * np.sin(2 * np.pi * xy[1])
    ]),
    name="rastrigin"
)

Ackley = Function(
    func=lambda xy: -A_ack * np.exp(-B_ack * np.sqrt((xy[0]**2 + xy[1]**2) / 2)) - 
                                       np.exp((np.cos(C_ack * xy[0]) + np.cos(C_ack * xy[1])) / 2) + 
                                       A_ack + np.exp(1),
    grad_func=lambda xy: np.array([
        A_ack * B_ack * np.exp(-B_ack * np.sqrt((xy[0]**2 + xy[1]**2) / 2)) * (xy[0] / np.sqrt(xy[0]**2 + xy[1]**2)) + 
        np.exp((np.cos(C_ack * xy[0]) + np.cos(C_ack * xy[1])) / 2) * C_ack * np.sin(C_ack * xy[0]),
        
        A_ack * B_ack * np.exp(-B_ack * np.sqrt((xy[0]**2 + xy[1]**2) / 2)) * (xy[1] / np.sqrt(xy[0]**2 + xy[1]**2)) + 
        np.exp((np.cos(C_ack * xy[0]) + np.cos(C_ack * xy[1])) / 2) * C_ack * np.sin(C_ack * xy[1])
    ]),
    name="ackley"
)


Bohachevsky = Function(
    func=lambda xy: xy[0]**2 + 2 * xy[1]**2 - 0.3 * np.cos(3 * np.pi * xy[0]) - 0.4 * np.cos(4 * np.pi * xy[1]) + 0.7,
    grad_func=lambda xy: np.array([
        2 * xy[0] + 0.9 * np.sin(3 * np.pi * xy[0]),
        4 * xy[1] + 0.8 * np.sin(4 * np.pi * xy[1])
    ]),
    name="bohachevsky"
)


Trid = Function(
    func=lambda xy: (xy[0] - 1)**2 + (xy[1] - xy[0]**2)**2 + (xy[0] - 1) * (xy[1] - 1),
    grad_func=lambda xy: np.array([
        2 * (xy[0] - 1) + (xy[1] - 1),
        2 * (xy[1] - xy[0]**2) * -2 * xy[0] + (xy[0] - 1)
    ]),
    name="trid"
)


RotatedHyperEllipsoid = Function(
    func=lambda xy: (xy[0]**2 + xy[1]**2)**2,
    grad_func=lambda xy: np.array([
        4 * xy[0] * (xy[0]**2 + xy[1]**2),
        4 * xy[1] * (xy[0]**2 + xy[1]**2)
    ]),
    name="rotated_hyper_ellipsoid"
)
