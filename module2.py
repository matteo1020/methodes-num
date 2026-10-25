# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 11:04:01 2026

@author: Session
"""
import numpy as np

# ce module a pour bbut d'implémenter la recontruction constante et reconstruction linéaire
# et aussi d'implémenter euler explicite et range kutta
""" Pour faire un test
c = np.zeros_like(domain, dtype=np.float64)
c_inlet_value = 1.0

# l'entrée est la ligne 1 (la ligne 0 est la bordure de zéros ajoutée par laplace_pot)
inlet = np.zeros_like(domain, dtype=bool)
inlet[1, :] = (domain[1, :] == 2)

c[inlet] = c_inlet_value
"""


def calculer_dcdt(c, u, v, schema, dx, dy, domain):
    """
    Bilan de flux de volumes finis, forme conservative : dc/dt = -div(u c).
    Convention : u > 0 vers la droite (j croissant), v > 0 vers le bas (i croissant).
    schema : 'rc constante' (upwind) ou 'rc linéaire' (pente limitée par Barth-Jespersen)
    """
    ny, nx = c.shape
    c_face_u = np.zeros((ny, nx+1))       # bords : u = 0, la valeur n'a pas d'importance
    c_face_v = np.zeros((ny+1, nx))

    if schema == 'rc constante':
        c_face_u[:, 1:-1] = np.where(u[:, 1:-1] > 0, c[:, :-1], c[:, 1:])
        c_face_v[1:-1, :] = np.where(v[1:-1, :] > 0, c[:-1, :], c[1:, :])

    elif schema == 'rc linéaire':
        fluide = domain > 0
        pmax = np.where(fluide, c, -np.inf)   # un mur ne doit pas compter dans le max
        pmin = np.where(fluide, c,  np.inf)   # ni dans le min

        # max et min sur la cellule et ses voisines fluides (on part de la cellule elle-même)
        c_max, c_min = c.copy(), c.copy()
        # valeurs des voisines pour les pentes (un mur voisin est remplacé par la cellule elle-même)
        c_E, c_W, c_N, c_S = c.copy(), c.copy(), c.copy(), c.copy()

        # voisine de droite (j+1) reçue par la cellule j
        c_max[:, :-1] = np.maximum(c_max[:, :-1], pmax[:, 1:])
        c_min[:, :-1] = np.minimum(c_min[:, :-1], pmin[:, 1:])
        c_E[:, :-1]   = np.where(fluide[:, 1:], c[:, 1:], c[:, :-1])
        # voisine de gauche (j-1)
        c_max[:, 1:] = np.maximum(c_max[:, 1:], pmax[:, :-1])
        c_min[:, 1:] = np.minimum(c_min[:, 1:], pmin[:, :-1])
        c_W[:, 1:]   = np.where(fluide[:, :-1], c[:, :-1], c[:, 1:])
        # voisine du haut (i-1)
        c_max[1:, :] = np.maximum(c_max[1:, :], pmax[:-1, :])
        c_min[1:, :] = np.minimum(c_min[1:, :], pmin[:-1, :])
        c_N[1:, :]   = np.where(fluide[:-1, :], c[:-1, :], c[1:, :])
        # voisine du bas (i+1)
        c_max[:-1, :] = np.maximum(c_max[:-1, :], pmax[1:, :])
        c_min[:-1, :] = np.minimum(c_min[:-1, :], pmin[1:, :])
        c_S[:-1, :]   = np.where(fluide[1:, :], c[1:, :], c[:-1, :])

        # incréments autorisés
        dmax = c_max - c        # >= 0
        dmin = c_min - c        # <= 0

        # pentes centrées
        sx = (c_E - c_W) / (2*dx)
        sy = (c_S - c_N) / (2*dy)

        # limiteur de Barth-Jespersen : minimum sur les 4 faces de la cellule
        phi = np.ones_like(c)
        for d2 in (sx*dx/2, -sx*dx/2, sy*dy/2, -sy*dy/2):       # Est, Ouest, Sud, Nord
            dm = np.where(d2 > 0, dmax, dmin)                    # incrément autorisé selon le signe
            d2_sur = np.where(d2 == 0, 1.0, d2)                  # évite 0/0
            phi = np.minimum(phi, np.where(d2 == 0, 1.0, np.minimum(1.0, dm/d2_sur)))

        # valeurs reconstruites sur les 4 faces de chaque cellule
        c_est   = c + phi*sx*dx/2
        c_ouest = c - phi*sx*dx/2
        c_sud   = c + phi*sy*dy/2
        c_nord  = c - phi*sy*dy/2

        # valeur à la face : côté amont
        c_face_u[:, 1:-1] = np.where(u[:, 1:-1] > 0, c_est[:, :-1], c_ouest[:, 1:])
        c_face_v[1:-1, :] = np.where(v[1:-1, :] > 0, c_sud[:-1, :], c_nord[1:, :])

    else:
        raise ValueError("schema inconnu : " + schema)

    # flux et bilan : communs à tous les schémas
    F_x = u * c_face_u * dy
    F_y = v * c_face_v * dx
    dcdt = (F_x[:, :-1] - F_x[:, 1:] + F_y[:-1, :] - F_y[1:, :]) / (dx*dy)
    return dcdt


def euler(c, u, v,dx, dy, dt,inlet, c_inlet_value, schema, domain):
    
    dcdt = calculer_dcdt(c, u, v, schema,dx, dy,domain)
    
    c1 = c  + dt*dcdt
    
    c1[inlet] = c_inlet_value
    
    return c1
 
    
def rk2(c, u, v, dx, dy, dt, inlet, c_inlet_value, schema, domain):
    k1 = calculer_dcdt(c, u, v, schema, dx, dy, domain)
    c_essai = c + dt*k1
    c_essai[inlet] = c_inlet_value                  # l'état intermédiaire respecte aussi l'entrée
    k2 = calculer_dcdt(c_essai, u, v, schema, dx, dy, domain)
    c1 = c + dt*0.5*(k1 + k2)
    c1[inlet] = c_inlet_value
    return c1