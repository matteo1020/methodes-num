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


def calculer_dcdt(c, u, v,schema, dx, dy):
    """
    Bilan de flux de volumes finis, reconstruction constante par cellule
    et décentrage amont (upwind). Forme conservative : dc/dt = -div(u c).
    Convention : u > 0 vers la droite (j croissant), v > 0 vers le bas (i croissant).
    """
    ny, nx = c.shape

    # on crée les matrices qui reprennent les valeurs de c en fonction de la vitesse
    c_face_u = np.zeros((ny, nx+1))       # bords : u = 0, donc la valeur n'a pas d'importance
    c_face_v = np.zeros((ny+1, nx))
    
    #shema reconstruction constante
    #si vitesse >0, alors on prend la valeur de c à gauche, sinon à droite ( logique pour de l'advection)
    if schema == 'rc constante':
        c_face_u[:, 1:-1] = np.where(u[:, 1:-1] > 0, c[:, :-1], c[:, 1:])
        c_face_v[1:-1, :] = np.where(v[1:-1, :] > 0, c[:-1, :], c[1:, :])
        
    #schema reconstruction linéaire
    elif schema == 'rc linéaire':
        c_face_u[:, 1:-1] = 0.5 * (c[:, :-1] + c[:, 1:])
        c_face_v[1:-1, :] = 0.5 * (c[:-1, :] + c[1:, :])
    else:
        raise ValueError("schema inconnu : " + schema)
        
        
        # on calcule les flux ( entrant, sortant,...)
    F_x = u * c_face_u * dy               # face verticale : longueur dy
    F_y = v * c_face_v * dx               # face horizontale : longueur dx

    # 3. Bilan par cellule
    #    entrant par l'ouest (F_x[:, :-1]) et le nord (F_y[:-1, :])
    #    sortant par l'est (F_x[:, 1:]) et le sud (F_y[1:, :])
    dcdt = (F_x[:, :-1] - F_x[:, 1:] + F_y[:-1, :] - F_y[1:, :]) / (dx*dy)

    return dcdt

def euler(u, v, c,dx, dy, dt,inlet, c_inlet_value, schema):
    
    dcdt = calculer_dcdt(c, u, v, schema,dx, dy)
    
    c1 = c  + dt*dcdt
    
    c1[inlet] = c_inlet_value
    
    return c1
 