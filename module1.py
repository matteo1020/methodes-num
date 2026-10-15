# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 12:22:24 2026

@author: Session
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

def charger_donnees() :
    pot = np.load('out_pot.npy')
    domain = np.load('out_domain.npy')
    return pot, domain

pot, domain = charger_donnees()

def calculer_vitesses(pot, domain, dx=1., dy=1.):
    """
    Calcule les vitesses normales aux faces et les vitesses au centre des cellules
    à partir du potentiel.

    Retourne : u, v (aux faces), u_c, v_c (aux centres)
    """
    ny, nx = pot.shape

    #je vais faire une dérivée centrée d'ordre 2 de précision ( car on cherche les valeurs des vitesses aux faces de chaque cases)

    #créons les vecteurs vitesses selon x et y
    u = np.zeros((ny, nx+1))
    v = np.zeros((ny+1, nx))

    #on fait la différence entre tout sauf la première valeur - tout sauf la dernière valeur : pot_i+1-pot_i
    #on met tout dans u sauf à la première et dernière valeur
    u[:, 1:-1] = (pot[:, 1:] - pot[:, :-1]) / dx
    v[1:-1, :] = -(pot[1:, :] - pot[:-1, :]) / dy

    #on crée un masque booléen qui dit si les deux cellules de part et d'autre d'une face sont dans le fluide (True si pas de paroi)
    fluide = domain > 0                          # True si la cellule est dans le fluide
    masque_u = fluide[:, 1:] & fluide[:, :-1]    # forme (ny, nx-1)
    masque_v = fluide[1:, :] & fluide[:-1, :]    # forme (ny-1, nx)

    #on met maintenant les valeurs des vitesses = 0 partout où le masque est false
    u[:, 1:-1] = u[:, 1:-1] * masque_u
    v[1:-1, :] = v[1:-1, :] * masque_v

    #nous avions calculé la vitesse sur les faces, maintenant calculons celle-ci au centre des cellules :
    u_c = 0.5 * (u[:, 1:] + u[:, :-1])   # forme (ny, nx)
    v_c = 0.5 * (v[1:, :] + v[:-1, :])   # forme (ny, nx)
    
    return u, v, u_c, v_c

def verifier_divergence(u, v, domain, dx=1., dy=1.):
    
    #Calcule la divergence discrète de chaque cellule et son maximum
    #dans les cellules fluides intérieures.

    #Retourne : div, max_div_interieur
    
    #vérification que la divergence est nulle
    div =(u[:, 1:] - u[:, :-1])/dx + (v[1:, :] - v[:-1, :])/dy
    interieur = (domain == 1)
    max_div = np.abs(div[interieur]).max() 
    if max_div > 1e-12 :
        sys.exit("Divergence trop grande : arrêt du programme")
    return div, max_div

def graph_vitesse(u_c, v_c) :
    norme = np.sqrt(u_c**2 + v_c**2)
    #représentation graphique
    plt.figure(figsize=(10, 8))
    plt.imshow(norme)
    plt.colorbar(label='norme de la vitesse')
    plt.title('Norme de la vitesse au centre des cellules')
    plt.show()
    
    return
