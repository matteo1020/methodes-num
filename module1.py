# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 12:22:24 2026

@author: Session
"""
import numpy as np
import matplotlib.pyplot as plt


def charger_donnees() :
    pot = np.load('out_pot.npy')
    domain = np.load('out_domain.npy')
    return pot, domain

pot, domain = charger_donnees()

dx = 1
dy = 1

ny, nx = pot.shape

#je vais faire une dérivée centrée d'ordre 2 de précision ( car on cherche les valeurs des vitesses aux faces de chaque cases)

#créons les vecteurs vitesses selon x et y
u = np.zeros((ny,nx+1))
v = np.zeros((ny+1,nx))


#on fait la différence entre tout sauf la première valeur - tout sauf la dernière valeur : pot_i+1-pot_i
#on met tout dans u sauf à la premièreet dernière valeur
u[:,1:-1] = (pot[:,1:]-pot[:,:-1])/dx

v[1:-1,:] = (pot[1:,:]-pot[:-1,:])

#on crée  un masque booléen qui dit si la case est a ses voisins qui sont des bords ou pas ( true si pas de bord)
fluide = domain > 0                      # True si la cellule est dans le fluide
masque_u = fluide[:, 1:] & fluide[:, :-1]   # forme (ny, nx-1)

masque_v = fluide[1:,:] & fluide[:-1,:]

#on met maintnenant les valeurs des vitesses =0 partout où le masque est false
u[:,1:-1] = u[:,1:-1]*masque_u

v[1:-1,:] = v[1:-1,:]*masque_v

#nous avions calculé la vitesse sur les faces, maintenant calculons celle-ci au  neoud :
u_c = 0.5 * (u[:, 1:] + u[:, :-1])   # forme (ny, nx)
v_c = 0.5 * (v[1:, :] + v[:-1, :])   # forme (ny, nx)

norme = np.sqrt(u_c**2 + v_c**2)

"""
#représentation graphique
plt.figure(figsize=(10, 8))
plt.imshow(norme)
plt.colorbar(label='norme de la vitesse')
plt.title('Norme de la vitesse au centre des cellules')
plt.show()
"""
#vérificationn que la divergence est nulle 
div = (u[:, 1:] - u[:, :-1])/dx + (v[1:, :] - v[:-1, :])/dy
interieur = (domain == 1)
print("max |div| intérieur :", np.abs(div[interieur]).max())