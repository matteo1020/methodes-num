# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 11:55:05 2026

@author: Session
"""
import numpy as np
from laplace import compute
import module1 as m1
import module2 as m2
import module3  as m3
from tqdm import tqdm

### Paramètres du problème :
    
###labyrinthe
lab = 'laby3.jpg'

#choix  de la reconstruction spacial ( soit "rc constante" ou "rc linéaire")
schema = "rc linéaire"
#choix euler ou rk2
methode_temp = "rk2"

#a regler en fonction de la taille du labyrinthe pour pas stocker trop  mais quand même assez
pas_video = 50

""" vrai main"""

#premier truc à faire dans tous les cas ( chargement des données)
compute(lab, 'out')
pot, domain = m1.charger_donnees()

#calcul des vitesses
u, v, u_c, v_c = m1.calculer_vitesses(pot, domain)

#affichage du graohe des vitesses
m1.graph_vitesse(u_c, v_c)

#verifier divergence
m1.verifier_divergence(u, v, domain)

#initialisation de c
c = np.zeros_like(domain, dtype=np.float64)
c_inlet_value = 1.0


inlet = np.zeros_like(domain, dtype=bool)
inlet[1, :] = (domain[1, :] == 2)

c[inlet] = c_inlet_value

#taille des mailles 
dx =1
dy = 1


max_u = np.max(np.abs(u))
max_v = np.max(np.abs(v))

#coéfficient de sécurité qui nous éloigne de l'instabilité
CFL = 0.5
dt = CFL / (max_u/dx + max_v/dy)
print(f"max_u = {max_u}, max_v = {max_v}, dt = {dt}")



ny, nx = domain.shape
sortie = np.zeros_like(domain, dtype=bool)
sortie[ny-2, :] = (domain[ny-2, :] == 2)
amont_sortie = np.zeros_like(domain, dtype=bool)
amont_sortie[ny-3, :] = sortie[ny-2, :]

#borne maximale de pas temporels (pour pas tourner dans le vide si une condition ne marche pas)
nb_pas =100000

seuil = 0.99

#variables pour stocker les images
frames = [c.copy()]          # état initial
temps  = [0.0]

if methode_temp == "euler" :
    for n in tqdm(range(nb_pas), desc='Advection'):
        c = m2.euler(c, u, v, dx, dy, dt, inlet, c_inlet_value, schema,domain)
        if (n + 1) % pas_video == 0:
            frames.append(c.copy())     # .copy() indispensable, sinon toutes les frames seraient identiques
            temps.append((n + 1) * dt)
        if c[amont_sortie].max() >= seuil:
            print("Arrêt au pas", n)
            break
if methode_temp == "rk2" :
    for n in tqdm(range(nb_pas), desc='Advection'):
        c = m2.rk2(c, u, v, dx, dy, dt, inlet, c_inlet_value, schema,domain)
        if (n + 1) % pas_video == 0:
            frames.append(c.copy())     # .copy() indispensable, sinon toutes les frames seraient identiques
            temps.append((n + 1) * dt)
        if c[amont_sortie].max() >= seuil:
            print("Arrêt au pas", n)
            break
m3.creer_video(frames, temps, domain, nom='advection_upwind_euler')