# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 11:55:05 2026

@author: Session
"""
import numpy as np
from laplace import compute
import module1 as m1
import module2 as m2




### vrai main

#premier truc à faire dans tous les cas ( chargement des données)
compute('labysimp.png', 'out')
pot, domain = m1.charger_donnees()

#calcul des vitesses
u, v, u_c, v_c = m1.calculer_vitesses(pot, domain)

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

#attention a changggggggggggeeeer !!!!!
dt = 0.5

#choix  de la reconstruction spacial ( soit "rc constante" ou "rc linéaire")
schema = "rc constante"

#choix du nombre de pas temporels
nb_pas =1000

for n in range(nb_pas):
    c = m2.euler(u, v,c, dx, dy, dt, inlet, c_inlet_value, schema)