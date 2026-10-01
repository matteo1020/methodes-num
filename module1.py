# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 12:22:24 2026

@author: Session
"""
import numpy as np


def charger_donnees() :
    pot = np.load('out_pot.npy')
    domain = np.load('out_domain')
    return pot, domain