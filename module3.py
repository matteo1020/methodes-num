# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 15:25:27 2026

@author: Session
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def creer_video(frames, temps, domain, nom='advection', fps=20):
    """
    Crée une animation de la concentration c au cours du temps.
    frames : liste de tableaux c (ny, nx), temps : liste des instants correspondants
    """
    # les murs (domain == 0) sont masqués pour qu'ils apparaissent en blanc
    def masquer(c):
        return np.ma.masked_where(domain == 0, c)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(masquer(frames[0]), vmin=0, vmax=1, cmap='viridis')
    fig.colorbar(im, ax=ax, label='concentration c')
    titre = ax.set_title(f't = {temps[0]:.2f}')

    def maj(k):
        im.set_data(masquer(frames[k]))      # on ne redessine que les données
        titre.set_text(f't = {temps[k]:.2f}')
        return im, titre

    ani = animation.FuncAnimation(fig, maj, frames=len(frames), blit=False)

    
    ani.save(nom + '.gif', writer='pillow', fps=fps)
    print('Vidéo enregistrée :', nom + '.gif')

    plt.close(fig)