#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   utils.py
@Time    :   2025/02/19 08:07:01
@Author  :   Dario B. Hug
@Contact :   dario.hug@acentauri.ch
@Desc    :   Utils of the Simulations II Course on the SPH
'''

import numpy as np
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, pos_x, pos_y):
        self.pos_x = pos_x
        self.pos_y = pos_y

    def get_pos(self):
        return (self.pos_x, self.pos_y)

class Cell:
    def __init__(self, pos_bl_x, pos_bl_y, pos_tr_x, pos_tr_y,particles = None):
        self.pos_bl_x = pos_bl_x
        self.pos_bl_y = pos_bl_y
        self.pos_tr_x = pos_tr_x
        self.pos_tr_y = pos_tr_y
        self.particles = particles if particles is not None else []
        self.children = []
        
    def get_boundry(self):
        return [(self.pos_bl_x, self.pos_bl_y), (self.pos_tr_x, self.pos_tr_y)]
    
    def print_boundry(self):
        print(f"{(self.pos_bl_x, self.pos_tr_y)}  ---  {(self.pos_tr_x, self.pos_tr_y)} \n \n {(self.pos_bl_x, self.pos_bl_y)}  ---  {(self.pos_tr_x, self.pos_bl_y)}")

    def split_cell(self, dimension_x, max_particles = 8):

        if len(self.particles) <= max_particles:
            return

        # Splitting in X direction
        if dimension_x:
            # Find middle of cell in x-Dimension
            mid_x = 0.5 * (self.pos_bl_x + self.pos_tr_x)

            subcells = [
                Cell(self.pos_bl_x, self.pos_bl_y, mid_x, self.pos_tr_y),
                Cell(mid_x, self.pos_bl_y, self.pos_tr_x, self.pos_tr_y),
            ]

            for particle in self.particles:
                x, y = particle.get_pos()
                if x < mid_x: subcells[0].particles.append(particle) #Append to lower Side
                else: subcells[1].particles.append(particle) #Append to upper Side

        else: #Splitting in Y-Direction

            mid_y = 0.5 * (self.pos_bl_y + self.pos_tr_y)

            subcells = [
                Cell(self.pos_bl_x, self.pos_bl_y, self.pos_tr_x, mid_y),
                Cell(self.pos_bl_x, mid_y, self.pos_tr_x, self.pos_tr_y)
            ]

            for particle in self.particles: 
                x, y = particle.get_pos()
                if y < mid_y: subcells[0].particles.append(particle)
                else: subcells[1].particles.append(particle)
    
        self.children = subcells 
        self.particles = [] #Clear current Cell

        for child in self.children: 
            if len(child.particles) >= max_particles:
                child.split_cell( not dimension_x, max_particles)

def plot_cells(ax, cell):

    # Draw the cell boundary
    rect = plt.Rectangle(
        (cell.pos_bl_x, cell.pos_bl_y),  # Bottom-left corner
        cell.pos_tr_x - cell.pos_bl_x,   # Width
        cell.pos_tr_y - cell.pos_bl_y,   # Height
        edgecolor='black',
        linewidth=1,
        fill=False
    )
    ax.add_patch(rect)

    # Plot particles as red dots
    for particle in cell.particles:
        x, y = particle.get_pos()
        ax.plot(x, y, 'ro', markersize=3)

    # Recursively plot children
    for sub_cell in cell.children:
        plot_cells(ax, sub_cell)