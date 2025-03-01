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
from heapq import *


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



"""         ---               week 02                   ---     """


# Priority queue
# Min heap structure -> smalles elmt is always at root
class prioq:
    def __init__(self, k):
        self.heap = []
        sentinel = (-np.inf, None, np.array([0.0,0.0]))
        for i in range(k):
            heappush(self.heap, sentinel)

    def replace(self, dist2, particle, dr):
        """Heapraplce() automatically restrucures heap after an elmt is pushed
        Function therefore ensures that smalles elmt is at root"""
        heapreplace(self.heap, (dist2, particle, dr))
    
    def key(self):
        """Distance is negative, so prioq has stores the 
        "farthest" distance at heap[0] since it is the most negative"""
        return self.heap[0][0]

def celldist2(cell: Cell, r):
    """Calculates the squared minimum distance between a particle
    position and this node."""
    r = np.array(r)

    # Define high and low boundaries of the cell
    r_high = np.array([cell.pos_tr_x, cell.pos_tr_y])  # Top-right corner
    r_low = np.array([cell.pos_bl_x, cell.pos_bl_y])   # Bottom-left corner

    # Compute minimum distance in each dimension
    d1 = r - r_high
    d2 = r_low - r

    # Ensure distances are positive (outside the box) or zero (inside the box)
    d1 = np.maximum(d1, d2)
    d1 = np.maximum(d1, np.zeros_like(d1))

    # Return squared distance
    return np.dot(d1, d1)


def neighbor_search_periodic(pq, root, r, period):
    # walk the closest image first (at offset=[0, 0])
    for y in [0.0, -period[1], period[1]]:
        for x in [0.0, -period[0], period[0]]:
            rOffset = np.array([x, y])
            neighbor_search(pq, root, r, rOffset)

def neighbor_search(pq: prioq, root:Cell, r, rOffset):
    """ Recursively checking cell and children cells to fill up prioq"""
    if not root.children:
        for particle in root.particles:
            pos_x, pos_y = particle.get_pos()
            dr_x = (pos_x + rOffset[0]) - r[0]
            dr_y = (pos_y + rOffset[1]) - r[1]
            distance = -(dr_x**2 + dr_y**2) #negativee to ensure prioq behaviour

            if distance > pq.key():
                pq.replace(distance, particle, (dr_x, dr_y))
    
    else:
        for child in root.children:
            if -celldist2(child, r- rOffset) > pq.key():
                neighbor_search(pq, child, r, rOffset)


