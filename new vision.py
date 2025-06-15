import numpy as np
import matplotlib.pyplot as plt
import random
import math
from mpmath import mp

size = 16 ; a = 1.42 ; L = np.sqrt(3) * a / 2 # L = round(np.sqrt(3) * a / 2 ,3)

def graphene_lattice(size): 
    xx, yy = 0, 0
    hx, hy = 0, 0
    carbon_atoms = [] ; hollow_sites = [] 
    for y in np.arange(0, size * 2 + 2):
        for x in np.arange(0, (2 * size + 1) / 2):
            # Carbon atoms
            if y % 2 == 0:
                xx += a if x % 2 == 0 else 2 * a
            else:
                xx += np.sqrt(3) * L if x == 0 else (2 * a if x % 2 == 0 else a)
            carbon_atoms.append((xx , yy))
            

            # Hollow sites
            if x < (2 * size + 1) / 4 - 1 and y != 0 and y != size * 2 + 2 - 1:
                if y % 2 == 0:
                    hx += 2 * a if x == 0 else 3 * a
                else:
                    hx += 7 * a / 2 if x == 0 else 3 * a
                hollow_sites.append((hx , hy))
                

        xx = 0 ; yy += L
        hx = 0 ; hy += L

        hollows_idx = {tuple(pos): i for i, pos in enumerate(hollow_sites)}
    return np.array(carbon_atoms), np.array(hollow_sites), hollows_idx

def plot_graphene_lattice_with_Li(carbon_atoms, Li_positions):
    # this fucntion plot carbon_atoms and Li_positions

    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    # Plot carbon atoms
    ax.scatter(carbon_atoms[:, 0], carbon_atoms[:, 1], c="black", s=10, label="Carbon Atoms")
    ax.scatter(Li_positions[:, 0], Li_positions[:, 1], c="blue", s=15, label="Lithium Ions")

    # Draw carbon-carbon bonds
    for i, carbon_atom in enumerate(carbon_atoms):
        for j, neighbor_atom in enumerate(carbon_atoms):
            if i != j:
                distance = np.linalg.norm(carbon_atom - neighbor_atom)
                if 1.3 <= distance <= 1.5:
                    ax.plot([carbon_atom[0], neighbor_atom[0]], [carbon_atom[1], neighbor_atom[1]], color="gray", linewidth=1, alpha=0.3)

    ax.set_xlabel("x-axis (A)", fontsize=12)
    ax.set_ylabel("y-axis (A)", fontsize=12)   
    ax.set_title("Graphene Lattice", fontsize=14)
    ax.axis("equal")
    ax.legend()
    plt.show()

def PBC(i,size) : 
    # Periodic Boundary Conditions
    if i < size / 2 :
         


carbon_atoms, hollow_sites, hollows_idx = graphene_lattice(size)
print("Carbon Atoms:", carbon_atoms)
print("Hollow Sites:", hollow_sites)
print("Hollows Index:", hollows_idx)    