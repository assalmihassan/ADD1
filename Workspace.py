import numpy as np
import matplotlib.pyplot as plt
import random
import math
from mpmath import mp

size = 16 ; a = 1.42 ; L = round(np.sqrt(3) * a / 2 ,3)

def graphene_lattice(size): 
    a = 1.42 ; L = round(np.sqrt(3) * a / 2 ,3) 
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
            #carbon_atoms.append((xx , yy))
            carbon_atoms.append((round(xx, 3), round(yy, 3)))

            # Hollow sites
            if x < (2 * size + 1) / 4 - 1 and y != 0 and y != size * 2 + 2 - 1:
                if y % 2 == 0:
                    hx += 2 * a if x == 0 else 3 * a
                else:
                    hx += 7 * a / 2 if x == 0 else 3 * a
                #hollow_sites.append((hx , hy))
                hollow_sites.append((round(hx, 3), round(hy, 3)))

        xx = 0 ; yy += L
        hx = 0 ; hy += L
        hollows_idx = {tuple(np.round(pos, 3)): i for i, pos in enumerate(hollow_sites)}
    return np.array(carbon_atoms), np.array(hollow_sites) , hollows_idx

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

def PBC(size, site):
    #this function for Periodic Boundary Conditions
    a = 1.42 ; L = round(np.sqrt(3) * a / 2 ,3) ; x,y = site #L = np.sqrt(3) * a / 2 ; x,y = site 
    x_min = a ; x_max = x_max = a * math.ceil(2 * size - ( (size/2) -1 )) # a * 2 * size - ( (size/2) -1 )
    y_min = L ; y_max = 2 * size * L  # = 32 L for size = 16 #28 * L
    if x < x_min : 
        dx = abs(x_min - x) ; x = x_max - dx
    elif x > x_max :
        dx = abs(x - x_max) ; x = x_min + dx
    if y > y_max :
        dy = abs(y - y_max)
        y = dy
    elif y < y_min :
        dy = abs(y)
        y = y_max - dy
    
    return round(x, 3), round(y, 3)

def preprocess_hollow_sites(hollow_sites):
    # Create a fast lookup: rounded (x, y) -> index
    site_lookup = {tuple(np.round(site, 3)): i for i, site in enumerate(hollow_sites)}
    return site_lookup


def find_valid_neighbors(pos, hollow_sites, Li_distribution, size) :
    neighbors = [
        (pos[0] - (3 * a / 2), pos[1] - L),  # h1
        (pos[0] + (3 * a / 2), pos[1] - L),  # h2
        (pos[0] - (3 * a / 2), pos[1] + L),  # h3
        (pos[0] + (3 * a / 2), pos[1] + L),  # h4
        (pos[0], pos[1] - (2 * L)),          # h5
        (pos[0], pos[1] + (2 * L)),          # h6
        ]
    neighbors = [PBC(size, site) for site in neighbors]
    valid_neighbors = []

    idx_neighbors = [None] * len(neighbors)
    valid_idx_neighbors = []
    # search for indexes of neighbors
    for i, neighbor in enumerate(neighbors):
        for j, site in enumerate(hollow_sites):
            if np.allclose(neighbor, site, atol=1e-1):
                idx_neighbors[i] = j
                break
    # search for valid neighbors
    for idx in idx_neighbors : 
        if Li_distribution[idx] is None : 
            valid_idx_neighbors.append(idx)
            valid_neighbors.append(hollow_sites[idx])

    return valid_neighbors , valid_idx_neighbors


carbon_atoms, hollow_sites, hollows_idx = graphene_lattice(size)

print(f"carbon_atomes : {carbon_atoms}")
print(f"hollow_sites : {hollow_sites}")
print(f"site_lookup : {hollows_idx}")

print(f"carbon_atoms.shape: {carbon_atoms.shape}")
print(f"hollow_sites.shape: {hollow_sites.shape}")
print(f"site_lookup: {len(hollows_idx)} hollow sites found.")
print(f"length of hollow_sites: {len(hollow_sites)}")

condition1 = set(range(size +8, size * size - size , size))
print("Condition 1:", condition1)

condition2 = [int(size + size/2 - 1 + i * 16) for i in range(size-2)]

print("Indices for condition 2:", condition2)

# plot_graphene_lattice_with_Li(carbon_atoms, hollow_sites)