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
    return np.array(carbon_atoms), np.array(hollow_sites)

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
    for i, neighbor in enumerate(neighbors) :
        for j, site in enumerate(hollow_sites) :
            if np.allclose(neighbor, site, atol=1e-1):
                idx_neighbors[i] = j
                break
    # search for valid neighbors
    for i, idx in enumerate(idx_neighbors) :
        if Li_distribution[idx] is None :
            # add the condition here
            pos2 = hollow_sites[idx]
            if i == 0 : 
                neighbors2 = [ 
                    (pos2[0] - (3 * a / 2), pos2[1] - L),  # h1
                    (pos2[0] + (3 * a / 2), pos2[1] - L),  # h2
                    (pos2[0] - (3 * a / 2), pos2[1] + L),  # h3
                    (pos2[0], pos2[1] - (2 * L)),          # h5
                    (pos2[0], pos2[1] + (2 * L)),          # h6
                ]
                #Add verification
                neighbors2 = [PBC(size, site) for site in neighbors2]
                idx_neighbors2 = [None] * len(neighbors2)
                for k, n2 in enumerate(neighbors2):
                    for j, site in enumerate(hollow_sites):
                        if np.allclose(n2, site, atol=1e-1):
                            idx_neighbors2[k] = j
                            break
                for idx2 in idx_neighbors2:
                    if Li_distribution[idx2] is not None :
                        break
                else :
                    valid_idx_neighbors.append(idx)
                    valid_neighbors.append(hollow_sites[idx])

            elif i == 1 :
                neighbors2 = [
                    (pos2[0] - (3 * a / 2), pos2[1] - L),  # h1
                    (pos2[0] + (3 * a / 2), pos2[1] - L),  # h2
                    (pos2[0] + (3 * a / 2), pos2[1] + L),  # h4
                    (pos2[0], pos2[1] - (2 * L)),          # h5
                    (pos2[0], pos2[1] + (2 * L)),          # h6

                ]
                #Add verification
                neighbors2 = [PBC(size, site) for site in neighbors2]
                idx_neighbors2 = [None] * len(neighbors2)
                for k, n2 in enumerate(neighbors2):
                    for j, site in enumerate(hollow_sites):
                        if np.allclose(n2, site, atol=1e-1):
                            idx_neighbors2[k] = j
                            break
                for idx2 in idx_neighbors2:
                    if Li_distribution[idx2] is not None :
                        break
                else :
                    valid_idx_neighbors.append(idx)
                    valid_neighbors.append(hollow_sites[idx])

            elif i == 2 :
                neighbors2 = [
                    (pos2[0] - (3 * a / 2), pos2[1] - L),  # h1
                    (pos2[0] - (3 * a / 2), pos2[1] + L),  # h3
                    (pos2[0] + (3 * a / 2), pos2[1] + L),  # h4
                    (pos2[0], pos2[1] - (2 * L)),          # h5
                    (pos2[0], pos2[1] + (2 * L)),          # h6
                ]
                #Add verification
                neighbors2 = [PBC(size, site) for site in neighbors2]
                idx_neighbors2 = [None] * len(neighbors2)
                for k, n2 in enumerate(neighbors2):
                    for j, site in enumerate(hollow_sites):
                        if np.allclose(n2, site, atol=1e-1):
                            idx_neighbors2[k] = j
                            break
                for idx2 in idx_neighbors2:
                    if Li_distribution[idx2] is not None :
                        break
                else :
                    valid_idx_neighbors.append(idx)
                    valid_neighbors.append(hollow_sites[idx])

            elif i == 3 :
                neighbors2 = [
                    (pos2[0] + (3 * a / 2), pos2[1] - L),  # h2
                    (pos2[0] - (3 * a / 2), pos2[1] + L),  # h3
                    (pos2[0] + (3 * a / 2), pos2[1] + L),  # h4
                    (pos2[0], pos2[1] - (2 * L)),          # h5
                    (pos2[0], pos2[1] + (2 * L)),          # h6
                ]
                #Add verification
                neighbors2 = [PBC(size, site) for site in neighbors2]
                idx_neighbors2 = [None] * len(neighbors2)
                for k, n2 in enumerate(neighbors2):
                    for j, site in enumerate(hollow_sites):
                        if np.allclose(n2, site, atol=1e-1):
                            idx_neighbors2[k] = j
                            break
                for idx2 in idx_neighbors2:
                    if Li_distribution[idx2] is not None :
                        break
                else :
                    valid_idx_neighbors.append(idx)
                    valid_neighbors.append(hollow_sites[idx])

            elif i == 4 :
                neighbors2 = [
                    (pos2[0] - (3 * a / 2), pos2[1] - L),  # h1
                    (pos2[0] + (3 * a / 2), pos2[1] - L),  # h2
                    (pos2[0] - (3 * a / 2), pos2[1] + L),  # h3
                    (pos2[0] + (3 * a / 2), pos2[1] + L),  # h4
                    (pos2[0], pos2[1] - (2 * L)),          # h5
                ]
                #Add verification
                neighbors2 = [PBC(size, site) for site in neighbors2]
                idx_neighbors2 = [None] * len(neighbors2)
                for k, n2 in enumerate(neighbors2):
                    for j, site in enumerate(hollow_sites):
                        if np.allclose(n2, site, atol=1e-1):
                            idx_neighbors2[k] = j
                            break
                for idx2 in idx_neighbors2:
                    if Li_distribution[idx2] is not None :
                        break
                else :
                    valid_idx_neighbors.append(idx)
                    valid_neighbors.append(hollow_sites[idx])
            else : 
                neighbors2 = [
                    (pos2[0] - (3 * a / 2), pos2[1] - L),  # h1
                    (pos2[0] + (3 * a / 2), pos2[1] - L),  # h2
                    (pos2[0] - (3 * a / 2), pos2[1] + L),  # h3
                    (pos2[0] + (3 * a / 2), pos2[1] + L),  # h4
                    (pos2[0], pos2[1] + (2 * L)),          # h6

                ] 
                #Add verification
                neighbors2 = [PBC(size, site) for site in neighbors2]
                idx_neighbors2 = [None] * len(neighbors2)
                for k, n2 in enumerate(neighbors2):
                    for j, site in enumerate(hollow_sites):
                        if np.allclose(n2, site, atol=1e-1):
                            idx_neighbors2[k] = j
                            break
                for idx2 in idx_neighbors2:
                    if Li_distribution[idx2] is not None :
                        break
                else :
                    valid_idx_neighbors.append(idx)
                    valid_neighbors.append(hollow_sites[idx])
    return valid_neighbors , valid_idx_neighbors

def find_CLMB_interaction(pos, hollow_sites, Li_distribution, size) :
    clmb_intr=[
        (pos[0] - (3 * a / 2), pos[1] - L),  # h1
        (pos[0] + (3 * a / 2), pos[1] - L),  # h2
        (pos[0] - (3 * a / 2), pos[1] + L),  # h3
        (pos[0] + (3 * a / 2), pos[1] + L),  # h4
        (pos[0], pos[1] - (2 * L)),          # h5
        (pos[0], pos[1] + (2 * L)),          # h6

        (pos[0] - (3 * a / 2), pos[1] - (3 * L)),  # h21
        (pos[0] + (3 * a / 2), pos[1] - (3 * L)),  # h22
        (pos[0] - (3 * a), pos[1]),                # h23
        (pos[0] + (3 * a), pos[1]),                # h24
        (pos[0] - (3 * a / 2), pos[1] + (3 * L)),  # h25
        (pos[0] + (3 * a / 2), pos[1] + (3 * L)),  # h26

        (pos[0], pos[1] - 4 * L),
        (pos[0], pos[1] + 4 * L),
        (pos[0] - 3 * a, pos[1] - 2 * L),
        (pos[0] + 3 * a, pos[1] - 2 * L),
        (pos[0] - 3 * a, pos[1] + 2 * L),
        (pos[0] + 3 * a, pos[1] + 2 * L),
    ]

    clmb_intr = [PBC(size, site) for site in clmb_intr]
    valid_clmb_intr=[]

    idx_clmb_intr = [None] * len(clmb_intr)
    valid_idx_clmb_intr = []
    # search for indexes of clmb_int
    for i, clmb in enumerate(clmb_intr):
        for j, site in enumerate(hollow_sites):
            if np.allclose(clmb, site, atol=1e-1):
                idx_clmb_intr[i] = j
                break
    
    for idx in idx_clmb_intr : 
        if Li_distribution[idx] is not None : #then ther is an interaction 
            valid_idx_clmb_intr.append(idx)
            valid_clmb_intr.append(hollow_sites[idx])
    
    return valid_clmb_intr, valid_idx_clmb_intr

def calc_Transition_Rates(i, hollow_sites, Li_distribution, valid_neighbors, valid_clmb_intr) : 
    T = 294; K_b = 1.380649e-23; h = 6.62607015e-34
    k_e = 8.99e9; e = 1.602e-19; K2 = k_e * e*e
    p_cst = (2*K_b*T) / h

    r_diff = 0 ; sum_rates=0 ; Transition_Rates=[]
    for site in valid_neighbors :
        print(f"site = {site}")
        point_c = (np.array(hollow_sites[i]) + np.array(site)) / 2 
        if len(valid_clmb_intr) == 0 :
            r_diff = 0
        else : 
            for Li in valid_clmb_intr :
                raa = np.linalg.norm(np.array(Li_distribution[i]) - np.array(Li)) ; ra = raa *1e-10
                rcc = np.linalg.norm(np.array(point_c) - np.array(Li)) ; rc = rcc *1e-10
                r_diff += (1/rc) - (1/ra)

        E_m = K2 * r_diff + 0.23*1.602e-19 
        exp = - E_m / ( T * K_b )
        pi = p_cst * mp.exp(exp)
        Transition_Rates.append(pi) 
        sum_rates += pi
    return Transition_Rates, sum_rates

def jump(Transition_Rates, valid_neighbors, valid_idx_neighbors) :
    Cum_Transition_Rates = np.cumsum(Transition_Rates)
    random_nbr = random.random()
    event_selection = random_nbr * Cum_Transition_Rates[-1] 
    
    selected_jump = next(i for i, p in enumerate(Cum_Transition_Rates) if event_selection <= p)

    selected_site = valid_neighbors[selected_jump]
    idx_selected_site = valid_idx_neighbors[selected_jump]

    return selected_site , idx_selected_site

def plot_msd_vs_time(time_history, msd_list):
    

    plt.figure(figsize=(8, 5))
    plt.plot(time_history, msd_list)#, marker='o')
    plt.xlabel('Time (s)')
    plt.ylabel('MSD (cm$^2$)')
    plt.title('Mean Squared Displacement vs Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
#####################################################################################
#####################################################################################
# This code generates a graphene lattice and places a single lithium ion at a random hollow site.
#####################################################################################
#####################################################################################

carbon_atoms, hollow_sites = graphene_lattice(size)
Li_distribution = [None] * len(hollow_sites)
# Li_distribution = np.zeros(len(hollow_sites), dtype=int)
Li_positions = [] 

index = random.randint(0, len(hollow_sites) - 1)
Li_distribution[index] = hollow_sites[index]

# Li_positions.append(hollow_sites[index])
Li_positions = [hollow_sites[index]] 
time_history = [0.0]

initial_position = np.array(Li_positions)

# print(f"hollow_sites: {hollow_sites}")
# print(f"Li_positions: {Li_positions}")

clock = 0.0; delta_t = 0
steps = 1
for step in range(steps):
    print(f"Step {step + 1}:")
    
    for idx, li_pos in enumerate(Li_distribution):
        if li_pos is not None:
            print(f"Li ion exists at hollow site: {li_pos}")
            valid_neighbors , valid_idx_neighbors = find_valid_neighbors(li_pos, hollow_sites, Li_distribution, size)
            print(f"Valid neighbors for Li ion at {li_pos}: {valid_neighbors}")
            if len(valid_neighbors) == 0 :
                print("X"*120) ; print(f"No Availible neighbors ==> This Li ion can't jump") ; print("X"*120)
                continue
            else :
                print(f"IIIIIi iiii iiii iiii = {li_pos}")
                valid_clmb_intr, valid_idx_clmb_intr = find_CLMB_interaction(li_pos, hollow_sites, Li_distribution, size)
                print(f"Valid CLMB interactions for Li ion at {li_pos}: {valid_clmb_intr}")
    
                Transition_Rates, sum_rates = calc_Transition_Rates(idx, hollow_sites, Li_distribution, valid_neighbors, valid_clmb_intr)
                print(f"Transition Rates for Li ion at {li_pos}: {Transition_Rates}")
                print(f"Sum of Transition Rates: {sum_rates}")

                random_number2 = random.random()
                delta_t = -math.log(random.random()) /sum_rates
                print("*"*120)
                print("delta_t:", delta_t)
                print("*"*120)
                clock += delta_t
                time_history.append(clock)

                selected_site , idx_selected_site = jump(Transition_Rates, valid_neighbors, valid_idx_neighbors)
                print(f"Selected jump site: {selected_site} with index {idx_selected_site}")
    
                Li_distribution[idx_selected_site] = hollow_sites[idx_selected_site]
                Li_distribution[idx] = None

                Li_positions.append(hollow_sites[idx_selected_site])

initial_position = np.array(Li_positions[0])
msd_list = [] #; displacements_squared = []
for r in Li_positions:
    delta_r = r - initial_position
    # displacements_squared.append(np.sum(delta_r**2))
    msd_list.append(np.sum(delta_r**2) * 1e-16)

# msd = np.mean(displacements_squared)
# msd_cm2 = msd * 1e-16  # Convert to cm^2

# current_position = hollow_sites[idx_selected_site]
# squared_displacements = np.linalg.norm(np.array(current_position) - np.array(initial_position))**2
# msd = np.mean(squared_displacements)
# msd_cm2 = msd * 1e-16
                
print(f"Simulation time: {float(clock):.5f} seconds")
diffusion_coefficient = msd_list[-1] / (4 * clock)
print(f"Diffusion Coefficient: {float(diffusion_coefficient):.5e} cm^2/s")

plot_msd_vs_time(time_history, msd_list)

plot_graphene_lattice_with_Li(carbon_atoms, np.array(Li_positions))