import numpy as np
import matplotlib.pyplot as plt
import random
import math
from mpmath import mp
import bisect

size = 16 ; a = 1.42 ; L = np.sqrt(3) * a / 2 # L = round(np.sqrt(3) * a / 2 ,3)
reservoir = 80 # 80 Li ions to be inserted
E_hop_eV = 0.23 ; E_hop_J = E_hop_eV * 1.602176634e-19  # Convert eV to J
E_ads_eV =  - 1 ; E_ads_J = E_ads_eV * 1.602176634e-19  # Convert eV to J
E_des_eV = 1 ; E_des_J = E_des_eV * 1.602176634e-19  # Convert eV to J
T = 294 # Temperature in Kelvin
alpha = 0.5 # adsorption probability OR charge transfer coefficient 
U_V = 0.0 # Voltage in Volts
U_0_V = 0.0 # Reference voltage in Volts
kB_eV = 8.617333262145e-5  # eV/K
KB_J = 1.380649e-23 # J/K
kBT_cst_J = KB_J * T  # Boltzmann constant times temperature in J
kBT_cst_eV = kB_eV * T  # Boltzmann constant times temperature in eV
h_eV = 4.1357e-15  # eV.s
f = kBT_cst_eV / h_eV # Attempt frequency in Hz or s-1
e_charge_eV = 1 # Elementary charge in eV
e_charge_C = 1.602176634e-19  # Elementary charge in C

epsilon_0 = 8.854187818e-12       # permittivité du vide en F/m
Cst_coulomb = 1 / (4 * np.pi * epsilon_0)  # constante de Coulomb en N.m²/C²

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
    
        #hollows_idx = {tuple(pos): i for i, pos in enumerate(hollow_sites)}
    hollow_sites_3 = np.round(hollow_sites, 3)
    return np.array(carbon_atoms), np.array(hollow_sites), np.array(hollow_sites_3)#, hollows_idx

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

def find_valid_neighbors(s,i, hollow_sites, Li_distribution):
    # s = size
        # Given the index i of a site, find the indices of its 6 neighboring 
        # sites, and then check which of those neighbors are empty.
    condition1 = set(range(s +8, s * s - s , s))
    condition2 = [int(s + s/2 - 1 + i * 16) for i in range(s-2)]
    if i < s/2 -1 :
        h1 =  i+(s*s)-(s/2) ; h2 =  i+(s*s)-(s/2)+1
        h3 = i + (s/2)      ; h4 =  i + 1 + (s/2)
        h5 =  i+(s*s)-s     ; h6 = i + s
    elif i == s/2 - 1 :
        h1 = (s*s) - 1         ; h2 = i+(s*s)-s+1 
        h3 = i+(s/2)        ; h4 = i+1
        h5 = i+(s*s)-s      ; h6 = i + s
    elif i == s/2 :
        h1 = i-1            ; h2 = i-(s/2) 
        h3 = i-1+s          ; h4 = i+(s/2) 
        h5 = i+(s*s)-s      ; h6 = i+s 
    elif i < s : 
        h1 = i-1-(s/2)      ; h2 =  i-(s/2)
        h3 = i-1+(s/2)      ; h4 =  i+(s/2)
        h5 = i+(s*s)-s      ; h6 = i + s 
    elif i in condition1:
        h1 = i-1            ; h2 = i - (s/2) 
        h3 = i-1+s          ; h4 = i + (s/2)
        h5 =  i - s         ; h6 = i + s
    elif i == (s*s) - (s/2) :
        h1 = i-1            ; h2 = i - (s/2) 
        h3 = i-(s*s)+s-1    ; h4 = i-(s*s)+(s/2)
        h5 = i-s            ; h6 = i-(s*s)+s
    elif i in condition2:
        h1 = i - (s/2)      ; h2 = i + 1 - s
        h3 = i + (s/2)      ; h4 = i + 1
        h5 = i - s          ; h6 = i + s
    elif i ==  s*s - 1 - s/2 :
        h1 = i-(s/2)        ; h2 = i+1-s
        h3 = i+(s/2)        ; h4 = i+1 
        h5 = i - s          ; h6 = i-(s*s) + s
    elif (s*s)-s <= i <=(s*s)-2-(s/2) :
        h1 = i - (s/2)      ; h2 = i + 1 - (s/2)
        h3 = i + (s/2)      ; h4 = i+ 1 + (s/2)
        h5 = i - s          ; h6 =  i+s-s*s
    elif s*s - s/2 +1 <= i <= s*s - 1 :
        h1 = i-1-(s/2)      ; h2 = i-s/2
        h3 = i-1-(s*s)+(s/2)  ; h4 = i-(s*s)+(s/2)  
        h5 = i-s            ; h6 = i-(s*s)+s
    else:
        h1 = i - s/2         ; h2 = i + 1 - s/2
        h3 = i + s/2         ; h4 = i + 1 + s/2
        h5 = i - s           ; h6 = i + s
    
    neighbors = [int(h1), int(h2), int(h3), int(h4), int(h5), int(h6)]
    valid_neighbors = []
    valid_neighbors_xy = []
    for h in neighbors:
        if Li_distribution[h] is None:
            valid_neighbors.append(h)
            valid_neighbors_xy.append(hollow_sites[h])
            
    return valid_neighbors, valid_neighbors_xy

def PBC(site):
    #this function for Periodic Boundary Conditions
    a = 1.42 ; x,y = site
    L = np.sqrt(3) * a / 2 #L = round(np.sqrt(3) * a / 2 ,3) ; x,y = site #L = np.sqrt(3) * a / 2 ; x,y = site 
    x_min = a ; x_max = 25 * a
    y_min = L ; y_max = 32 * L
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

def find_CLMB_interaction(pos, hollow_sites, Li_distribution, size) : 
    valid_clmb_intr = [] # List to store valid CLMB interaction positions (x,y)
    valid_clmb_intr_idx = [] # List to store indices "n_i" of valid CLMB interactions

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
    clmb_intr = np.round(clmb_intr, 3)  # Round the positions to 3 decimal places
    clmb_intr = [PBC(site) for site in clmb_intr]

    for clmb_site in clmb_intr:
        for j, hollow_site in enumerate(hollow_sites):
            if np.allclose(clmb_site, hollow_site, atol=1e-1): 
                if Li_distribution[j] is not None:  # Check if the site is occupied by a Li ion
                    valid_clmb_intr.append(clmb_site)
                    valid_clmb_intr_idx.append(j)  # Store the index of the valid CLMB

    return valid_clmb_intr, valid_clmb_intr_idx

def Hop_Rates(site_idx, hollow_sites, valid_neighbors_idx, clmb_idx):
    i_hop_rates = [] # List to store hopping rates  
    for destination in valid_neighbors_idx:
        point_c = (np.array(hollow_sites[destination]) + np.array(hollow_sites[site_idx])) / 2
        r_diff = 0.0
        if len(clmb_idx) > 0:
            for ion in clmb_idx:
                raa = np.linalg.norm(np.array(hollow_sites[site_idx]) - np.array(hollow_sites[ion])) ; ra = raa * 1e-10  # Convert to meters
                rcc = np.linalg.norm(np.array(hollow_sites[ion]) - point_c) ; rc = rcc * 1e-10  # Convert to meters
                r_diff += (1/rc) - (1/ra)
        E_m = E_hop_J + r_diff * Cst_coulomb * e_charge_C * e_charge_C  # in Joules
        R_hop_dest_j = f * np.exp(- E_m / kBT_cst_J)  # Hopping rate in s-1
        i_hop_rates.append(R_hop_dest_j)  # Append the hopping rate for this destination
    return i_hop_rates

def Adsorption_Rate(U_V, U_0_V) :
    E_m_eV = E_ads_eV - alpha * e_charge_C * (U_V - U_0_V) 
    i_Adsortion_rate = f * np.exp( - E_m_eV / kB_eV ) 
    return i_Adsortion_rate  # Return the adsorption rate for this site

def Desorption_Rate(U_V, U_0_V) :
    E_m_eV = E_des_eV + (1 - alpha) * e_charge_C * (U_V - U_0_V) 
    i_Desorption_rate = f * np.exp( - E_m_eV / kB_eV ) 
    return i_Desorption_rate  # Return the desorption rate for this site

def Hop_Analysis(Li_distribution, hollow_sites) :
    Hop_Rates_list = [] # List to store hopping rates
    Hop_Event_Catalog = [] # List to store hopping events
    for i, site in enumerate(Li_distribution):
        if site is not None: # if Li ion is present
            # as soon as I found an ion I'll search for its valid neighbors : 
            valid_neighbors_idx, valid_neighbors_xy = find_valid_neighbors(size, i, hollow_sites, Li_distribution)
            # valid_neighbors contient les indices des voisins valides
            # valid_neighbors_xy contient les coordonées des voisins valides
            if len(valid_neighbors_idx) == 0 :
                continue  # if no valid neighbors, skip loop iteration and go the to next site
            """
            as soon as I found an ion I'll search for its valid neighbors : 
                each valid neigbor is a potential destination for hopping
            and I will calculate the hopping rate for each valid neighbor   
                to do so, I will first find the CLMB interaction for this site
            and then I will calculate the hopping rate for each valid neighbor/destination
            then append the hopping rates to Hop_Rates_list
            """
            clmb_pos_xy, clmb_idx = find_CLMB_interaction(hollow_sites_3[i], hollow_sites, Li_distribution, size)
            i_hop_rates = Hop_Rates(i, hollow_sites, valid_neighbors_idx, clmb_idx)
            Hop_Rates_list.append(i_hop_rates)  # Append the hopping rates for this site
            Hop_Event_Catalog_i = [ ['hop', i, distination] for distination in valid_neighbors_idx]
            Hop_Event_Catalog.append(Hop_Event_Catalog_i) # Append the hopping events for this site
    
    return Hop_Rates_list, Hop_Event_Catalog

def Adsorption_and_Desorption_analysis(Li_distribution, reservoir):
    Adsorption_Rates_list = []  # List to store adsorption rates
    Adsorption_Event_Catalog = []  # List to store adsorption events

    Desorption_Rates_list = []  # List to store desorption rates
    Desorption_Event_Catalog = []  # List to store desorption events
    
    adsorption_sites_idx = [8+ 14 * i for i in range(size)]  # Indices of adsorption sites
    
    for left_site_idx in adsorption_sites_idx :
        if Li_distribution[left_site_idx] is None: #if the adsorption site is empty then I can insert a Li ion
            if reservoir > 0:
                i_Asorption_event_catalog = ['adsorption', left_site_idx]
                Adsorption_Event_Catalog.append(i_Asorption_event_catalog)
                i_Adsorption_rate = Adsorption_Rate(U_V, U_0_V)
                Adsorption_Rates_list.append(i_Adsorption_rate)
        else:
            i_Desoption_event_catalog = ['desorption', left_site_idx]
            Desorption_Event_Catalog.append(i_Desoption_event_catalog)
            i_Desorption_rate = Desorption_Rate(U_V, U_0_V)
            Desorption_Rates_list.append(i_Desorption_rate)               
    
    return Adsorption_Rates_list, Adsorption_Event_Catalog, Desorption_Rates_list, Desorption_Event_Catalog

def Events_Analysis(Li_distribution, hollow_sites, Hop_Rates_list, Hop_Event_Catalog, Adsorption_Rates_list, Adsorption_Event_Catalog, Desorption_Rates_list, Desorption_Event_Catalog):
    Rates_list = []  # List to store all rates
    Events_Catalog = []  # List to store all events
    Cumulative_rate_list = []

    #Rates_list = Hop_Rates_list + Adsorption_Rates_list + Desorption_Rates_list
    flat_Hop_Rates_list = [rate for sublist in Hop_Rates_list for rate in sublist]
    print(f"flat_Hop_Rates_list: {flat_Hop_Rates_list}\n")
    Rates_list = flat_Hop_Rates_list + Adsorption_Rates_list + Desorption_Rates_list

    Events_Catalog = Hop_Event_Catalog + Adsorption_Event_Catalog + Desorption_Event_Catalog
    # if len(Hop_Rates_list) > 0:
    #     Events_Catalog.append(Hop_Event_Catalog)
    # if len(Adsorption_Rates_list) > 0:
    #     Events_Catalog.append(Adsorption_Event_Catalog) 
    # if len(Desorption_Rates_list) > 0:
    #     Events_Catalog.append(Desorption_Event_Catalog)
    #Events_Catalog.append([Hop_Event_Catalog, Adsorption_Event_Catalog, Desorption_Event_Catalog])
    #Events_Catalog.extend([Hop_Event_Catalog, Adsorption_Event_Catalog, Desorption_Event_Catalog])    
    # Events_Catalog = [ event for sublist in Hop_Event_Catalog for event in sublist] + \
    #                  [event for sublist in Adsorption_Event_Catalog for event in sublist] + \
    #                  [event for sublist in Desorption_Event_Catalog for event in sublist]

    print(f"Rates_list: {Rates_list}\n")
    print(f"Events_Catalog: {Events_Catalog}\n")
    # Rates_list = Hop_Rates_list + Adsorption_Rates_list + Desorption_Rates_list
    # Events_Catalog = Hop_Event_Catalog + Adsorption_Event_Catalog + Desorption_Event_Catalog

    Cumulative_rate_list = np.cumsum(Rates_list)  # cumulative sum of all rates
    R_total = Cumulative_rate_list[-1]  # Total rate is the last element of the cumulative rate list

    delta_t = - math.log(random.random()) / R_total # time step

    random_2 = random.uniform(0, 1) * R_total # random.uniform(0, 1) gives a random float between 0.0 and 1.0

    """
    This is a highly efficient binary search algorithm to find the first 
    element in cumulative_rate_list that is greater than u 

    The index of this element is our winning index
    => the function bisect_right does exactly this
    """ 
    winning_index = next(i for i, p in enumerate(Cumulative_rate_list) if random_2 <= p)
    #winning_index = bisect.bisect_right(Cumulative_rate_list, random_2)
    print(f"Winning index: {winning_index}\n")
    winning_event = Events_Catalog[winning_index]
    print("*" * 50)
    print(f"Winning event: {winning_event}\n")
    print("*" * 50)
    """
    winning_event is a table : one of the following tables
        winning_event = [ 'hop' , current_site_index, destination _site_index]
        winning_event = [ 'adsorption' , adsorption_site_index]
        winning_event = [ 'desorption' , desorption_site_index]
    """
    
    return winning_index, winning_event, delta_t

"""
This a modelisation of how the analysis will work :


Rates = ['R_hop_1', 'R_hop_2', 'R_hop_3', 'R_hop_4', 'R_hop_5', 'R_ads_1', 'R_ads_2', 'R_des_3', 'R_des_1', 'R_des_2']
Events_catalog=[]

Ratess = [1,2,3,4,5,6,7]  # Example rates for hopping and adsorption/desorption events

hop_events_vatalog=[['hop', 0, 1], ['hop', 0, 2], ['hop', 0, 3], ['hop', 0, 4], ['hop', 0, 5]]
adsorption_events_vatalog=[['adsorption', 1], ['adsorption', 2], ['adsorption', 3]]
desorption_events_vatalog=[['desorption', 1], ['desorption', 2]]
print("length of T1:", len(hop_events_vatalog))
print("length of T2:", len(hop_events_vatalog))
print("length of T3:", len(hop_events_vatalog))
#T.extend([T1, T2, T3])  # Extend T with T1, T2, and T3
Events_catalog = hop_events_vatalog + adsorption_events_vatalog + desorption_events_vatalog  # Concatenate T1, T2, and T3 into T
print("Rates:", Rates)
print("Events_catalog:", Events_catalog)
print("Events_catalog[0]:", Events_catalog[0])
print("Events_catalog[4]:", Events_catalog[4])
print("Events_catalog[6]:", Events_catalog[6])
print("Events_catalog[6][0]:", Events_catalog[6][0]) 

Cumulative_rate_list = np.cumsum(Ratess)  # Cumulative sum of rates
R_total = Cumulative_rate_list[-1]  # Total rate is the last element of
print("Cumulative_rate_list:", Cumulative_rate_list)
print("R_total:", R_total)  # Total rate is the last element of the cumulative
"""

carbon_atoms, hollow_sites, hollow_sites_3 = graphene_lattice(size)
Li_distribution = [None] * len(hollow_sites)
simulation_time = 0.0
###################################################################
####
#### MAIN KMC ANALYSIS
####
#################################################################### 
steps = 20 # Number of steps to run the simulation
for step in range(steps):
    print(f"Step {step + 1}/{steps} - Simulation time: {simulation_time:.2f} seconds\n")
    Hop_Rates_list, Hop_Event_Catalog = Hop_Analysis(Li_distribution, hollow_sites)
    print(f"Hop_Rates_list: {Hop_Rates_list}\n")
    print(f"Hop_Event_Catalog: {Hop_Event_Catalog}\n")
    Adsorption_Rates_list, Adsorption_Event_Catalog, Desorption_Rates_list, Desorption_Event_Catalog = Adsorption_and_Desorption_analysis(Li_distribution, reservoir)
    print(f"Adsorption_Rates_list: {Adsorption_Rates_list}\n")
    print(f"Adsorption_Event_Catalog: {Adsorption_Event_Catalog}\n")
    print(f"Desorption_Rates_list: {Desorption_Rates_list}\n")
    print(f"Desorption_Event_Catalog: {Desorption_Event_Catalog}\n")
    winning_index, winning_event, delta_t = Events_Analysis(Li_distribution, hollow_sites, Hop_Rates_list, Hop_Event_Catalog, Adsorption_Rates_list, Adsorption_Event_Catalog, Desorption_Rates_list, Desorption_Event_Catalog)
    
    simulation_time += delta_t  # Update the simulation time with the time step

    event_type = winning_event[0]
    print(f"Winning event: {winning_event}, Event type: {event_type}, Delta t: {delta_t:.4f} seconds\n")
    """
    winning_event is a table : one of the following tables
        winning_event = [ 'hop' , current_site_index, destination _site_index]
        winning_event = [ 'adsorption' , adsorption_site_index]
        winning_event = [ 'desorption' , desorption_site_index]
    """
    if event_type == 'hop':
        Li_distribution[ winning_event[2] ] = hollow_sites[winning_event[2]]  # Move Li ion to the destination site
        Li_distribution[ winning_event[1] ] = None  # Empty the current site
        #######
        # Here I may add some updates
        #######

    elif event_type == 'adsorption':
        Li_distribution[ winning_event[1] ] = hollow_sites[winning_event[1]]
        reservoir -= 1
        if reservoir == 0:
            print("Reservoir is empty, no more Li ions to insert.\n")
        #######
        # Here I may add some updates
        #######

    else :
    # elif event_type == 'desorption':
        Li_distribution[ winning_event[1] ] = None
        reservoir += 1
        #######
        # Here I may add some updates
        #######

print("Simulation completed.\n")
print("simulation_time:", simulation_time, "seconds\n")
plot_graphene_lattice_with_Li(carbon_atoms, Li_positions=np.array([pos for pos in Li_distribution if pos is not None]))