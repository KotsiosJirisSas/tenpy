

import os
import re

folder = "/mnt/users/dperkovic/DMRG_modified/data_load/Landau_level_2_bands_V0_perturbation/"

params = []

for filename in os.listdir(folder):
    match = re.search(
        r"V0=([-+]?\d*\.?\d+)_d=([-+]?\d*\.?\d+)_Lx=([-+]?\d*\.?\d+)",
        filename
    )

    if match:
        V0 = float(match.group(1))
        xi = float(match.group(2))
        Lx = float(match.group(3))

        params.append((V0, Lx, xi))



print(params)


with open('jobs.sh','w') as f:
    
    mem=3
    
    for V0,Lx,d in params:
        

        f.write('python3 /mnt/users/dperkovic/DMRG_modified/tenpy/run_idmrg/create_graphs/create_MPO_graph_2_LL_V0.py '+str(Lx)+' '+str(V0)+" "+str(d)+"\n")