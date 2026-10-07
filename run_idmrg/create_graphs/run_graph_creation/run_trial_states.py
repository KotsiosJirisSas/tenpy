import numpy as np






#V0s=[0.0]
Lxs=[15]
Lxs=np.linspace(10,24,15)

with open('jobs.sh','w') as f:
    
    mem=3

    for Lx in Lxs:

        print(Lx)
        #f.write('python3 /mnt/users/dperkovic/DMRG_modified/tenpy/run_idmrg/create_graphs/create_MPO_graph_pseudopotentials.py '+str(Lx)+" 1 1 1 \n")
        #f.write('python3 /mnt/users/dperkovic/DMRG_modified/tenpy/run_idmrg/create_graphs/create_MPO_graph_pseudopotentials.py '+str(Lx)+" 1 1 0\n")
        f.write('python3 /mnt/users/dperkovic/DMRG_modified/tenpy/run_idmrg/create_graphs/create_MPO_graph_pseudopotentials.py '+str(Lx)+" 1 -0.5 1\n")
        