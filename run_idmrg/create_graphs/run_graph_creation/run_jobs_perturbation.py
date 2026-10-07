import numpy as np





V0s=np.linspace(0,-0.5,6)
Lxs=[15]
ds=[10]

with open('jobs.sh','w') as f:
    
    mem=3
    
    for V0 in V0s:
        for Lx in Lxs:
            for d in ds:
                print(Lx)
                f.write('python3 /mnt/users/dperkovic/DMRG_modified/tenpy/run_idmrg/create_graphs/create_MPO_graph_2_LL_V0.py '+str(Lx)+' '+str(V0)+" "+str(d)+"\n")