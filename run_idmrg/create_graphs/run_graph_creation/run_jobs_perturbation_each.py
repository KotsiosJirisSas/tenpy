import numpy as np







ds=[20]

Lxs=[14]
V0s=np.linspace(-0.9,-1.5,16)
print(V0s)

V0s=np.linspace(0.0,-1.2,41)
print(V0s)
V0s=[-0.8,-0.83,-0.85,-0.88]
V0s=[-0.91]
with open('jobs.sh','w') as f:
    
    mem=3
    
    for V0 in V0s:
        for Lx in Lxs:
            for d in ds:
                print(Lx)
                f.write('python3 /mnt/users/dperkovic/DMRG_modified/tenpy/run_idmrg/create_graphs/create_MPO_graph_2_LL_V0_each.py '+str(Lx)+' '+str(V0)+" "+str(d)+"\n")