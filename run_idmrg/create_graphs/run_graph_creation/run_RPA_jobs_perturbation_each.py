import numpy as np







ds=[20]

Lxs=[15]
V0s=np.linspace(-0.9,-1.5,16)
print(V0s)


V0s=[0.0,-0.2]
Lxs=np.linspace(10,22,13)
print(Lxs)
V0s=np.linspace(0.0,-0.30,25)
Lxs=[14]
with open('jobs.sh','w') as f:
    
    mem=3
    
    for V0 in V0s:
        for Lx in Lxs:
            for d in ds:
                print(Lx)
                #f.write('/usr/bin/python3  /mnt/users/dperkovic/DMRG_modified/tenpy/run_idmrg/create_graphs/create_MPO_graph_2_LL_V0_each_RPA_screened.py '+str(Lx)+' '+str(V0)+" "+str(d)+"\n")
                f.write('addqueue -c "5 min" -m 4 /usr/bin/python3  /mnt/users/dperkovic/DMRG_modified/tenpy/run_idmrg/create_graphs/create_MPO_graph_2_LL_V0_each_RPA_screened.py '+str(Lx)+' '+str(V0)+" "+str(d)+"\n")