import numpy as np





mus=np.linspace(-2,0,11)
#mus=np.append(mus,[3,-3])
mus=np.append(mus,[-3])
Lxs=[15]
with open('jobs.sh','w') as f:
    
    mem=3
    
    for mu in mus:
        for Lx in Lxs:
            print(Lx)
            #f.write('addqueue -q short -c "10 min" -m '+str(mem)+' /usr/bin/python3 /mnt/users/dperkovic/DMRG_modified/tenpy/run_idmrg/create_MPO_graph_screened.py '+str(Lx)+" "+str(round(V0,2))+" "+str(a)+" "+str(b)+" "+str(xi)+"\n")
            f.write('python3 /mnt/users/dperkovic/DMRG_modified/tenpy/run_idmrg/create_graphs/create_MPO_graph_landau_level_mixing.py '+str(Lx)+' '+str(mu)+"\n")
                