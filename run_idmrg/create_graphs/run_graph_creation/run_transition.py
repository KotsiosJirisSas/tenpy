import numpy as np





path='/mnt/users/dperkovic/DMRG_modified/data_load/weak_strong_transition_trial_states/'
#V0s=[0.0]
Lxs=[18]
V0s=np.linspace(-0.3,0.3,17)


V0s=np.linspace(0.3,1,8)

#V0s=np.linspace(-1,-0.3,8)
print(V0s)
with open('jobs.sh','w') as f:
    
    mem=3

    for Lx in Lxs:
        for V0 in V0s:

            print(Lx)

            f.write('python3 /mnt/users/dperkovic/DMRG_modified/tenpy/run_idmrg/create_graphs/create_MPO_graph_trial_transition.py '+str(Lx)+" 1 "+str(V0)+" 1\n")
            