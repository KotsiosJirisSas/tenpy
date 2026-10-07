import numpy as np
import os
#################
######## we create job script such that it avoids new compute modes. that is, if code runs into error
####### we just rerun the code until we hit the old compute node which will do what we want it to do
##################


mus=np.linspace(0,-1.0,11)
num_sites=[6]



path="/mnt/users/dperkovic/DMRG_modified/infinite_DMRG_data/V0_coulumb_correct/"
#path="/mnt/users/dperkovic/DMRG_modified/infinite_DMRG_data/nu=1_3/"
files=[]

# Get list of file names (excluding directories)
for name in os.listdir(path):
    if 'MPS_data' in name:
        files.append(name[:-4])
###############
###### last parameter just resubmits the shit
##########



path_save='/mnt/users/dperkovic/DMRG_modified/tenpy/segment_DMRG/data/RPA_charge_gap/'
charges=[[0,0,0],[0,2,0],[0,-2,0],[0,0,2],[0,0,-2]]
added_sites=8

Ls=[150,151,152,153,154,155,156,157,158,159,160]

halves=[152]
Ls=[152]

charges=[[0,0,2],[0,0,-2]]
halves=[97,95]
Ls=[96]
with open('jobs.sh','w') as f:  

    for file_load in files:
       
        for charge in charges:
            if 'chi=1000'  in file_load and 'MPS_data' in file_load and 'a=10.0_b=0.0' in file_load:
                print(file_load)
                for L in Ls:

                    for half in halves:
                        file_load_full=path+file_load
                    
                        file_save=path_save+file_load+'_pstate='+str(charge[0])+'_'+str(charge[1])+"_"+str(charge[2])+'_L='+str(L)
                        
                        f.write('addqueue  -q short -c "5 hmin" -m 40 /usr/bin/python3 /mnt/users/dperkovic/DMRG_modified/tenpy/segment_DMRG/multilayer_segment_final.py '+str(added_sites)+" "+file_load_full+' '+ file_save+' '+str(L)+' '+str(half)+' '+str(charge[0])+" " +str(charge[1])+" " +str(charge[2])+'\n')
                        
