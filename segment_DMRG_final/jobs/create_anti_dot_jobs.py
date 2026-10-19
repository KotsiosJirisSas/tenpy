import numpy as np
import os
from itertools import product

#################
######## we create job script such that it avoids new compute modes. that is, if code runs into error
####### we just rerun the code until we hit the old compute node which will do what we want it to do
##################

import math


path="/mnt/users/dperkovic/DMRG_modified/infinite_DMRG_data/nu=2_3_antidot_project/"
#path="/mnt/users/dperkovic/DMRG_modified/infinite_DMRG_data/nu=1_3/"
files=[]

# Get list of file names (excluding directories)
for name in os.listdir(path):
    if 'MPS_data' in name:
        files.append(name[:-4])
###############
###### last parameter just resubmits the shit
##########



path_save='/mnt/users/dperkovic/DMRG_modified/segment_DMRG_results/nu=2_3_antidot_project/'


second = [0]


#first = list(range(280*3, 380*3 + 1, 6))  

#first = list(range( 0*3,1000*3 + 1, 27))  

first = list(range( -800*3,800*3 + 1, 27*3))  

charges = [[q1, q2] for q1, q2 in product(first, second)]

#quit()
#added_sites=60

#halves=[39,45,48,54,57,60]
halves = [15,24,33,42,51,60,69,78,87]
#halves=[63,66,81]

#halves = [93, 102,111,120,129]
#quit()
Ls=[270]
#halves=[120]
#'SHIFTED_BY_1.5_'

#-200,-400

#quit()
Us=[4.0]
chis=[1000]
x0s=[50]
with open('jobs.sh','w') as f:  

    for file_load in files:
        for x0 in x0s:
            for charge in charges:
                if 'chi=3000'  in file_load and 'MPS_data' in file_load and 'd=6' in file_load:
                        print(file_load)
                    
                        for L in Ls:
                            for U0 in Us:
                                for chi in chis:
                                    for half in halves:
                                        
                                        added_sites = 3*math.ceil((math.sqrt(2*abs(charge[0]) + 4)/3)) - 3   # = 2R-3, R = your old value
                                        #added_sites=2
                                        print(added_sites)
                                        file_load_full=path+file_load
                                    
                                        file_save=path_save+file_load+'_charge='+str(charge[0])+'_'+str(charge[1])+"_"+'_L='+str(L)+'_U='+str(U0)+'_added_sites='+str(added_sites)+'_chi_seg='+str(chi)+'_interface_position='+str(half)+'_x0='+str(x0)+'_width=15'
                                        
                                        f.write('addqueue  -c "7 hour" -m 14 /usr/bin/python3 /mnt/users/dperkovic/DMRG_modified/tenpy/segment_DMRG_final/multilayer_segment_final.py '+str(added_sites)+" "+file_load_full+' '+ file_save+' '+str(L)+' '+str(half)+' '+str(chi)+' '+str(U0)+' '+str(x0)+' '+str(charge[0])+" " +str(charge[1])+'\n')
                                        
