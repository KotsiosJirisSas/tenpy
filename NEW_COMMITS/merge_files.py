import numpy
import os
import pickle
import numpy as np
data_new={}

with open("/mnt/users/dperkovic/quantum_hall_dmrg/data_load/pf_apf_final/Data.pkl",'rb')as f:
    obj = pickle.load(f)
print(obj.keys())

with open("/mnt/users/dperkovic/quantum_hall_dmrg/data_load/pf_apf_0101_root/Data_new.pkl",'rb') as f:
    obj2 = pickle.load(f,encoding='latin1')

print(obj2['APf'].keys())

data_new['MPS_Bs']=obj2['Pf']['MPS_Bs']
data_new['MPS_Ss']=obj2['Pf']['MPS_Ss']
data_new['MPS_qflat']=obj2['Pf']['MPS_qflat']
print(data_new['MPS_qflat'])

"""
from collections import Counter
import numpy as np
arr = np.array(data_new['MPS_qflat'])  # Shape: (N, D)
# Convert to a 2D array if it's not already
arr = np.array([np.ravel(x) for x in arr])

# Compare each row to the previous one
diffs = np.any(arr[1:] != arr[:-1], axis=1)

# New elements start at index 0 and where diffs is True (offset by 1)
new_element_indices = np.flatnonzero(np.r_[True, diffs])

print(new_element_indices)
"""
data_new['MPS_2_Bs']=obj2['APf']['MPS_Bs']
data_new['MPS_2_Ss']=obj2['APf']['MPS_Ss']
data_new['MPS_2_qflat']=obj2['APf']['MPS_qflat']


obj2['Pf']['parameters']['root_config']=np.array([0,1,0,1])
data_new['Parameters']=obj2['Pf']['parameters']
#print(obj2['Pf']['parameters']['root_config'])
#quit()
data_new['permutations']=obj2['Pf']['permutations']
data_new['graph']=obj2['Pf']['graph']
data_new['MPO_B']=obj2['Pf']['MPO_B']
data_new['indices']=obj2['Pf']['indices']

with open("/mnt/users/dperkovic/quantum_hall_dmrg/data_load/pf_apf_0101_root/Data.pkl", 'wb') as f:
    pickle.dump(data_new, f)
quit()