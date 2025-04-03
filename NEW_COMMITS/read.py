import sys
import numpy as np
k=0

K_sectors=[]
shifts=[]
with open('/mnt/users/dperkovic/quantum_hall_dmrg/tenpy/NEW_COMMITS/L=302_Pf_Apf_truncated_charges_success=4.txt','r') as f:
    lines=f.readlines()
 
    for k in range(len(lines)//3):
        print(3*k)
        print(lines[3*k][7:-2].split(' '))
        vala=int(lines[3*k][7:-2].split(' ')[0])

        valb=int(lines[3*k][8:-2].split(' ')[-1])
        K_sector=float(lines[3*k+2][10:])
        print(K_sector)
        ind=np.where(np.abs(np.array(K_sectors)-K_sector)<0.1)[0]
        if len(ind)==0:
            K_sectors.append(K_sector)
            shifts.append((4,vala,valb))

print(np.sort(K_sectors)*(8*np.pi**2)/(18**2))