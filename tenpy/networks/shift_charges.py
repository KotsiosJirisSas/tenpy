import sys
#sys.path.append('/mnt/users/dperkovic/DMRG_modified/tenpy/tenpy/linalg') 
from tenpy.linalg.charges import  LegCharge
import numpy as np
#SHIFT CHARGES
def shift_charges(qflat2,shift):
    qflat=[]

    for i in range(len(qflat2)):

        kopy=[]
        for m in range(len(qflat2[i])):
            if m==0:
                shifted=qflat2[i][0]+shift*np.sum(qflat2[i][1:])
                kopy.append(shifted)
                #HOW DO YOU SHIFT QFLAT2 ACTUALLY,
                #talk to konstantinos in the morning
            else:
                kopy.append(qflat2[i][m])#+charge_shift[m])
        qflat.append(kopy)
    return qflat

def shift_charges_individually(qflat2,index,shift):
    qflat=[]

    for i in range(len(qflat2)):

        kopy=[]
        for m in range(len(qflat2[i])):
            if m==0:
                shifted=qflat2[i][0]+shift*np.sum(qflat2[i][index])
                kopy.append(shifted)
                #HOW DO YOU SHIFT QFLAT2 ACTUALLY,
                #talk to konstantinos in the morning
            else:
                kopy.append(qflat2[i][m])#+charge_shift[m])
        qflat.append(kopy)
    return qflat


def shift_charges_leg(leg,L):
    #returns the leg with K sector charges shifted!
    qflat=leg.to_qflat()

    qflat=shift_charges(qflat,L)
    chargeinfo=leg.chinfo
    
    leg=LegCharge.from_qflat(chargeinfo,qflat,qconj=leg.qconj)
    return leg

def shift_charges_leg_individually(leg,L,index):
    #returns the leg with K sector charges shifted!
    qflat=leg.to_qflat()

    qflat=shift_charges_individually(qflat,index,L)
    
    chargeinfo=leg.chinfo
  
    leg=LegCharge.from_qflat(chargeinfo,qflat,qconj=leg.qconj)
    return leg

def translate_by_appropriate_amount(legs_moved,leg_fixed,bunch):

    #bunch or not bunch
    #translates the charges in the appropriate manner
    #legs_moved get their charges changes to be the same as those at legs_fixes
    legs_new=[]
    

    ###########
    #assumption here is that we move by a full unit cell
    # that all charges that are not K, are N1,N2 etc
    #that there is equal number of N1,N2 etc sites in the unit cell
    ##########
    ind=0
   
   
    qflat=leg_fixed.to_qflat()
    for i in range(len(qflat)):
        net_charge=np.sum(qflat[i][1:])
        if net_charge!=0:
            ind=i
            break;
  
    qflat=leg_fixed.to_qflat()[ind]
    net_charge=np.sum(qflat[1:])
  

    #print(net_charge)
    for i,leg in enumerate(legs_moved):
        #print(leg)
        
        try:
            qflat2=leg.to_qflat()[ind]
        except:
            qflat2=[0,0]
        if i==0:
            if net_charge!=0:
                
                shift=(qflat[0]-qflat2[0])//net_charge
                #if shift!=0: 
                #    print(shift)
                """
                if shift!=0 and shift !=3 and shift!=-3:
                    print('$'*30)
                    print(shift)
                """
            else:
                
                shift=0
        #print('shiftara'*30)
        #print(shift)
      
        lega=shift_charges_leg(leg,shift)
        
        if bunch[i]:
           
            _,lega=lega.sort(bunch=True)
        
        legs_new.append(lega)
    return legs_new


def translate_all_legs(legs_moved,num_sites,bunch):

    #bunch or not bunch
    #translates the charges in the appropriate manner
    #legs_moved get their charges changes to be the same as those at legs_fixes
    legs_new=[]
    

    ###########
    #assumption here is that we move by a full unit cell
    # that all charges that are not K, are N1,N2 etc
    #that there is equal number of N1,N2 etc sites in the unit cell
    ##########
    ind=0
   
   

    #print(net_charge)
    for i,leg in enumerate(legs_moved):
     
   
        
        lega=shift_charges_leg(leg,num_sites)
        
        if bunch[i]:
           
            _,lega=lega.sort(bunch=True)
        legs_new.append(lega)
    return legs_new
"""
This commented out function within DMRG can shift All of the required variables to be withing the unit cell
################
#####instead of this can potentially just shift everything backwards that needs shifting backwards
#### physical charges
#################

############
### ENSURE THAT BEFORE EVERY RUN OUR SITES ARE DEFINED WITHIN THE SAME CONVENTION
### THUS IS WE ARE CONSTRUCTING EVIRONMENT FOR SITE I, THE MPS MUST MATCH IT, and
#so does the MPO
#print('TRANSLATION IS ON ON ON ON ON ON ON ON'*30)
C1=self.env.sites[0].leg.to_qflat()[0][1]
for i in range(L):


Ki=self.env.sites[i].leg.to_qflat()[0][0]

K_psi=self.psi._B[i].legs[1].to_qflat()[0][0]
if Ki!=K_psi:
    bunch=[leg.bunched for leg in self.psi._B[i].legs]
    self.psi._B[i].legs=translate_all_legs(self.psi._B[i].legs,(Ki-K_psi)//C1,bunch)
K_W=self.env.H._W[i].legs[-1].to_qflat()[0][0]


if K_W!=Ki:
    bunch=[leg.bunched for leg in self.env.H._W[i].legs]
    self.env.H._W[i].legs=translate_all_legs(self.env.H._W[i].legs,(Ki-K_W)//C1,bunch)

#############
#### NEED TO CHECK WHETHER ENVIRONMENT NEEDS T0 BE TRANSLATED
##########
key = self.env._RP_keys[self.env._to_valid_index(i)]
RP = self.env.cache.get(key, None)

if RP:
    site2=RP.legs[1].to_qflat()
    slite= self.env.H._W[i].legs[1].to_qflat()
    ##########
    ###check what needs shifting and shift it.
    #match environment to MPO
    ######
    non_zero_ind=np.flatnonzero(slite[:,1])[0]
    
    if not np.all(site2==slite):
        bunch=[leg.bunched for leg in self.env.cache[key].legs]

        
        shift=(slite[non_zero_ind,0]-site2[non_zero_ind,0])/site2[non_zero_ind,1]
        self.env.cache[key].legs=translate_all_legs(self.env.cache[key].legs,shift,bunch)
        

if RP:
    self.env.H._W[i].legs[1].test_contractible(RP.legs[1])
    
key = self.env._LP_keys[self.env._to_valid_index(i)]
LP = self.env.cache.get(key, None)
#if i==0:
#    print(LP)
if LP:
    site2=LP.legs[1].to_qflat()
    slite=self.env.H._W[i].legs[0].to_qflat()
    ##########
    ###check what needs shifting and shift it
    ###
    ######
    non_zero_ind=np.flatnonzero(slite[:,1])[0]
    if not np.all(site2==slite):
        bunch=[leg.bunched for leg in self.env.cache[key].legs]
        shift=(slite[non_zero_ind,0]-site2[non_zero_ind,0])/site2[non_zero_ind,1]
        self.env.cache[key].legs=translate_all_legs(self.env.cache[key].legs,shift,bunch)
    
if LP:
    self.env.H._W[i].legs[0].test_contractible(LP.legs[1])
#print('check consistency !!!')
#print(RP)

#print(self.psi._B[i])
#print(self.env.H._W[i])
#####
# at the end of the sweep we are left with three environments
#TWO OF THOSE WILL indeed match everything that has been done 
#because they are constructed for the new psi
#ONE WILL NOT MATCH SINCE IT IT FROM OLD ENVIRONMENT 
#THIS NON-MATCHING ENVIRONMENT THEN GETS CONTRACTED AND becomes new env?
"""