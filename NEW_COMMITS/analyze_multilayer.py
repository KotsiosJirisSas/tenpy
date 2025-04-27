
import h5py
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import curve_fit
import plot_single_density
def linear_model(x, m, c):
    return m * x + c



def load_data(name):
    #PUT THE CORRECT PATH IN
    file_path='/mnt/users/dperkovic/quantum_hall_dmrg/segment_data/'
    
    with h5py.File(file_path+name+'.h5', 'r') as h5_file:
        # List all datasets and groups in the file
        #print("Keys in the file:", list(h5_file.keys()))

        # Access a dataset or group
        data={}
       
        keys=['density','energy']#,'entanglement_spectrum']
        for dataset_name in keys:
         
            if dataset_name=='density':# or dataset_name=='entanglement_spectrum':
                print(dataset_name)
                data[dataset_name]= h5_file[dataset_name][:]
            if dataset_name=='energy':
                data[dataset_name]= h5_file[dataset_name][...]
            if dataset_name=='entanglement_spectrum':
                keys_of_dict = list(h5_file[dataset_name])
           
                dikt={}
                for k in keys_of_dict:
                    #print(kaka)
                    dikt[k]=list(h5_file[dataset_name][k])
                    print(dikt)
                    quit()
                data[dataset_name]=dikt
                
      
       
    return data;



def plot_densities_multilayer():
    #fix path
    path='/mnt/users/dperkovic/quantum_hall_dmrg/segment_data/pf_apf/'
    dir_list = os.listdir(path)
    dir_list.sort()



    dipoles=[]

    energies=[]
    for name in dir_list:
        val=''
        print(name)
        if  "multilayer" in name:
          
            name=name[:-3] 
            
        
            data=load_data(name)
            
            
          
        
            print("*"*30)
            print(name)
            
            density=data['density']
           
              
             
            
            dist=np.arange(len(density))//2
            ind=np.where(dist<250)
        
            #calculate the dipole
            Ly=18
            dipole=np.sum((density[ind]-1/3)*dist[ind])*8*np.pi**2/Ly**2
   
            print("*"*30)
            print(name)
    
            print('NUMBER of excess number of electrons:')
            print(np.sum(density-1/3))
            
            
            energies.append(data['energy'])
            dipoles.append(dipole)

            #plot all the densities
            plot_single_density.plot_density(density,name,N_layer=2)
                

    plt.figure()
   
    name='energ_dipole'
    #save_graph_data(name,dipoles,energies,list(mus))
    
    #print(params)
    dipoles=np.array(dipoles)
   
    a=np.argsort(dipoles)
 
    ps=np.array(dipoles)[a]
    Es=np.array(energies)[a]
    print(Es)

    
    plt.figure()
    plt.title('Energy against dipole for multilayer interface')
    plt.plot(ps, Es-np.min(Es),c='#1f77b4', marker='o',markerfacecolor='none', linestyle='-', label="data")
   
    plt.ylabel(r"$E$"+' \energy of a state')
    plt.xlabel(r"$\frac{p}{L_y}/\frac{e}{4\pi}$")
    
    #Plot Haldanes value as a vertical line
    xs=np.ones(10)*(-1)
    ys=np.linspace(0,np.max(Es)+0.003-np.min(Es),10)
    plt.plot(xs,ys,ls='--',color='orange',label="Haldane's value")
    plt.legend()
    plt.savefig('/mnt/users/dperkovic/quantum_hall_dmrg/tenpy/NEW_COMMITS/figures/'+name+'.png')
   
plot_densities_multilayer()