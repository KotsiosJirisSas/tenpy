import h5py
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import curve_fit

def linear_model(x, m, c):
    return m * x + c

def load_data(name):
    file_path='/mnt/users/dperkovic/quantum_hall_dmrg/segment_data/'
    # Open the HDF5 file in read mode
    with h5py.File(file_path+name+'.h5', 'r') as h5_file:
        # List all datasets and groups in the file
        print("Keys in the file:", list(h5_file.keys()))

        # Access a dataset or group
        data={}
        keys=['density','energy']
        for dataset_name in keys:
            #print(dataset_name)
            #print(h5_file[dataset_name])
            if dataset_name=='density':
                data[dataset_name]= h5_file[dataset_name][:]
            else:
                data[dataset_name]= h5_file[dataset_name][...]
                #print(data[dataset_name])
            #print("Data shape:", data.shape)
            #print("Data type:", data.dtype)
            #print("Data values:", data)
       
    return data;
def plot_densities():
    path = "/mnt/users/dperkovic/quantum_hall_dmrg/segment_data"


    dir_list = os.listdir(path)
    dir_list.sort()



    dipoles=[]
    mus=[]
    for name in dir_list:
        if 'linear_potential_added_' in name and "['empty'" not in name:#and 'Lx_16_QH_nu_1_3' in name:
            name=name[:-3] 
            data=load_data(name)
            #ind1=name.index('mu_')+3
            #ind2=name.index('_data')
            print(name)
            #mu=float(name[ind1:ind2])
            #mu=0.1
            mu=0.1
            if mu<0.4:
                density=data['density']

                E=data['energy']
                print(E)
                dipole=np.sum((density-1/3)*np.arange(len(density)))*8*np.pi**2/18**2
                #dipoles.append(dipole)
                if E<0:
                    dipoles.append(E)
                    #print(name[ind1:ind2])
                
                    mus.append(mu)
                plt.figure()
                plt.scatter(np.arange(len(data['density'])),data['density'])
                plt.savefig('density_'+name+'.png')
    plt.figure()
    print(len(dipoles))
    mus=-(len(dipoles)-1-np.arange(len(dipoles)))*2*np.pi/18
    #mus=(4-np.arange(5))*2*np.pi/18
    #mus=np.array([3,0])*2*np.pi/16
    #mus=(5-np.arange(6))*2*np.pi/16
    #dipoles.append(0.35)
    plt.scatter(mus,dipoles)
    
    params, params_covariance = curve_fit(linear_model, mus, dipoles)
    print(params)
   
    xvals=np.linspace(0, np.max(mus),10)
    ys=linear_model(xvals,params[0],params[1])
    print(4*np.pi/18)
    print(4*np.pi/16)
    print(params)
    #plt.plot(xvals,ys,ls='--',color='red',label='y='+str(params[0])[:5]+'*x+'+str(params[1])[:5])
    plt.title('Energy against momentum sector')
    #plt.title('Dipole against size of transfered momentum from GS for Coulumb,\nexpected slope=0.785, expected intercept=1/3')
    plt.xlabel(r"$\Delta k_y$")
    plt.ylabel(r"E")
    #plt.ylabel(r"$\frac{p}{L_y} / \frac{e}{4\pi} $")
    plt.legend()
    plt.savefig('dipoles.png')
plot_densities()
quit()

name='segment_Lx_16_QH_nu_1_3_Gs_v_-0.1'
data=load_data(name)
print(data)
plt.sca

#h5_to_dict_with_attrs(data)
quit()
print(data.keys())
#print(data.keys())