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
        keys=['density']
        for dataset_name in keys:
            print(dataset_name)
            print(h5_file[dataset_name])
            data[dataset_name]= h5_file[dataset_name][:]
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
        if 'finite' in name:
            name=name[:-3] 
            data=load_data(name)
            #ind1=name.index('mu_')+3
            #ind2=name.index('_data')
            mu=0.1
            #mu=float(name[ind1:ind2])
            if mu<0.4:
                density=data['density']
                dipole=np.sum((density-1/3)*np.arange(len(density)))*8*np.pi**2/18**2
                density_2=np.zeros(len(density))
                for i in range(len(density)):
                    if i%3==1:
                        density_2[i]=1
                dipole=np.sum((density)*np.arange(len(density)))-np.sum((density_2)*np.arange(len(density)))
                dipoles.append(dipole)
             
                #print(name[ind1:ind2])
                
                mus.append(mu)
                plt.figure()
                plt.scatter(np.arange(len(data['density'])),data['density'])
                plt.savefig('density_'+name+'.png')
    plt.figure()
    mus=np.array(mus)/2
    plt.scatter(mus,dipoles)
    
    params, params_covariance = curve_fit(linear_model, mus, dipoles)
    print(params)
    xvals=np.linspace(0, np.max(mus),10)
    ys=linear_model(xvals,params[0],params[1])
    print(params)
    #plt.plot(xvals,ys,ls='--',color='red',label='y='+str(params[0])[:5]+'*x')
    plt.title('Dipole against the amplitude of potential step \n for'+str())
    plt.xlabel('V_0')
    plt.ylabel(r"$\frac{p}{L_y} / \frac{e}{4\pi} $")
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