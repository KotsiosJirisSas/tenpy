import h5py
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import curve_fit

def linear_model(x, m, c):
    return m * x + c


def h5_to_dict(obj):
    result = {}
    for key in obj.keys():
        item = obj[key]
        if isinstance(item, h5py.Group):
            # Recursively process groups
            result[key] = h5_to_dict(item)
        elif isinstance(item, h5py.Dataset):
            data = item[()]
            # Decode bytes to string if it's a string dataset
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            # Handle object arrays or other complex types if needed
            result[key] = data
    return result

def load_data(name):
    
    
    file_path='/mnt/users/dperkovic/DMRG_modified/infinite_DMRG_data/1_3_1_3_decouple/'

    with h5py.File(file_path+name+'.h5', 'r') as h5_file:
        # List all datasets and groups in the file
        #print("Keys in the file:", list(h5_file.keys()))

        # Access a dataset or group
        data={}
       
        keys=['density','energy','entanglement_entropy']#,'entanglement_spectrum']
        for dataset_name in keys:
         
            if dataset_name=='density':# or dataset_name=='entanglement_spectrum':
                print(dataset_name)
                data[dataset_name]= h5_file[dataset_name][:]
            
            if dataset_name=='entanglement_entropy':# or dataset_name=='entanglement_spectrum':
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
                    
                data[dataset_name]=dikt
                
      
       
    return data;
def save_graph_data(name,dipoles,energies,Ks):
    with open(name+".txt", "w") as f:
        f.write("dipole:\n")
        f.write(str(dipoles)+"\n")
        f.write("Energies:\n")
        f.write(str(energies)+"\n")
        f.write("Ks:\n")
        f.write(str(Ks)+"\n")

def load_ES(name):

    file_path= '/mnt/users/dperkovic/DMRG_modified/infinite_DMRG_data/1_3_1_3_decouple/'
    #with h5py.File(file_path+name+'.h5', 'r') as f:
    #    data_dict = h5_to_dict(f)
    #print(data_dict.keys())
    #quit()
    print('IN BRO')
    with h5py.File(file_path+name+'.h5', 'r') as obj:
        # List all datasets and groups in the file
        data=h5_to_dict(obj)
    return data
def plot_entanglement_spectra():
   
    path='/mnt/users/dperkovic/DMRG_modified/infinite_DMRG_data/1_3_1_3_decouple/'
    dir_list = os.listdir(path)
    dir_list.sort()
    energies=[]
    for name in dir_list:
        val=''
        print(name)
        if  "MPS" not in name:
          
            name=name[:-3] 
            data=load_data(name)
            
            print("*"*30)
            print(name)
            
            

            data=load_ES(name)
            ES=data['entanglement_spectrum']
            energies.append(data['energy'])
            
        
            
       
            data=reconfigure_ES(ES[str(0)])
            ES_plot(data,file_path='/mnt/users/dperkovic/DMRG_modified/tenpy/run_idmrg/tests/ES/ES_site='+str(0)+'_'+str(name[:-4])+'.png')
                

def ES_plot(es,file_path='data/ES.png'):
    minEE = np.array([ list(sec[0]) + [ np.sum(np.exp(-2*sec[1])) ] for sec in es])
    N = minEE[np.argmax(minEE[:, -1]), 1:-1]
    print('Minimum entanglement entropy charge sector:',minEE[np.argmax(minEE[:, -1]),:],N)     # pick only the number sectors
    pts = []
    for sec in es:
        if np.all(sec[0][1:] == N) :
            pts = pts + [ [sec[0][0], p] for p in sec[1] ]
    pts = np.array(pts)
    if len(pts) == 0: pts = np.zeros((0, 2), dtype=int)
    #pts[:, 0] /= float(psi.L) #rescales momentum by unit cell length
    wh, wv = 1, 0.01
    plt_kwargs = {'markersize':10, 'color':'navy', 'marker':[(-wh, -wv), (wh, -wv), (wh, wv), (-wh, wv)], 'linewidth':0.}
    plt_kwargs2 = {'markersize':10, 'color':'red', 'marker':[(-wh, -wv), (wh, -wv), (wh, wv), (-wh, wv)], 'linewidth':0.}
    plt.plot(pts[:, 0], pts[:, 1],**plt_kwargs)
    plt.xlabel(r'$K$', fontsize=18)
    plt.ylabel(r'$E_i$', fontsize=18, rotation='horizontal')
    plot_title = 'es'
    plt.title(plot_title)
    #directory_name = "results"
    #current_directory = os.getcwd()
    #new_directory_path = os.path.join(current_directory, directory_name)
    #file_name = 'ES.png'
    #file_path = os.path.join(new_directory_path, file_name)
    #file_path = 'data/ES.png'
    plt.savefig(file_path,dpi=400)
    plt.close()
    return

def reconfigure_ES(ES):
    data=[]
    print('IN')
    for key in ES.keys():
        print(key)
        #print(ES[key])
        print(ES[key]['0'])

        
        data.append([ES[key]['0'],ES[key]['1']])
   
   
    return data


plot_entanglement_spectra()
