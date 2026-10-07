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
    
    
    file_path='/mnt/users/dperkovic/DMRG_modified/infinite_DMRG_data/V0_coulumb_correct/shift/'
    #file_path='/mnt/users/dperkovic/DMRG_modified/infinite_DMRG_data/nu=1_5/'
    with h5py.File(file_path+name+'.h5', 'r') as h5_file:
        # List all datasets and groups in the file
        #print("Keys in the file:", list(h5_file.keys()))

        # Access a dataset or group
        data={}
       
        keys=['density','energy','entanglement_entropy','correlation_length','model_par']#,'entanglement_spectrum']
        for dataset_name in keys:
           
            if dataset_name=='density':# or dataset_name=='entanglement_spectrum':
              
                data[dataset_name]= h5_file[dataset_name][:]
            
            if dataset_name=='entanglement_entropy':
                print(dataset_name)
                #try:
                data[dataset_name]= h5_file[dataset_name][:]
                print( h5_file[dataset_name][:])
                #except:
                #    data[dataset_name]=0.0
            if dataset_name=='energy'or dataset_name=='correlation_length':
                try:
                    data[dataset_name]= h5_file[dataset_name][...]
                except:
                    data[dataset_name]= 0.0
                    pass
                
            if dataset_name=='entanglement_spectrum':
                keys_of_dict = list(h5_file[dataset_name])
           
                dikt={}
                for k in keys_of_dict:
                    #print(kaka)
                    dikt[k]=list(h5_file[dataset_name][k])
                    
                data[dataset_name]=dikt
            if dataset_name=='model_par':
                keys_of_dict = list(h5_file[dataset_name])
           
                dikt={}
                for k in keys_of_dict:
                    #print(kaka)
                    dikt[k]=h5_file[dataset_name][k]
                    
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

    file_path= '/mnt/users/dperkovic/DMRG_modified/infinite_DMRG_data/V0_coulumb_correct/shift/'
    #with h5py.File(file_path+name+'.h5', 'r') as f:
    #    data_dict = h5_to_dict(f)
    #print(data_dict.keys())
    #quit()
    #file_path='/mnt/users/dperkovic/DMRG_modified/infinite_DMRG_data/nu=1_5/'
    with h5py.File(file_path+name+'.h5', 'r') as obj:
        # List all datasets and groups in the file
        data=h5_to_dict(obj)
    return data
def plot_entanglement_spectra():
   
    path='/mnt/users/dperkovic/DMRG_modified/infinite_DMRG_data/V0_coulumb_correct/shift/'
    #path='/mnt/users/dperkovic/DMRG_modified/infinite_DMRG_data/nu=1_5/'
    dir_list = os.listdir(path)
    dir_list.sort()
    energies=[]
    chis=[]
    for name in dir_list:
        val=''
      
        if  ("MPS" not in name and 'Environment' not in name and 'Envs' not in name):# and 'Coulumb' in name and 'chi=4000' in name and 'xi=2.0' in name and '[1,0,0,1,0,0,0,0]' in name:# and 'V0-0.19' not in name:# and 'chi=2000' in name:#f and 'chi=2000'  in name  and 'xi=1' in name:# and 'chi=300' in name:
          
            name=name[:-3] 
            print("*"*30)
            print(name)
           
            data=load_data(name)
            #0.05884633958877217
            #0.06038391477237681
            #0.05872705845382242
       
          
         
            print(data['density'])
            print('energy:',data['energy'])
            print(data.keys())
            index=name.index('chi=')
            #index2=name.index('_site')
            print(name[index+4:])
          
            #index1=name.index('V0=')+3
            #ndex2=name.index('_b=0')
            #index2=name.index('Lx=')
            #chi=float(name[index+4:])
            #print(name[index1:index2])
            #chi=float(name[index1:index2])
            chi=-1 
            #print(chi)
            try:
                print('correl length:')
                print(data['correlation_length']/(2*np.pi))
                if chi>-100:
                    #print(data['model_par']['layers'])
                    chis.append(chi)
                    print('length',data['correlation_length']/(2*np.pi))
                    energies.append(data['correlation_length']/(2*np.pi))
                    val1=chi
                    val2=data['correlation_length']/(2*np.pi)
            except:
                val=0
                energies.append(0)
        
            data=load_ES(name)
            ES=data['entanglement_spectrum']
            
            
            #energies.append(data['entanglement_entropy'][0])

            site=2
            val1=0
            val2=0
            data=reconfigure_ES(ES[str(site)])
            
            ES_plot(data,val1,val2,file_path='/mnt/users/dperkovic/DMRG_modified/tenpy/run_idmrg/'+str(name)+"_site="+str(site)+'.png')
    #print(energies)
    #quit()
    #chis=[15,16,17,18,19]
    plt.title('correlation length against V0 for Coulumb for 10010000 \n sector, chi=4000 xi=2.0')
    plt.xlabel('V_0')
    plt.ylabel('epsilon/l_B')
    plt.scatter(chis,energies)
    plt.ylim(1,15)
    plt.yscale('log')
    plt.savefig('baaaa.png')   

def ES_plot(es,val1,val2,file_path='data/ES.png'):
    minEE = np.array([ list(sec[0]) + [ np.sum(np.exp(-2*sec[1])) ] for sec in es])
    N = minEE[np.argmax(minEE[:, -1]), 1:-1]
    #print('Minimum entanglement entropy charge sector:',minEE[np.argmax(minEE[:, -1]),:],N)     # pick only the number sectors
    pts = []
    #print('N:',N)

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
    #print(pts[:, 0])
    #print()
 
    #2,4,10
    plt.title('V0='+str(val1)+' epsilon='+str(val2))
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
        #print(key)
        #print(ES[key])
        #print('IN')
        #print(ES[key]['0'][0])
        """
        if ES[key]['0'][0]==2 and ES[key]['0'][1]==2:
            print(np.log(ES[key]['1']**2))
            #print(ES[key]['1'])
        """  
        #print(ES[key]['1'])
        
        
        data.append([ES[key]['0'],ES[key]['1']])
   
   
    return data


def plot_against_correlation():
   
    path='/mnt/users/dperkovic/DMRG_modified/infinite_DMRG_data/V0_coulumb_correct/'
    dir_list = os.listdir(path)
    dir_list.sort()
    energies=[]
    chis=[]
    for name in dir_list:
        val=''
        label='V0=-0.7'
        if  "MPS" not in name and 'a=10' in name and label in name and '[1,0,0,1' in name:# and '[1,0,1' not in name:# and 'a=12.0' in name:# and 'chi=300' in name:
          
            name=name[:-3] 
            print("*"*30)
            print(name)
           
            data=load_data(name)
       
          
         
            print(data['density'])
            print('energy:',data['energy'])
            print(data.keys())
            index=name.index('chi=')+4
            #index2=name.index('_site')
            print(name[index:])
          
       
            #index2=name.index('_b=0')
       
            #chi=float(name[index+4:])
            chi=float(name[index:])
            print(chi)
          
            try:
                if chi>-100:
                    #print(data['model_par']['layers'])
                    chis.append(chi)
                    print('length',data['correlation_length']/(2*np.pi))
                    energies.append(data['correlation_length']/(2*np.pi))
                    val1=chi
                    val2=data['correlation_length']/(2*np.pi)
            except:
                val=0
                energies.append(0)
        
            data=load_ES(name)
            ES=data['entanglement_spectrum']
            
            
            #energies.append(data['entanglement_entropy'][0])

            site=1
            #data=reconfigure_ES(ES[str(site)])
            #ES_plot(data,val1,val2,file_path='/mnt/users/dperkovic/DMRG_modified/tenpy/run_idmrg/'+str(name)+"_site="+str(site)+'.png')
    #print(energies)
    #quit()
    #chis=[15,16,17,18,19]
    plt.title('Correlation length scaling with bond dimension '+label)
    plt.xlabel('chi')
    plt.ylabel('epsilon/l_B')
    plt.scatter(chis,energies)
    #plt.ylim(1.7,3.5)
    plt.savefig('Correlation_length_bond_'+label+'.png') 

#plot_against_correlation()
#quit()
plot_entanglement_spectra()
