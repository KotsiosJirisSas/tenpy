
import h5py
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def f_ramp_with_dip2(v,x):
    Lseg=200

    if x < int(Lseg/6.):
        return v*(x/(Lseg/6.)-1)
    else:
        return v*(1+0.4*np.exp(-0.0001*(x-48.333)**6)-1)
    
def plot_density_integral(density,distance,name,filling=1/3):

    plt.figure()

    
    integrals=[]
    for x in distance:
        integral=np.sum(density[distance<x]-filling)
        integrals.append(integral)

  
    
    plt.plot(distance, integrals,c='#1f77b4', marker='o',markerfacecolor='none', linestyle='-', label="Main plot")
    plt.plot(distance, distance*0,c='#1f77b4', linestyle='--', label="Main plot")
    plt.plot(distance, distance*0-1/3,  linestyle='--', color='goldenrod', label="Main plot")
    plt.ylabel(r"$\Delta N$")
    plt.xlabel(r"$X_i$")
    plt.savefig('temporary_figures/'+name+'.png')

def plot_density_with_EE_spectrum(density,distance,EEs,name):

   

    # Create the main plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(distance, density, label="Main plot")
    ax.set_title("Main Plot with Inset")
    ax.legend()

    # Create an inset plot
    inset_ax = inset_axes(ax, width="30%", height="30%", loc="upper right")  # width and height are percentages
    inset_ax.plot(EEs['60'], 'r', label="Inset plot")
    inset_ax.set_title("Inset", fontsize=8)
    inset_ax.tick_params(axis='both', which='major', labelsize=8)
    plt.savefig('temporary_figures/'+name+'.png')
    plt.show()



def plot_density(density,name,mu,N_layer=1,filling=1/2,potential=True):
    distance=np.arange(len(density))//N_layer

    fig, ax = plt.subplots(figsize=(6, 4))

  
    ax.plot(distance,density,c='#1f77b4', marker='o',markerfacecolor='none', linestyle='-', label="Main plot")
    ax.set_title(r"$p=$"+str(mu),fontsize=15)
    
    ax.set_ylabel(r"$\nu$",fontsize=15)
    ax.set_xlabel(r"$X_i$",fontsize=15)

    max_x=120
    #ax.set_xlim(-2,max_x)
    integrals=[]
    for x in distance:
        integral=np.sum(density[distance<x]-filling)
        integrals.append(integral)
    #ax2=ax.twinx()
    ax.plot(distance,distance*0+0.5,ls='--')
    #ax.set_xlim(40,250)
    #ax2.scatter(distance,np.array(integrals), color='r')
    if potential==False:
        v=0.2
        V=[]
        for x in distance:
            V.append(-f_ramp_with_dip2(v,x))
        print('INNNNN')
        #ax2=ax.twinx()
        inset_position = [0.7, 0.23, 0.2, 0.2]
        ax2=fig.add_axes(inset_position)
        ax2.set_xlim(-2,max_x)
        ax2.plot(distance, V,'r',markerfacecolor='none', linestyle='-', label="Main plot")

        ax2.set_xlabel(r"$X_i$")

        custom_labels_y=['0','0.2']
        ax2.set_ylabel(r"$V$")
        ax2.yaxis.set_label_coords(-0.05, 0.6)
        ax2.set_yticks([0,0.2])
        ax2.set_xticks([])
   


    inset_position = [0.65, 0.6, 0.28, 0.28]  # [x0, y0, width, height] in normalized figure coordinates

    # Create the inset plot
    inset_ax = fig.add_axes(inset_position)
    #inset_ax = inset_axes(ax, width="40%", height="40%", loc=(5,5))  # width and height are percentages
    inset_ax.plot(distance,integrals, 'r', label="Inset plot")
    inset_ax.plot(distance,np.array(integrals)*0, ls='--', label="Inset plot")
    inset_ax.plot(distance,np.array(integrals)*0+1/4, c='goldenrod',ls='--', label="Inset plot")
    inset_ax.set_ylabel(r"$\Delta N$")
    inset_ax.set_xlabel(r"$X_i$")
    #inset_ax.set_xlim(-2,max_x)
    #inset_ax.set_xticks([])
    #inset_ax.set_yticks([0,-2/3])
    #custom_labels_y=['0','-2/3']
    #inset_ax.yaxis.set_label_coords(-0.03, 0.9)
    #inset_ax.set_yticklabels(custom_labels_y)
   
    #inset_ax.set_title("Inset", fontsize=8)
    plt.tight_layout()
    #inset_ax.tick_params(axis='minor', which='x', labelsize=8)
    plt.savefig('temporary_figures/_'+name+'.png')


def plot_energy(Ks,Es,ps,name):
   
    fig, ax = plt.subplots(figsize=(6, 4))
    #plt.figure()
  
    plt.plot(Ks,Es,c='#1f77b4', marker='o',markerfacecolor='none', linestyle='-', label="Main plot")
    #plt.title(r"$K=$"+str(mu),fontsize=15)
    
    plt.ylabel(r"$E$",fontsize=15)
    plt.xlabel(r"$K$",fontsize=15)
    
    #plt.xlim(-2,100)
   
    #ax2=ax.twinx()
    #ax2.scatter(distance,np.array(integrals), color='r')

    inset_position = [0.24, 0.27, 0.27, 0.27]  # [x0, y0, width, height] in normalized figure coordinates

    # Create the inset plot
    inset_ax = fig.add_axes(inset_position)
    #inset_ax = inset_axes(ax, width="40%", height="40%", loc=(5,5))  # width and height are percentages
    inset_ax.plot(Ks,ps, 'r', label="Inset plot")
    #inset_ax.plot(distance,np.array(integrals)*0, ls='--', label="Inset plot")
    #inset_ax.plot(distance,np.array(integrals)*0-1/3, c='goldenrod',ls='--', label="Inset plot")
    inset_ax.set_ylabel(r"$\frac{p}{L_y}/\frac{e}{4\pi}$")
    inset_ax.set_xlabel(r"$K$")
    #inset_ax.set_xlim(-2,100)
    #inset_ax.set_xticks([])
    #inset_ax.set_yticks([-5,0,-1/3,5])
    #custom_labels_y=['-5','0','-1/3','5']
    inset_ax.yaxis.set_label_coords(-0.1, 0.5)
    #inset_ax.set_yticklabels(custom_labels_y)
    #inset_ax.set_title("Inset", fontsize=8)
    plt.tight_layout()
    #inset_ax.tick_params(axis='minor', which='x', labelsize=8)
    plt.savefig(name+'.png')