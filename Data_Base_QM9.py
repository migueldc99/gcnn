# -*- coding: utf-8 -*-
"""
@author: Miguel Dalmau Casañal
"""


# Importing numpy
import numpy as np

# Importing matplotlib for graphics and fixing the default size of plots
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['mathtext.fontset']='cm'
matplotlib.rcParams.update({'font.size': 18})
plt.rcParams["figure.figsize"] = (10,8)

# In case we need sympy
from sympy import init_printing
init_printing(use_latex=True)

import tarfile   # to work with files .tar
import glob      # to search directories


r"""
my_tar = tarfile.open('dsgdb9nsd.xyz.tar')  #my_tar.tar.gz
my_tar.extractall('C:/Users/Portatil/Documents/Máster/TFM/Trabajo/1_Base_Datos/Ficheros_xyz')
my_tar.close()
"""

### First, we read the files to get the histograms ###

dipolar_mom = []    # dipolar moment
gap = []            # gap
energy_0K = []      # energy at 0 K

# we create an array that stores the path of all files 
files_directories = glob.glob('./Ficheros_xyz/dsgdb9nsd_*.xyz')

for i in range (0,133884):
  
    file = open(files_directories[i], 'r')  # we open the file
    datos = file.readlines()                # we generate an array who saves in each posicition a line with characters
    properties = datos[1].split()           # split the elements of line 2 (here there are the properties)
    
    n_atoms = float(datos[0])
    dipolar_mom.append(float(properties[5]))   
    gap.append(float(properties[9]))
    energy_0K.append(float(properties[12])/n_atoms)


histo_dipolar_mom = np.array(dipolar_mom)  # turn it into numpy.array to work with it later
histo_gap = np.array(gap)
histo_energy_0K = np.array(energy_0K)


### HISTOGRAMS ###

fig = plt.figure(figsize=(19,7))


ax1 = fig.add_subplot(121)

ax1.spines['left'].set_linewidth(1.75)
ax1.spines['right'].set_linewidth(1.75)
ax1.spines['top'].set_linewidth(1.75)
ax1.spines['bottom'].set_linewidth(1.75)

ax1.tick_params(top = True, right=True, bottom = True, direction="in", width=1.6, length=4.5)  #para las marcas

ax1.set_xlabel(r'Dipolar moment $\mu$ (D)', fontdict = {'fontsize':24, 'color':'k'})
ax1.set_ylabel(r'P($\mu$)', fontdict = {'fontsize':24, 'color':'k'})
ax1.text(0,0.32,"$a)$", fontsize=27)
ax1.set_xlim([-0.5,8.5])
ax1.set_ylim([0.0,0.35])
ax1.set_xticks([0,1,2,3,4,5,6,7,8,9,10])

ax1.hist(histo_dipolar_mom, bins = 65, range=[0, 10.5], density = True, ec = 'k', linewidth=1.3)


ax2 = fig.add_subplot(122)

ax2.spines['left'].set_linewidth(1.75)
ax2.spines['right'].set_linewidth(1.75)
ax2.spines['top'].set_linewidth(1.75)
ax2.spines['bottom'].set_linewidth(1.75)

ax2.tick_params(top = True, right=True, bottom = True, direction="in", width=1.6, length=4.5)

ax2.set_xlabel(r'Internal energy per particle at 0 K (Ha)', fontdict = {'fontsize':24, 'color':'k'})
ax2.set_ylabel(r'P(energy)', fontdict = {'fontsize':24, 'color':'k'})
ax2.text(-85,0.1,"$b)$", fontsize=27)

ax2.hist(histo_energy_0K, bins = 50, density = True, ec = 'k', linewidth=1.3)


plt.show()
#plt.savefig('DB.pdf',bbox_inches='tight')


#Now, the probability distribution function for the gap 
fig, ax = plt.subplots()

ax.spines['left'].set_linewidth(1.7)
ax.spines['right'].set_linewidth(1.75)
ax.spines['top'].set_linewidth(1.75)
ax.spines['bottom'].set_linewidth(1.75)

ax.tick_params(top = True, right=True, bottom = True, direction="in", width=1.6, length=4.5)

ax.set_xlabel(r'Gap, $\epsilon_{gap}$ (Ha) ', fontdict = {'fontsize':24, 'color':'k'})
ax.set_ylabel(r'P(gap)', fontdict = {'fontsize':24, 'color':'k'})
ax.set_xlim([0.09,0.42])
ax.set_ylim([0.,10.])

ax.hist(histo_gap, bins = 50, range=[0.1, 0.4], density = True, ec = 'k', linewidth=1.3)

plt.show()
#plt.savefig('gap(1).pdf',bbox_inches='tight')