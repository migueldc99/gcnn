import numpy as np
import torch
from torch_geometric.data import Data
from glob import glob
import re
import bz2
import pickle

from molecularGraphs import generateNodeFeatures, Molecule2Graph

"""
This script transforms molecule files into graph data and stores the graph info into a file, one for 
each molecule, for later use in fitting, validating and testing the graph NN model
""" 

# first generate the node features

species = ['H', 'C', 'N', 'O', 'F']

nodeFeatures = ['atomic_number', 'covalent_radius', 'vdw_radius', 'electron_affinity', 'en_pauling']

specFeatures = generateNodeFeatures( species, nodeFeatureList = nodeFeatures )

# now loop over files, read the info from each file and generate the appropriate graph; store

files = glob('*.xyz')

#

pattern = 'dsgdb9nsd'
replace = 'grph'

for n, file in enumerate(files):
    
    print(n,file)

    graph = Molecule2Graph( file, specFeatures, nMaxNeighbours = 12 )

    if graph.is_directed():   # graph should be undirected, so if this happens, sign alarm!
       txt = file + ' generates a directed graph!!!'
       print(txt)

    idNum = re.search('\d{6}',file)
    fileOut = 'grph_'+idNum[0]+'.bz2'

    fileObject = bz2.BZ2File( fileOut, 'wb' )
    pickle.dump( graph, fileObject )
    fileObject.close()
     
