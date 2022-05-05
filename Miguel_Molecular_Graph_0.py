# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 18:08:21 2022

@author: Miguel
"""

# Importing numpy
import numpy as np
from numpy import *

# Importing matplotlib for graphics and fixing the default size of plots
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # colors in graphics
from matplotlib.ticker import AutoMinorLocator

matplotlib.rcParams['mathtext.fontset']='cm'
matplotlib.rcParams.update({'font.size': 18})
plt.rcParams["figure.figsize"] = (10,8)

# In case we need sympy
from sympy import init_printing
init_printing(use_latex=True)

# For working with well stabished parameters
from mendeleev import elements
from scipy.constants import physical_constants


import tarfile   # treatment of files .tar
import glob      # searching directories
import re        # regular expresions

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from Features import Features

# define a conversion factor from Hartrees to eV
Hartree2eV = physical_constants['Hartree energy in eV'][0]

r""" 
The following dictionary defines the atomic ref energies, needed 
to calculate atomisation energies. These values are in Hartree
"""

atomRef = dict( H = -0.500273, C = -37.846772, N = -54.583861, 
                O = -75.064579, F = -99.718730 )



def getMolecule(fileName: str) -> Tuple[int, int, List[str], float, float, float]:

    r""" 
    This function opens a file of the QM-9 database and processes it, returning the molecule structure
    in xyz format, a molecule identifier (tag), and vectors containing the entire list of molecular
    properties.

    Arg:

       fileName (string): the path to the file where the molecule information is stored.

    Returns:

       moleculeID (integer): integer identifying the molecule number in the database.
       n_atoms (integer): number of atoms in the molecule.
       elements (List[string]): the chemical elements to which each atom belongs (len = n_atoms).
       coordinates (np.array(float)[n_atoms, 3]): atomic positions.
       properties (np.array(float)[:]): molecular properties; scalar properties + frequencies (see database documentation).
       charge (np.array(float)[n_atoms]): Mulliken charges of atoms.
    """


    #First, the ID of the molecule and its number of atoms

    file = open(fileName, 'r')      # we open the file
    
    data = file.readlines()         # textlines are stored in data, each row contains a string of characteres (1 line)

    n_atoms = int(data[0])          # number of atoms is specified in 1st line (component #0)

    words = data[1].split()
    moleculeID = int(words[1])
    molecularData = np.array(words[2:], dtype = float)


    # Now, the corresponding elements and coords for the n_atoms

    elements = []                                        # elements label
    coordinates = np.zeros((n_atoms, 3), dtype = float)  # coordinates [Angstrom]
    charge = np.zeros((n_atoms), dtype = float)          # Mulliken charges (e)
    
    for i in range(n_atoms):

        line = re.sub(r"\*\^","e", data[i+2]) # this prevents stupid exponential lines in the data base
             # re.sub (pattern, replacement, chain, mark = 0) # Out: chain with 'replacement' (can be string, function...) instead of 'pattern'
        words = line.split()
        
        elements.append(words[0])

        x = float(words[1])
        y = float(words[2])
        z = float(words[3])
        c = float(words[4])

        coordinates[i,0:3] = x, y, z
        charge[i] = c


    # Finally, the vibrational frequencies, in cm^-1

    frequencies = np.array(data[n_atoms+2].split(), dtype = float)

    # we pack all the molecular data into a single array of properties

    properties = np.expand_dims(np.concatenate((molecularData, frequencies)), axis = 0)

    return moleculeID, n_atoms, elements, coordinates, properties, charge




def Molecule2Graph(fileName: str, atoms_features: Dict[str,float], edge_features: Features, bond_angle_features: Features, 
                    dihedral_features: Features = None, covalentRadii = None, nMaxNeighbours: int = 6 ) -> dgl.graph:
    

    r"""

    This function reads molecule information from the database file and 
    converts it to DGL (Deep Graph Learning) graph. A graph can be 
    constructed in one of two ways:

    1. Chemically intuitive way: a node (atom) has edges only to 
       other nodes that are at a distance that is 1.1 times the 
       sum of their respective covalent radii away. In this mode
       edges will correspond to chemical bonds. To activate this 
       mode it is necessary to pass the dictionary covalentRadii;
       if it is not passed or is set to None, the second mode is
       activated (see below).

    2. Geometrical way: each node will have a minimum of nMaxNeighbours
       nearest nodes; this is a minimum number because the graph 
       should be undirected, meaning that if j is neighbour of i, 
       i must be a neighbour of j, even if i is not among the 
       nMaxNeighbours nearest neighbours of j; therefore the 
       actual number of neighbours of a node is at least equal
       to nMaxNeighbours, but in fact can be slightly larger.

    Args:

       fileName (string): the path to the file where the molecule 
           information is stored.

       atoms_features: a dictionary (keys=elements symbols, values = node 
           features) containing the node features.

       edge_features (Features): instance of Features class to 
           calculate edge features related to the distance between
           the nodes i-j defining the edge.

       bond_angle_features (Features): instance to compute bond-angle
           features for each node, akin to a histogram of bond-angles
           formed by the nearest-neighbour nodes of node i. If an atom
           i has only one neighbour, it does not define any bond angles
           so its bond_angle features would be zero. These kind of 
           features are associated to nodes, and are concatenated to
           the "chemical" features specifying elements, etc.

       dihedral_features (Features): similar to bond_angle_features above,
           an instance of Features to encode information about the 
           dihedral angles formed by the neighbours k of node i and l of 
           node j, where i-j define the edge; these are edge features, 
           to be concatenated to the distance-related edge features;
           if not given, default None, in which case they are not used.
 
       covalentRadii: dictionary dict( elements:radius in Angstrom ); if 
           not given (None), neighbours are instead found using the next
           parameter, choosing the nearest nMaxNeighbours to each node.

       nMaxNeighbours: number of nearest neighbours (this is a minimum 
           number; in general it may be slightly larger so as to ensure 
           that graphs are undirected (if i is neighbour of j, j is a 
           neighbour of i).

    Return:
        
        moleculeGraph (dgl.graph): a graph containing the molecule features.
    """
    
    moleculeID, n_atoms, elements, positions, properties, charges = getMolecule(fileName)
    
    # the total number of node features is given by the elements features 

    n_features = atoms_features[elements[0]].size 
    nodeFeatures = torch.zeros( (n_atoms, n_features), dtype = torch.float32 )

    # atoms will be graph nodes; edges will be created for every neighbour of i that is among the nearest 
    # nMaxNeighbours neighbours of atom i 

    # first we loop over all pairs of atoms and calculate the matrix of squared distances

    dij2 = np.zeros((n_atoms, n_atoms), dtype = float)

    for i in range(n_atoms-1):
        for j in range(i+1, n_atoms):

            rij = positions[j,:] - positions[i,:]
            dij2[i,j] = dij2[j,i] = np.dot(rij, rij)

            #dij2[i,j] = rij2] 
            #dij2[j,i= rij2

    if (covalentRadii):  # use sum of covalent radii as criterion for edges
        
        nMax = 12
        nNeighbours = np.zeros((n_atoms), dtype = int) # neighbour to which we are pointing, nearest=0, second nearest=1 and so on     
        neighbourDistance = np.zeros((n_atoms, nMax), dtype = float)   # distances matrix
        neighbourIndex = np.zeros((n_atoms, nMax), dtype = int)        # neighbours matrix

        node0 = []    # starting node, node i
        node1 = []    # node pointed to by i, node j

        for i in range(n_atoms-1):
            for j in range(i+1, n_atoms):

                dcut = 1.1*(covalentRadii[elements[i]] + covalentRadii[elements[j]])
                dcut2 = dcut*dcut

                # we consider two nodes are neighbours if its separation dist^2 is less than dcut^2
                
                if (dij2[i,j] <= dcut2):  

                    node0.append(i)
                    node1.append(j)

                    node0.append(j) # we want the graph undirected
                    node1.append(i)

                    dij = np.sqrt(dij2[i,j])  # distance between the two nodes

                    neighbourDistance[i, nNeighbours[i]] = dij
                    neighbourDistance[j, nNeighbours[j]] = dij

                    neighbourIndex[i, nNeighbours[i]] = j
                    neighbourIndex[j, nNeighbours[j]] = i

                    nNeighbours[i] += 1
                    nNeighbours[j] += 1

        edge_index = torch.tensor([node0, node1], dtype = torch.long)

        _, num_edges = edge_index.shape


    else:  # we fall on this if covalentRadii = None
    
        if n_atoms <= nMaxNeighbours:
            nMax = n_atoms - 1
        else: 
            nMax = nMaxNeighbours

        neighbourIndex = np.zeros((n_atoms, nMax), dtype = int)
        
        node0 = [] 
        node1 = []
        
        # for each atom, select only the nearest nMax neighbours; this requires sorting the squared distances
        
        for i in range(n_atoms):

            sqrdDistances = dij2[i,:]
            indices = np.argsort(sqrdDistances)

            neighbourIndex[i,0:nMax] = indices[1:nMax+1]  # indices starts by 1 because pos=0 corresponds to dii=0 (the atom with itself)
        
        for i in range(n_atoms):
            for n in range(nMax):
                
                j = neighbourIndex[i,n]
                
                node0.append(i)
                node1.append(j)
        
        # before we can create the graph, we need to make sure that the graph will be undirected, i.e. that
        # if atom j is a neighbour of i, i is a neighbour of j; this is not necessarily the case when fixing
        # the number of neighbours, so we must impose it 

        edges = torch.tensor( [node0, node1], dtype = torch.long )

        edge_index = to_undirected( edge_index = edges )

        _, num_edges = edge_index.shape

        # then, to construct the geometric features for nodes and edges we need to know 
        # number of neighbours, neighbour indices and distances for each atom

        nNeighbours = np.zeros((n_atoms), dtype = int)
        neighbourIndex = np.zeros((n_atoms, n_atoms), dtype = int)
        neighbourDistance = np.zeros((n_atoms, n_atoms), dtype = float)

        for n in range(num_edges):

            i = edge_index[0,n]
            j = edge_index[1,n]

            neighbourDistance[i,nNeighbours[i]] = np.sqrt(dij2[i,j])
            
            neighbourIndex[i,nNeighbours[i]] = j
            
            nNeighbours[i] += 1

            # yet the graph is undirected, as we loop edge_index we are taking into account edges i->j and then j->i
    
    
    """ # construct node geometric features; the name is confusing, as these will really
    # be edge features; the node features consist only of elements properties
        
    nBAFeatures = bond_angle_features.n_features
    nodeGeometricFeatures = np.zeros( ( n_atoms, nBAFeatures ) )

    for i in range( n_atoms ):

        anglehist = np.zeros( ( nBAFeatures ), dtype = float )

        for n in range( nNeighbours[i] - 1 ):
            
            j = neighbourIndex[i,n]

            rij = positions[j,:] - positions[i,:]
            dij = neighbourDistance[i,n]

            for m in range( n+1, nNeighbours[i] ):

                k = neighbourIndex[i,m]

                rik = positions[k,:] - positions[i,:]
                dik = neighbourDistance[i,m]

                costhetaijk = np.dot( rij, rik ) / ( dij * dik )
 
                thetaijk = np.arccos( costhetaijk )

                anglehist += bond_angle_features.u_k( thetaijk )

        nodeGeometricFeatures[i,:] = anglehist
        nodeFeatures[i,:] = torch.from_numpy( atoms_features[elements[i]] )
    
    
    # now, based on the edge-index information, we can construct the edge attributes

    bond_features = []

    for n in range(num_edges):

        i = edge_index[0,n]
        j = edge_index[1,n]

        dij = np.sqrt(dij2[i,j])       # neighbourDistance[i,j] ????

        features = np.concatenate((edge_features.u_k(dij), nodeGeometricFeatures[i,:], nodeGeometricFeatures[j,:]))

        bond_features.append(features)


    # finally, if we have attributes for bond angles to be added to edges... 

    if dihedral_features:

        angles = []

        nDAFeatures = dihedral_features.n_features

        for n in range( num_edges ):

            i = edge_index[0,n]
            j = edge_index[1,n]

            ri = positions[i,:]
            rj = positions[j,:]

            anglehist = np.zeros( ( nDAFeatures ), dtype = float )

            for ni in range( nNeighbours[i] ):

                k = neighbourIndex[i,ni]

                if k == j: continue  # k must be different from j

                dik = ( positions[k,:] - ri ) / neighbourDistance[i,ni]

                for nj in range( nNeighbours[j] ):

                    l = neighbourIndex[j,nj]

                    # cannot define a dihedral if l == k, or l == i
                    if l == k or l == i: continue

                    djl = ( positions[l,:] - rj ) / neighbourDistance[j,nj]

                    cosijkl = np.dot( dik, djl )

                    thetaijkl = np.arccos( cosijkl )

                    anglehist += dihedral_features.u_k( thetaijkl )

            total_bond_features = \
               np.concatenate( ( bond_features[n], anglehist ) )

            bond_features[n] = total_bond_features
    """


    # calculate the atomisation energy, the property we'll study later

    moleculeRef = 0.0

    for i in range(n_atoms):
        
        moleculeRef += atomRef[elements[i]]

    atomisationEnergy = Hartree2eV*(properties[0,10] - moleculeRef)  # internal energy at 0 k
    
    
    # now we can create a graph object (dgl.graph)

    pos = torch.from_numpy(positions)
    edge_attr = torch.tensor(bond_features, dtype = torch.float32)
    # fit = torch.tensor(( atomisationEnergy / float(n_atoms) ), dtype = torch.float32)
    fit = torch.tensor(atomisationEnergy, dtype = torch.float32)
    
    # the line above is for total energy for the whole graph; 
    # if you want energy/atom uncomment line above
    # and comment previous line

    # Using ytorch geometric,
    # moleculeGraph = Data(x = nodeFeatures, y = fit, edge_index = edge_index, edge_attr = edge_attr, pos = pos)
    # Using dgl,
    moleculeGraph = dgl.graph((edge_index[0],edge_index[1]), num_nodes= n_atoms)
    
    moleculeGraph.ndata['coordinates'] = pos
    moleculeGraph.ndata['nodeFeatures'] = nodeFeatures
    moleculeGraph.ndata['fit'] = fit
    moleculeGraph.edata['edge_attr'] = edge_attr


    return moleculeGraph    