
import numpy as np
import torch
from torch_geometric.data import Data
import re

import EdgeFeatureFunction as eff
from molecularGraphs import xyz2Graph

def getConfiguration( configuration ):

    nAtoms = int(configuration[0])  # number of atoms is specified in 1st line

    species = []                                           # species label
    coordinates = np.zeros( (nAtoms, 3), dtype = float )   # coordinates in Angstrom

    # below extract chemical labels, coordinates and charges

    m = 0

    for n in range(2, nAtoms+2):

        line = re.sub(r"\*\^","e", configuration[n]) # this prevents stupid exponential lines in the data base

        words = line.split()

        species.append(words[0])

        x = float( words[1] )
        y = float( words[2] )
        z = float( words[3] )

        coordinates[m,0] = x
        coordinates[m,1] = y
        coordinates[m,2] = z

        m += 1

    return nAtoms, species, coordinates

""" 
This function takes a trajectory and energy file from the c7o2H10_md database and 
returns a dataset consisting of a graph for each configuration in the trajectory
"""

def getTrajectory( mdFile, energyFile, specFeatures, nNeighbours = 6, nEdFeat = 1 ):

    # first read the energies

    efile = open( energyFile, 'r' )

    elines = efile.readlines()

    efile.close()

    nConfigurations = len(elines) - 1 # first line is a comment

    energies = np.zeros( ( nConfigurations ), dtype = float ) 

    for n, line in enumerate( elines ):
        if n > 0:
           words = line.split() 
           energies[n-1] = float( words[0] )
    
    # now get each configuration of the trajectory file, and generate a graph for each

    tfile = open( mdFile, 'r' )
 
    tlines = tfile.readlines()

    tfile.close()

    nLinesConf = int(len( tlines ) / nConfigurations )

    trajectorySet = []

    nConf = 0

    for n in range( 0, len(tlines), nLinesConf ):

        configuration = tlines[n:n+nLinesConf]

        nAtoms, labels, positions = getConfiguration( configuration )

        graph = xyz2Graph( nAtoms, labels, positions, None, specFeatures, \
               nMaxNeighbours = nNeighbours, nEdgeFeatures = nEdFeat )

        trajectorySet.append( graph )

        nConf += 1

    return trajectorySet 

