import torch
import numpy as np
from torch_geometric.data import Data
from glob import glob
import re
from typing import List

from gcnn.graphs.base import MolecularGraphs, get_molecule

# Atom reference energies (Hartree) for atomization energy calculation
# Extracted from MolecularGraphs class for standalone use in transforms
_atom_ref = dict(
    H=-0.500273, C=-37.846772, N=-54.583861, O=-75.064579, F=-99.718730
)

r""" Classes providing different flavours of database energy normalisation """

class scaleAndShift( object ):

    r""" scales and shifts the data to be in the range [-1,1] """

    def __init__( self, minValue: float, maxValue: float ) -> None:

        self.min = minValue
        self.max = maxValue
        self.delta = self.max - self.min

    def __call__( self, data: Data ) -> Data:

        r""" forward transform (scale to within [-1,1]) """

        y = data.y.item()

        ty = 2. * ( y - self.min ) / self.delta - 1.

        data.y = torch.tensor( ty, dtype = torch.float32 )

        return data

    def unscale( self, x: float ) -> float:

        r""" undo previous transformation to get back to original data """

        E = self.min + self.delta * ( x + 1. ) / 2.

        return E


class standardize( object ):

    r""" transforms the data such that xnew = ( x - xmean ) / std(x) """

    def __init__( self, meanValue: float, stdValue: float ) -> None:

        self.mean = meanValue
        self.std = stdValue

    def __call__( self, data: Data ) -> Data:

        r""" forward transform xnew = ( x - xmean ) / std(x) """

        y = data.y.item()

        ty = ( y - self.mean ) / self.std 

        data.y = torch.tensor( ty, dtype = torch.float32 )

        return data

    def unscale( self, x: float ) -> float:

        """ undo previous transformation to get original data """

        E = self.mean + self.std * x 

        return E


def SetUpDataTransform( transformData: str, directories: List[str] ) -> object:

    r""" This function sets up and returns the type of transformation to be applied to the data """

    std = re.compile( 'std', re.IGNORECASE )
    scale = re.compile( 'scale', re.IGNORECASE )

    if transformData and transformData != None :

        eMin, eMax, eMean, eSTD = analyseDataBase( directories )

        if re.search( std, transformData ):

            transform = standardize( eMean, eSTD )

        elif re.search( scale, transformData ):

            transform = scaleAndShift( eMin, eMax )

        else:

            transform = None

    else:

        transform = None

    return transform


def analyseDataBase( directories: List[str] ) -> float:

    r""" this function scans the data to fit and returns the minimum and maximum energies for scale-shift """

    files = []

    for directory in directories:

        txt = directory + '*.xyz'

        dirFiles = glob( txt )

        files += dirFiles

    e = []

    for file in files:

        _, nAt, labels, _, properties, _ = get_molecule( file )

        moleculeRef = 0.0

        for n in range( nAt ):

            moleculeRef += _atom_ref[labels[n]]

        # Index 10 = U0 (internal energy at 0 K), consistent with covalent.py/generalised.py
        atomisationEnergy = properties[0,10] - moleculeRef

        e.append( atomisationEnergy )

    energy = np.array( e )

    eMin = np.min( energy )
    eMax = np.max( energy )

    eMean = np.mean( energy )
    eSTD = np.std( energy )

    return eMin, eMax, eMean, eSTD
