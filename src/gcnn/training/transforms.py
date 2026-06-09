import torch
import numpy as np
from torch_geometric.data import Data
from typing import List

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


def setup_transform(transform_name: str, dataset) -> object:
    """Set up a data transform by computing statistics from a loaded dataset.

    Args:
        transform_name: One of 'standardize', 'scale', or None/False.
            - 'standardize' or 'std': normalizes y to zero mean, unit variance.
            - 'scale': scales y to [-1, 1] range.
        dataset: A loaded dataset (e.g. CachedGraphDataset) with .y attributes.

    Returns:
        A transform object (callable) or None.
    """
    if not transform_name:
        return None

    # Collect all target energies from the dataset
    energies = np.array([dataset[i].y.item() for i in range(len(dataset))])

    name_lower = transform_name.lower()

    if 'std' in name_lower or 'standard' in name_lower:
        return standardize(np.mean(energies), np.std(energies))
    elif 'scale' in name_lower:
        return scaleAndShift(np.min(energies), np.max(energies))
    else:
        print(f"WARNING: unknown transform '{transform_name}', skipping.")
        return None
