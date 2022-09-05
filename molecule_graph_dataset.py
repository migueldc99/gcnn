
from glob import glob
import random

import torch
from torch_geometric.data import Data, Dataset

from molecular_graphs import MolecularGraphs

featureList = ["atomic_number"]

class MoleculeGraphDataSet(Dataset):

    r"""Data set class to load molecular graph data"""

    def __init__(
        self,
        database_dir: str,
        graphs: MolecularGraphs,
        nMaxEntries: int = None,
        seed: int = 42,
        transform: object = None,
        pre_transform: object = None,
        pre_filter: object = None,
    ) -> None:

        r"""

        Args:

           database_dir (str): the directory where the data files reside

           graphs: an object of class MolecularGraphs whose function is
                  to read each file in the data-base and return a
                  graph constructed according to the particular way
                  implemented in the class object (see MolecularGraphs
                  for a description of the class and derived classes)

           nMaxEntries (int): optionally used to limit the number of clusters
                  to consider; default is all

           seed (int): initialises the random seed for choosing randomly
                  which data files to consider; the default ensures the
                  same sequence is used for the same number of files in
                  different runs

        """

        super().__init__(database_dir, transform, pre_transform, pre_filter)

        self.database_dir = database_dir

        self.graphs = graphs

        filenames = database_dir + "/*.xyz"

        files = glob(filenames)

        self.n_molecules = len(files)

        r"""
        filenames contains a list of files, one for each cluster in
        the database if nMaxEntries != None and is set to some integer
        value less than n_molecules, then nMaxEntries clusters are
        selected randomly for use.
        """

        if nMaxEntries and nMaxEntries < self.n_molecules:

            self.n_molecules = nMaxEntries
            random.seed(seed)
            self.filenames = random.sample(files, nMaxEntries)

        else:

            self.n_molecules = len(files)
            self.filenames = files

    def len(self) -> int:
        r"""return the number of entries in the database"""

        return self.n_molecules

    def get(self, idx: int) -> Data:

        r"""
        This function loads from file the corresponding data for entry
        idx in the database and returns the corresponding graph read
        from the file
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.filenames[idx]

        molecule_graph = self.graphs.molecule2graph(file_name)

        return molecule_graph

    def get_file_name(self, idx: int) -> str:

        r"""Returns the cluster data file name"""

        return self.filenames[idx]
