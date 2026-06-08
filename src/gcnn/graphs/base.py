
from abc import ABC, abstractmethod
import re
from typing import Dict, List, Tuple

import numpy as np
from torch_geometric.data import Data
from mendeleev import element
from scipy.constants import physical_constants

# first some useful functions

def get_dihedral_angle(
        rki: np.ndarray, rij: np.ndarray, rjl: np.ndarray
) -> float:

    r"""
    given vectors rki, rij, rjl, defined in the
    moiety k-i-j-l, it returns the dihedral angle
    """

    # WARNING: Avoid using np.cross: this can cause a numerical error
    # by sometimes resulting in abs(cosphi)= 1 + delta, delta ~ 1.0e-16
    # which is sufficient to cause np.arccos to fail.
    # vkiij = np.cross( rki, rij )
    vkiij0 = rki[1] * rij[2] - rki[2] * rij[1]
    vkiij1 = rki[2] * rij[0] - rki[0] * rij[2]
    vkiij2 = rki[0] * rij[1] - rki[1] * rij[1]

    vkiij = np.array([vkiij0, vkiij1, vkiij2], dtype=np.float)

    nkiij = np.sqrt(np.dot(vkiij, vkiij))

    # vijjl = np.cross( rij, rjl )
    vijjl0 = rij[1] * rjl[2] - rij[2] * rjl[1]
    vijjl1 = rij[2] * rjl[0] - rij[0] * rjl[2]
    vijjl2 = rij[0] * rjl[1] - rij[1] * rjl[0]

    vijjl = np.array([vijjl0, vijjl1, vijjl2], dtype=np.float)

    nijjl = np.sqrt(np.dot(vijjl, vijjl))

    cosphi = np.dot(vkiij, vijjl) / (nkiij * nijjl)

    phi = np.arccos(cosphi)

    return phi

def get_molecule(
    file_name: str
) -> Tuple[int, int, List[str], float, float, float]:

    r"""
    this script opens a file of the GDB-9 database and processes it,
    returning the molecule structure in xyz format, a molecule identifier
    (tag), and a vector containing the entire list of molecular properties

    Args:

        file_name (str): filename containing the molecular information

    returns:

        molecule_id (int): integer identifying the molecule number
        in the database n_atoms (int): number of atoms in the molecule
        species (List[str]): the species of each atom (len = n_atoms)
        coordinates (np.array(float)[n_atoms,3]): atomic positions
        properties (np.array(float)[:]): molecular properties, see
        database docummentation charge (np.array(float)[n_atoms]):
        Mulliken charges of atoms

    """

    with open(file_name, "r") as file_in:
        lines = file_in.readlines()

    n_atoms = int(lines[0])  # number of atoms is specified in 1st line

    words = lines[1].split()

    molecule_id = int(words[1])

    molecular_data = np.array(words[2:], dtype=float)

    species = []  # species label
    coordinates = np.zeros((n_atoms, 3), dtype=float)  # coordinates in Angstrom
    charge = np.zeros((n_atoms), dtype=float)  # Mulliken charges (e)

    # below extract chemical labels, coordinates and charges

    m = 0

    for n in range(2, n_atoms + 2):

        line = re.sub(
            r"\*\^", "e", lines[n]
        )  # this prevents stupid exponential lines in the data base

        words = line.split()

        species.append(words[0])

        x = float(words[1])
        y = float(words[2])
        z = float(words[3])

        c = float(words[4])

        coordinates[m, :] = x, y, z

        charge[m] = c

        m += 1

    # finally obtain the vibrational frequencies, in cm^-1

    frequencies = np.array(lines[n_atoms + 2].split(), dtype=float)

    # we pack all the molecular data into a single array of properties

    properties = np.expand_dims(
       np.concatenate((molecular_data, frequencies)), axis=0
    )

    return molecule_id, n_atoms, species, coordinates, properties, charge

# now abstract base class MolecularGraphs

class MolecularGraphs(ABC):

    r"""

    This is an Abstract Base Class (abc) that allows easy construction
    of derived classes implementing different strategies to turn
    molecule structural information into a graph. The base class
    implement two helper functions:

    generate_node_features: sets up the arrays of node features for each
          chemical species

    get_molecule: given a file name storing a molecule data, it reads
          the data; used by molecule2graph

    Derived classes need to implement method molecule2graph, that
    takes as input an input file containing the molecular information
    and returns a torch_geometric.data Data object (graph) constructed
    according to the molecule2graph implementation.

    """

    def __init__(
        self,
        node_feature_list: List[str],
        n_total_node_features: int = 10,
        pooling: str = "add",
    ) -> None:

        r"""

        Initialises an instance of the class. It needs to be passed the list
        of chemical species that may be found in the training files, the
        chemical feature list of each node, which is a list of data
        that Mendeleev can understand (see below), and a total number for
        the node features; besides the chemical/physical features, nodes
        can be assigned initial numerical features that are specific to each
        species and that have no physico-chemical significance (see
        generate_node_features() below for details)

        Args:

           species: a list of species for which nodeFeatures are desired;
             this can be an array of atomic numbers (ints)
             of an array of standard chemical labels (characters)

           node_feature_list (List[str], default empty): contains
             mendeleev data commands to select the required features
             for each feature e.g. node_feature_list = ['atomic_number',
             'atomic_radius', 'covalent_radius']

           n_total_node_features (int, 10): the total number of features per
             node, counting those listed in node_feature_list (if any) and
             the numerical ones.

           pooling (str = 'add' ): indicates the type of pooling to be
             done over nodes to estimate the fitted property; usually
             this will be 'add', meaning that the property prediction
             is done by summing over nodes; the only other contemplated
             case is 'mean', in which case the prediction is given by
             averaging over nodes. WARNING: this must be done in a
             concerted way (the same) in the GNN model definition!

        """

        self.species = ['H', 'C', 'N', 'O', 'F']

        self.node_feature_list = node_feature_list

        self.n_total_node_features = n_total_node_features

        self.spec_features = self.generate_node_features()

        self.pooling = pooling

        # define a conversion factor from Hartrees to eV
        self.Hartree2eV = physical_constants["Hartree energy in eV"][0]

        r"""
        The following dictionary defines the atomic ref energies, needed
        to calculate atomisation energies. These values are in Hartree
        """

        self.atom_ref = dict(
            H=-0.500273, C=-37.846772, N=-54.583861, O=-75.064579, F=-99.718730
        )

    def generate_node_features(self) -> Dict[str, float]:

        r"""

        This function generates initial node features for a molecular graph

        It returns a dictionary where the keys are the chemical symbol and
        the values are an array of dimensions (n_node_features), such that
        initial nodes in the graph can have their features filled according
        to their species. The array of features thus created is later used
        in molecule2graph to generate the corresponding molecular graph.

        """

        n_species = len(self.species)
        n_features = len(self.node_feature_list)
        n_node_features = self.n_total_node_features - n_features

        # generate an element object for each species

        spec_list = []

        for spec in self.species:
            spec_list.append(element(spec))

        x = np.pi * np.linspace(0.0, 1.0, n_node_features)

        factor = np.ones((n_features), dtype=float)

        for n in range(n_features):

            if re.search("radius", self.node_feature_list[n]):
                # if feature is a distance, convert from pm to Ang
                factor[n] = 0.01

        # now we can loop over each individual species, create its feature
        # vector and store it in spec_features

        # we want to have node features normalised in the range [-1:1]

        values = np.zeros((n_features, n_species), dtype=float)

        for n, spec in enumerate(spec_list):

            for m in range(n_features):

                command = "spec." + self.node_feature_list[m]
                values[m, n] = eval(command)

        # now detect the maximum and minimum values for each feature
        # over the list of species we have

        features_max = np.zeros(n_features, dtype=float)
        features_min = np.zeros(n_features, dtype=float)

        for m in range(n_features):
            features_max[m] = np.max(values[m, :])
            features_min[m] = np.min(values[m, :])

        # normalise values

        for n in range(n_species):
            for m in range(n_features):

                values[m, n] = (
                    2.0
                    * (values[m, n] - features_min[m])
                    / (features_max[m] - features_min[m])
                    - 1.0
                )

        spec_features = {}

        for n, spec in enumerate(spec_list):

            amplitude = 0.1 * float(spec.period)
            group = float(spec.group_id)

            vec2 = amplitude * np.sin(group * x)

            spec_features[spec.symbol] = np.concatenate((values[:, n], vec2))

        # we are done

        return spec_features

    @abstractmethod
    def molecule2graph(self, file_name: str) -> Data:

        r"""

        This method must be implemented in derived classes. Its purpose
        is to take a file-name as input containing molecular structural
        information and return a Data (graph) object representing the
        same molecule.

        Args:

          file_name (string): path to file containing molecule data

        returns: molecular graph (Data instance)

        """

        raise NotImplementedError
