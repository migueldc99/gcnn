
from typing import Dict, List

import numpy as np
import torch
from torch_geometric.data import Data
from mendeleev import element

from features import Features
from molecular_graphs import get_dihedral_angle, get_molecule
from molecular_graphs import MolecularGraphs


class CovalentMolecularGraphs(MolecularGraphs):

    r"""

    A class to read molecule information from the database file
    and convert it to torch_geometric Data graph. In this class, graphs
    are constructed in a chemically intuitive way: a node (atom) has edges
    only to other nodes that are at a distance that is up to alpha times the
    sum of their respective covalent radii away, where alpha is
    a factor >= 1 (default 1.1). In this mode edges will correspond
    to chemical bonds. Covalent radii are extracted from Mendeleev for
    each listed species.

    """

    def __init__(
        self,
        edge_features: Features,
        bond_angle_features: Features,
        dihedral_features: Features = None,
        nodeFeatureList: List[str] = ['atomic_number'],
        nTotalNodeFeatures: int = 10,
        n_max_neighbours: int = 6,
        alpha: float = 1.1,
        pooling: str = "add",
    ) -> None:

        # initialise the base class

        super().__init__(nodeFeatureList, nTotalNodeFeatures, pooling)

        self.edge_features = edge_features
        self.bond_angle_features = bond_angle_features
        self.dihedral_features = dihedral_features
        self.n_max_neighbours = n_max_neighbours
        self.alpha = alpha  # alpha is the scaling factor for bond (edge)
        # critera, i.e. two atoms are bonded if their
        # separation is r <= alpha*(rc1 + rc2), where
        # rci are the respective covalent radii

        self.covalent_radii = self.get_covalent_radii()

    def get_covalent_radii(self) -> Dict[str, float]:

        r"""

        sets up and returns a dictionary of covalent radii (in Ang)
        for the list of species in its argument

        returns:

          covalent_radii: dict of covalent radius for eash species (in Angstrom)

        """

        covalent_radii = {}

        for label in self.species:

            spec = element(label)

            covalent_radii[label] = spec.covalent_radius / 100.0
            # mendeleev stores radii in pm, hence the factor

        return covalent_radii

    def molecule2graph(self, fileName: str) -> Data:

        r"""

        A function to read molecule information from the database file and
        convert it to torch_geometric Data graph. In this particular class
        graphs are constructed in the following way:

        Chemically intuitive way: a node (atom) has edges only to
           other nodes that are at a distance that is up to alpha times the
           sum of their respective covalent radii away. In this mode
           edges will correspond to chemical bonds. To activate this
           mode it is necessary to pass the dictionary covalent_radii;
           if it is not passed or is set to None, the second mode is
           activated (see below).

        Args:

        fileName (string): the path to the file where the molecule
           information is stored in file.

        """

        (
            molecule_id,
            n_atoms,
            labels,
            positions,
            properties,
            charges,
        ) = get_molecule(fileName)

        # the total number of node features is given by the species features

        n_features = self.spec_features[labels[0]].size
        node_features = torch.zeros((n_atoms, n_features), dtype=torch.float32)

        # atoms will be graph nodes; edges will be created for every
        # neighbour of i that is among the nearest
        # n_max_neighbours neighbours of atom i

        # first we loop over all pairs of atoms and calculate the matrix
        # of squared distances

        dij2 = np.zeros((n_atoms, n_atoms), dtype=float)

        for i in range(n_atoms - 1):
            for j in range(i + 1, n_atoms):

                rij = positions[j, :] - positions[i, :]
                rij2 = np.dot(rij, rij)

                dij2[i, j] = rij2
                dij2[j, i] = rij2

        n_max = 12 # expected maximum number of neighbours bonded to node
        n_neighbours = np.zeros((n_atoms), dtype=int)
        neighbour_distance = np.zeros((n_atoms, n_max), dtype=float)
        neighbour_index = np.zeros((n_atoms, n_max), dtype=int)

        node0 = []
        node1 = []

        for i in range(n_atoms - 1):

            for j in range(i + 1, n_atoms):

                dcut = self.alpha * (
                    self.covalent_radii[labels[i]] + self.covalent_radii[labels[j]]
                )

                dcut2 = dcut * dcut

                if dij2[i, j] <= dcut2:

                    node0.append(i)
                    node1.append(j)

                    node0.append(j)
                    node1.append(i)

                    dij = np.sqrt(dij2[i, j])

                    neighbour_distance[i, n_neighbours[i]] = dij
                    neighbour_distance[j, n_neighbours[j]] = dij

                    neighbour_index[i, n_neighbours[i]] = j
                    neighbour_index[j, n_neighbours[j]] = i

                    n_neighbours[i] += 1
                    n_neighbours[j] += 1

        edge_index = torch.tensor([node0, node1], dtype=torch.long)

        _, num_edges = edge_index.shape

        # construct node geometric features; the name is confusing, as these will really
        # be edge features; the node features consist only of species properties

        n_ba_features = self.bond_angle_features.n_features()

        node_geometric_features = np.zeros((n_atoms, n_ba_features))

        for i in range(n_atoms):

            anglehist = np.zeros((n_ba_features), dtype=float)

            for n in range(n_neighbours[i] - 1):

                j = neighbour_index[i, n]

                rij = positions[j, :] - positions[i, :]
                dij = neighbour_distance[i, n]

                for m in range(n + 1, n_neighbours[i]):

                    k = neighbour_index[i, m]

                    rik = positions[k, :] - positions[i, :]
                    dik = neighbour_distance[i, m]

                    costhetaijk = np.dot(rij, rik) / (dij * dik)

                    thetaijk = np.arccos(costhetaijk)

                    anglehist += self.bond_angle_features.u_k(thetaijk)

            node_geometric_features[i, :] = anglehist
            node_features[i, :] = torch.from_numpy(self.spec_features[labels[i]])

        # now, based on the edge-index information, we can construct the edge attributes

        bond_features = []

        for n in range(num_edges):

            i = edge_index[0, n]
            j = edge_index[1, n]

            dij = np.sqrt(dij2[i, j])

            features = np.concatenate(
                (
                    self.edge_features.u_k(dij),
                    node_geometric_features[i, :],
                    node_geometric_features[j, :],
                )
            )

            bond_features.append(features)

        # finally, if we have attributes for dihedral angles to be added to edges..

        if self.dihedral_features:

            n_da_features = self.dihedral_features.n_features()

            for n in range(num_edges):

                i = edge_index[0, n]
                j = edge_index[1, n]

                ri = positions[i, :]
                rj = positions[j, :]

                rij = rj - ri

                anglehist = np.zeros((n_da_features), dtype=float)

                for ni in range(n_neighbours[i]):

                    k = neighbour_index[i, ni]

                    if k == j:
                        continue  # k must be different from j

                    rki = ri - positions[k, :]

                    for nj in range(n_neighbours[j]):

                        l = neighbour_index[j, nj]

                        # cannot define a dihedral if l == k, or l == i
                        if l in (k, i):
                            continue

                        rjl = positions[l, :] - rj

                        thetaijkl = get_dihedral_angle(rki, rij, rjl)

                        anglehist += self.dihedral_features.u_k(thetaijkl)

                total_bond_features = np.concatenate((bond_features[n], anglehist))

                bond_features[n] = total_bond_features

        # it is apparently faster to convert numpy arrays to tensors than
        # to convert arrays of numpys, so let's do it this way

        features = np.array(bond_features)

        # calculate the atomisation energy

        molecule_ref = 0.0

        for n in range(n_atoms):

            molecule_ref += self.atom_ref[labels[n]]

        atomisation_energy = self.Hartree2eV * (properties[0, 10] - molecule_ref)

        # now we can create a graph object (Data)

        edge_attr = torch.tensor(features, dtype=torch.float32)

        if self.pooling == "add":
            y = torch.tensor(atomisation_energy, dtype=torch.float32)
        else:
            y = torch.tensor((atomisation_energy / float(n_atoms)), dtype=torch.float32)

        pos = torch.from_numpy(positions)

        molecule_graph = Data(
            x=node_features, y=y, edge_index=edge_index, edge_attr=edge_attr, pos=pos
        )

        return molecule_graph


# register this derived class as subclass of MolecularGraphs

MolecularGraphs.register(CovalentMolecularGraphs)
