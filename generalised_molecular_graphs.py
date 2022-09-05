
from typing import Dict, List

import numpy as np
import torch
from torch_geometric.data import Data
from mendeleev import element

from features import Features
from molecular_graphs import get_dihedral_angle, get_molecule
from molecular_graphs import MolecularGraphs

class GeneralisedMolecularGraphs(MolecularGraphs):

    r"""

    WARNING!!!! The line-graph and line(line-graph) constructed
    below (bond graph and angle graph) are incomplete and do not
    at present allow to correctly propagate information from
    bond angles and dihedral angles back to edges and nodes.

    A class to read molecule information from the database file
    and convert it to torch_geometric Data graph. This is similar
    in many respects to CovalentMolecularGraphs class, in that we
    use the same criterium for defining edges, but different
    in that the graphs created are a generalised form of the
    Data structure, which stores additional arrays for a structural
    description. Specifically, the Data structure defining the graph
    now is augmented to include additional features describing the
    line graph formed by edges as new nodes and bond-angles as
    edges, and optionally a second line graph (line graph of the
    line graph), in which bond angles are nodes and torsional or
    dihedral angles are edges. It is assumed that both line graphs
    are treated as undirected, so every angle is counted twice (e.g.
    in the water molecule there would be two bond angles and zero
    dihedral angles; in the ethane molecule there would be 24 bond
    angles and 18 dihedral angles). In particular we add the following
    to Data:

       num_angles: the number of identified bond angles (twice the
          number of real bond angles in the molecule)
       bond_angle_index[2,num_angles]: the adjacency matrix of the
          line graph formed by edges and bond-angles
       bond_angle_attr[num_angles,n_features]: the attributes of the
          bond angles

       if dihedral features are active (they are optional), Data also
       includes:

       num_dihedral: number of identified dihedral angles (twice the
          number of real dihedral angles, as graph is undirected)
       dihedral_index[2,num_dihedral]
       dihedral_attr[num_dihedral,n_features]

    """

    def __init__(
        self,
        edge_features: Features,
        bond_angle_features: Features,
        dihedral_features: Features = None,
        node_feature_list: List[str] = ['atomic_number'],
        n_total_node_features: int = 10,
        n_max_neighbours: int = 6,
        alpha: float = 1.1,
        pooling: str = "add",
    ) -> None:

        # initialise the base class

        super().__init__(node_feature_list, n_total_node_features, pooling)

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

    def molecule2graph(self, file_name: str) -> Data:

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

        file_name (string): the path to the file where the molecule
           information is stored in file.

        """

        (
            molecule_id,
            n_atoms,
            labels,
            positions,
            properties,
            charges,
        ) = get_molecule(file_name)

        # the total number of node features is given by the species features

        n_features = self.spec_features[labels[0]].size
        node_features = torch.zeros((n_atoms, n_features), dtype=torch.float32)

        for i in range(n_atoms):

            node_features[i, :] = torch.from_numpy(self.spec_features[labels[i]])

        # atoms will be graph nodes; edges will be created for every
        # neighbour of i that is among the nearest
        # n_max_neighbours neighbours of atom i

        # first we loop over all pairs of atoms and calculate
        # the matrix of squared distances

        dij2 = np.zeros((n_atoms, n_atoms), dtype=float)

        for i in range(n_atoms - 1):
            for j in range(i + 1, n_atoms):

                rij = positions[j, :] - positions[i, :]
                rij2 = np.dot(rij, rij)

                dij2[i, j] = rij2
                dij2[j, i] = rij2

        n_max = 12
        n_neighbours = np.zeros((n_atoms), dtype=int)
        neighbour_distance = np.zeros((n_atoms, n_max), dtype=float)
        neighbour_index = np.zeros((n_atoms, n_max), dtype=int)

        bond = -1 * np.ones((n_atoms, n_atoms), dtype=int)
        # this array identifies the bond number linking atoms i and j

        node0 = []
        node1 = []

        edge_length = []

        num_bonds = 0

        for i in range(n_atoms - 1):

            for j in range(i + 1, n_atoms):

                dcut = self.alpha * (
                    self.covalent_radii[labels[i]] + self.covalent_radii[labels[j]]
                )

                dcut2 = dcut * dcut

                if dij2[i, j] <= dcut2:

                    bond[i, j] = num_bonds
                    bond[j, i] = num_bonds

                    num_bonds += 1

                    dij = np.sqrt(dij2[i, j])

                    # features.append(dij)

                    neighbour_distance[i, n_neighbours[i]] = dij
                    neighbour_distance[j, n_neighbours[j]] = dij

                    neighbour_index[i, n_neighbours[i]] = j
                    neighbour_index[j, n_neighbours[j]] = i

                    n_neighbours[i] += 1
                    n_neighbours[j] += 1

                    edge_length.append(dij)

                    node0.append(i)
                    node1.append(j)

        # right now node0 and node1 would give an undirected graph; we need
        # to concatenate to create nodei = node0+node1, nodej=node1+node0 to
        # have an undirected graph

        edge_index = torch.tensor([node0 + node1, node1 + node0], dtype=torch.long)

        _, num_edges = edge_index.shape

        # now obtain the edge features

        bond_features = np.zeros(
            (num_edges, self.edge_features.n_features()), dtype=np.float32
        )

        edge_length = edge_length + edge_length

        for n in range(num_edges):

            bond_features[n, :] = self.edge_features.u_k(edge_length[n])

        # here we will calculate the angle_index array, or in other words, the
        # line graph, where edges in edge_index become nodes, and bond angles
        # become edges; note that the line graph is also undirected, so each
        # edge (bond angle) appears twice

        n_bond_angle = 0
        angles = []
        angle_bond0 = []
        angle_bond1 = []
        angle_bonds = -1 * np.ones((num_bonds, num_bonds), dtype=int)

        for i in range(n_atoms):

            ri = positions[i, :]

            for n in range(n_neighbours[i] - 1):

                j = neighbour_index[i, n]

                rij = positions[j, :] - ri
                dij = neighbour_distance[i, n]

                bond_ij = bond[i, j]

                for m in range(n + 1, n_neighbours[i]):

                    k = neighbour_index[i, m]

                    rik = positions[k, :] - ri
                    dik = neighbour_distance[i, m]

                    bond_ik = bond[i, k]

                    angle_bond0.append(bond_ij)
                    angle_bond1.append(bond_ik)

                    costhetajik = np.dot(rij, rik) / (dij * dik)

                    thetajik = np.arccos(costhetajik)

                    angles.append(thetajik)

                    angle_bonds[bond_ij, bond_ik] = n_bond_angle
                    angle_bonds[bond_ik, bond_ij] = n_bond_angle

        bond_angle_index = torch.tensor(
            [angle_bond0 + angle_bond1, angle_bond1 + angle_bond0], dtype=torch.long
        )

        _, num_angles = bond_angle_index.shape

        bond_angles = angles + angles

        # calculate the bond-angle features

        angle_features = np.zeros(
            (num_angles, self.bond_angle_features.n_features()), dtype=np.float32
        )

        for n in range(num_angles):

            angle_features[n, :] = self.bond_angle_features.u_k(bond_angles[n])

        # finally, dihedral angles

        if self.dihedral_features:

            # rather than looping through edges, because the graph is undirected
            # we'll just go over bonds...

            angles = []

            angle0 = []
            angle1 = []

            for n in range(num_bonds - 1):

                i, j = edge_index[:, n]

                ri = positions[i, :]
                rj = positions[j, :]

                for m in range(n + 1, num_bonds):

                    k, l = edge_index[:, m]

                    if i in (k,l):
                        continue  # all atoms must
                    if j in (k,l):
                        continue  # be distinct

                    rk = positions[k, :]
                    rl = positions[l, :]

                    bond_jk = bond[j, k]
                    bond_jl = bond[j, l]
                    bond_ik = bond[i, k]
                    bond_il = bond[i, l]

                    if bond_jk >= 0:  # i-j-k-l

                        angle0.append(angle_bonds[n, bond_jk])
                        angle1.append(angle_bonds[bond_jk, m])

                        rij = rj - ri
                        rjk = rk - rj
                        rkl = rl - rk

                        thetaijkl = get_dihedral_angle(rij, rjk, rkl)

                        angles.append(thetaijkl)

                    elif bond_jl >= 0:  # i-j-l-k

                        angle0.append(angle_bonds[n, bond_jl])
                        angle1.append(angle_bonds[bond_jl, m])

                        rij = rj - ri
                        rjl = rl - rj
                        rlk = rk - rl

                        thetaijkl = get_dihedral_angle(rij, rjl, rlk)

                        angles.append(thetaijkl)

                    elif bond_ik >= 0:  # j-i-k-l

                        angle0.append(angle_bonds[n, bond_ik])
                        angle1.append(angle_bonds[bond_ik, m])

                        rji = ri - rj
                        rik = rk - ri
                        rkl = rl - rk

                        thetaijkl = get_dihedral_angle(rji, rik, rkl)

                        angles.append(thetaijkl)

                    elif bond_il >= 0:  # j-i-l-k

                        angle0.append(angle_bonds[n, bond_il])
                        angle1.append(angle_bonds[bond_il, m])

                        rji = ri - rj
                        ril = rl - ri
                        rlk = rk - rl

                        thetaijkl = get_dihedral_angle(rji, ril, rlk)

                        angles.append(thetaijkl)

            dihedral_index = torch.tensor(
                [angle0 + angle1, angle1 + angle0], dtype=torch.long
            )

            _, num_dihedral = dihedral_index.shape

            dihedral_angles = angles + angles

            # now obtain the dihedral angle features

            torsion_features = np.zeros(
                (num_dihedral, self.dihedral_features.n_features()), dtype=np.float32
            )

            for n in range(num_dihedral):

                angle = dihedral_angles[n]

                torsion_features[n, :] = self.dihedral_features.u_k(angle)

        # calculate the atomisation energy

        molecule_ref = 0.0

        for n in range(n_atoms):

            molecule_ref += self.atom_ref[labels[n]]

        atomisation_energy = self.Hartree2eV * (properties[0, 10] - molecule_ref)

        # now we can create a graph object (Data)

        edge_attr = torch.from_numpy(bond_features)
        bond_angle_attr = torch.from_numpy(angle_features)

        if self.dihedral_features:
            dihedral_attr = torch.from_numpy(torsion_features)
        else:
            dihedral_attr = None

        if self.pooling == "add":
            y = torch.tensor(atomisation_energy, dtype=torch.float32)
        else:
            y = torch.tensor((atomisation_energy / float(n_atoms)), dtype=torch.float32)

        pos = torch.from_numpy(positions)

        # we will create a bond graph (line-graph) where bonds are nodes and
        # bond angles are edges with appropriate edge features; this will
        # be added to the graph as a new feature

        bond_graph = Data(
            num_nodes=(edge_index.size(1) // 2),
            edge_index=bond_angle_index,
            edge_attr=bond_angle_attr,
        )

        if self.dihedral_features:

            angle_graph = Data(
                num_nodes=(bond_angle_index.size(1) // 2 // 2),
                edge_index=dihedral_index,
                edge_attr=dihedral_attr,
            )

        else:

            angle_graph = Data(num_nodes=0)

        molecule_graph = Data(
            x=node_features,
            y=y,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            line_graph=bond_graph,
            angle_graph=angle_graph,
        )

        return molecule_graph


# register this derived class as a subclass of MolecularGraphs

MolecularGraphs.register(GeneralisedMolecularGraphs)
