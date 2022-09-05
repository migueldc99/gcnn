
from typing import List

from features import Features
from covalent_molecular_graphs import CovalentMolecularGraphs
from generalised_molecular_graphs import GeneralisedMolecularGraphs
from geometric_molecular_graphs import GeometricMolecularGraphs
from molecular_graphs import MolecularGraphs

def set_up_molecular_graphs(
    graph_type: str,
    edge_features: Features,
    bond_angle_features: Features,
    dihedral_features: Features = None,
    node_feature_list: List[str] = ['atomic_number'],
    n_total_node_features: int = 10,
    n_max_neighbours: int = 6,
    alpha: float = 1.1,
    pooling: str = "add",
) -> MolecularGraphs:

    r"""

    A MolecularGraphs factory.

      graph_type (str) specifies the type of graph to be constructed:

      graph_type = 'geometric': a geometric graph construction in which
                  edges are set up between the nearest n_max_neighbours of
                  every node; n_max_neighbours is really a minimum number of
                  neighbours, because the graph must be undirected, so it may
                  be that additional neighbours are added in order to
                  ensure undirectedness.

      graph_type = 'covalent': this is the 'chemical' graph rep., in
                  which an edge corresponds to a chemical bond; edges
                  are placed between nodes separated by a distance
                  equal or smaller than the sum of covalent radii
                  times alpha, i.e. rij < alpha(rci + rcj); again the
                  graph is undirected, so every bond is represented as
                  two edges.

      graph_type = 'generalised': this is different to the previous two, in
                  that the graph contains also a line graph for the
                  bond angles (identified by bond_angle_index and
                  bond_angle_attr), and optionally, a second lineline
                  graph for dihedral angles. This is as yet experimental.

    """

    if graph_type == "geometric":

        graphs = GeometricMolecularGraphs(
            edge_features,
            bond_angle_features,
            dihedral_features,
            node_feature_list,
            n_total_node_features,
            n_max_neighbours,
            pooling,
        )

    elif graph_type == "covalent":

        graphs = CovalentMolecularGraphs(
            edge_features,
            bond_angle_features,
            dihedral_features,
            node_feature_list,
            n_total_node_features,
            n_max_neighbours,
            alpha,
            pooling,
        )

    elif graph_type == "generalised":

        graphs = GeneralisedMolecularGraphs(
            edge_features,
            bond_angle_features,
            dihedral_features,
            node_feature_list,
            n_total_node_features,
            n_max_neighbours,
            alpha,
            pooling,
        )

    else:  # we make this the default case

        graphs = CovalentMolecularGraphs(
            species,
            edge_features,
            bond_angle_features,
            dihedral_features,
            node_feature_list,
            n_total_node_features,
            n_max_neighbours,
            alpha,
            pooling,
        )

    return graphs
