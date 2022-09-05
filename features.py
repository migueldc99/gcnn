from typing import Dict, Tuple

import numpy as np

r"""
A general class to define features, which can be distance, or
angle-based, or general. Their general expression is adopted
from Xie & Grossman

.. math::
      u_k(x) = \exp\left(-(x - \mu_k)^2/\sigma^2)

where the n_features :math:`\mu_k` points are evenly distributed
between x_min and x_max

Also provided a function that returns instances of different types
of features (edges, bond-angle, dihedral-angle)

"""

class VariableOutOfBounds(Exception):
    pass

class Features:

    r"""

    Args:

       x_min (float): the lower limit of the argument range
       x_max (float): the upper limit
       n_features (int): total number of feature values
       sigma (float): the Gaussian inverse exponent of the features
          (see above)
       norm (bool): if True, the features are normalised in the
           following sense:
           .. math::
              \sum_k^n_features u_k(x) = 1

    """

    def __init__(
        self,
        x_min: float = 0.0,
        x_max: float = 1.0,
        n_features: int = 40,
        sigma: float = 0.2,
        norm: bool = False,
    ) -> None:

        self._x_min = x_min
        self._x_max = x_max
        self._n_features = n_features
        self._sigma = sigma
        self._norm = norm

        self.points = np.linspace(self._x_min, self._x_max, self._n_features)

    def u_k(self, x: float) -> np.ndarray:

        """the edge feature evaluated at all edge feature points"""

        if x < self._x_min or x > self._x_max:
            raise VariableOutOfBounds('Value {} out of bounds.'.format(x))

        val = (x - self.points) / self._sigma
        val2 = -val * val

        feature = np.exp(val2)

        if self._norm:
            norm = np.sum(feature)
            if norm > 0.0:
                feature /= norm

        return feature

    def du_k(self, x: float) -> np.ndarray:

        """the edge feature derivative evaluated at all edge feature points"""

        if x < self._x_min or x > self._x_max:
            raise VariableOutOfBounds('Value {} out of bounds.'.format(x))

        val = (x - self.points) / self._sigma
        val2 = -val * val

        du = -2.0 * val * np.exp(val2) / self._sigma

        return du

    def parameters(self) -> Dict:

        """interrogate the instance about its defining parameters"""

        return {
           "x_min": self._x_min,
           "x_max": self._x_max,
           "n_features": self._n_features,
           "sigma": self._sigma,
           "norm": self._norm,
        }

    def n_features(self) -> int:
        """return number of features"""
        return self._n_features


def set_up_features(input_data: Dict) -> Tuple[Features]:

    r"""

    This function gets passed the input data, searches for the
    definitions of edge, bond angle and dihedral angle feature instances
    and returns them

    """

    # first edge features (features depending only on bond-distance )

    edge_features = input_data.get("EdgeFeatures", None)

    if edge_features is None:

        edges = Features()

    else:

        r_min = edge_features.get("r_min", 0.0)
        r_max = edge_features.get("r_max", 8.0)
        n_edge_features = edge_features.get("n_features", 10)
        sigma = edge_features.get("sigma", 0.2)
        norm = edge_features.get("norm", False)

        edges = Features(
            x_min=r_min, x_max=r_max, n_features=n_edge_features,
            sigma=sigma, norm=norm
        )

    # bond-angle features

    angle_features = input_data.get("AngleFeatures", None)

    if angle_features is None:

        bond_angle = Features(x_min=0.0, x_max=np.pi, norm=True)

    else:

        theta_min = angle_features.get("theta_min", 0.0)
        theta_max = angle_features.get("theta_max", np.pi)
        n_angle = angle_features.get("n_features", 10)
        sigma = angle_features.get("sigma", 0.1)
        norm = angle_features.get("norm", True)

        bond_angle = Features(
            x_min=theta_min, x_max=theta_max, n_features=n_angle,
            sigma=sigma, norm=norm
        )

    # finally, dihedral angle features, if given (if not, return as None)

    dihedral_features = input_data.get("DihedralFeatures", None)

    if dihedral_features:

        theta_min = dihedral_features.get("theta_min", 0.0)
        theta_max = dihedral_features.get("theta_min", np.pi)
        n_dihedral = angle_features.get("n_features", 10)
        sigma = angle_features.get("sigma", 0.1)
        norm = angle_features.get("norm", True)

        dihedral_angle = Features(
            x_min=theta_min, x_max=theta_max, n_features=n_dihedral,
            sigma=sigma, norm=norm
        )

    return edges, bond_angle, dihedral_angle
