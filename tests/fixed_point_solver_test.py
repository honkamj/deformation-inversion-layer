"""Tests for fixed point solvers"""

from unittest import TestCase

from torch import Tensor, ones_like, rand, sqrt
from torch.testing import assert_close

from deformation_inversion_layer.fixed_point_iteration import (
    AndersonSolver,
    MaxElementWiseAbsStopCriterion,
)


class FixedPointSolverTests(TestCase):
    """Tests for fixed point solvers"""

    def test_anderson_solver(self):
        """Test that anderson solver converges to correct value"""

        def func(tensor: Tensor, out: Tensor) -> None:
            out[:] = (7 / tensor + tensor) / 2

        initial_value = rand(5, 10, 10) + 4
        assert_close(
            sqrt(7 * ones_like(initial_value)),
            AndersonSolver(
                stop_criterion=MaxElementWiseAbsStopCriterion(threshold=1e-6)
            ).solve(func, initial_value=initial_value),
        )
