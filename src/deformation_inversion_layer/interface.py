"""Interface definitions"""

from abc import ABC, abstractmethod
from typing import Callable

from torch import Tensor


class IInterpolator(ABC):
    """Interpolates values on regular grid in voxel coordinates"""

    @abstractmethod
    def __call__(self, volume: Tensor, coordinates: Tensor) -> Tensor:
        """Interpolate

        Args:
            volume: Volume to be interpolated with shape
                (batch_size, *channel_dims, dim_1, ..., dim_{n_dims}). Dimension
                order is the same as the coordinate order of the coordinates
            coordinates: Interpolation coordinates with shape
                (batch_size, n_dims, *target_shape)

        Returns:
            Interpolated volume with shape (batch_size, *channel_dims, *target_shape)
        """


class IFixedPointSolver(ABC):
    """Interface for fixed point solvers"""

    @abstractmethod
    def solve(
        self,
        fixed_point_function: Callable[[Tensor, Tensor], None],
        initial_value: Tensor,
    ) -> Tensor:
        """Solve fixed point problem

        Args:
            fixed_point_function: Function to be iterated, the function should store
                its output in-place to the Tensor of the second argument
            initial_value: Initial iteration value

        Returns:
            Solution of the fixed point iteration
        """


class IFixedPointStopCriterion(ABC):
    """Defines stopping criterion for fixed point iteration"""

    @abstractmethod
    def should_stop_after(
        self,
        previous_iteration: Tensor,
        current_iteration: Tensor,
        iteration_to_end: int,
    ) -> bool:
        """Return whether iterating should be stopped at end of an iteration

        After first evaluation of the fixed point function iteration_to_end == 0

        Args:
            previous_iteration: Previous output of the fixed point iteration
            current_iteration: Current output of the fixed point iteration
            iteration_to_end: Index of the iteration which ended

        Returns:
            Whether the iteration should be stopped
        """

    @abstractmethod
    def should_stop_before(self, iteration_to_start: int) -> bool:
        """Return whether iterating should be continued at beginning of an iteration

        Before first evaluation of the fixed point function iteration_to_start == 0

        Args:
            iteration_to_start: Index of the iteration being started

        Returns:
            Whether the iteration should be stopped
        """
