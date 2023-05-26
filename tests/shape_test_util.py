"""Util for testing correct handling of shapes"""

from itertools import chain, product
from typing import Iterator, Sequence

from torch import Tensor, broadcast_to


class BroadcastShapeTestingUtil:
    """Namespace for testing utilities for tensors being correctly broadcasted"""
    BATCH_SHAPES = [(1,), (3,), (5,)]
    SPATIAL_SHAPES = [tuple(), (2,), (2, 3)]

    @staticmethod
    def _anchor_dim(n_dims: int, leading_dims: int) -> int:
        return min(leading_dims, n_dims - 1)

    @staticmethod
    def _num_spatial_dimensions(n_dims: int, num_leading_dims: int) -> int:
        return max(n_dims - num_leading_dims - 1, 0)

    @classmethod
    def _broadcast_to_by_leading_dims(
        cls, tensor: Tensor, shape: Sequence[int], num_leading_dims: int
    ) -> Tensor:
        """Broadcasts tensor to shape based on leading dimensions"""
        anchor_dim = cls._anchor_dim(tensor.ndim, num_leading_dims)
        dims_to_add = cls._num_spatial_dimensions(len(shape), num_leading_dims) - cls._num_spatial_dimensions(
            tensor.ndim, num_leading_dims
        )
        new_shape = tensor.shape[: anchor_dim + 1] + (1,) * dims_to_add + tensor.shape[anchor_dim + 1 :]
        if new_shape:
            tensor = tensor.view(new_shape)
        return broadcast_to(tensor, tuple(shape))

    @classmethod
    def expand_tensor_shapes_for_testing(
            cls,
            *tensors: Tensor
        ) -> Iterator[Sequence[Tensor]]:
        """Expand tensor shapes from batch and spatial size

        E.g: Input tensors with shapes (3, 2) and (3, 3), yield:
        (1, 3, 2), (1, 3, 3)
        (5, 3, 2), (5, 3, 3),
        (1, 3, 2, 2), (1, 3, 3, 2)
        ...
        """
        shape_iterator = chain(
            product(
                cls.BATCH_SHAPES,
                cls.SPATIAL_SHAPES),
            [(tuple(), tuple())]
        )
        for batch_shape, spatial_shape in shape_iterator:
            reshaped_tensors = [
                cls._broadcast_to_by_leading_dims(
                    tensor,
                    batch_shape + tuple(tensor.shape) + spatial_shape,
                    tensor.ndim)
                for tensor in tensors
            ]
            yield reshaped_tensors
