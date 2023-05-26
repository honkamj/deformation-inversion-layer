"""Tests for dense deformation primitives"""

from unittest import TestCase

from shape_test_util import BroadcastShapeTestingUtil
from torch import device, rand, tensor
from torch.testing import assert_close

from deformation_inversion_layer.interpolator.algorithm import (
    generate_voxel_coordinate_grid, interpolate)


class InterpolationTests(TestCase):
    """Tests for interpolation"""

    GRID_SHAPES = (
        (1, 2, 2, 2),
        (1, 3, 15),
        (3, 3, 15, 16),
        (3, 2, 15, 16, 17),
        (3, 2, 15, 16, 17),
        (3, 2, 15, 16, 17),
        (2,)
    )
    VOLUME_SHAPES = (
        (1, 2, 2, 2),
        (2, 5, 13, 14, 15),
        (3, 2, 13, 14, 15),
        (1, 2, 13, 14),
        (1, 13, 14),
        (3, 5, 7, 13, 14),
        (3, 5, 7, 13, 14),
    )
    TARGET_SHAPES = (
        (1, 2, 2, 2),
        (2, 5, 15),
        (3, 2, 15, 16),
        (3, 2, 15, 16, 17),
        (3, 15, 16, 17),
        (3, 5, 7, 15, 16, 17),
        (3, 5, 7)
    )
    VOLUME = tensor(
        [
            [
                [1.0, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]
            ],
            [
                [13, 14, 15, 16],
                [17, 18, 19, 20],
                [21, 22, 23, 24]
            ]
        ]
    )
    POINTS = (
        tensor([0.5, 1.5, 2.0]),
        tensor([0.0, 0.0, 0.0]),
        tensor([1.0, 2.0, 3.0])
    )
    VALUES = (
        tensor(
            (7 + 11 + 19 + 23) / 4
        ),
        tensor(1.0),
        tensor(24.0)
    )

    def test_consistency_with_grid_generation(self) -> None:
        """Check that interpolation methods are consistent with grid generation"""
        shape = (15, 16, 17)
        voxel_grid = generate_voxel_coordinate_grid(shape, device('cpu'))
        assert_close(
            interpolate(voxel_grid, grid=voxel_grid),
            voxel_grid
        )

    def test_shape_consistency_for_interpolaton(self) -> None:
        """Check that shapes produced are correct"""
        for grid_shape, volume_shape, target_shape in zip(
                self.GRID_SHAPES,
                self.VOLUME_SHAPES,
                self.TARGET_SHAPES):
            grid = rand(*grid_shape) * 30
            volume = rand(*volume_shape)
            interpolated = interpolate(volume, grid=grid)
            assert_close(
                interpolated.shape,
                target_shape
            )

    def test_correct_values_generated(self) -> None:
        """Check that correct values are interpolated with different shapes"""
        for n_channels in range(1, 3):
            for grid, target in zip(self.POINTS, self.VALUES):
                target = target.expand(n_channels)
                volume = self.VOLUME.expand(n_channels, *self.VOLUME.shape)
                for grid, target in\
                        BroadcastShapeTestingUtil.expand_tensor_shapes_for_testing(
                            grid,
                            target):
                    if target.ndim > 1:
                        batched_volume = volume.expand(target.size(0), *volume.shape)
                    else:
                        batched_volume = volume[None]
                        target = target[None]
                    assert_close(
                        interpolate(batched_volume, grid),
                        target)

    def test_correct_values_generated_without_channels(self) -> None:
        """Check that correct values are interpolated with different shapes
        when volume has no channels"""
        for grid, target in zip(self.POINTS, self.VALUES):
            for grid, target in\
                    BroadcastShapeTestingUtil.expand_tensor_shapes_for_testing(
                        grid,
                        target):
                if target.ndim > 0:
                    batched_volume = self.VOLUME.expand(target.size(0), *self.VOLUME.shape)
                else:
                    batched_volume = self.VOLUME[None]
                    target = target[None]
                assert_close(
                    interpolate(batched_volume, grid),
                    target)
