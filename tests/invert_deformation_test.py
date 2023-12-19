"""Tests for fixed point inversion of displacement field"""

from functools import partial
from unittest import TestCase

from torch import Generator, Tensor, rand, zeros_like
from torch.autograd import gradcheck
from torch.testing import assert_close

from deformation_inversion_layer.fixed_point_invert_deformation import (
    DeformationInversionArguments,
    fixed_point_invert_deformation,
)
from deformation_inversion_layer.fixed_point_iteration import (
    AndersonSolver,
    MaxElementWiseAbsStopCriterion,
    RelativeL2ErrorStopCriterion,
)
from deformation_inversion_layer.interpolator import LinearInterpolator
from deformation_inversion_layer.interpolator.algorithm import (
    generate_voxel_coordinate_grid,
)


class FixedPointInversionTests(TestCase):
    """Tests for fixed point inversion"""

    def test_composition_zero(self) -> None:
        """Test that composition is zero"""
        generator = Generator().manual_seed(1337)
        test_input = (2 * rand(1, 2, 5, 5, generator=generator) - 1) * 0.25
        interpolator = LinearInterpolator()
        inverted = fixed_point_invert_deformation(
            test_input,
            arguments=DeformationInversionArguments(
                interpolator=LinearInterpolator(),
                forward_solver=AndersonSolver(
                    stop_criterion=MaxElementWiseAbsStopCriterion(threshold=1e-6)
                ),
                backward_solver=AndersonSolver(
                    stop_criterion=MaxElementWiseAbsStopCriterion(threshold=1e-6)
                ),
            ),
        )
        composition = (
            interpolator(
                test_input,
                generate_voxel_coordinate_grid((5, 5), test_input.device) + inverted,
            )
            + inverted
        )
        assert_close(composition, zeros_like(composition))

    def test_displacement_field_grad(self) -> None:
        """Test that gradients are correct when only input displacement field
        requires gradients"""
        generator = Generator().manual_seed(1337)
        test_ddf = (2 * rand(1, 2, 5, 5, generator=generator) - 1) * 0.25
        fixed_point_invert_displacement_field_ = partial(
            fixed_point_invert_deformation,
            arguments=DeformationInversionArguments(
                interpolator=LinearInterpolator(),
                forward_solver=AndersonSolver(
                    stop_criterion=MaxElementWiseAbsStopCriterion(threshold=1e-6)
                ),
                backward_solver=AndersonSolver(
                    stop_criterion=MaxElementWiseAbsStopCriterion(threshold=1e-6)
                ),
            ),
        )
        self.assertTrue(
            gradcheck(
                fixed_point_invert_displacement_field_,
                test_ddf.double().requires_grad_(),
                eps=1e-3,
                atol=1e-5,
                check_undefined_grad=False,
            )
        )

    def test_coordinates_grad(self) -> None:
        """Test that gradients are correct when only coordinates require
        gradients"""
        generator = Generator().manual_seed(1337)
        test_ddf = (2 * rand(1, 2, 5, 5, generator=generator) - 1) * 0.25
        test_coordinates = 2 * rand(1, 2, 2, 2, generator=generator)
        arguments = DeformationInversionArguments(
            interpolator=LinearInterpolator(),
            forward_solver=AndersonSolver(
                stop_criterion=MaxElementWiseAbsStopCriterion(threshold=1e-5)
            ),
            backward_solver=AndersonSolver(
                stop_criterion=RelativeL2ErrorStopCriterion(threshold=1e-5)
            ),
        )

        def fixed_point_invert_displacement_field_(coordinates: Tensor) -> Tensor:
            return fixed_point_invert_deformation(
                displacement_field=test_ddf.double(),
                arguments=arguments,
                coordinates=coordinates,
            )

        self.assertTrue(
            gradcheck(
                fixed_point_invert_displacement_field_,
                test_coordinates.double().requires_grad_(),
                eps=1e-5,
                atol=1e-5,
                check_undefined_grad=False,
            )
        )

    def test_both_grad(self) -> None:
        """Test that gradients are correct when both input displacement field
        and coordinates require gradients"""
        generator = Generator().manual_seed(1337)
        test_ddf = (2 * rand(1, 2, 5, 5, generator=generator) - 1) * 0.25
        test_coordinates = 2 * rand(1, 2, 2, 2, generator=generator)
        arguments = DeformationInversionArguments(
            interpolator=LinearInterpolator(),
            forward_solver=AndersonSolver(
                stop_criterion=MaxElementWiseAbsStopCriterion(threshold=1e-5)
            ),
            backward_solver=AndersonSolver(
                stop_criterion=RelativeL2ErrorStopCriterion(threshold=1e-5)
            ),
        )

        def fixed_point_invert_displacement_field_(
            displacement_field: Tensor, coordinates: Tensor
        ) -> Tensor:
            return fixed_point_invert_deformation(
                displacement_field=displacement_field,
                arguments=arguments,
                coordinates=coordinates,
            )

        self.assertTrue(
            gradcheck(
                fixed_point_invert_displacement_field_,
                (
                    test_ddf.double().requires_grad_(),
                    test_coordinates.double().requires_grad_(),
                ),
                eps=1e-5,
                atol=1e-5,
                check_undefined_grad=False,
            )
        )
