"""Tests for fixed point inversion of displacement field"""

from functools import partial
from unittest import TestCase

from torch import Generator, rand, zeros_like
from torch.autograd import gradcheck
from torch.testing import assert_close

from deformation_inversion_layer.fixed_point_invert_deformation import (
    DeformationInversionArguments, fixed_point_invert_deformation)
from deformation_inversion_layer.fixed_point_iteration import (
    AndersonSolver, MaxElementWiseAbsStopCriterion,
    RelativeL2ErrorStopCriterion)
from deformation_inversion_layer.interpolator import LinearInterpolator
from deformation_inversion_layer.interpolator.algorithm import \
    generate_voxel_coordinate_grid


class FixedPointInversionTests(TestCase):
    """Tests for fixed point inversion"""

    def test_composition_zero(self) -> None:
        """Test that composition is zero"""
        from os import environ
        generator = Generator().manual_seed(1337)
        test_input = (2 * rand(1, 2, 5, 5, generator=generator) - 1) * 0.25
        interpolator = LinearInterpolator()
        inverted = fixed_point_invert_deformation(
            test_input,
            arguments=DeformationInversionArguments(
                interpolator=LinearInterpolator(),
                forward_solver=AndersonSolver(
                    stop_criterion=MaxElementWiseAbsStopCriterion(threshold=1e-6)),
                backward_solver=AndersonSolver(
                    stop_criterion=MaxElementWiseAbsStopCriterion(threshold=1e-6))
            )
        )
        composition = interpolator(
            test_input,
            generate_voxel_coordinate_grid((5, 5), test_input.device) + inverted
        ) + inverted
        assert_close(
            composition,
            zeros_like(composition)
        )

    def test_grad(self) -> None:
        """Test that gradients are correct"""
        generator = Generator().manual_seed(1337)
        test_input = (2 * rand(1, 2, 5, 5, generator=generator) - 1) * 0.25
        fixed_point_invert_displacement_field_ = partial(
            fixed_point_invert_deformation,
            arguments=DeformationInversionArguments(
                interpolator=LinearInterpolator(),
                forward_solver=AndersonSolver(
                    stop_criterion=MaxElementWiseAbsStopCriterion(threshold=1e-6)
                ),
                backward_solver=AndersonSolver(
                    stop_criterion=RelativeL2ErrorStopCriterion(threshold=1e-6)
                )
            )
        )
        self.assertTrue(
            gradcheck(
                fixed_point_invert_displacement_field_,
                test_input.double().requires_grad_(),
                eps=1e-5,
                atol=1e-5,
                check_undefined_grad=False
            )
        )
