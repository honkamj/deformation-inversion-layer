"""Fixed point deformation inversion function"""

from functools import partial
from typing import Optional

from torch import Tensor
from torch import dtype as torch_dtype
from torch import enable_grad, zeros_like
from torch.autograd import grad
from torch.autograd.function import Function, FunctionCtx, once_differentiable

from .fixed_point_iteration import (AndersonSolver,
                                    MaxElementWiseAbsStopCriterion,
                                    RelativeL2ErrorStopCriterion)
from .interface import IFixedPointSolver, IInterpolator
from .interpolator import LinearInterpolator
from .interpolator.algorithm import generate_voxel_coordinate_grid


class DeformationInversionArguments:
    """Arguments for deformation fixed point inversion

    Args:
        interpolator: Interpolator with which to interpolate the input
            displacement field
        forward_solver: Fixed point solver for the forward pass
        backward_solver: Fixed point solver for the backward pass
        forward_dtype: Data type to use for the solver in the forward pass, by default
            the data type of the input is used
        backward_dtype: Data type to use for the solver in the backward pass, by default
            the data type of the input is used
    """

    def __init__(
            self,
            interpolator: Optional[IInterpolator] = None,
            forward_solver: Optional[IFixedPointSolver] = None,
            backward_solver: Optional[IFixedPointSolver] = None,
            forward_dtype: Optional[torch_dtype] = None,
            backward_dtype: Optional[torch_dtype] = None
        ) -> None:
        self.interpolator = LinearInterpolator() if interpolator is None else interpolator
        self.forward_solver = AndersonSolver(
            stop_criterion=MaxElementWiseAbsStopCriterion(),
        ) if forward_solver is None else forward_solver
        self.backward_solver = AndersonSolver(
            stop_criterion=RelativeL2ErrorStopCriterion(),
        ) if backward_solver is None else backward_solver
        self.forward_dtype = forward_dtype
        self.backward_dtype = backward_dtype


def fixed_point_invert_deformation(
    displacement_field: Tensor,
    arguments: Optional[DeformationInversionArguments] = None,
    initial_guess: Optional[Tensor] = None,
) -> Tensor:
    """Fixed point invert displacement field

    Args:
        displacement_field: Displacement field describing the deformation to invert with shape
            (batch_size, n_channels, dim_1, ..., dim_{n_dims})
        arguments: Arguments for fixed point inversion
        initial_guess: Initial guess for inverted displacement field, if not given, negative
            of the displacement field is used
    
    Returns:
        Inverted displacement field with shape (batch_size, n_channels, dim_1, ..., dim_{n_dims})
    """
    return _FixedPointInvertDisplacementField.apply(
        displacement_field,
        DeformationInversionArguments() if arguments is None else arguments,
        initial_guess
    )


class _FixedPointInvertDisplacementField(Function):
    @staticmethod
    def _forward_fixed_point_mapping(
        inverted_displacement_field: Tensor,
        displacement_field: Tensor,
        interpolator: IInterpolator,
        voxel_coordinate_grid: Tensor,
    ) -> Tensor:
        return -interpolator(
            volume=displacement_field,
            coordinates=voxel_coordinate_grid + inverted_displacement_field,
        )

    @staticmethod
    def _backward_fixed_point_mapping(
        vjp_estimate: Tensor,
        inverted_displacement_field: Tensor,
        forward_fixed_point_output: Tensor,
        grad_output: Tensor,
    ) -> Tensor:
        return (
            grad(
                outputs=forward_fixed_point_output,
                inputs=inverted_displacement_field,
                grad_outputs=vjp_estimate,
                retain_graph=True,
            )[0]
            + grad_output
        )

    @staticmethod
    def forward(  # type: ignore # pylint: disable=arguments-differ
        ctx: FunctionCtx,
        displacement_field: Tensor,
        arguments: DeformationInversionArguments,
        initial_guess: Optional[Tensor],
    ):
        dtype = (
            displacement_field.dtype if arguments.forward_dtype is None else arguments.forward_dtype
        )
        type_converted_displacement_field = displacement_field.to(dtype)
        inverted_displacement_field = arguments.forward_solver.solve(
            partial(
                _FixedPointInvertDisplacementField._forward_fixed_point_mapping,
                displacement_field=type_converted_displacement_field,
                voxel_coordinate_grid=generate_voxel_coordinate_grid(
                    displacement_field.shape[2:], displacement_field.device, dtype=dtype
                ),
                interpolator=arguments.interpolator,
            ),
            initial_value=(
                -type_converted_displacement_field if initial_guess is None else initial_guess
            ),
        ).to(displacement_field.dtype)
        grad_needed, _, _ = ctx.needs_input_grad  # type: ignore
        if grad_needed:
            ctx.save_for_backward(displacement_field, inverted_displacement_field)
            ctx.arguments = arguments  # type: ignore
            ctx.dtype = dtype  # type: ignore
        return inverted_displacement_field

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):  # type: ignore # pylint: disable=arguments-differ
        grad_needed, _, _ = ctx.needs_input_grad
        if grad_needed:
            displacement_field: Tensor
            inverted_displacement_field: Tensor
            (
                displacement_field,
                inverted_displacement_field,
            ) = ctx.saved_tensors
            arguments: DeformationInversionArguments = ctx.arguments
            del ctx
            dtype = (
                displacement_field.dtype
                if arguments.backward_dtype is None
                else arguments.backward_dtype
            )
            original_dtype = displacement_field.dtype
            displacement_field = displacement_field.to(dtype).detach()
            inverted_displacement_field = inverted_displacement_field.to(dtype).detach()
            grad_output = grad_output.to(dtype)
            if arguments.backward_solver is None:
                raise RuntimeError("Backward solver not specified!")
            with enable_grad():
                displacement_field.requires_grad_(True)
                inverted_displacement_field.requires_grad_(True)
                forward_fixed_point_output = (
                    _FixedPointInvertDisplacementField._forward_fixed_point_mapping(
                        inverted_displacement_field=inverted_displacement_field,
                        displacement_field=displacement_field,
                        interpolator=arguments.interpolator,
                        voxel_coordinate_grid=generate_voxel_coordinate_grid(
                            displacement_field.shape[2:], displacement_field.device, dtype=dtype
                        ),
                    )
                )
                displacement_field.requires_grad_(False)
                fixed_point_solved_gradient = arguments.backward_solver.solve(
                    partial(
                        _FixedPointInvertDisplacementField._backward_fixed_point_mapping,
                        inverted_displacement_field=inverted_displacement_field,
                        forward_fixed_point_output=forward_fixed_point_output,
                        grad_output=grad_output,
                    ),
                    initial_value=zeros_like(grad_output),
                )
                displacement_field.requires_grad_(True)
                inverted_displacement_field.requires_grad_(False)
                output_grad = grad(
                    outputs=forward_fixed_point_output,
                    inputs=displacement_field,
                    grad_outputs=fixed_point_solved_gradient,
                    retain_graph=False,
                )[0]
                return output_grad.to(original_dtype), None, None
        return None, None, None
