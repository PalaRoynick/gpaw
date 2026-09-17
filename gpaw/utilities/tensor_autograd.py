"""Eager tensor operations with a small reverse-mode result graph."""

from __future__ import annotations

import numpy as np

def _split_einsum(subscripts: str) -> tuple[str, str, str]:
    """Split one explicit binary einsum into left, right, and output labels."""
    inputs, output = subscripts.replace(' ', '').split('->')
    left, right = inputs.split(',')
    return left, right, output


def _labels(subscripts: str) -> str:
    """Return explicit result labels without NumPy's ellipsis notation."""
    return subscripts.replace('...', '')


def _value(operand):
    """Return an operation result's array or an ordinary operand unchanged."""
    if isinstance(operand, TensorResult):
        return operand.value
    return operand


def _sum_to_shape(value, shape):
    """Undo broadcasting before passing a derivative to its operand."""
    while value.ndim > len(shape):
        value = value.sum(axis=0)
    axes = tuple(axis for axis, (actual, wanted) in enumerate(
        zip(value.shape, shape)) if wanted == 1 and actual != 1)
    if axes:
        value = value.sum(axis=axes, keepdims=True)
    return value.reshape(shape)


class TensorResult:
    """Forward tensor value whose instance implements its reverse rule."""

    __array_priority__ = 1000

    def __init__(self, engine: 'Engine', value, indices: str):
        self.engine = engine
        self.value = value
        self.indices = indices
        self.parents: tuple[TensorResult, ...] = ()

    def backward(self, gradient=None) -> None:
        """Propagate one reverse pass through the reachable result graph.

        Nodes are topologically ordered first.  Contributions from every
        downstream use are accumulated before a node's local reverse rule is
        evaluated, so a shared branch is visited exactly once.
        """
        if gradient is None:
            gradient = self.engine.xp.ones_like(self.value)

        order = []
        visited = set()

        def visit(node):
            identity = id(node)
            if identity in visited:
                return
            visited.add(identity)
            for parent in node.parents:
                visit(parent)
            order.append(node)

        visit(self)
        gradients = {id(self): gradient}
        for node in reversed(order):
            node_gradient = gradients.pop(id(node))
            for parent, contribution in node._backward(node_gradient):
                identity = id(parent)
                previous = gradients.get(identity)
                if previous is None:
                    gradients[identity] = contribution
                else:
                    gradients[identity] = previous + contribution

    def _backward(self, gradient):
        """Return ``(parent, cotangent)`` pairs for this operation's VJP.

        Subclasses implement only this local derivative.  Graph traversal,
        shared-node accumulation, and ordering belong to ``backward()``.
        """
        raise NotImplementedError

    def _logical_key(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        batch_rank = self.value.ndim - len(self.indices)
        return (slice(None),) * batch_rank + key

    def __getitem__(self, key):
        """Index logical tensor axes while retaining implicit batch axes."""
        return self.value[self._logical_key(key)]

    def __setitem__(self, key, value) -> None:
        """Assign constants without changing the result's reverse rule."""
        self.value[self._logical_key(key)] = value

    def __sub__(self, other: 'TensorResult') -> 'SubtractResult':
        return SubtractResult(self, other)

    def __add__(self, other: 'TensorResult') -> 'AddResult':
        return AddResult(self, other)

    def __mul__(self, other) -> 'TensorResult':
        if isinstance(other, TensorResult):
            return MultiplyResult(self, other)
        constant = other
        return ScaleResult(self, constant)

    def __rmul__(self, other) -> 'TensorResult':
        return self * other

    def __isub__(self, constant):
        self.value[...] -= constant
        return self

    def slice(self, key, indices: str) -> 'SliceResult':
        return SliceResult(self, key, indices)


class InputTensor(TensorResult):
    """Differentiable leaf that writes into a caller-owned gradient array."""

    def __init__(self, engine: 'Engine', value, gradient, indices: str):
        super().__init__(engine, value, indices)
        self.gradient = gradient
        self._gradient_initialized = False

    def _backward(self, gradient):
        gradient = _sum_to_shape(gradient, self.gradient.shape)
        if self._gradient_initialized:
            self.gradient[...] += gradient
        else:
            self.gradient[...] = gradient
            self._gradient_initialized = True
        return ()


class EinsumResult(TensorResult):
    """Binary einsum result and the history required for its two VJPs."""

    def __init__(self,
                 engine: 'Engine',
                 subscripts: str,
                 left,
                 right,
                 optimize=True):
        left_indices, right_indices, indices = _split_einsum(subscripts)
        value = engine.contract(
            subscripts, _value(left), _value(right), optimize=optimize)
        super().__init__(engine, value, _labels(indices))
        self.left = left
        self.right = right
        self.left_indices = left_indices
        self.right_indices = right_indices
        self.output_indices = indices
        self.optimize = optimize
        self.parents = tuple(
            operand for operand in (left, right)
            if isinstance(operand, TensorResult))

    def _backward(self, gradient):
        contributions = []
        if isinstance(self.left, TensorResult):
            left_gradient = self.engine.contract(
                f'{self.output_indices},{self.right_indices}'
                f'->{self.left_indices}',
                gradient,
                _value(self.right),
                optimize=self.optimize)
            contributions.append((self.left, left_gradient))
        if isinstance(self.right, TensorResult):
            right_gradient = self.engine.contract(
                f'{self.left_indices},{self.output_indices}'
                f'->{self.right_indices}',
                _value(self.left),
                gradient,
                optimize=self.optimize)
            contributions.append((self.right, right_gradient))
        return contributions


class SubtractResult(TensorResult):
    """Difference of two differentiable tensors."""

    def __init__(self, left: TensorResult, right: TensorResult):
        super().__init__(left.engine, left.value - right.value, left.indices)
        self.left = left
        self.right = right
        self.parents = (left, right)

    def _backward(self, gradient):
        return ((self.left, gradient), (self.right, -gradient))


class AddResult(TensorResult):
    """Sum of two differentiable tensors."""

    def __init__(self, left: TensorResult, right: TensorResult):
        super().__init__(left.engine, left.value + right.value, left.indices)
        self.left = left
        self.right = right
        self.parents = (left, right)

    def _backward(self, gradient):
        return ((self.left, gradient), (self.right, gradient))


class MultiplyResult(TensorResult):
    """Elementwise product of two differentiable tensors."""

    def __init__(self, left: TensorResult, right: TensorResult):
        super().__init__(left.engine, left.value * right.value, left.indices)
        self.left = left
        self.right = right
        self.parents = (left, right)

    def _backward(self, gradient):
        return (
            (self.left, _sum_to_shape(
                gradient * self.right.value, self.left.value.shape)),
            (self.right, _sum_to_shape(
                gradient * self.left.value, self.right.value.shape)))


class ScaleResult(TensorResult):
    """Tensor multiplied by an ordinary broadcastable array."""

    def __init__(self, operand: TensorResult, scale):
        super().__init__(
            operand.engine, operand.value * scale, operand.indices)
        self.operand = operand
        self.scale = scale
        self.parents = (operand,)

    def _backward(self, gradient):
        contribution = _sum_to_shape(
            gradient * self.scale, self.operand.value.shape)
        return ((self.operand, contribution),)


class SliceResult(TensorResult):
    """View of a tensor whose reverse rule scatters into the source shape."""

    def __init__(self, operand: TensorResult, key, indices: str):
        super().__init__(operand.engine, operand.value[key], indices)
        self.operand = operand
        self.key = key
        self.parents = (operand,)

    def _backward(self, gradient):
        contribution = self.engine.xp.zeros_like(self.operand.value)
        contribution[self.key] += gradient
        return ((self.operand, contribution),)


class SumResult(TensorResult):
    """Reduction over one or more explicit tensor axes."""

    def __init__(self, operand: TensorResult, axes: tuple[int, ...]):
        axes = tuple(axis if axis >= 0 else operand.value.ndim + axis
                     for axis in axes)
        batch_rank = operand.value.ndim - len(operand.indices)
        logical_axes = {axis - batch_rank for axis in axes}
        indices = ''.join(
            label for position, label in enumerate(operand.indices)
            if position not in logical_axes)
        super().__init__(
            operand.engine, operand.value.sum(axis=axes), indices)
        self.operand = operand
        self.axes = axes
        self.parents = (operand,)

    def _backward(self, gradient):
        for axis in sorted(self.axes):
            gradient = self.engine.xp.expand_dims(gradient, axis)
        contribution = self.engine.xp.broadcast_to(
            gradient, self.operand.value.shape)
        return ((self.operand, contribution),)


class StackResult(TensorResult):
    """Stack equal-shaped results along one new labelled axis."""

    def __init__(self,
                 operands: tuple[TensorResult, ...],
                 axis: int,
                 indices: str):
        engine = operands[0].engine
        value = engine.xp.stack(
            [operand.value for operand in operands], axis=axis)
        super().__init__(engine, value, indices)
        self.operands = operands
        self.axis = axis
        self.parents = operands

    def _backward(self, gradient):
        return tuple(
            (operand, self.engine.xp.take(
                gradient, position, axis=self.axis))
            for position, operand in enumerate(self.operands))


class RadialDerivativeResult(TensorResult):
    """Origin-free radial finite difference and its exact transpose."""

    def __init__(self, values: TensorResult, spacing):
        xp = values.engine.xp
        shape = (spacing.shape[:-1] +
                 (1,) * (values.value.ndim - spacing.ndim) +
                 spacing.shape[-1:])
        spacing = spacing.reshape(shape)
        derivative = xp.empty_like(values.value[..., 1:])
        derivative[..., :-1] = 0.5 * (
            values.value[..., 2:] - values.value[..., :-2]
        ) / spacing[..., 1:-1]
        derivative[..., -1] = (
            values.value[..., -1] - values.value[..., -2]
        ) / spacing[..., -1]
        super().__init__(
            values.engine, derivative, values.indices[:-1] + 'h')
        self.values = values
        self.spacing = spacing
        self.parents = (values,)

    def _backward(self, gradient):
        contribution = self.engine.xp.zeros_like(self.values.value)
        centered = 0.5 * gradient[..., :-1] / self.spacing[..., 1:-1]
        contribution[..., 2:] += centered
        contribution[..., :-2] -= centered
        backward = gradient[..., -1] / self.spacing[..., -1]
        contribution[..., -1] += backward
        contribution[..., -2] -= backward
        return ((self.values, contribution),)


class Engine:
    """NumPy/CuPy execution backend for differentiable tensor operations."""

    def __init__(self, array_module=np):
        self.xp = array_module

    def requires_gradient(self, value, gradient, indices: str) -> InputTensor:
        """Create the differentiable leaf that receives the final gradient."""
        return InputTensor(self, value, gradient, indices)

    def contract(self, subscripts: str, left, right, *, optimize=True):
        """Execute a raw backend contraction without recording history."""
        return self.xp.einsum(
            subscripts, left, right, optimize=optimize)

    def einsum(self, subscripts: str, left, right,
               *, optimize=True) -> EinsumResult | object:
        """Contract two operands and retain history when either is active."""
        if not isinstance(left, TensorResult) and not isinstance(
                right, TensorResult):
            return self.contract(
                subscripts, left, right, optimize=optimize)
        return EinsumResult(self, subscripts, left, right, optimize)

    def sum(self, operand: TensorResult, axis) -> SumResult:
        """Reduce active tensor axes and create the corresponding result."""
        axes = axis if isinstance(axis, tuple) else (axis,)
        return SumResult(operand, axes)

    def stack(self,
              operands: tuple[TensorResult, ...],
              axis: int,
              indices: str) -> StackResult:
        """Stack active tensors along a new explicitly labelled axis."""
        return StackResult(operands, axis, indices)

    def radial_derivative(self,
                          values: TensorResult,
                          spacing) -> RadialDerivativeResult:
        """Apply the origin-free radial stencil as a differentiable result."""
        return RadialDerivativeResult(values, spacing)
