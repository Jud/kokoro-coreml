"""Shared coremltools op registrations for Kokoro TTS CoreML export."""

from coremltools.converters.mil.frontend.torch.torch_op_registry import (
    _TORCH_OPS_REGISTRY,
    register_torch_op,
)
from coremltools.converters.mil import Builder as mb


def register_missing_torch_ops():
    """Register torch ops that coremltools doesn't handle natively."""
    if "new_ones" not in _TORCH_OPS_REGISTRY:
        @register_torch_op
        def new_ones(context, node):
            size = context[node.inputs[1]]
            shape = mb.cast(x=size, dtype="int32", name=node.name + "_shape")
            result = mb.fill(shape=shape, value=1.0, name=node.name)
            context.add(result)

    if "new_zeros" not in _TORCH_OPS_REGISTRY:
        @register_torch_op
        def new_zeros(context, node):
            size = context[node.inputs[1]]
            shape = mb.cast(x=size, dtype="int32", name=node.name + "_shape")
            result = mb.fill(shape=shape, value=0.0, name=node.name)
            context.add(result)

    if "multiply" not in _TORCH_OPS_REGISTRY:
        @register_torch_op
        def multiply(context, node):
            from coremltools.converters.mil.frontend.torch.ops import mul
            mul(context, node)
