import torch
import ctypes
from ctypes import POINTER, Structure, c_int32, c_void_p
from libinfiniop import (
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    create_handle,
    open_lib,
    to_tensor,
    get_test_devices,
    check_error,
    rearrange_if_needed,
    rearrange_tensor,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
)


class RerrangeDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopRearrangeDescriptor_t = POINTER(RerrangeDescriptor)


def test(
    lib
):
    handle = create_handle(lib)
    x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
    y = torch.zeros(x.shape, dtype=torch.float32)

    x = rearrange_tensor(x, [1, 3])
    y = rearrange_tensor(y, [2, 1])
    print(f"x_shape: {x.shape}")
    print(f"x_stride: {x.stride()}")
    print(f"y_stride: {y.stride()}")
    print(f"x: {x}")

    x_tensor, y_tensor = [to_tensor(tensor, lib) for tensor in [x, y]]

    descriptor = infiniopRearrangeDescriptor_t()
    check_error(
        lib.infiniopCreateRearrangeDescriptor(
            handle, ctypes.byref(descriptor), y_tensor.descriptor, x_tensor.descriptor
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [x_tensor, y_tensor]:
        tensor.destroyDesc(lib)

    def lib_rearrange():
        check_error(
            lib.infiniopRearrange(descriptor, y_tensor.data, x_tensor.data, None)
        )

    lib_rearrange()

    print(f"y: {y}")

    check_error(lib.infiniopDestroyRearrangeDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    lib = open_lib()

    lib.infiniopCreateRearrangeDescriptor.restype = c_int32
    lib.infiniopCreateRearrangeDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopRearrangeDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopRearrange.restype = c_int32
    lib.infiniopRearrange.argtypes = [
        infiniopRearrangeDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyRearrangeDescriptor.restype = c_int32
    lib.infiniopDestroyRearrangeDescriptor.argtypes = [infiniopRearrangeDescriptor_t]

    test(lib)
