import ctypes
import numpy as np
from torch_mlir.runtime import UnrankedMemRefDescriptor, make_nd_memref_descriptor
from torch_mlir.execution_engine import ExecutionEngine

import re
import os

class RefBackend_HwaccCall_Adaptor:

    SEARCH_FOR_FUNC_PREFIX = "linalg_matmul_"

    memref_type_to_np_dtype = {
        # "mrf16": np.float16,    # TODO: not supported
        # "mrf32": np.float32,    # TODO: not supported
        # "mrf64": np.float64,    # TODO: not supported
        # "mri1": np.bool_,   # TODO: not supported
        # "mri8": np.int8,    # TODO: not supported
        # "mri32": np.int32,  # TODO: not supported
        # "mri64": np.int64,  # TODO: not supported
        # "mrc32": np.complex64,  # TODO: not supported
        # "mrc64": np.complex128  # TODO: not supported
    }
    elemental_type_to_ctype = {
        # "i1": ctypes.c_bool,  # TODO: not supported
        # "i8": ctypes.c_byte,  # TODO: not supported
        # "i64": ctypes.c_int,  # TODO: not supported
        "f32": ctypes.c_float,
        # "f64": ctypes.c_double  # TODO: not supported
    }

    memref_type_view_pattern = r"^view((\d+x)+)(f32)$"

    def get_hwacc_funcs(self, module):
        return_prefix_len = len(self.SEARCH_FOR_FUNC_PREFIX)
        hwacc_funcs = []
        with module.context:
            for func in module.body:
                # Returns strings of the form `"refbackend.."` so `"` is deleted.
                func_name = str(func.attributes["sym_name"]).replace('"', '')
                if func_name[:return_prefix_len] == self.SEARCH_FOR_FUNC_PREFIX:
                    hwacc_funcs.append(func_name)
        return hwacc_funcs

    def get_ctype_func(self, func_name):
        return_prefix_len = len(self.SEARCH_FOR_FUNC_PREFIX)
        ret_types = func_name[return_prefix_len:].split("_")
        ctypes_arg = [None]
        for type in ret_types:
            if type in self.elemental_type_to_ctype:
                ctypes_arg.append(self.elemental_type_to_ctype[type])
            elif type in self.memref_type_to_np_dtype:
                # TODO: UnrankedMemRefDescriptor is not supported in c lib
                # therefore, no implementation provided. 
                ctypes_arg.append(ctypes.POINTER(UnrankedMemRefDescriptor))
            else:
                match = re.match(self.memref_type_view_pattern, type)
                if match:
                    digits = match.group(1).split('x')
                    digits.pop() if len(digits) > 1 else 0
                    digits = [int(x) for x in digits]
                    type_str = match.group(3)
                    print(f"{digits=}")
                    print(f"{type_str=}")
                    ctypes_arg.append(
                        ctypes.POINTER(
                            make_nd_memref_descriptor(
                                len(digits),                            # rank
                                self.elemental_type_to_ctype[type_str]  # dtype
                            )
                        )
                    )
                else:
                    assert False, f"Not supported type: {type}"
        print(f"{ctypes_arg=}")
        return ctypes.CFUNCTYPE(*ctypes_arg), ret_types

    def __init__(self):
        library_paths = os.getenv('LD_LIBRARY_PATH').split(':')
        self.c_lib = None

        # Find & Locate `libEslHwacc.so`
        for path in library_paths:
            library_path = os.path.join(path, 'libEslHwacc.so')

            if not os.path.exists(library_path):
                continue
            try:
                # load and save in c_lib. 
                self.c_lib = ctypes.CDLL(library_path)
            except OSError:
                # try next if cannot be loaded
                continue

            # check if required symbol exist.
            if hasattr(self.c_lib, 'hwacc_debug') and hasattr(self.c_lib, 'matmul_f32'):
                break
            else:
                if os.name == 'nt':  # Only Windows need to FreeLibrary
                    ctypes.windll.kernel32.FreeLibrary(self.c_lib._handle)
                self.c_lib = None

        assert self.c_lib is not None, "No compatible `libEslHwacc.so` in `LD_LIBRARY_PATH`. "

    def register(self, remote_ee : ExecutionEngine, module):
        hwacc_funcs = self.get_hwacc_funcs(module)
        print(f"{hwacc_funcs=}")

        for hwacc_func in hwacc_funcs:
            print(f"{hwacc_func=}")
            ctype_wrapper, ret_types = self.get_ctype_func(hwacc_func)
            print(f"{ctype_wrapper=}")
            print(f"{ret_types=}")
            remote_ee.register_runtime(hwacc_func, ctype_wrapper(self.c_lib.hwacc_debug))
            # remote_ee.register_runtime(hwacc_func, ctype_wrapper(self.c_lib.matmul_f32))
