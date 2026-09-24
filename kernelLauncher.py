import argparse
import ctypes
import math
from contextlib import contextmanager

import numpy as np
import torch

# python launchKernel.py --hsaco kernel.hsaco --kernel Attn_p2   --grid 32,64,1 --block 128,1,1
# ============================================================
# HIP Runtime
# ============================================================

hip = ctypes.CDLL("libamdhip64.so")

hipError_t = ctypes.c_int
hipModule_t = ctypes.c_void_p
hipFunction_t = ctypes.c_void_p
hipStream_t = ctypes.c_void_p


# hipModuleLoad
hip.hipModuleLoad.argtypes = [
    ctypes.POINTER(hipModule_t),
    ctypes.c_char_p,
]
hip.hipModuleLoad.restype = hipError_t


# hipModuleGetFunction
hip.hipModuleGetFunction.argtypes = [
    ctypes.POINTER(hipFunction_t),
    hipModule_t,
    ctypes.c_char_p,
]
hip.hipModuleGetFunction.restype = hipError_t


# hipModuleLaunchKernel
hip.hipModuleLaunchKernel.argtypes = [
    hipFunction_t,
    ctypes.c_uint,  # gridDimX
    ctypes.c_uint,  # gridDimY
    ctypes.c_uint,  # gridDimZ
    ctypes.c_uint,  # blockDimX
    ctypes.c_uint,  # blockDimY
    ctypes.c_uint,  # blockDimZ
    ctypes.c_uint,  # sharedMemBytes
    hipStream_t,    # stream
    ctypes.POINTER(ctypes.c_void_p),  # kernelParams
    ctypes.POINTER(ctypes.c_void_p),  # extra
]
hip.hipModuleLaunchKernel.restype = hipError_t


# hipModuleUnload
hip.hipModuleUnload.argtypes = [
    hipModule_t,
]
hip.hipModuleUnload.restype = hipError_t


# hipDeviceSynchronize
hip.hipDeviceSynchronize.argtypes = []
hip.hipDeviceSynchronize.restype = hipError_t


# hipGetErrorString
hip.hipGetErrorString.argtypes = [
    hipError_t,
]
hip.hipGetErrorString.restype = ctypes.c_char_p


# ============================================================
# HIP error checking
# ============================================================

def hip_check(err):
    if err != 0:
        msg = hip.hipGetErrorString(err)
        if msg:
            msg = msg.decode()
        raise RuntimeError( "HIP error {}: {}".format(err,msg ))


# ============================================================
# Parse X,Y,Z
# ============================================================

def parse_xyz(value):
    values = value.split(",")
    if len(values) != 3:
        raise argparse.ArgumentTypeError(
            "Expected X,Y,Z, for example: 32,64,1"
        )
    return (
        int(values[0]),
        int(values[1]),
        int(values[2]),
    )


# ============================================================
# Launch HSACO
# ============================================================

@contextmanager
def loaded_hsaco(
    hsaco_path,
    kernel_name,
    kernel_args,
    grid,
    block,
    shared_mem_bytes=0,
):
    """Keep the module and argument storage alive across repeated launches."""
    torch.cuda.init()
    stream = torch.cuda.current_stream()
    module = hipModule_t()
    # --------------------------------------------------------
    # Load HSACO
    # --------------------------------------------------------
    hip_check(hip.hipModuleLoad(ctypes.byref(module),hsaco_path.encode()))
    try:
        # ----------------------------------------------------
        # Find kernel
        # ----------------------------------------------------
        func = hipFunction_t()
        hip_check(hip.hipModuleGetFunction(ctypes.byref(func),module,kernel_name.encode()))
        # ----------------------------------------------------
        # Build kernel parameters
        #
        # equivalent C++:
        #
        # void *params[] = {
        #     &arg0,
        #     &arg1,
        #     ...
        # };
        # ----------------------------------------------------

        arg_storage = []
        for arg in kernel_args:
            # torch tensor -> GPU pointer
            if isinstance(arg, torch.Tensor):
                if not arg.is_cuda:
                    raise ValueError("Tensor kernel argument must be on GPU")
                value = ctypes.c_void_p(arg.data_ptr())
            # scalar argument such as:
            #
            # ctypes.c_int32(...)
            # ctypes.c_int64(...)
            # ctypes.c_float(...)
            #
            elif isinstance(arg, ctypes._SimpleCData):
                value = arg
            else:
                raise TypeError("Unsupported kernel argument type: {}".format(type(arg)))
            arg_storage.append(value)

        # kernelParams
        params = (ctypes.c_void_p * len(arg_storage))()

        for i in range(len(arg_storage)):
            params[i] = ctypes.cast(
                ctypes.byref(
                    arg_storage[i]
                ),
                ctypes.c_void_p
            )

        # ----------------------------------------------------
        # Grid / block
        # ----------------------------------------------------
        gx, gy, gz = grid
        bx, by, bz = block
        hip_stream = hipStream_t(stream.cuda_stream)

        def launch():
            hip_check(
                hip.hipModuleLaunchKernel(
                    func,
                    gx,gy,gz,bx,by,bz,
                    shared_mem_bytes,
                    hip_stream,
                    params,
                    None
                )
            )

        yield launch
    finally:
        # No queued launch may outlive its module, including on errors.
        try:
            stream.synchronize()
        finally:
            hip_check(hip.hipModuleUnload(module))


def launch_hsaco(
    hsaco_path,
    kernel_name,
    kernel_args,
    grid,
    block,
    shared_mem_bytes=0,
):
    """Compatibility helper for a single launch; use loaded_hsaco to benchmark."""
    with loaded_hsaco(
        hsaco_path, kernel_name, kernel_args, grid, block, shared_mem_bytes
    ) as launch:
        st = torch.cuda.Event(enable_timing=True)
        et = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream()
        torch.cuda.synchronize()
        st.record(stream)
        launch()
        et.record(stream)
        et.synchronize()
        return st.elapsed_time(et)


def benchmark_attention(baseline_fn, kernel_fn, warmup=10, repeat=50):
    """Return the baseline output and per-launch GPU event times in ms."""
    if warmup < 0 or repeat <= 0:
        raise ValueError("warmup must be non-negative and repeat must be positive")

    stream = torch.cuda.current_stream()
    baseline = None
    times = {"baseline": [], "kernel": []}
    functions = {"baseline": baseline_fn, "kernel": kernel_fn}
    # Initialize event resources before recording any samples.
    events = {
        name: (torch.cuda.Event(enable_timing=True),
               torch.cuda.Event(enable_timing=True))
        for name in functions
    }
    with torch.inference_mode():
        for pair in events.values():
            for event in pair:
                event.record(stream)
        for _ in range(warmup):
            baseline = baseline_fn()
            kernel_fn()
        torch.cuda.synchronize()

        for i in range(repeat):
            # Alternate which implementation runs first to reduce order bias.
            order = ("baseline", "kernel") if i % 2 == 0 else ("kernel", "baseline")
            for name in order:
                st, et = events[name]
                st.record(stream)
                result = functions[name]()
                et.record(stream)
                et.synchronize()
                times[name].append(st.elapsed_time(et))
                if name == "baseline":
                    baseline = result

    return baseline, times["baseline"], times["kernel"]


def print_timing_summary(name, samples):
    print(
        "{} (ms): min={:.6f}, median={:.6f}, p90={:.6f}, max={:.6f}".format(
            name, np.min(samples), np.median(samples),
            np.percentile(samples, 90), np.max(samples)
        )
    )

# ============================================================
# Attention baseline
#
# Q: [B,H,S,D]
# K: [B,H,D,S]
# V: [B,H,S,D]
#
# GEMM
#   ->
# scale
#   ->
# causal mask
#   ->
# softmax
#   ->
# GEMM
# ============================================================

def attention_baseline(q,k,v,):
    D = q.shape[-1]
    S = q.shape[-2]
    # --------------------------------------------------------
    # GEMM 1
    #
    # [B,H,S,D] @ [B,H,D,S]
    #
    # ->
    #
    # [B,H,S,S]
    # --------------------------------------------------------
    score = torch.matmul(q,k)
    # --------------------------------------------------------
    # scale
    # --------------------------------------------------------
    scale = 1.0 / math.sqrt(D)
    score = score * scale
    # --------------------------------------------------------
    # causal mask
    #
    # Keep:
    #
    # x - -
    # x x -
    # x x x
    #
    # upper triangular area becomes -inf
    # --------------------------------------------------------

    mask = torch.triu(
        torch.ones(S,S,dtype=torch.bool,device=q.device),
        diagonal=1
    )

    score = score.masked_fill(
        mask,
        float("-inf")
    )

    # --------------------------------------------------------
    # softmax
    #
    # Use fp32 internally.
    # --------------------------------------------------------
    prob = torch.softmax(score,dim=-1,dtype=torch.float32)
    # Convert back to fp16 for GEMM2
    prob = prob.to(torch.float16)
    # --------------------------------------------------------
    # GEMM 2
    #
    # [B,H,S,S] @ [B,H,S,D]
    #
    # ->
    #
    # [B,H,S,D]
    # --------------------------------------------------------
    output = torch.matmul(prob,v)
    return output


def attention_baseline_sdpa(q, k, v):
    """PyTorch SDPA baseline for Q/V [B,H,S,D] and K [B,H,D,S].

    PyTorch selects the attention backend for the current device and dtype.
    K layout conversion is included when timing this function.
    """
    # Contiguous head dimensions allow fused attention backends to be used.
    key = k.transpose(-2, -1).contiguous()
    return torch.nn.functional.scaled_dot_product_attention(
        q,
        key,
        v,
        dropout_p=0.0,
        is_causal=True,
    )


# ============================================================
# Compare
# ============================================================

def compare_results(
    hsaco_output,
    baseline_output,
):

    print("")
    print("========================================")
    print("Correctness")
    print("========================================")

    # Make sure dtypes are identical for comparison
    hsaco_output = hsaco_output.float()
    baseline_output = baseline_output.float()

    # --------------------------------------------------------
    # torch.allclose
    # --------------------------------------------------------

    passed = torch.allclose(
        hsaco_output,
        baseline_output,
        rtol=1e-2,
        atol=1e-2
    )

    print(
        "torch.allclose(atol=1e-2, rtol=1e-2): {}".format(
            passed
        )
    )

    # --------------------------------------------------------
    # Additional error information
    # --------------------------------------------------------

    diff = torch.abs(
        hsaco_output - baseline_output
    )

    max_abs_error = diff.max().item()
    mean_abs_error = diff.mean().item()

    print(
        "max abs error : {}".format(
            max_abs_error
        )
    )

    print(
        "mean abs error: {}".format(
            mean_abs_error
        )
    )

    # --------------------------------------------------------
    # Count mismatched elements using same allclose condition
    #
    # |a-b| <= atol + rtol * |b|
    # --------------------------------------------------------

    tolerance = (
        1e-2
        + 1e-2 * torch.abs(baseline_output)
    )

    mismatch = diff > tolerance

    mismatch_count = mismatch.sum().item()
    total_count = mismatch.numel()

    print(
        "mismatch count: {} / {}".format(
            mismatch_count,
            total_count
        )
    )

    # --------------------------------------------------------
    # First values
    # --------------------------------------------------------

    print("")
    print("HSACO first 16 values:")

    print(
        hsaco_output.flatten()[:16]
    )

    print("")
    print("Baseline first 16 values:")

    print(
        baseline_output.flatten()[:16]
    )

    print("")

    if passed:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")

    print("========================================")

    return passed


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Launch AMD HSACO attention kernel"
    )

    # --------------------------------------------------------
    # HSACO
    # --------------------------------------------------------

    parser.add_argument(
        "--hsaco",
        required=True,
        help="Path to HSACO file"
    )

    parser.add_argument(
        "--kernel",
        default="Attn_p2",
        help="Kernel symbol name"
    )

    # --------------------------------------------------------
    # launch configuration
    # --------------------------------------------------------

    parser.add_argument(
        "--grid",
        type=parse_xyz,
        required=True,
        help="Grid dimensions X,Y,Z, example: 32,64,1"
    )

    parser.add_argument(
        "--block",
        type=parse_xyz,
        required=True,
        help="Block dimensions X,Y,Z, example: 64,1,1"
    )

    parser.add_argument(
        "--shared-mem",
        type=int,
        default=0,
        help="Dynamic shared memory bytes"
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device ID"
    )

    parser.add_argument(
        "--warmup", type=int, default=10,
        help="Untimed warmup launches per implementation (default: 10)"
    )
    parser.add_argument(
        "--repeat", type=int, default=50,
        help="Timed samples per implementation (default: 50)"
    )

    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")

    # ========================================================
    # Device
    # ========================================================
    torch.cuda.set_device(
        args.device
    )
    torch.cuda.init()
    # ========================================================
    # Attention shape
    #
    # Current Attn_p2:
    #
    # Q   [1,32,4096,128]
    # K   [1,32,128,4096]
    # V   [1,32,4096,128]
    # OUT [1,32,4096,128]
    # ========================================================

    B = 1
    H = 32
    S = 4096
    D = 128

    print("")
    print("========================================")
    print("Configuration")
    print("========================================")

    print(
        "HSACO       : {}".format(
            args.hsaco
        )
    )

    print(
        "Kernel      : {}".format(
            args.kernel
        )
    )

    print(
        "Grid        : {}".format(
            args.grid
        )
    )

    print(
        "Block       : {}".format(
            args.block
        )
    )

    print(
        "Shared mem  : {}".format(
            args.shared_mem
        )
    )

    print(
        "Shape       : B={}, H={}, S={}, D={}".format(
            B,
            H,
            S,
            D
        )
    )

    print("========================================")

    # ========================================================
    # Input
    # ========================================================

    print("GPU         : {}".format(torch.cuda.get_device_name(args.device)))
    print("PyTorch     : {} (HIP: {})".format(torch.__version__, torch.version.hip))
    print("Benchmark   : warmup={}, repeat={}, alternating order".format(
        args.warmup, args.repeat
    ))
    print("SDPA timing includes K transpose/contiguous conversion.")

    torch.manual_seed(0)

    q = torch.randn(B,H,S,D,device="cuda",dtype=torch.float16)

    # NOTE:
    #
    # K is already transposed:
    #
    # [B,H,D,S]
    #
    k = torch.randn(B,H,D,S,device="cuda",dtype=torch.float16)
    v = torch.randn(B,H,S,D,device="cuda",dtype=torch.float16)
    out = torch.empty(B,H,S,D,device="cuda",dtype=torch.float16)

    # ========================================================
    # Baseline
    # ========================================================

    print("")
    print("Benchmarking PyTorch SDPA and HSACO...")
    with loaded_hsaco(
        hsaco_path=args.hsaco,
        kernel_name=args.kernel,
        kernel_args=[q,k,v,out],
        grid=args.grid,
        block=args.block,
        shared_mem_bytes=args.shared_mem
    ) as launch:
        baseline, time_base, time_ours = benchmark_attention(
            lambda: attention_baseline_sdpa(q, k, v),
            launch,
            warmup=args.warmup,
            repeat=args.repeat,
        )

    print_timing_summary("HSACO", time_ours)
    print_timing_summary("PyTorch SDPA", time_base)
    time_our_mid = np.median(time_ours)
    time_base_mid = np.median(time_base)
    if time_our_mid > 0:
        print(f"{time_our_mid=}, {time_base_mid=}, speedup={time_base_mid / time_our_mid}x")
    else:
        print("HSACO median is zero; speedup is unavailable.")

    # ========================================================
    # Correctness
    # ========================================================

    passed = compare_results(out,baseline)

    # Return non-zero so shell scripts can detect failure
    if not passed:
        raise SystemExit(1)

# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()
