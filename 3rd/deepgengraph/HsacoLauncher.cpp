#include <hip/hip_runtime.h>
#include <cstddef>
#include <iostream>


int main(int argc, char** argv) {
    int device_count;
    if(argc < 2){
        std::cerr << "usage: " << argv[0] << " <kernel.hsaco>\n";
        return -1;
    }
    hipGetDeviceCount(&device_count);

    if (device_count == 0) {
        std::cerr << "No HIP device found\n";
        return 1;
    }

    int device_id;
    hipGetDevice(&device_id);
    hipSetDevice(device_id);

    // load hsaco
    hipModule_t module;
    hipError_t err =hipModuleLoad( &module, argv[1]);
    if (err != hipSuccess) {
        std::cerr 
            << "hipModuleLoad failed: "
            << hipGetErrorString(err)
            << std::endl;
        return 1;
    }

    // get kernel
    hipFunction_t kernel;

    err = hipModuleGetFunction(
            &kernel,
            module,
            "Attn_p2"
        );
    if (err != hipSuccess) {
        std::cerr
            << "hipModuleGetFunction failed: "
            << hipGetErrorString(err)
            << std::endl;
        hipModuleUnload(module);
        return 1;
    }
    std::cout 
        << "kernel loaded successfully\n";

    /*
       kernel:

       define void @Attn_p2(
           ptr addrspace(1) %0,  // q:   tensor<1x4096x32x128xf16>
           ptr addrspace(1) %1,  // k:   tensor<1x4096x32x128xf16>
           ptr addrspace(1) %2,  // v:   tensor<1x4096x32x128xf16>
           ptr addrspace(1) %3   // out: tensor<1x4096x32x128xf16>
       )
    */

    constexpr size_t batch = 1;
    constexpr size_t seq_len = 4096;
    constexpr size_t head_num = 32;
    constexpr size_t head_dim = 128;
    constexpr size_t f16_bytes = 2;
    constexpr size_t tensor_bytes =
        batch * seq_len * head_num * head_dim * f16_bytes;

    void *q = nullptr;
    void *k = nullptr;
    void *v = nullptr;
    void *out = nullptr;

    auto cleanup_buffers = [&]() {
        if (out) { hipFree(out); }
        if (v) { hipFree(v); }
        if (k) { hipFree(k); }
        if (q) { hipFree(q); }
    };

    err = hipMalloc(&q, tensor_bytes);
    if (err != hipSuccess) {
        std::cerr << "hipMalloc q failed: " << hipGetErrorString(err) << std::endl;
        hipModuleUnload(module);
        return 1;
    }
    err = hipMalloc(&k, tensor_bytes);
    if (err != hipSuccess) {
        std::cerr << "hipMalloc k failed: " << hipGetErrorString(err) << std::endl;
        cleanup_buffers();
        hipModuleUnload(module);
        return 1;
    }
    err = hipMalloc(&v, tensor_bytes);
    if (err != hipSuccess) {
        std::cerr << "hipMalloc v failed: " << hipGetErrorString(err) << std::endl;
        cleanup_buffers();
        hipModuleUnload(module);
        return 1;
    }
    err = hipMalloc(&out, tensor_bytes);
    if (err != hipSuccess) {
        std::cerr << "hipMalloc out failed: " << hipGetErrorString(err) << std::endl;
        cleanup_buffers();
        hipModuleUnload(module);
        return 1;
    }

    err = hipMemset(q, 0, tensor_bytes);
    if (err != hipSuccess) {
        std::cerr << "hipMemset q failed: " << hipGetErrorString(err) << std::endl;
        cleanup_buffers();
        hipModuleUnload(module);
        return 1;
    }
    err = hipMemset(k, 0, tensor_bytes);
    if (err != hipSuccess) {
        std::cerr << "hipMemset k failed: " << hipGetErrorString(err) << std::endl;
        cleanup_buffers();
        hipModuleUnload(module);
        return 1;
    }
    err = hipMemset(v, 0, tensor_bytes);
    if (err != hipSuccess) {
        std::cerr << "hipMemset v failed: " << hipGetErrorString(err) << std::endl;
        cleanup_buffers();
        hipModuleUnload(module);
        return 1;
    }
    err = hipMemset(out, 0, tensor_bytes);
    if (err != hipSuccess) {
        std::cerr << "hipMemset out failed: " << hipGetErrorString(err) << std::endl;
        cleanup_buffers();
        hipModuleUnload(module);
        return 1;
    }

    void *args[] = {
        &q,
        &k,
        &v,
        &out,
    };

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, 0);

    /*
       log.log:
       gridDim = [1, 32, 32]
       thread_num = 128
    */

    err =
    hipModuleLaunchKernel(
        kernel,
        // grid
        1,
        32,
        32,
        // block
        128,
        1,
        1,
        // dynamic shared memory
        0,
        // stream
        0,
        // args
        args,
        nullptr
    );

    if (err != hipSuccess) {
        std::cerr
            << "launch failed: "
            << hipGetErrorString(err)
            << std::endl;
        cleanup_buffers();
        hipEventDestroy(stop);
        hipEventDestroy(start);
        hipModuleUnload(module);
        return 1;
    }

    hipEventRecord(stop,0);
    hipEventSynchronize(stop);
    float ms;
    hipEventElapsedTime(
        &ms,
        start,
        stop
    );
    std::cout
        << "kernel time = "
        << ms
        << " ms\n";

    cleanup_buffers();
    hipEventDestroy(stop);
    hipEventDestroy(start);
    hipModuleUnload(module);
    return 0;
}
