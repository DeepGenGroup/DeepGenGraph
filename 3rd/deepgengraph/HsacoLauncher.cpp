#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>


static uint16_t float_to_half_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exp = (bits >> 23) & 0xffu;
    uint32_t mant = bits & 0x7fffffu;

    if (exp == 0xffu) {
        if (mant == 0) {
            return static_cast<uint16_t>(sign | 0x7c00u);
        }
        return static_cast<uint16_t>(sign | 0x7c00u | (mant >> 13) | 1u);
    }

    int half_exp = static_cast<int>(exp) - 127 + 15;
    if (half_exp >= 0x1f) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    if (half_exp <= 0) {
        if (half_exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant |= 0x800000u;
        const int shift = 14 - half_exp;
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) {
            ++half_mant;
        }
        return static_cast<uint16_t>(sign | half_mant);
    }

    uint32_t half = sign | (static_cast<uint32_t>(half_exp) << 10) | (mant >> 13);
    if (mant & 0x1000u) {
        ++half;
    }
    return static_cast<uint16_t>(half);
}

static float half_bits_to_float(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    int exp = (value >> 10) & 0x1fu;
    uint32_t mant = value & 0x03ffu;
    uint32_t bits = 0;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ffu;
            bits = sign
                | (static_cast<uint32_t>(exp + 127 - 15) << 23)
                | (mant << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign
            | (static_cast<uint32_t>(exp + 127 - 15) << 23)
            | (mant << 13);
    }

    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

static void print_tensor_sample(
    const char *name,
    const std::vector<uint16_t> &data,
    size_t count
) {
    const size_t sample_count = std::min(count, data.size());
    std::cout << name << " first " << sample_count << " values:";
    std::cout << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < sample_count; ++i) {
        std::cout << ' ' << half_bits_to_float(data[i]);
    }
    std::cout << '\n';
}

static void print_tensor_summary(
    const char *name,
    const std::vector<uint16_t> &data
) {
    float min_value = std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
    size_t finite_count = 0;
    size_t nonfinite_count = 0;

    for (uint16_t value : data) {
        const float fp32 = half_bits_to_float(value);
        if (std::isfinite(fp32)) {
            min_value = std::min(min_value, fp32);
            max_value = std::max(max_value, fp32);
            ++finite_count;
        } else {
            ++nonfinite_count;
        }
    }

    std::cout << name << " summary: ";
    if (finite_count > 0) {
        std::cout
            << "min=" << min_value
            << ", max=" << max_value
            << ", finite_count=" << finite_count;
    } else {
        std::cout << "no finite values";
    }
    std::cout << ", nonfinite_count=" << nonfinite_count << '\n';
}

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
    constexpr size_t tensor_elems =
        batch * seq_len * head_num * head_dim;
    constexpr size_t tensor_bytes =
        tensor_elems * sizeof(uint16_t);
    constexpr size_t sample_count = 32;

    std::vector<uint16_t> h_q(tensor_elems);
    std::vector<uint16_t> h_k(tensor_elems);
    std::vector<uint16_t> h_v(tensor_elems);
    std::vector<uint16_t> h_out(tensor_elems, 0);

    for (size_t i = 0; i < tensor_elems; ++i) {
        // h_q[i] = float_to_half_bits((static_cast<int>(i % 17) - 8) * 0.01f);
        // h_k[i] = float_to_half_bits((static_cast<int>(i % 19) - 9) * 0.01f);
        // h_v[i] = float_to_half_bits((static_cast<int>(i % 23) - 11) * 0.01f);
        h_q[i] = float_to_half_bits(0.01f);
        h_k[i] = float_to_half_bits(0.01f);
        h_v[i] = float_to_half_bits(0.01f);
    }

    print_tensor_sample("q input", h_q, sample_count);
    print_tensor_sample("k input", h_k, sample_count);
    print_tensor_sample("v input", h_v, sample_count);

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

    err = hipMemcpy(q, h_q.data(), tensor_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "hipMemcpy q H2D failed: " << hipGetErrorString(err) << std::endl;
        cleanup_buffers();
        hipModuleUnload(module);
        return 1;
    }
    err = hipMemcpy(k, h_k.data(), tensor_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "hipMemcpy k H2D failed: " << hipGetErrorString(err) << std::endl;
        cleanup_buffers();
        hipModuleUnload(module);
        return 1;
    }
    err = hipMemcpy(v, h_v.data(), tensor_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "hipMemcpy v H2D failed: " << hipGetErrorString(err) << std::endl;
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
        1, 32, 32,
        // block
        128, 1, 1,
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

    err = hipMemcpy(h_out.data(), out, tensor_bytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "hipMemcpy out D2H failed: " << hipGetErrorString(err) << std::endl;
        cleanup_buffers();
        hipEventDestroy(stop);
        hipEventDestroy(start);
        hipModuleUnload(module);
        return 1;
    }

    print_tensor_sample("out", h_out, sample_count);
    print_tensor_summary("out", h_out);

    cleanup_buffers();
    hipEventDestroy(stop);
    hipEventDestroy(start);
    hipModuleUnload(module);
    return 0;
}
