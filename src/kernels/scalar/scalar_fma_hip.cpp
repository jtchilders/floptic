1:#ifdef FLOPTIC_HAS_HIP
2:
3:#include "floptic/kernel_base.hpp"
4:#include "floptic/kernel_registry.hpp"
5:#include "floptic/timer.hpp"
6:#include <hip/hip_runtime.h>
7:#include <hip/hip_fp16.h>
8:#include <hip/hip_bfloat16.h>
9:#include <vector>
10:#include <iostream>
11:#include <cstdint>
12:
13:namespace floptic {
14:
15:// ============================================================================
16:// FMA dispatch
17:// ============================================================================
18:
19:__device__ __forceinline__ double do_fma(double a, double b, double c) { return fma(a, b, c); }
20:__device__ __forceinline__ float  do_fma(float a, float b, float c)   { return fmaf(a, b, c); }
21:__device__ __forceinline__ _Float16 do_fma(_Float16 a, _Float16 b, _Float16 c) {
22:    return __hfma(a, b, c);
23:}
24:__device__ __forceinline__ hip_bfloat16 do_fma(hip_bfloat16 a, hip_bfloat16 b, hip_bfloat16 c) {
25:    // hip_bfloat16 doesn't have native FMA on all archs; use float path
26:    float fa = static_cast<float>(a);
27:    float fb = static_cast<float>(b);
28:    float fc = static_cast<float>(c);
29:    return hip_bfloat16(fmaf(fa, fb, fc));
30:}
31:
32:// ============================================================================
33:// Type conversion helpers
34:// ============================================================================
35:
36:template <typename T>
37:__device__ __forceinline__ T make_val(double v);
38:
39:template <> __device__ __forceinline__ double       make_val<double>(double v)       { return v; }
40:template <> __device__ __forceinline__ float        make_val<float>(double v)        { return static_cast<float>(v); }
41:template <> __device__ __forceinline__ _Float16     make_val<_Float16>(double v)     { return static_cast<_Float16>(v); }
42:template <> __device__ __forceinline__ hip_bfloat16 make_val<hip_bfloat16>(double v) { return hip_bfloat16(static_cast<float>(v)); }
43:
44:// ============================================================================
45:// HIP Kernels
46:// ============================================================================
47:
48:template <typename T>
49:__global__ void scalar_fma_throughput_kernel(T* __restrict__ sink, int64_t iters) {
50:    T a = make_val<T>(1.0000001);
51:    T b = make_val<T>(0.9999999);
52:    T r0 = make_val<T>(1.0);
53:    T r1 = make_val<T>(2.0);
54:    T r2 = make_val<T>(3.0);
55:    T r3 = make_val<T>(4.0);
56:    T r4 = make_val<T>(5.0);
57:    T r5 = make_val<T>(6.0);
58:    T r6 = make_val<T>(7.0);
59:    T r7 = make_val<T>(8.0);
60:
61:    for (int64_t i = 0; i < iters; i++) {
62:        r0 = do_fma(a, r0, b);
63:        r1 = do_fma(a, r1, b);
64:        r2 = do_fma(a, r2, b);
65:        r3 = do_fma(a, r3, b);
66:        r4 = do_fma(a, r4, b);
67:        r5 = do_fma(a, r5, b);
68:        r6 = do_fma(a, r6, b);
69:        r7 = do_fma(a, r7, b);
70:    }
71:
72:    int idx = threadIdx.x + blockIdx.x * blockDim.x;
73:    T sum = do_fma(r0, make_val<T>(1.0), r1);
74:    sum = do_fma(sum, make_val<T>(1.0), r2);
75:    sum = do_fma(sum, make_val<T>(1.0), r3);
76:    sum = do_fma(sum, make_val<T>(1.0), r4);
77:    sum = do_fma(sum, make_val<T>(1.0), r5);
78:    sum = do_fma(sum, make_val<T>(1.0), r6);
79:    sum = do_fma(sum, make_val<T>(1.0), r7);
80:    sink[idx] = sum;
81:}
82:
83:template <typename T>
84:__global__ void scalar_fma_latency_kernel(T* __restrict__ sink, int64_t iters) {
85:    T a = make_val<T>(1.0000001);
86:    T b = make_val<T>(0.9999999);
87:    T r = make_val<T>(1.0);
88:
89:    for (int64_t i = 0; i < iters; i++) {
90:        r = do_fma(a, r, b);
91:    }
92:
93:    int idx = threadIdx.x + blockIdx.x * blockDim.x;
94:    sink[idx] = r;
95:}
96:
97:// ============================================================================
98:// Kernel Runner
99:// ============================================================================
100:
101:template <typename T>
102:static float run_hip_benchmark(const std::string& mode,
103:                               int blocks, int threads_per_block,
104:                               int64_t iters) {
105:    int total_threads = blocks * threads_per_block;
106:    T* d_sink;
107:    hipMalloc(&d_sink, total_threads * sizeof(T));
108:
109:    hipEvent_t start, stop;
110:    hipEventCreate(&start);
111:    hipEventCreate(&stop);
112:
113:    hipEventRecord(start);
114:    if (mode == "latency") {
115:        scalar_fma_latency_kernel<<<blocks, threads_per_block>>>(d_sink, iters);
116:    } else {
117:        scalar_fma_throughput_kernel<<<blocks, threads_per_block>>>(d_sink, iters);
118:    }
119:    hipEventRecord(stop);
120:    hipEventSynchronize(stop);
121:
122:    float ms = 0.0f;
123:    hipEventElapsedTime(&ms, start, stop);
124:
125:    hipEventDestroy(start);
126:    hipEventDestroy(stop);
127:    hipFree(d_sink);
128:
129:    return ms;
130:}
131:
132:static float dispatch_benchmark(Precision prec, const std::string& mode,
133:                                int blocks, int tpb, int64_t iters) {
134:    switch (prec) {
135:        case Precision::FP64: return run_hip_benchmark<double>(mode, blocks, tpb, iters);
136:        case Precision::FP32: return run_hip_benchmark<float>(mode, blocks, tpb, iters);
137:        case Precision::FP16: return run_hip_benchmark<_Float16>(mode, blocks, tpb, iters);
138:        case Precision::BF16: return run_hip_benchmark<hip_bfloat16>(mode, blocks, tpb, iters);
139:        default:
140:            std::cerr << "  ERROR: Unsupported precision for scalar_fma HIP" << std::endl;
141:            return 0.0f;
142:    }
143:}
144:
145:// ============================================================================
146:// Kernel class
147:// ============================================================================
148:
149:class ScalarFmaHip : public KernelBase {
150:public:
151:    std::string name() const override { return "scalar_fma"; }
152:    std::string category() const override { return "scalar"; }
153:    std::string backend() const override { return "hip"; }
154:
155:    std::vector<Precision> supported_precisions() const override {
156:        return {Precision::FP64, Precision::FP32, Precision::FP16, Precision::BF16};
157:    }
158:
159:    std::vector<std::string> supported_modes() const override {
160:        return {"throughput", "latency"};
161:    }
162:
163:    KernelResult run(const KernelConfig& config,
164:                     const DeviceInfo& device,
165:                     int measurement_trials) override {
166:        int dev_idx = 0;
167:        auto pos = device.id.find(':');
168:        if (pos != std::string::npos)
169:            dev_idx = std::stoi(device.id.substr(pos + 1));
170:        hipSetDevice(dev_idx);
171:
172:        int cus = device.compute_units;
173:        int tpb = config.gpu_threads_per_block > 0 ? config.gpu_threads_per_block : 256;
174:        // AMD wavefront = 64 threads, so blocks_per_CU is different
175:        int bpcu = config.gpu_blocks_per_sm > 0 ? config.gpu_blocks_per_sm : 4;
176:        int blocks = config.gpu_blocks > 0 ? config.gpu_blocks : cus * bpcu;
177:
178:        int64_t iters = config.iterations;
179:        int total_threads = blocks * tpb;
180:
181:        // FLOPs per trial
182:        int fma_chains = (config.mode == "latency") ? 1 : 8;
183:        double flops_per_trial = (double)total_threads * iters * fma_chains * 2.0;
184:
185:        // Warmup
186:        for (int w = 0; w < 10; w++) {
187:            dispatch_benchmark(config.precision, config.mode, blocks, tpb, iters);
188:        }
189:        hipDeviceSynchronize();
190:
191:        // Measure
192:        std::vector<double> times;
193:        times.reserve(measurement_trials);
194:        for (int t = 0; t < measurement_trials; t++) {
195:            float ms = dispatch_benchmark(config.precision, config.mode, blocks, tpb, iters);
196:            times.push_back(ms);
197:        }
198:
199:        std::sort(times.begin(), times.end());
200:
201:        KernelResult result;
202:        result.min_time_ms = times.front();
203:        result.max_time_ms = times.back();
204:        result.median_time_ms = times[times.size() / 2];
206:
207:        result.gflops = (flops_per_trial / (result.median_time_ms * 1e-3)) / 1e9;
208:
209:        // Peak comparison
210:        std::string peak_key = precision_to_string(config.precision);
211:        auto it = device.theoretical_peak_gflops.find(peak_key);
212:        if (it != device.theoretical_peak_gflops.end() && it->second > 0) {
213:            result.peak_percent = (result.gflops / it->second) * 100.0;
214:        }
215:
216:        return result;
217:    }
218:};
219:
220:REGISTER_KERNEL(ScalarFmaHip);
221:
222:// Force-link
223:namespace force_link {
224:    void scalar_fma_hip_link() {
225:        volatile auto* p = &KernelRegistry::instance();
226:        (void)p;
227:    }
228:}
229:
230:} // namespace floptic
231:
232:#endif // FLOPTIC_HAS_HIP
