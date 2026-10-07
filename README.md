# Inference Engineering Concepts

Welcome to the **Inference Engineering Concepts** curriculum. This guide serves as a comprehensive roadmap for understanding, building, and optimizing Large Language Model (LLM) inference engines—from foundational math to GPU kernel-level programming and production-scale deployment.

---

## Table of Contents
- [Part I — Foundations](#part-i--foundations)
- [Part II-A — Single-Engine Internals](#part-ii-a--single-engine-internals)
- [Part II-B — Scale-Out and Production](#part-ii-b--scale-out-and-production)
- [Part III — GPU and Kernel Level](#part-iii--gpu-and-kernel-level)
- [Part IV — Capstone and Specialisation](#part-iv--capstone-and-specialisation)
- [Appendices](#appendices)

---

## Part I — Foundations

### Chapter 1. Transformer Inference Math
* **1.1** Parameter counting for decoder-only transformers
* **1.2** FLOPs per token: ~2·N·P for inference vs 6·N·P for training
* **1.3** Prefill as GEMM vs decode as GEMV
* **1.4** Memory traffic per decode step (weights + KV reads)
* **1.5** Theoretical TTFT and inter-token latency (ITL) derivation
* **1.6** Sampling: greedy, temperature, top-k/top-p, and their cost
* **💻 1.7 Project:** Inference calculator + from-scratch decoder (naive → cached)

### Chapter 2. Roofline Model and GPU Architecture
* **2.1** Compute-bound vs memory-bound vs overhead-bound
* **2.2** Arithmetic intensity and the ridge point
* **2.3** GPU anatomy: SMs, warps, tensor cores
* **2.4** Memory hierarchy: HBM → L2 → shared memory → registers
* **2.5** Why batching raises arithmetic intensity
* **2.6** Precision and peak FLOPs (BF16 vs FP8 vs FP4)
* **2.7** Interconnects: PCIe, NVLink, InfiniBand/RDMA
* **💻 2.8 Project:** Bandwidth/TFLOPs measurement + GEMM batch-size roofline sweep

### Chapter 3. KV Cache and Attention Variants
* **3.1** KV cache purpose and sizing formula
* **3.2** MHA, MQA, GQA
* **3.3** MLA (DeepSeek latent attention)
* **3.4** Sliding-window, hybrid local/global, and cross-layer KV sharing
* **3.5** Weights vs KV as the memory limiter at long context
* **3.6** Max-concurrency calculation under a memory budget
* **💻 3.7 Project:** Preallocated KV + GQA in your decoder; validate against vLLM's KV-capacity log

### Chapter 4. Metrics, Workloads and Benchmarking
* **4.1** TTFT, TPOT/ITL, E2E latency, throughput (req/s, tok/s)
* **4.2** Goodput and SLO definition
* **4.3** Aggregate vs per-user throughput under continuous batching
* **4.4** Workload modelling: ISL/OSL distributions, Poisson arrivals, prefix sharing
* **4.5** Offline vs online vs semi-online workloads
* **4.6** Benchmark hygiene: warm-up, version pinning, reproducibility
* **4.7** Nondeterminism and batch-variance in inference
* **💻 4.8 Project:** Latency–throughput curves + goodput report for your deployment

> 🏆 **Gate I:** A number-driven breakdown of your own service.

---

## Part II-A — Single-Engine Internals

### Chapter 5. Continuous Batching and Paged KV
* **5.1** Static vs iteration-level scheduling (Orca)
* **5.2** Selective batching
* **5.3** PagedAttention: blocks, block tables, fragmentation
* **5.4** Preemption: recompute vs swap
* **5.5** Engine architecture: API server, tokenizer, scheduler, model runner
* **5.6** vLLM V1 engine core and KV-cache manager walkthrough
* **5.7** Code reading: `nano-vllm` → `Mini-SGLang`
* **💻 5.8 Project:** Toy engine with an asyncio queue, scheduler loop, paged KV, streaming, OpenAI API

### Chapter 6. Prefix Caching and Scheduling Policy
* **6.1** Hash-based prefix caching (vLLM)
* **6.2** RadixAttention (SGLang)
* **6.3** Chunked prefill and stall-free batching (Sarathi-Serve)
* **6.4** Prefill–decode interference
* **6.5** Fairness, priority and admission control
* **6.6** Overlap (CPU/GPU) scheduling
* **💻 6.7 Project:** Add prefix cache + chunked prefill; measure TTFT/TPOT tails

### Chapter 7. Speculative and Structured Decoding
* **7.1** Speculative sampling and the losslessness proof
* **7.2** Speedup model: acceptance rate × draft cost
* **7.3** Draft models, n-gram / prompt lookup
* **7.4** Medusa heads
* **7.5** EAGLE, EAGLE-2, EAGLE-3
* **7.6** Why speedup collapses at high batch size
* **7.7** Grammar-constrained decoding (FSM, XGrammar) and its overhead
* **💻 7.8 Project:** Vanilla spec decoding + EAGLE-3 batch-size sweep + JSON-schema benchmark

### Chapter 8. Quantization
* **8.1** Number formats: INT8, FP8 (E4M3/E5M2), INT4, MXFP4, NVFP4
* **8.2** Weight-only vs weight+activation quantization: which regime each helps
* **8.3** Outliers and LLM.int8()
* **8.4** SmoothQuant
* **8.5** GPTQ
* **8.6** AWQ
* **8.7** Kernels that make formats fast: Marlin, FP8/FP4 tensor-core GEMMs
* **8.8** KV-cache quantization (FP8, KIVI, NVFP4 KV)
* **8.9** Calibration and quality evaluation beyond perplexity
* **💻 8.10 Project:** BF16 / FP8 / INT4 / FP8-KV comparison + decision memo

### Chapter 9. LoRA and Multi-Tenant Serving
* **9.1** Merged vs unmerged adapters
* **9.2** Segmented/grouped GEMM (Punica, S-LoRA)
* **9.3** Adapter paging and memory
* **💻 9.4 Project:** 1 / 8 / 64 adapters throughput study

> 🏆 **Gate II-A:** Toy engine with all features + benchmark vs vLLM + explanation of the gap.

---

## Part II-B — Scale-Out and Production

### Chapter 10. Distributed Inference
* **10.1** Tensor parallelism (column/row-parallel layouts)
* **10.2** Pipeline parallelism and bubbles in inference
* **10.3** Replica-level data parallelism
* **10.4** Expert parallelism and all-to-all
* **10.5** Collectives: all-reduce, all-gather, all-to-all; NCCL internals
* **10.6** Communication cost modelling (NVLink vs PCIe vs IB)
* **10.7** Choosing TP × DP (× PP) layouts for an SLO
* **💻 10.8 Project:** Manual TP layers + TP vs DP benchmark + nccl-tests fit

### Chapter 11. MoE and Long-Context Inference
* **11.1** Routing, gating and load imbalance
* **11.2** Grouped GEMM for experts
* **11.3** EP vs TP for experts; wide-EP + DP-attention
* **11.4** Long-context bottlenecks
* **11.5** Context/sequence parallelism, ring attention
* **11.6** KV offload and multi-tier KV (CPU / DRAM / SSD)
* **11.7** Sparse and hybrid attention for long context
* **💻 11.8 Project:** MoE TP vs EP with expert-load logging; long-context sweep

### Chapter 12. Disaggregation and Cluster Routing
* **12.1** Prefill/decode disaggregation (DistServe, Splitwise)
* **12.2** KV transfer cost and when disaggregation loses
* **12.3** KV-cache-centric architectures (Mooncake)
* **12.4** KV-aware and prefix-aware routing
* **12.5** NVIDIA Dynamo: router, KV block manager, NIXL
* **12.6** llm-d: inference scheduler, Gateway API Inference Extension, well-lit paths
* **💻 12.7 Project:** Aggregated vs disaggregated goodput under two workloads

### Chapter 13. Production Operations on Kubernetes
* **13.1** Autoscaling signals: queue depth, KV utilisation, SLO burn
* **13.2** HPA/KEDA with custom metrics
* **13.3** Cold start: image pull, weight loading, CUDA graph capture, snapshots
* **13.4** Gang scheduling for multi-node replicas (LeaderWorkerSet, Grove)
* **13.5** Observability: metrics, tracing, per-request logs
* **13.6** Failure modes: OOM, stuck requests, head-of-line blocking
* **13.7** Cost per million tokens at the SLO
* **💻 13.8 Project:** EKS deployment with prefix-aware routing, autoscaler and cost dashboard

### Chapter 14. Multimodal and Audio Inference
* **14.1** Encoder compute vs LLM prefill
* **14.2** Token inflation from images and audio
* **14.3** Encoder caching and batching
* **14.4** Encoder–LLM disaggregation
* **14.5** Streaming audio in/out and real-time constraints
* **💻 14.6 Project:** Qwen2-Audio end-to-end trace + one measured optimisation

> 🏆 **Gate II-B:** Whiteboard design of a full serving architecture with numbers.

---

## Part III — GPU and Kernel Level

### Chapter 15. Profiling
* **15.1** PyTorch profiler
* **15.2** Nsight Systems: timelines, launch gaps, NCCL overlap
* **15.3** Nsight Compute: roofline, memory charts, occupancy
* **15.4** NVTX annotation
* **💻 15.5 Project:** nsys/ncu analysis of a vLLM decode step

### Chapter 16. Triton
* **16.1** Programming model: blocks, tiles, masks
* **16.2** Fused softmax, RMSNorm, SwiGLU
* **16.3** Matmul and autotuning
* **16.4** FlashAttention-2 forward in Triton
* **16.5** Paged decode attention reading a block table
* **16.6** Triton's limits
* **💻 16.7 Project:** Kernels + GPU MODE leaderboard submission

### Chapter 17. CUDA Fundamentals
* **17.1** Thread/block/grid model
* **17.2** Coalescing and memory access patterns
* **17.3** Shared-memory tiling and bank conflicts
* **17.4** Occupancy and register pressure
* **17.5** Warp primitives, reductions, scans
* **17.6** Streams, async copies, events
* **17.7** PyTorch custom ops (C++ extensions)
* **💻 17.8 Project:** SGEMM ladder from naive to ≥60–70% of cuBLAS

### Chapter 18. Attention Kernels
* **18.1** Online softmax and IO-aware tiling (FA1)
* **18.2** Work partitioning (FA2)
* **18.3** Hopper asynchrony: TMA, warp specialisation, FP8 (FA3)
* **18.4** Blackwell bottlenecks: exponential unit, shared-memory bandwidth (FA4)
* **18.5** Flash-decoding / split-KV
* **18.6** FlashInfer: paged/ragged KV, JIT templates, load balancing
* **18.7** FlexAttention
* **💻 18.8 Project:** Split-KV decode kernel vs FlashInfer

### Chapter 19. GEMMs, CUTLASS/CuTe and Low-Precision Kernels
* **19.1** Tensor-core MMA shapes
* **19.2** Swizzled layouts and pipelining (cp.async, TMA)
* **19.3** Epilogue fusion
* **19.4** Grouped GEMM for MoE
* **19.5** Dequant-in-kernel (Marlin-style W4A16)
* **19.6** CUTLASS and CuTe-DSL
* **💻 19.7 Project:** FP8 GEMM with fused epilogue vs cuBLASLt

### Chapter 20. Compilation and Launch Overhead
* **20.1** Why small-batch decode is CPU-bound
* **20.2** CUDA graphs: capture constraints, bucketing, memory pools
* **20.3** torch.compile: Dynamo, Inductor, fusion
* **20.4** Dynamic shapes and graph breaks
* **💻 20.5 Project:** CUDA-graph decode buckets in the toy engine + trace attribution

> 🏆 **Gate III (Kernel Track):** One kernel integrated into vLLM/SGLang with an end-to-end benchmark.

---

## Part IV — Capstone and Specialisation

### Chapter 21. Capstone Projects *(choose two)*
* **21.1** Merged PR to vLLM / SGLang / FlashInfer / TRT-LLM / llm-d / Dynamo
* **21.2** Published mini-engine with a design doc
* **21.3** Reproducible public benchmark study
* **21.4** EKS reference architecture write-up
* **21.5** Kernel project with leaderboard or upstream result
* **21.6** Domain-trained EAGLE-3 drafter

### Chapter 22. Specialisation Tracks
* **22.1** Serving/platform *(+ Go or Rust)*
* **22.2** Performance/framework *(+ C++)*
* **22.3** Kernels *(+ PMPP depth, competitions)*

### Chapter 23. Systems-Language Fluency
* **23.1** C++ for CUDA extensions and engine code
* **23.2** Rust (Dynamo core) or Go (platform tooling)

---

## Appendices
* **A.** **Hardware reference sheet:** H100 / H200 / B200 FLOPs, HBM bandwidth, NVLink
* **B.** **Formula sheet:** FLOPs, KV bytes, roofline, all-reduce cost
* **C.** **Paper reading list by chapter** (must-read vs optional)
* **D.** **Codebase reading order:** `nano-vllm` → `Mini-SGLang` → `gpt-fast` → `vLLM V1` → `SGLang`
* **E.** **Benchmark report template**
* **F.** **Pitfalls checklist**
* **G.** **Staying current:** Conferences, blogs, communities
