# 🦥 Unsloth Puzzles Challenge

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4](https://img.shields.io/badge/pytorch-2.4-ee4c2c.svg)](https://pytorch.org/)
[![Triton](https://img.shields.io/badge/triton-3.0-blue.svg)](https://github.com/openai/triton)
[![Kaggle](https://img.shields.io/badge/kaggle-verified-blue.svg)](https://www.kaggle.com/code/ashutosh0x/unsloth-puzzles-final-verification)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A comprehensive implementation and optimization suite for the Unsloth AI coding challenges, focusing on high-performance kernels, memory-efficient backpropagation, and multi-GPU sharding.

---

## 🛠️ Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" />
  <img src="https://img.shields.io/badge/NVIDIA-%2376B900.svg?style=for-the-badge&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/Triton-%23000000.svg?style=for-the-badge&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white" />
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" />
</p>

---

## 🧩 Challenges Overview

### ⚡ Challenge A: NF4 Triton Kernel
Implemented a custom Triton GPU kernel for dequantizing **4-bit NormalFloat (NF4)** weights. This kernel matches the bit-exact logic of Unsloth's optimized implementations.
- **Optimization**: Memory-coalesced loads and block-parallel processing.
- **Performance**: Efficient per-block scaling with absmax constants.

### ⛓️ Challenge B: FSDP2 + QLoRA
Configured **Fully Sharded Data Parallel (v2)** for 4-bit quantized training. 
- **Efficiency**: Shards the model across multiple GPUs (e.g., T4 x2) to enable training of larger context lengths.
- **Verification**: Tested on Kaggle dual-T4 setup.

### 🚀 Challenge C: torch.compile Optimization
Eliminated graph breaks to achieve **fullgraph=True** compilation. 
- **Impact**: Significant reduction in kernel dispatch overhead and better utilization of hardware accelerators.

### 🛠️ Challenge D: Llama 3.1 Tool Calling (Bounty Code)
Provided production-grade integration for **Llama 3.1 Instruct** tool calling.
- **Fixes**: Corrected Jinja templates and added dynamic special token detection (`<|python_tag|>`, `<|eom_id|>`).
- **PR Ready**: Refined based on maintainer-style code audits to ensure dynamic compatibility.

### 🧠 Challenge E: Memory-Efficient Backprop
Implemented a custom `torch.autograd` function for **chunked language modeling loss**.
- **VRAM Reduction**: Saves **~50%** peak VRAM during loss computation by avoiding the materialization of the full vocab-sized logit tensor.
- **Correctness**: Verified with gradient unit tests (atol < 1e-5).

---

## 📊 Verification & Benchmarks

All solutions are verified for mathematical correctness and performance.
- **Detailed Evaluation**: [EVALUATION.md](./EVALUATION.md)
- **Technical Documentation**: [DOCUMENTATION.md](./DOCUMENTATION.md)
- **Kaggle Workbook**: [FINAL_VERIFICATION.ipynb](https://www.kaggle.com/code/ashutosh0x/unsloth-puzzles-final-verification)

---

## 📬 Contact & Contributions
Developed by [Ashutosh Kumar Singh](https://github.com/Ashutosh0x). If you're a maintainer at Unsloth, feel free to reach out regarding the Challenge D PR!
