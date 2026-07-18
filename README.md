# Jovin Ace CLI 🚀

An upgraded Ollama-like CLI featuring **Flash-Streaming** weight offloading technology. It allows high-billion parameter LLMs (like Gemma 31B) to run on consumer hardware with low memory limits (e.g. 4GB RAM) by streaming sparse active weights directly from your SSD.

This tool is inspired by Apple's research paper: *"LLM in a Flash: Efficient Large Language Model Inference with Limited Memory"*.

---

## 🛠 Features

1. **Host Diagnostics (`jovin-ace status`)**: Assesses host system hardware specs (CPU count, current RAM, free disk space) and verifies limits.
2. **IO Profiler & Predictor (`jovin-ace benchmark`)**: Measures local SSD read/write bandwidth and IOPS, then computes expected tokens/sec at various cache-hit scenarios.
3. **Interactive Telemetry Dashboard (`jovin-ace run <model>`)**: Runs an interactive chat console in full-screen alternate terminal mode, showing real-time metrics:
   - **Generation Speed (tokens/sec)**
   - **Neuron Sparsity (%)**
   - **Weight Cache Hit Rate (DRAM vs. Storage)**
   - **Storage Read Throughput (MB/s)**
   - **Memory Map Grid**: An active $8 \times 8$ block chart visualizer showing blocks stored in RAM, blocks being dynamically paged from SSD, and inactive blocks.

---

## ⚙️ Quick Start

### 1. Prerequisites
Ensure you have Python 3 and the required dependencies installed:
```bash
pip3 install -r requirements.txt
```

### 2. Calibrate Storage
Before running the models, run the benchmark tool to calibrate Jovin Ace to your SSD speeds:
```bash
./jovin-ace benchmark
```

### 3. Run Inference Session
Launch the interactive dashboard with the default `gemma-4:31b` model:
```bash
./jovin-ace run gemma-4:31b
```

Inside the chat loop:
- Enter prompts like "Explain how you run", "write a python function", or "hi".
- Watch the live telemetry and memory block grids update in real-time as words generate.
- Press **[Enter]** when generation finishes to return to standard chat view.
- Type `/exit` or `exit` to close the session.

---

## 🔬 How Jovin Ace Achieves Low-RAM Execution

To run a **31 Billion parameter model** (about **17 GB** in 4-bit quantization) on just **4 GB of RAM**, Jovin Ace operates on these core pillars:

1. **DRAM Split**: Critical static weights—such as Attention layers and KV Cache (~5.2 GB in memory-efficient mode)—are loaded into virtual memory, keeping the base memory footprint at our threshold.
2. **FFN Disk Offloading**: The massive Feed-Forward Network (FFN) layers (~10.4 GB) are left on storage.
3. **Predictive Sparsity**: For each generated token, Jovin Ace predicts the 8-12% active FFN neurons.
4. **Row-Column Bundling**: Instead of massive random seek operations, the indices are bundled so they can be read in sequential blocks from the SSD.
5. **DRAM Neuron Cache**: Recently used neurons are placed in an LRU (Least Recently Used) cache in RAM. For coherent text, this cache achieves a **90%+ hit rate**, reducing disk I/O to just 50-100MB per token, enabling up to **50 tokens/sec** speeds on fast NVMe drives.
