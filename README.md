# Jetson_Orin_Nano_Super_TTS_Benchmark_Overview

## LLM LLama.cpp benchmark overview
### Gemma3 1B
```
  Model Load Time (Warm-up): 1.488 seconds
  Benchmarking Prompt 1...
    TTFT (Prompt 1): 0.271 seconds
    TPS (Prompt 1, after first token): 24.08 (Generated Tokens: 128)
  Benchmarking Prompt 2 (with context)...
    TTFT (Prompt 2): 0.109 seconds
    TPS (Prompt 2, after first token): 25.15 (Generated Tokens: 128)
--------------------------------------------------
```
### Gemma3 4B
```
  Model Load Time (Warm-up): 7.523 seconds
  Benchmarking Prompt 1...
    TTFT (Prompt 1): 0.171 seconds
    TPS (Prompt 1, after first token): 14.90 (Generated Tokens: 128)
  Benchmarking Prompt 2 (with context)...
    TTFT (Prompt 2): 0.301 seconds
    TPS (Prompt 2, after first token): 14.73 (Generated Tokens: 128)
--------------------------------------------------
```
### Llama-3.2 3B
```
  Model Load Time (Warm-up): 2.024 seconds
  Benchmarking Prompt 1...
    TTFT (Prompt 1): 0.145 seconds
    TPS (Prompt 1, after first token): 21.90 (Generated Tokens: 128)
  Benchmarking Prompt 2 (with context)...
    TTFT (Prompt 2): 0.274 seconds
    TPS (Prompt 2, after first token): 21.28 (Generated Tokens: 128)
--------------------------------------------------
```
### Nemotron-Mini-4B
```
  Model Load Time (Warm-up): 2.377 seconds
  Benchmarking Prompt 1...
    TTFT (Prompt 1): 0.122 seconds
    TPS (Prompt 1, after first token): 18.85 (Generated Tokens: 128)
  Benchmarking Prompt 2 (with context)...
    TTFT (Prompt 2): 0.292 seconds
    TPS (Prompt 2, after first token): 18.25 (Generated Tokens: 128)
--------------------------------------------------
```
### Phi-3.5 Mini
```
  Model Load Time (Warm-up): 1.764 seconds
  Benchmarking Prompt 1...
    TTFT (Prompt 1): 0.169 seconds
    TPS (Prompt 1, after first token): 20.11 (Generated Tokens: 128)
  Benchmarking Prompt 2 (with context)...
    TTFT (Prompt 2): 0.435 seconds
    TPS (Prompt 2, after first token): 18.86 (Generated Tokens: 128)
--------------------------------------------------
```

### Phi-4 Mini
```
  Model Load Time (Warm-up): 2.387 seconds
  Benchmarking Prompt 1...
    TTFT (Prompt 1): 0.135 seconds
    TPS (Prompt 1, after first token): 18.06 (Generated Tokens: 128)
  Benchmarking Prompt 2 (with context)...
    TTFT (Prompt 2): 0.351 seconds
    TPS (Prompt 2, after first token): 17.43 (Generated Tokens: 128)
```

### Nemotron-4B
```
  Model Load Time (Warm-up): 2.301 seconds
  Benchmarking Prompt 1...
    TTFT (Prompt 1): 0.141 seconds
    TPS (Prompt 1, after first token): 17.87 (Generated Tokens: 128)
  Benchmarking Prompt 2 (with context)...
    TTFT (Prompt 2): 0.345 seconds
    TPS (Prompt 2, after first token): 17.03 (Generated Tokens: 128)
--------------------------------------------------
```

### Qwen3.5 0.8B
```
  Model Load Time (Warm-up): 3.205 seconds
  Benchmarking Prompt 1...
    TTFT (Prompt 1): 0.318 seconds
    TPS (Prompt 1, after first token): 9.80 (Generated Tokens: 32)
  Benchmarking Prompt 2 (with context)...
    TTFT (Prompt 2): 0.219 seconds
    TPS (Prompt 2, after first token): 10.09 (Generated Tokens: 32)
--------------------------------------------------
```

### Qwen3.5 2B
```
  Model Load Time (Warm-up): 5.416 seconds
  Benchmarking Prompt 1...
    TTFT (Prompt 1): 0.092 seconds
    TPS (Prompt 1, after first token): 6.95 (Generated Tokens: 32)
  Benchmarking Prompt 2 (with context)...
    TTFT (Prompt 2): 0.079 seconds
    TPS (Prompt 2, after first token): 7.01 (Generated Tokens: 32)
--------------------------------------------------
```

### Qwen3.5 4B
```
  Model Load Time (Warm-up): 22.598 seconds
  Benchmarking Prompt 1...
    TTFT (Prompt 1): 0.190 seconds
    TPS (Prompt 1, after first token): 3.43 (Generated Tokens: 32)
  Benchmarking Prompt 2 (with context)...
    TTFT (Prompt 2): 0.153 seconds
    TPS (Prompt 2, after first token): 3.44 (Generated Tokens: 32)
--------------------------------------------------
```


## Benchmark Results

### General Benchmark Parameters
- Benchmarked through llama-server.
- Context targets: 64, 256, 1024, 4096 tokens.
- Output token target: 256 tokens.
- Repetitions per context: 10.
- API mode: completion.
- Deterministic decode settings: temperature=0.0, top-k=1, top-p=1.0.
- Reserved llama.cpp context size is computed as: ctx-size = input_target + output_tokens + ctx_headroom (128).
- Prompt requests no reasoning trace or <think> tags.
- Any remaining <think> traces are stripped from saved outputs before rating.
- TTFT measured client-side; PP and TG parsed from llama-server timing logs.

### Qwen3.5-0.8B-Q4_K_M
- Model path: /models/Qwen3.5-0.8B-Q4_K_M.gguf
- Load time across context benchmarks (s): 3.378 +/- 0.026
- Human rating across contexts: 2.00 +/- 0.00

| Context target | Reserved ctx-size | TTFT ms mean +/- std | PP tok/s mean +/- std | TG tok/s mean +/- std |
|---:|---:|---:|---:|---:|
| 64 | 448 | 86.69 +/- 6.13 | 834.88 +/- 34.31 | 40.96 +/- 0.07 |
| 256 | 640 | 196.26 +/- 4.32 | 1373.74 +/- 32.08 | 41.11 +/- 0.02 |
| 1024 | 1408 | 687.74 +/- 4.67 | 1527.07 +/- 10.31 | 40.92 +/- 0.05 |
| 4096 | 4480 | 2667.99 +/- 6.73 | 1559.49 +/- 2.92 | 39.95 +/- 0.08 |

### Qwen3.5-2B-Q4_K_M
- Model path: /models/Qwen3.5-2B-Q4_K_M.gguf
- Load time across context benchmarks (s): 4.699 +/- 1.372
- Human rating across contexts: 3.25 +/- 1.50

| Context target | Reserved ctx-size | TTFT ms mean +/- std | PP tok/s mean +/- std | TG tok/s mean +/- std |
|---:|---:|---:|---:|---:|
| 64 | 448 | 110.53 +/- 7.59 | 637.15 +/- 10.77 | 28.65 +/- 0.02 |
| 256 | 640 | 273.50 +/- 1.74 | 971.65 +/- 6.63 | 28.39 +/- 0.02 |
| 1024 | 1408 | 999.27 +/- 1.59 | 1042.37 +/- 1.52 | 28.20 +/- 0.03 |
| 4096 | 4480 | 3911.90 +/- 5.40 | 1058.54 +/- 1.38 | 27.82 +/- 0.03 |

### Qwen3.5-4B-Q4_K_M
- Model path: /models/Qwen3.5-4B-Q4_K_M.gguf
- Load time across context benchmarks (s): 12.977 +/- 6.017
- Human rating across contexts: 4.00 +/- 0.00

| Context target | Reserved ctx-size | TTFT ms mean +/- std | PP tok/s mean +/- std | TG tok/s mean +/- std |
|---:|---:|---:|---:|---:|
| 64 | 448 | 219.39 +/- 16.04 | 308.55 +/- 4.09 | 14.05 +/- 0.03 |
| 256 | 640 | 624.95 +/- 2.69 | 415.96 +/- 1.19 | 13.91 +/- 0.02 |
| 1024 | 1408 | 2400.84 +/- 4.08 | 429.57 +/- 0.73 | 13.82 +/- 0.01 |
| 4096 | 4480 | 9521.57 +/- 6.06 | 432.11 +/- 0.28 | 13.59 +/- 0.01 |

### Qwen3.5-4B-UD-Q4_K_XL
- Model path: /models/Qwen3.5-4B-UD-Q4_K_XL.gguf
- Load time across context benchmarks (s): 22.798 +/- 24.254
- Human rating across contexts: 4.00 +/- 0.00

| Context target | Reserved ctx-size | TTFT ms mean +/- std | PP tok/s mean +/- std | TG tok/s mean +/- std |
|---:|---:|---:|---:|---:|
| 64 | 448 | 221.88 +/- 21.83 | 308.43 +/- 3.72 | 13.92 +/- 0.06 |
| 256 | 640 | 621.73 +/- 1.77 | 418.27 +/- 1.20 | 13.82 +/- 0.01 |
| 1024 | 1408 | 2389.13 +/- 2.31 | 431.72 +/- 0.38 | 13.75 +/- 0.02 |
| 4096 | 4480 | 9475.84 +/- 4.91 | 434.18 +/- 0.20 | 13.52 +/- 0.01 |

<!-- LLAMA_SERVER_BENCHMARK_RESULTS_END -->

## TTS Benchmark overview

Table for generation Generation "Hello World!" (duration)
Engine | RTF | Speed (s) | Speed (tps) | RT tps req | tps / req | RAM used
---|---|---|---|---|---|---|
Fish-Speech v1.5 | ? | ? | 4,8 | 24 | 0.2 | ?
Orpheus TTS 3b | ? | ? | 19 | 83 | 0,23 | ?
Oute-TTS 0.6B (vLLM) | 4.93 | 0.29 | 40 | 150 | 0.26 | 4.5 GB
Llama-Oute-TTS 1B (vLLM) | --- | --- | --- | --- | --- | OOM
CSM | ? | ? | ? | ? | ? |
Dia | ? | ? | ? | ? | ? |
Kokoro | ? | ? | ? | ? | ? |
Piper | ? | ? | ? | ? | ? |
Riva TTS | ? | ? | ? | ? | ? |
IMS Toucan | ? | ? | ? | ? | ? |
Parler-Mini | ? | ? | ? | ? | ? |
**NeuTTS Nano German (F32)** | **12.51** | **15.26** | **4.0** | **50** | **0.08** | **2575 MB**

Table for generation "Hello World, how are you today? I hope your life is as wonderful and beautiful as mine. No sarcasm."

### NeuTTS Nano German (German Language)


| Variant | Input | RTF | Inference (s) | TPS | RT tps req | tps/req | RAM |
|---------|-------|-----|---------------|-----|-----------|---------|-----|
| **F32** | Short (1.2s) | 12.51 | 15.26 | 4.0 | 50 | 0.08 | 2575 MB |
| **F32** | Long (9.1s) | 11.79 | 107.77 | 4.2 | 50 | 0.08 | 2575 MB |

======================================================================
BENCHMARK SUMMARY - NeuTTS Nano German on Jetson Orin Nano Super
======================================================================
  Model Load Time (Warm-up): 15.734 seconds
  RAM used by model: 2575 MB

Engine | RTF | Speed (s) | Speed (tps) | RT tps req | tps / req | RAM used
---|---|---|---|---|---|---|
NeuTTS Nano German (short) | 12.51 | 15.26 | 4.0 | 50 | 0.08 | 2575 MB
NeuTTS Nano German (long) | 11.79 | 107.77 | 4.2 | 50 | 0.08 | 2575 MB
=========

## Smolkartoffel-135M

### How to install
 - Start pytorch container  `jetson-containers run --volume /home/jorn/Documents/kokoro:/app -p 8085:8085 $(autotag pytorch)`
 - Install xcodec2 `pip install xcodec2`
 - Run code `python -m app/smolkartoffel/smolkartoffel.py`

---
