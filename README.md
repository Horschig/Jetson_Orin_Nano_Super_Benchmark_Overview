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


## TTS Benchmark overview


Table for generation Generation "Hello World!" (duration)
Engine | RTF | Speed (s) | Speed (tps) | RT tps req | tps / req | RAM used
---|---|---|---|---|---|
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

Table for generation "Hello World, how are you today? I hope your life is as wonderful and beautiful as mine. No sarcasm."



## Smolkartoffel-135M

### How to install
 - Start pytorch container  `jetson-containers run --volume /home/jorn/Documents/kokoro:/app -p 8085:8085 $(autotag pytorch)`
 - Install xcodec2 `pip install xcodec2`
 - Run code `python -m app/smolkartoffel/smolkartoffel.py`
