# Smolkartoffel-135M

# Model
https://huggingface.co/SebastianBodza/SmolKartoffel-135M-v0.1

# Codec
https://huggingface.co/HKUSTAudio/xcodec2

# Memory usage
`MiB Mem :   7619,9 total,    160,4 free,   5572,9 used,   1886,6 buff/cache`

# Loading time
```
Loading models to cuda...
  Tokenizer loading time: 95.173 seconds
  SmolKartoffel model loading time: 1.849 seconds
  XCodec2 model loading time: 11.411 seconds
Core models loaded successfully.
Total startup loading time: 108.434 seconds
```

# Inference time
```
Received TTS request for: "Hello world"
  Time for input preparation and tokenization: 0.002 seconds
  Time for SmolKartoffel model.generate(): 41.136 seconds
  Time for decoding LLM output and extracting speech IDs: 0.001 seconds
  Time for XCodec2 model.decode_code(): 0.261 seconds
  Time for saving audio: 0.011 seconds
Total time for /v1/audio/speech: 41.545 seconds
```

```
Received TTS request for: "Hello World, how are you today? I hope your life is as wonderful and beautiful as mine. No sarcasm."
  Time for input preparation and tokenization: 0.003 seconds
  Time for SmolKartoffel model.generate(): 36.639 seconds
  Time for decoding LLM output and extracting speech IDs: 0.001 seconds
  Time for XCodec2 model.decode_code(): 0.257 seconds
  Time for saving audio: 0.010 seconds
Total time for /v1/audio/speech: 37.042 seconds

```


