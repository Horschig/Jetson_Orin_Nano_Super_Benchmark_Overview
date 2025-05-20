# Jetson_Orin_Nano_Super_TTS_Benchmark_Overview

## Bnechmark overview


Table for generation Generation "Hello World!" (duration)
Engine | RTF | Speed (s) | Speed (tps) | RT tps req | tps / req
---|---|---|---|---|---|
Fish-Speech v1.5 | ? | ? | 4,8 | 24 | 0.2
Orpheus TTS 3b | ? | ? | 19 | 83 | 0,23
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
