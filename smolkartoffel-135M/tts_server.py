# /app/tts_service/main_api.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForCausalLM
from xcodec2.modeling_xcodec2 import XCodec2Model
import os
import tempfile
import time
import torchaudio # Added for potential voice cloning

# --- Model Configuration ---
SMOLKARTOFFEL_MODEL_ID = "SebastianBodza/SmolKartoffel-135M-v0.1"
XCODEC_MODEL_ID = "srinivasbilla/xcodec2" # Using the one from Spaces app
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Global Model Instances & Voice Clone Placeholders ---
tokenizer = None
smol_model = None
codec_model = None

# --- Placeholder for Voice Cloning Feature ---
# This would store the processed speech prompt tokens from the reference audio
GLOBAL_VOICE_CLONE_SPEECH_PROMPT_TOKENS_STR = None
# Path to a hardcoded wave file for voice cloning (edit path as needed)
VOICE_CLONE_REFERENCE_WAV_PATH = "/app/reference_voice.wav" # Example path

app = FastAPI()

# --- Helper function from SmolKartoffel example (used by cloning too) ---
def ids_to_speech_tokens_str(speech_ids_tensor):
    """Converts a tensor of speech IDs to a list of <|s_ID|> token strings."""
    speech_tokens_list = []
    for speech_id in speech_ids_tensor.tolist(): # Iterate over tensor elements
        speech_tokens_list.append(f"<|s_{speech_id}|>")
    return speech_tokens_list

@app.on_event("startup")
async def load_models_and_voice_prompt():
    global tokenizer, smol_model, codec_model, GLOBAL_VOICE_CLONE_SPEECH_PROMPT_TOKENS_STR
    print(f"Loading models to {DEVICE}...")
    
    overall_load_start_time = time.time()

    try:
        tokenizer_load_start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(SMOLKARTOFFEL_MODEL_ID)
        tokenizer_load_end_time = time.time()
        print(f"  Tokenizer loading time: {tokenizer_load_end_time - tokenizer_load_start_time:.3f} seconds")

        smol_model_load_start_time = time.time()
        smol_model = AutoModelForCausalLM.from_pretrained(
            SMOLKARTOFFEL_MODEL_ID,
            trust_remote_code=True,
        )
        smol_model.to(DEVICE)
        smol_model.eval()
        smol_model_load_end_time = time.time()
        print(f"  SmolKartoffel model loading time: {smol_model_load_end_time - smol_model_load_start_time:.3f} seconds")

        codec_model_load_start_time = time.time()
        codec_model = XCodec2Model.from_pretrained(XCODEC_MODEL_ID)
        codec_model.to(DEVICE)
        codec_model.eval()
        codec_model_load_end_time = time.time()
        print(f"  XCodec2 model loading time: {codec_model_load_end_time - codec_model_load_start_time:.3f} seconds")
        
        print("Core models loaded successfully.")

        # --- Optional: Load and process hardcoded voice cloning reference WAV ---
        # To enable, uncomment the following block and ensure VOICE_CLONE_REFERENCE_WAV_PATH is correct.
        '''
        if os.path.exists(VOICE_CLONE_REFERENCE_WAV_PATH):
            print(f"Loading voice cloning reference WAV: {VOICE_CLONE_REFERENCE_WAV_PATH}")
            vc_load_start_time = time.time()
            try:
                waveform, sample_rate = torchaudio.load(VOICE_CLONE_REFERENCE_WAV_PATH)
                waveform = waveform.to(DEVICE)

                # Ensure mono and correct sample rate (16kHz for XCodec2)
                if waveform.size(0) > 1: # If stereo
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000).to(DEVICE)
                    waveform = resampler(waveform)
                
                # Normalize (optional, but good practice, similar to Spaces app)
                # waveform_max_abs = torch.max(torch.abs(waveform))
                # if waveform_max_abs > 0:
                #    waveform = waveform / waveform_max_abs
                
                # Trim or pad if necessary (e.g., first 15 seconds as in Spaces app)
                # max_prompt_sec = 15
                # if waveform.shape[1] / 16000 > max_prompt_sec:
                #     waveform = waveform[:, :16000 * max_prompt_sec]

                # Encode the prompt waveform to speech IDs using XCodec2
                with torch.inference_mode(): # Or torch.no_grad()
                    vq_code_prompt = codec_model.encode_code(input_waveform=waveform) # Shape: [1, 1, num_codes]
                
                # vq_code_prompt is likely shape [1, 1, num_codes], get the actual codes
                speech_ids_tensor = vq_code_prompt[0, 0, :] 
                
                # Convert integer speech IDs to <|s_ID|> token strings
                GLOBAL_VOICE_CLONE_SPEECH_PROMPT_TOKENS_STR = ids_to_speech_tokens_str(speech_ids_tensor)
                
                vc_load_end_time = time.time()
                print(f"  Voice cloning reference processed: {len(GLOBAL_VOICE_CLONE_SPEECH_PROMPT_TOKENS_STR)} speech prompt tokens generated.")
                print(f"  Time for voice clone prompt processing: {vc_load_end_time - vc_load_start_time:.3f} seconds")

            except Exception as e:
                print(f"Error loading or processing voice clone reference WAV: {e}")
                GLOBAL_VOICE_CLONE_SPEECH_PROMPT_TOKENS_STR = None # Ensure it's None if loading fails
        else:
            print(f"Voice cloning reference WAV not found at {VOICE_CLONE_REFERENCE_WAV_PATH}. Voice cloning from hardcoded file disabled.")
            GLOBAL_VOICE_CLONE_SPEECH_PROMPT_TOKENS_STR = None
        '''
        # --- End of Optional Voice Cloning WAV loading ---

    except Exception as e:
        print(f"Error during startup model loading: {e}")
        raise RuntimeError(f"Failed to load models: {e}")
    
    overall_load_end_time = time.time()
    print(f"Total startup loading time: {overall_load_end_time - overall_load_start_time:.3f} seconds")

# --- Helper function from SmolKartoffel example (used in API response) ---
def extract_speech_ids(speech_tokens_str_list): # Takes a list of token strings
    speech_ids = []
    for token_str in speech_tokens_str_list:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            try:
                num_str = token_str[4:-2]
                num = int(num_str)
                speech_ids.append(num)
            except ValueError:
                print(f"Warning: Could not parse speech ID from token: {token_str}")
    return speech_ids

class TTSRequest(BaseModel):
    model: str 
    input: str
    voice: str = "default"
    # Add a flag for future voice cloning requests
    # use_voice_clone_prompt: bool = False # Example

@app.post("/v1/audio/speech")
async def generate_speech(request: TTSRequest):
    endpoint_start_time = time.time()

    if not tokenizer or not smol_model or not codec_model:
        raise HTTPException(status_code=503, detail="Models are not loaded yet. Please try again shortly.")

    input_text = request.input
    print(f"Received TTS request for: \"{input_text}\"")

    # --- Determine if using voice clone prompt (future enhancement) ---
    # For now, this example assumes voice cloning uses the global prompt if loaded.
    # A more robust API would take a parameter in TTSRequest.
    use_cloned_voice = GLOBAL_VOICE_CLONE_SPEECH_PROMPT_TOKENS_STR is not None
    # if request.use_voice_clone_prompt and GLOBAL_VOICE_CLONE_SPEECH_PROMPT_TOKENS_STR is None:
    # print("Warning: Requested voice cloning, but no reference prompt loaded.")
    # use_cloned_voice = False # Fallback if requested but not available


    try:
        with torch.inference_mode():
            prep_start_time = time.time()
            
            # --- Prepare assistant's opening content ---
            assistant_content_start = "<|SPEECH_GENERATION_START|>"
            num_prompt_tokens = 0

            # --- OPTIONAL: Incorporate voice clone prompt ---
            # If GLOBAL_VOICE_CLONE_SPEECH_PROMPT_TOKENS_STR is populated (i.e., reference audio loaded and processed)
            # and we want to use it for this request:
            if use_cloned_voice:
                # Concatenate the speech prompt tokens to the assistant's starting message
                assistant_content_start += "".join(GLOBAL_VOICE_CLONE_SPEECH_PROMPT_TOKENS_STR)
                num_prompt_tokens = len(GLOBAL_VOICE_CLONE_SPEECH_PROMPT_TOKENS_STR)
                print(f"  Using {num_prompt_tokens} pre-loaded speech prompt tokens for voice cloning.")
            # --- End of OPTIONAL voice clone prompt ---

            # For voice cloning, the input_text might also include the transcript of the reference audio.
            # The SmolKartoffel Spaces app does this: `input_text = prompt_text + " " + target_text`
            # For simplicity here, we're just using the target_text with the audio prompt.
            # You might need to adjust `formatted_text` if the model expects the transcript too.
            formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"
            
            chat_template = [
                {"role": "user", "content": "Convert the text to speech:" + formatted_text},
                {"role": "assistant", "content": assistant_content_start} # Uses the potentially modified start
            ]

            input_ids = tokenizer.apply_chat_template(
                chat_template,
                tokenize=True,
                return_tensors='pt',
                # continue_final_message=True # As seen in Spaces app. May be beneficial.
            ).to(DEVICE)
            prep_end_time = time.time()
            print(f"  Time for input preparation and tokenization: {prep_end_time - prep_start_time:.3f} seconds")

            speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
            if speech_end_id is None: speech_end_id = tokenizer.eos_token_id
            if speech_end_id is None: raise RuntimeError("EOS token ID not found.")

            smol_gen_start_time = time.time()
            outputs = smol_model.generate(
                input_ids,
                max_length=2048, 
                eos_token_id=speech_end_id,
                do_sample=True, top_p=0.9, temperature=0.8,
            )
            smol_gen_end_time = time.time()
            print(f"  Time for SmolKartoffel model.generate(): {smol_gen_end_time - smol_gen_start_time:.3f} seconds")
            
            # --- Adjust slicing if voice prompt tokens were used ---
            # `input_ids.shape[1]` is the length of the tokenized chat_template.
            # If `assistant_content_start` included N prompt tokens, then those N tokens are part of input_ids.
            # The generation `outputs` contains `input_ids` followed by newly generated tokens.
            # We want to extract only the newly generated tokens *after* the speech prompt (if any).
            
            # `generated_ids` should contain only the *newly generated speech tokens* by the LLM,
            # not the input prompt tokens (text part) nor the speech prompt tokens (audio part, if used).
            
            # The logic for slicing generated_ids:
            # Start slicing *after* all input_ids (which includes text prompt and potentially audio prompt tokens).
            # `outputs` has shape [batch_size, sequence_length]. `outputs[0]` is the sequence.
            # `input_ids.shape[1]` is the length of the entire input sequence fed to `generate`.
            
            # Original slicing (when no audio prompt is part of input_ids *before* generation):
            # generated_ids_llm_output = outputs[0, input_ids.shape[1]:]

            # If an audio prompt (num_prompt_tokens) was part of the `assistant_content_start`
            # and thus part of `input_ids`, then `model.generate` continues from there.
            # The `outputs` will contain the original `input_ids` followed by the new generation.
            # We are interested in the tokens *after* `input_ids`.
            generated_ids_llm_output = outputs[0, input_ids.shape[1]:]

            # The Spaces app's `infer` function has a slightly different slicing when a prompt is used:
            # `generated_ids = outputs[0][input_ids.shape[1] - len(speech_ids_prefix) : -1]`
            # This implies that `model.generate` might re-generate the speech_ids_prefix or that
            # `input_ids` to `model.generate` *already includes* the speech_ids_prefix as part of the target to be completed.
            # The key is how `apply_chat_template` with `continue_final_message=True` (if used) and the model itself
            # handle the `assistant_content_start` that includes speech tokens.

            # For our current setup where GLOBAL_VOICE_CLONE_SPEECH_PROMPT_TOKENS_STR are part of input_ids:
            # The generated tokens *after* the full input_ids sequence is what we care about.
            # So `outputs[0, input_ids.shape[1]:]` should give the new tokens.
            
            # Let's stick to the simpler interpretation for now:
            # These are the tokens generated *by the LLM* in this call.
            # If voice cloning was used, these tokens should follow the "style" of the prompt.
            # We still need to decode these LLM-generated tokens into xcodec speech IDs.

            decode_extract_start_time = time.time()
            # Decode the LLM-generated tokens (these are from SmolKartoffel's vocab, potentially including <|s_ID|> tokens)
            llm_generated_tokens_text_list = tokenizer.batch_decode(generated_ids_llm_output.unsqueeze(0), skip_special_tokens=False)[0].split()
            
            # Extract only the <|s_ID|> parts and convert to integer IDs for XCodec
            speech_ids_for_xcodec = extract_speech_ids(llm_generated_tokens_text_list)
            decode_extract_end_time = time.time()
            print(f"  Time for decoding LLM output and extracting speech IDs: {decode_extract_end_time - decode_extract_start_time:.3f} seconds")

            if not speech_ids_for_xcodec:
                print("No speech IDs were extracted from LLM output.")
                raise HTTPException(status_code=500, detail="Failed to generate speech IDs from LLM output.")

            # --- Combine with prompt if cloning (This part is tricky and depends on model behavior) ---
            # The Spaces app does: `gen_wav = gen_wav[:, :, prompt_wav.shape[1] :]` AFTER decoding.
            # This implies the XCodec decodes the *entire sequence* (prompt + new) and then they trim the audio.
            # An alternative is to only feed `speech_ids_for_xcodec` to the XCodec.
            # If the LLM is meant to *continue* the speech prompt, then `speech_ids_for_xcodec` are the new parts.

            # For now, let's assume `speech_ids_for_xcodec` are the *only* ones we need for XCodec from this generation step.
            # If voice cloning was intended to make the *style* transfer, and these are new words in that style.
            final_speech_ids_tensor = torch.tensor(speech_ids_for_xcodec).to(DEVICE).unsqueeze(0).unsqueeze(0)
            
            # If GLOBAL_VOICE_CLONE_SPEECH_PROMPT_TOKENS_STR was used and the LLM output is a *continuation*,
            # you might need to concatenate:
            # prompt_speech_ids = extract_speech_ids(GLOBAL_VOICE_CLONE_SPEECH_PROMPT_TOKENS_STR)
            # combined_speech_ids = prompt_speech_ids + speech_ids_for_xcodec
            # final_speech_ids_tensor = torch.tensor(combined_speech_ids).to(DEVICE).unsqueeze(0).unsqueeze(0)
            # This depends heavily on the model's training and expected input for voice cloning.
            # The Spaces app's approach of `gen_wav = gen_wav[:, :, prompt_wav.shape[1] :]` is safer if the model
            # implicitly regenerates or includes the prompt in its thinking.
            # However, if `GLOBAL_VOICE_CLONE_SPEECH_PROMPT_TOKENS_STR` was already part of the input to `smol_model.generate`,
            # then `speech_ids_for_xcodec` should be the *new* content.

            codec_decode_start_time = time.time()
            gen_wav = codec_model.decode_code(final_speech_ids_tensor) # Decode only the new speech tokens
            audio_data = gen_wav[0, 0, :].cpu().numpy()
            codec_decode_end_time = time.time()
            print(f"  Time for XCodec2 model.decode_code(): {codec_decode_end_time - codec_decode_start_time:.3f} seconds")

        file_save_start_time = time.time()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file:
            sf.write(tmp_audio_file.name, audio_data, 16000)
            temp_file_name = tmp_audio_file.name
        file_save_end_time = time.time()
        print(f"  Time for saving audio: {file_save_end_time - file_save_start_time:.3f} seconds")
        
        endpoint_end_time = time.time()
        print(f"Total time for /v1/audio/speech: {endpoint_end_time - endpoint_start_time:.3f} seconds")

        return FileResponse(temp_file_name, media_type="audio/wav", filename="output.wav", background=lambda: os.unlink(temp_file_name))

    except Exception as e:
        print(f"Error during TTS generation: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server on 0.0.0.0:8085...")
    uvicorn.run(app, host="0.0.0.0", port=8085)

