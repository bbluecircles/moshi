#!/usr/bin/env python3
"""
FastAPI wrapper for Moshi TTS
Loads models on startup and provides /generate_audio endpoint
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
import sphn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from safetensors.torch import save_file
from pathlib import Path as PathlibPath

from moshi.models.tts import TTSModel, DEFAULT_DSM_TTS_REPO, DEFAULT_DSM_TTS_VOICE_REPO
from moshi.models.loaders import CheckpointInfo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
tts_model: Optional[TTSModel] = None
mimi = None
cfg_coef_conditioning = None
cfg_is_no_text = True
cfg_is_no_prefix = True
model_config = {}
output_dir: PathlibPath = PathlibPath("generated_audio") 

class TTSRequest(BaseModel):
    """Request model for TTS generation"""
    turns: List[str] = Field(..., description="List of text turns to synthesize")
    voices: List[str] = Field(..., description="List of voice names corresponding to turns")
    id: Optional[str] = Field(default=None, description="Request ID for output file naming")
    
class TTSGenerationRequest(BaseModel):
    """Complete request model including generation parameters"""
    # Required fields
    turns: List[str] = Field(..., description="List of text turns to synthesize (conversation segments)")
    voices: List[str] = Field(..., description="List of voice names from kyutai/tts-voices repository")
    
    # Optional fields with defaults from CLI
    id: Optional[str] = Field(default=None, description="Request ID for output file naming")
    batch_size: int = Field(default=32, description="Batch size for inference")
    nq: int = Field(default=32, description="Number of codebooks to generate")
    temp: float = Field(default=0.6, description="Temperature for text and audio")
    cfg_coef: float = Field(default=2.0, description="CFG coefficient")
    max_padding: int = Field(default=8, description="Max padding in a row, in steps")
    initial_padding: int = Field(default=2, description="Initial padding, in steps")
    final_padding: int = Field(default=4, description="Amount of padding after the last word, in steps")
    padding_bonus: float = Field(default=0.0, description="Bonus for padding logits (-2 to 2)")
    padding_between: int = Field(default=1, description="Minimal padding between words")
    only_wav: bool = Field(default=True, description="Only return wav data, not debug files")
    return_file: bool = Field(default=True, description="Return audio as downloadable file")
    return_base64: bool = Field(default=True, description="Return audio as base64 encoded string")

class TTSResponse(BaseModel):
    """Response model for TTS generation"""
    success: bool
    message: str
    request_id: str
    audio_base64: Optional[str] = None
    audio_info: Optional[Dict[str, Any]] = None
    generation_time: float
    effective_speed: float

class ModelConfig(BaseModel):
    """Model configuration for initialization"""
    hf_repo: str = Field(default=DEFAULT_DSM_TTS_REPO, description="HF repo for pretrained models")
    voice_repo: str = Field(default=DEFAULT_DSM_TTS_VOICE_REPO, description="HF repo for voice embeddings")
    config: Optional[str] = Field(default=None, description="Local config JSON file path")
    tokenizer: Optional[str] = Field(default=None, description="Local tokenizer file path")
    mimi_weight: Optional[str] = Field(default=None, description="Local Mimi checkpoint path")
    moshi_weight: Optional[str] = Field(default=None, description="Local Moshi checkpoint path")
    device: str = Field(default="cuda", description="Device to run on")
    dtype: str = Field(default="bfloat16", description="Data type (bfloat16 or float16)")
    nq: int = Field(default=32, description="Number of codebooks")
    temp: float = Field(default=0.6, description="Temperature")
    cfg_coef: float = Field(default=2.0, description="CFG coefficient")
    max_padding: int = Field(default=8, description="Max padding")
    initial_padding: int = Field(default=2, description="Initial padding")
    final_padding: int = Field(default=4, description="Final padding")
    padding_bonus: float = Field(default=0.0, description="Padding bonus")

async def initialize_model(config: ModelConfig):
    """Initialize the TTS model with given configuration"""
    global tts_model, mimi, cfg_coef_conditioning, cfg_is_no_text, cfg_is_no_prefix, model_config
    
    try:
        logger.info("Initializing TTS model...")
        
        # Convert dtype string to torch dtype
        dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
        
        # Get checkpoint info
        checkpoint_info = CheckpointInfo.from_hf_repo(
            config.hf_repo, config.moshi_weight, config.mimi_weight, 
            config.tokenizer, config.config
        )
        
        # Initialize TTS model
        tts_model = TTSModel.from_checkpoint_info(
            checkpoint_info, 
            voice_repo=config.voice_repo,
            n_q=config.nq,
            temp=config.temp,
            cfg_coef=config.cfg_coef,
            max_padding=config.max_padding,
            initial_padding=config.initial_padding,
            final_padding=config.final_padding,
            padding_bonus=config.padding_bonus,
            device=config.device,
            dtype=dtype
        )
        
        # Configure CFG
        if tts_model.valid_cfg_conditionings:
            cfg_coef_conditioning = tts_model.cfg_coef
            tts_model.cfg_coef = 1.0
            cfg_is_no_text = False
            cfg_is_no_prefix = False
        else:
            cfg_is_no_text = True
            cfg_is_no_prefix = True
            
        mimi = tts_model.mimi
        model_config = config.dict()
        
        logger.info("TTS model initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize TTS model: {str(e)}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app"""
    # Startup
    default_config = ModelConfig()
    await initialize_model(default_config)
    yield
    # Shutdown
    logger.info("Shutting down TTS service")

# Create FastAPI app
app = FastAPI(
    title="Moshi TTS API",
    description="FastAPI wrapper for Moshi Text-to-Speech synthesis",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "Moshi TTS API",
        "status": "running",
        "model_loaded": tts_model is not None,
        "endpoints": ["/generate_audio", "/health", "/model_info", "/available_voices", "/examples"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": tts_model is not None,
        "device": model_config.get("device", "unknown")
    }

@app.get("/model_info")
async def model_info():
    """Get current model configuration"""
    return {
        "model_config": model_config,
        "model_loaded": tts_model is not None,
        "multi_speaker": tts_model.multi_speaker if tts_model else None,
        "valid_cfg_conditionings": tts_model.valid_cfg_conditionings if tts_model else None
    }

@app.get("/available_voices")
async def get_available_voices():
    """Get list of available voices from the voice repository"""
    if tts_model is None:
        raise HTTPException(status_code=500, detail="TTS model not loaded")
    
    try:
        # This would ideally fetch from the voice repository
        # For now, return common voice categories based on the repository structure
        voice_categories = {
            "vctk": "VCTK dataset voices (English speakers, p001-p376)",
            "expresso": "Expresso dataset voices with emotional variations",
            "cml-tts": "CML-TTS French voices", 
            "unmute-prod-website": "Unmute production voices"
        }
        
        # Example voice names based on the repository structure
        example_voices = [
            "p225",  # VCTK female
            "p226",  # VCTK male
            "p227",  # VCTK female
            "p228",  # VCTK female
            "p229",  # VCTK female
            "p230",  # VCTK female
            "p231",  # VCTK female
            "p232",  # VCTK male
            "p233",  # VCTK female
            "p234",  # VCTK female
            "ex01_happy",     # Expresso emotional
            "ex01_sad",       # Expresso emotional
            "ex01_angry",     # Expresso emotional
            "ex01_neutral",   # Expresso emotional
        ]
        
        return {
            "voice_categories": voice_categories,
            "example_voices": example_voices,
            "note": "Voice names should match files in the kyutai/tts-voices repository. Check https://huggingface.co/kyutai/tts-voices for the complete list.",
            "voice_repo": model_config.get("voice_repo", "kyutai/tts-voices")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching voice information: {str(e)}")

@app.get("/examples")
async def get_examples():
    """Get example requests for the TTS API"""
    return {
        "single_speaker": {
            "description": "Generate speech for a single speaker",
            "example": {
                "turns": ["Hello, welcome to our text-to-speech demonstration."],
                "voices": ["p225"],
                "temp": 0.7,
                "return_base64": True
            }
        },
        "dialogue": {
            "description": "Generate a conversation between two speakers", 
            "example": {
                "turns": [
                    "Good morning! How can I help you today?",
                    "Hi there! I'm looking for information about your services.",
                    "I'd be happy to help. What specific information are you looking for?"
                ],
                "voices": ["p225", "p226", "p225"],
                "temp": 0.6,
                "cfg_coef": 2.0,
                "return_base64": True
            }
        },
        "emotional_speech": {
            "description": "Generate emotional speech using Expresso voices",
            "example": {
                "turns": ["I'm so excited about this new technology!"],
                "voices": ["ex01_happy"],
                "temp": 0.8,
                "return_base64": True
            }
        },
        "multi_language": {
            "description": "Generate French speech (if using French voices)",
            "example": {
                "turns": ["Bonjour, comment allez-vous aujourd'hui?"],
                "voices": ["french_speaker_01"],
                "temp": 0.6,
                "return_base64": True
            }
        }
    }

@app.post("/model/reload")
async def reload_model(config: ModelConfig):
    """Reload model with new configuration"""
    try:
        await initialize_model(config)
        return {"success": True, "message": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")

@app.post("/generate_audio", response_model=TTSResponse)
async def generate_audio(request: TTSGenerationRequest):
    """Generate audio from text using TTS model"""
    if tts_model is None:
        raise HTTPException(status_code=500, detail="TTS model not loaded")
    
    # Generate unique ID if not provided
    request_id = request.id or str(uuid.uuid4())
    
    try:
        begin = time.time()
        
        # Prepare the request
        tts_request = TTSRequest(
            turns=request.turns,
            voices=request.voices,
            id=request_id
        )
        
        # Process single request (batch of 1)
        entries = tts_model.prepare_script(tts_request.turns, padding_between=request.padding_between)
        
        if tts_model.multi_speaker:
            print(f"Is MUTLI SPEAKER")
            voices = [tts_model.get_voice_path(voice) for voice in tts_request.voices]
        else:
            voices = []
            
        attributes = tts_model.make_condition_attributes(voices, cfg_coef_conditioning)
        
        # Handle prefixes
        prefixes = None
        if not tts_model.multi_speaker:
            if len(tts_request.voices) != 1:
                raise HTTPException(
                    status_code=400, 
                    detail="For this model, exactly one voice is required"
                )
            prefix_path = tts_model.get_voice_path(tts_request.voices[0])
            prefixes = [tts_model.get_prefix(prefix_path)]
        
        # Generate audio
        logger.info(f"Generating audio for request {request_id}")
        
        # Update model parameters for this request
        original_cfg_coef = tts_model.cfg_coef
        original_temp = tts_model.temp
        original_max_padding = tts_model.machine.max_padding
        original_initial_padding = tts_model.machine.initial_padding
        original_final_padding = tts_model.final_padding
        original_padding_bonus = tts_model.padding_bonus
        
        try:
            # Temporarily update model parameters
            if cfg_coef_conditioning is None:
                tts_model.cfg_coef = request.cfg_coef
            tts_model.temp = request.temp
            tts_model.machine.max_padding = request.max_padding
            tts_model.machine.initial_padding = request.initial_padding
            tts_model.final_padding = request.final_padding
            tts_model.padding_bonus = request.padding_bonus
            
            result = tts_model.generate(
                [entries], [attributes], prefixes=prefixes,
                cfg_is_no_prefix=cfg_is_no_prefix, cfg_is_no_text=cfg_is_no_text
            )
            
        finally:
            # Restore original parameters
            tts_model.cfg_coef = original_cfg_coef
            tts_model.temp = original_temp
            tts_model.machine.max_padding = original_max_padding
            tts_model.machine.initial_padding = original_initial_padding
            tts_model.final_padding = original_final_padding
            tts_model.padding_bonus = original_padding_bonus
        
        # Process the result
        frames = torch.cat(result.frames, dim=-1).cpu()
        total_duration = frames.shape[0] * frames.shape[-1] / mimi.frame_rate
        
        # Decode audio
        wav_frames = []
        with torch.no_grad(), tts_model.mimi.streaming(1):
            for frame in result.frames[tts_model.delay_steps:]:
                wav_frames.append(tts_model.mimi.decode(frame[:, 1:]))
        
        wavs = torch.cat(wav_frames, dim=-1)
        
        # Process the single result
        end_step = result.end_steps[0]
        if end_step is None:
            logger.warning(f"End step is None for request {request_id}")
            wav_length = wavs.shape[-1]
        else:
            wav_length = int((mimi.sample_rate * (end_step + tts_model.final_padding) / mimi.frame_rate))
        
        effective_duration = wav_length / mimi.sample_rate
        wav = wavs[0, :, :wav_length]
        
        start_step = 0
        if prefixes is not None:
            start_step = prefixes[0].shape[-1]
            start = int(mimi.sample_rate * start_step / mimi.frame_rate)
            wav = wav[:, start:]
        
        # Clamp audio values
        wav = wav.clamp(-1, 1).cpu().numpy()
        
        time_taken = time.time() - begin
        effective_speed = effective_duration / time_taken
        
        # Prepare response
        response_data = {
            "success": True,
            "message": "Audio generated successfully",
            "request_id": request_id,
            "generation_time": time_taken,
            "effective_speed": effective_speed
        }
        
        # Handle audio output
        if request.return_file:
            # Save audio to file and return as FileResponse
            file_extension = "wav"
                
            filename = f"{request_id}.{file_extension}"
            file_path = output_dir / filename
            
            # Save as WAV
            sphn.write_wav(str(file_path), wav, mimi.sample_rate)

            
            # Return file response with proper headers
            media_type = "audio/wav" if file_extension == "wav" else "audio/mpeg"
            
            return FileResponse(
                path=str(file_path),
                media_type=media_type,
                filename=filename,
                headers={
                    "X-Request-ID": request_id,
                    "X-Generation-Time": str(time_taken),
                    "X-Effective-Speed": str(effective_speed),
                    "X-Audio-Duration": str(effective_duration),
                    "X-Sample-Rate": str(int(mimi.sample_rate)),
                    "X-Channels": str(int(wav.shape[0]))
                }
            )
        
        # Add audio data
        if request.return_base64:
            # Convert to base64 using a temporary file approach
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Write WAV to temporary file
                sphn.write_wav(temp_path, wav, mimi.sample_rate)
                
                # Read the file and encode to base64
                with open(temp_path, 'rb') as f:
                    audio_bytes = f.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                response_data["audio_base64"] = audio_base64
            finally:
                # Clean up temporary file
                import os
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        # Add audio info
        response_data["audio_info"] = {
            "sample_rate": int(mimi.sample_rate),
            "duration": float(effective_duration),
            "channels": int(wav.shape[0]),
            "samples": int(wav.shape[1]),
            "end_step": end_step,
            "start_step": start_step
        }
        
        if not request.only_wav:
            # Add debug information
            response_data["debug_info"] = {
                "hf_repo": model_config.get("hf_repo"),
                "voice_repo": model_config.get("voice_repo"),
                "cfg_coef": request.cfg_coef,
                "temp": request.temp,
                "max_padding": request.max_padding,
                "transcript": result.all_transcripts[0],
                "consumption_times": result.all_consumption_times[0],
                "turns": tts_request.turns,
                "voices": tts_request.voices,
                "logged_text_tokens": result.logged_text_tokens[0]
            }
        
        logger.info(f"Generated audio for request {request_id} in {time_taken:.2f}s (speed: {effective_speed:.2f}x)")
        
        return TTSResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error generating audio for request {request_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")

@app.post("/generate_audio_batch")
async def generate_audio_batch(requests: List[TTSGenerationRequest]):
    """Generate audio for multiple requests in batch"""
    if tts_model is None:
        raise HTTPException(status_code=500, detail="TTS model not loaded")
    
    if not requests:
        raise HTTPException(status_code=400, detail="No requests provided")
    
    # For now, process sequentially - could be optimized for true batch processing
    results = []
    for request in requests:
        try:
            result = await generate_audio(request)
            results.append(result)
        except Exception as e:
            # Add failed result
            results.append({
                "success": False,
                "message": f"Failed: {str(e)}",
                "request_id": request.id or "unknown",
                "generation_time": 0.0,
                "effective_speed": 0.0
            })
    
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    
    parser = argparse.ArgumentParser(description="Moshi TTS FastAPI Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    # Model configuration arguments
    parser.add_argument("--hf-repo", default=DEFAULT_DSM_TTS_REPO, help="HF repo for models")
    parser.add_argument("--voice-repo", default=DEFAULT_DSM_TTS_VOICE_REPO, help="HF repo for voices")
    parser.add_argument("--config", help="Local config JSON file")
    parser.add_argument("--tokenizer", help="Local tokenizer file")
    parser.add_argument("--mimi-weight", help="Local Mimi checkpoint")
    parser.add_argument("--moshi-weight", help="Local Moshi checkpoint")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--half", action="store_true", help="Use float16 instead of bfloat16")
    
    args = parser.parse_args()
    
    # Update default model config based on CLI args
    ModelConfig.model_fields["hf_repo"].default = args.hf_repo
    ModelConfig.model_fields["voice_repo"].default = args.voice_repo
    if args.config:
        ModelConfig.model_fields["config"].default = args.config
    if args.tokenizer:
        ModelConfig.model_fields["tokenizer"].default = args.tokenizer
    if args.mimi_weight:
        ModelConfig.model_fields["mimi_weight"].default = args.mimi_weight
    if args.moshi_weight:
        ModelConfig.model_fields["moshi_weight"].default = args.moshi_weight
    ModelConfig.model_fields["device"].default = args.device
    if args.half:
        ModelConfig.model_fields["dtype"].default = "float16"
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )