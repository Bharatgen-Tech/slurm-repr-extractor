from transformers import PretrainedConfig, PreTrainedModel, AutoModel, AutoConfig
import torch
import os
import numpy as np
import json
import onnxruntime as ort
import tqdm
from huggingface_hub import snapshot_download

class IndicASRConfig(PretrainedConfig):
    model_type = "iasr"
    
    def __init__(self, ts_folder: str = "path", BLANK_ID: int = 256, RNNT_MAX_SYMBOLS: int = 10,
                 PRED_RNN_LAYERS: int = 2, PRED_RNN_HIDDEN_DIM: int = 640, SOS: int = 5632, **kwargs):
        super().__init__(**kwargs)
        self.ts_folder = ts_folder
        self.BLANK_ID = BLANK_ID
        self.RNNT_MAX_SYMBOLS = RNNT_MAX_SYMBOLS
        self.PRED_RNN_LAYERS = PRED_RNN_LAYERS
        self.PRED_RNN_HIDDEN_DIM = PRED_RNN_HIDDEN_DIM
        self.SOS = SOS
        if 'FRAME_DURATION_MS' not in kwargs:
            print('Please check FRAME_DURATION_MS. The timestamps can be inaccurate')
            fs = 0.04
        else:
            fs = kwargs['FRAME_DURATION_MS']
        self.FRAME_DURATION_MS = fs

class IndicASRModel(PreTrainedModel):
    config_class = IndicASRConfig

    def __init__(self, config, device, local_rank):
        super().__init__(config)
        
        # Load model components
        self.models = {}
        names = ['encoder']
        self.models = {}
        self.d = device 
        self.models['preprocessor'] = torch.jit.load(f'{config.ts_folder}/assets/preprocessor.ts', map_location=self.d)
        for n in tqdm.tqdm(names):
            component_name = f'{config.ts_folder}/assets/{n}.onnx'
        	# Check This later
            providers = [
            	("CUDAExecutionProvider", {"device_id": local_rank})
        	]            
            if os.path.exists(config.ts_folder):
                self.models[n] = ort.InferenceSession(component_name, providers = providers)
            else:
                self.models[n] = None   
                print('Failed to load', component_name)
    
    # def batched_encode(self, wavs, lengths):
    #    audio_signal, length = self.models['preprocessor'](input_signal=wavs.to(self.d), length=lengths.to(self.d))
    #    outputs, encoded_lengths = self.models['encoder'].run(['outputs', 'encoded_lengths'], {'audio_signal': audio_signal.cpu().numpy(), 'length': length.cpu().numpy()})
    #    return outputs, encoded_lengths
    def batched_encode(self, wavs, lengths):
    # === PREPROCESSOR STAGE ===
        print(f"\n{'='*60}")
        print(f"[DEBUG] PREPROCESSOR INPUT")
        print(f"{'='*60}")
        print(f"wavs: shape={wavs.shape}, dtype={wavs.dtype}, device={wavs.device}")
        print(f"lengths: shape={lengths.shape}, values={lengths.tolist()}")
        print(f"target device: {self.d}")
        
        try:
            audio_signal, length = self.models['preprocessor'](
                input_signal=wavs.to(self.d), 
                length=lengths.to(self.d)
            )
            print(f"✅ Preprocessor success")
            print(f"audio_signal: shape={audio_signal.shape}, dtype={audio_signal.dtype}")
            print(f"length: shape={length.shape}, values={length.tolist()}")
        except Exception as e:
            print(f"❌ Preprocessor failed: {e}")
            raise
        
        # === VALIDATION BEFORE ONNX ===
        print(f"\n{'='*60}")
        print(f"[DEBUG] PRE-ONNX VALIDATION")
        print(f"{'='*60}")
        
        # Check for invalid values
        has_nan = torch.isnan(audio_signal).any().item()
        has_inf = torch.isinf(audio_signal).any().item()
        has_zero_length = (length <= 0).any().item()
        
        print(f"audio_signal NaN: {has_nan}, Inf: {has_inf}")
        print(f"length has zeros/negatives: {has_zero_length}")
        
        if has_nan or has_inf:
            print(f"⚠️  CRITICAL: Invalid audio_signal detected!")
            print(f"   NaN indices: {torch.isnan(audio_signal).nonzero(as_tuple=True)}")
            print(f"   Inf indices: {torch.isinf(audio_signal).nonzero(as_tuple=True)}")
        
        if has_zero_length:
            print(f"⚠️  CRITICAL: Zero/negative lengths: {length[length <= 0]}")
        
        # Convert to numpy
        print(f"\nConverting to numpy...")
        audio_np = audio_signal.cpu().numpy()
        length_np = length.cpu().numpy()
        print(f"audio_np: shape={audio_np.shape}, dtype={audio_np.dtype}, C-contiguous={audio_np.flags['C_CONTIGUOUS']}")
        print(f"length_np: shape={length_np.shape}, dtype={length_np.dtype}, values={length_np}")
        
        # === ONNX ENCODER STAGE ===
        print(f"\n{'='*60}")
        print(f"[DEBUG] CALLING ONNX ENCODER")
        print(f"{'='*60}")
        print(f"Providers: {self.models['encoder'].get_providers()}")
        
        try:
            outputs, encoded_lengths = self.models['encoder'].run(
                ['outputs', 'encoded_lengths'], 
                {'audio_signal': audio_np, 'length': length_np}
            )
            print(f"✅ ONNX encoder success")
            print(f"outputs: shape={outputs.shape}")
            print(f"encoded_lengths: shape={encoded_lengths.shape}, values={encoded_lengths}")
            return outputs, encoded_lengths
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"❌ ONNX ENCODER FAILED")
            print(f"{'='*60}")
            print(f"Error: {type(e).__name__}: {str(e)}")
            print(f"Input shapes at failure:")
            print(f"  audio_np.shape: {audio_np.shape}")
            print(f"  length_np: {length_np}")
            print(f"GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")
            print(f"{'='*60}\n")
            raise

