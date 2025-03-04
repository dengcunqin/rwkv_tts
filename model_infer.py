import logging
import torch
import numpy as np
import torch.nn.functional as F
from typing import Union, Dict, List, Tuple, Optional
from decoder.models import VocosBackbone
from decoder.heads import ISTFTHead
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from transformers import WhisperFeatureExtractor
from speech_tokenizer.utils import extract_speech_token_wav
class tts_decode(torch.nn.Module):
    def __init__(self,device = 'cuda',load_whisper=False,*args, **kwargs):
        super().__init__()
        self.device = device
        if load_whisper:
            self.whisper_model = WhisperVQEncoder.from_pretrained('glm-4-voice-tokenizer').eval().to(self.device)
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained('glm-4-voice-tokenizer')
        self.codebook_weight = torch.load('codebook_weight.pt')
        self.backbone = VocosBackbone(input_channels= 1280,dim= 768,intermediate_dim= 2304,num_layers= 12,adanorm_num_embeddings= 4).to(self.device)
        self.head = ISTFTHead(dim= 768,n_fft= 5120,hop_length= 1280,padding='same').to(self.device)
        self.bandwidth_id = torch.tensor([0]).to(self.device)
    def codes_to_features(self,codes: torch.Tensor) -> torch.Tensor:
        if codes.dim() == 2:
            codes = codes.unsqueeze(1)
        # n_bins = self.whisper_model.config.quantize_vocab_size
        n_bins = 16384
        offsets = torch.arange(0, n_bins * len(codes), n_bins, device=codes.device)
        embeddings_idxs = codes + offsets.view(-1, 1, 1)
        features = torch.nn.functional.embedding(embeddings_idxs, self.codebook_weight).sum(dim=0)
        features = features.transpose(1, 2)
        return features
    def decode(self, features_input: torch.Tensor) -> torch.Tensor:
        x = self.backbone(features_input, bandwidth_id=self.bandwidth_id)
        audio_output = self.head(x)
        return audio_output
    def infer(
        self,
        speech: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        
        speech = speech
        self.whisper_model.eval()
        with torch.no_grad():
            audio_tokens = extract_speech_token_wav(self.whisper_model, self.feature_extractor, [speech])[0]
        return audio_tokens
    
    def infer_token(
        self,
        audio_tokens: list,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        with torch.no_grad():
            features = self.codes_to_features(torch.tensor([audio_tokens],dtype=torch.long,device=self.device))
            audio_output = self.decode(features)
        return audio_output
    


class get_rwkv_model():
    def __init__(self,device,load_whisper=False):
        self.device=device
        self.load_models(load_whisper)
    def get_model(self):
        return self.model

    def load_models(self,load_whisper):
        self.model=tts_decode(self.device,load_whisper)

        print('初始化模型成功')
        self.model.to(self.device)


    def infer_token(self,token:list):
        import time
        self.model.eval()
        with torch.no_grad():
            k=time.time()
            audio_output=self.model.infer_token(token)

        return audio_output
    



if __name__=='__main__':
    model=get_rwkv_model('cpu',for_trainning=False)
    try:
        model_dict = torch.load("pytorch_model.bin", map_location='cpu')
        model.model.load_state_dict(model_dict)
        print('成功加载模型')
    except:
        print('未加载最新模型')
