import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from model_infer import get_rwkv_model
import torch
import torchaudio
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__=='__main__':
    model=get_rwkv_model('cpu',load_whisper=True)
    model_dict = torch.load("model_decode.pt", map_location='cpu')
    model.model.load_state_dict(model_dict.state_dict(),strict=False)
    print('成功加载模型')
    
    model.model.eval()

    # 加载音频文件
    waveform, sample_rate = torchaudio.load('A2_0.wav')

    prompt_token = model.model.infer(waveform)

    print('prompt_token:',prompt_token)

    print('生成prompt完成')