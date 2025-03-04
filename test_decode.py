import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from model_infer import get_rwkv_model
import torch
import torchaudio
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__=='__main__':
    model=get_rwkv_model('cpu',for_trainning=False)
    model_dict = torch.load("model_decode.pt", map_location='cpu')
    model.model.load_state_dict(model_dict.state_dict())
    print('成功加载模型')
    
    model.model.eval()

    audio_output = model.infer_token(
[3071, 8271, 7661, 7661, 3727, 12184, 13201, 904, 3063, 1449, 10679, 15706, 14085, 938, 5728, 14336, 11286, 12925, 12861, 2377, 10473, 2404, 12249, 4409, 15513, 12072, 2350, 2350, 1587, 2427, 12072, 7661, 12476, 4134, 1662, 1271, 2147, 5087, 8406, 10669, 1807, 15819, 2056, 2537, 9442, 14677, 2278, 8442, 12905, 4, 2754, 4059, 6918, 15406, 6150, 2842, 12338, 279, 8761, 164, 1635, 10513, 12927, 12604, 11123, 12837, 9669, 11053, 7249, 15993, 8220, 1117, 10973, 2469, 10617, 4610, 15254, 3854, 1524, 7661, 7661, 12072, 12217, 3071, 8271, 7661, 7661, 3727, 12184, 13201, 904, 6691, 2677, 2220, 2307, 15921, 9306, 5625, 14269, 1501, 13001, 11084, 16037, 3639, 586, 15432, 8678, 11129, 1524, 12217, 10045, 10045]
    )
    torchaudio.save('output_rwkv.wav', audio_output, sample_rate=16000, encoding='PCM_S', bits_per_sample=16)

    print('生成音频完成')