# 该项目使用RWKV生成音频，具备语音克隆能力。
## RWKV7使用单卡4090训练12小时.
音频演示
lol_prompt.wav为prompt参考音频
output_lol.wav和output_lol_1.wav为生成的音频，由于训练最多使用三句话，所以三句话之后的音频效果比较差。
总体来说，由于训练时间比较少，音频合成效果有待提升。
glm-4-voice-tokenizer下载地址:https://www.modelscope.cn/models/ZhipuAI/glm-4-voice-tokenizer
其他模型下载地址：https://www.modelscope.cn/models/dengcunqin/test_tts
由于算力不足，本项目不再更新。