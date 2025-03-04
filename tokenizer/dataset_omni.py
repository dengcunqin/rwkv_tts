########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info

from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
import json

class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args

        self.tokenizer = TRIE_TOKENIZER('tokenizer/rwkv_tts_vocab.txt','tokenizer/unuse_vocab.txt','tokenizer/rwkv_tts_kokoro_vocab.txt')

        self.vocab_size=len(self.tokenizer.token2idx)+3

        with open('rwkv_train_data.txt','r',encoding='utf-8') as f:
            self.data_list = [x.strip() for x in f.readlines()]

        with open('rwkv_train_data_aishell3.txt','r',encoding='utf-8') as f:
            self.data_list_aishell3 = json.loads(f.read())
            self.data_list_aishell3_keys=list(self.data_list_aishell3.keys())
            



    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):

        # random_data = random.choice(self.data_list)
        # print(random_data)
        # random_data_json = json.loads(random_data)
        # text = random_data_json['user']
        # audio = random_data_json['audio']

        # train_text = f'<|user|>{text}<|endoftext|><|begin_of_audio|>{audio}<|end_of_audio|>'

        # print('>>>>>>>>>>>>',train_text)

        # exit()

        while True:
            try:
            
                if random.randint(0,3)==1:
                    random_key = random.choice(self.data_list_aishell3_keys)
                    data_1 = random.choices(self.data_list_aishell3[random_key],k=3)
                    dix=[]
                    for da in data_1:
                        da_json = json.loads(da)
                        text = da_json['user'].strip()
                        audio = da_json['audio'].strip()
                        # print('>>>>>>>>>>>>>>>>>>',text,audio)
                        dix_1 = self.tokenizer.tokenizer_2.encode('<|user|>')+self.tokenizer.encode(text)+self.tokenizer.tokenizer_2.encode(f'<|endoftext|><|begin_of_audio|>{audio}<|end_of_audio|>')
                        if len(dix)+len(dix_1)<513:
                            dix += dix_1
                else:
                    random_data = random.choice(self.data_list)
                    random_data_json = json.loads(random_data)

                    # random_ratio = random.randint(1,3)
                    text = random_data_json['user'].strip()
                    audio = random_data_json['audio'].strip()
                    # train_text = f'<|user|>{text}<|endoftext|><|begin_of_audio|>{audio}<|end_of_audio|>'

                    # print(text,audio)
                    # self.tokenizer.tokenizer_2.encode('<|user|>')
                    # self.tokenizer.encode(text)
                    # self.tokenizer.tokenizer_2.encode(f'<|endoftext|><|begin_of_audio|>{audio}<|end_of_audio|>')


                    dix = self.tokenizer.tokenizer_2.encode('<|user|>')+self.tokenizer.encode(text)+self.tokenizer.tokenizer_2.encode(f'<|endoftext|><|begin_of_audio|>{audio}<|end_of_audio|>')
                # print(dix)

                # dix = dix * random_ratio

                if len(dix)>513:continue

                # 目标长度
                target_length = 513

                # 如果数组长度小于目标长度，进行填充
                if len(dix) < target_length:
                    padded_arr = np.pad(dix, (0, target_length - len(dix)), mode='constant', constant_values=225) # 直接在后面pad<|end_of_audio|>。
                # 如果数组长度大于目标长度，进行截断
                else:
                    padded_arr = dix[:target_length]
            
                dix=padded_arr



                x = torch.tensor(dix[:-1], dtype=torch.long)
                y = torch.tensor(dix[1:], dtype=torch.long)
                break
            except:
                print('读取错误')
                continue

        return x, y
