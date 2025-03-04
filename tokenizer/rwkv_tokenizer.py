########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

class TRIE:
    __slots__ = tuple("ch,to,values,front".split(","))
    to:list
    values:set
    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for ch in range(256)]
        self.values = set()
        self.front = front

    def __repr__(self):
        fr = self
        ret = []
        while(fr!=None):
            if(fr.ch!=None):
                ret.append(fr.ch)
            fr = fr.front
        return "<TRIE %s %s>"%(ret[::-1], self.values)
    
    def add(self, key:bytes, idx:int=0, val=None):
        if(idx == len(key)):
            if(val is None):
                val = key
            self.values.add(val)
            return self
        ch = key[idx]
        if(self.to[ch] is None):
            self.to[ch] = TRIE(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx+1, val=val)
    
    def find_longest(self, key:bytes, idx:int=0):
        u:TRIE = self
        ch:int = key[idx]
        
        while(u.to[ch] is not None):
            u = u.to[ch]
            idx += 1
            if(u.values):
                ret = idx, u, u.values
            if(idx==len(key)):
                break
            ch = key[idx]
        return ret

import re

digit_map = {
            '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
            '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
        }
digit_map_1 = {
            '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
            '五': '5', '六': '6', '七': '7', '八': '8', '九': '9'
        }

def replace_match(match):
            number_str = match.group(1)  # 提取数字部分
            chinese_number = ''.join(digit_map[d] for d in number_str)  # 转换成中文
            return f"'''audio{chinese_number}'''"
def replace_match_1(match_1):
            number_str = match_1.group(1)  # 提取数字部分
            chinese_number = ''.join(digit_map_1[d] for d in number_str)  # 转换成中文
            return f"<|audio_{chinese_number}|>"

class TRIE_TOKENIZER():
    def __init__(self, file_name,unuse_file_name,kokoro_file_name):

        self.tokenizer_2 = TRIE_TOKENIZER_NEXT(kokoro_file_name)

        with open(unuse_file_name, "r", encoding="utf-8") as f:
            self.unuse_lines = [x.strip('\n') for x in f.readlines()]

        with open('tokenizer/text_2_vocab.txt', "r", encoding="utf-8") as f:
            self.text2vocab_dict={}
            i=0
            for x in f.readlines():
                x1 = x.strip('\n').split('\t')
                i+=1
                self.text2vocab_dict[i]=x1[1]

        self.idx2token = {}
        sorted = [] # must be already sorted
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k,v in self.idx2token.items():
            self.token2idx[v] = int(k)

        self.root = TRIE()
        for t, i in self.token2idx.items():
            _ = self.root.add(t, val=(t, i))

    def replace_chars_with_single_space(self,input_str):
        # input_str = re.sub(r'<\|audio_(\d+)\|>', replace_match, input_str)
    
        # 遍历字符串，将存在于列表中的字符替换为空格
        for char in self.unuse_lines:
            input_str = input_str.replace(char, ' ')

        # input_str = re.sub(r"'''audio([一二三四五六七八九零]+)'''", replace_match_1, input_str)
        # 使用正则表达式将连续的空格替换为单个空格
        input_str = re.sub(r'\s+', ' ', input_str)
        # 去除字符串两端的多余空格

        # print(input_str)

        
        return input_str.strip()


    def encodeBytes(self, src:bytes):
        idx:int = 0
        tokens = []
        while (idx < len(src)):
            _idx:int = idx
            idx, _, values = self.root.find_longest(src, idx)
            assert(idx != _idx)
            _, token = next(iter(values))            
            tokens.append(token)
        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src:str):
        src_1 = self.replace_chars_with_single_space(src.lower())
        data = self.encodeBytes(src_1.encode("utf-8"))
        filtered_data = [x for x in data if x != 1]
        src_2 = ' '.join([self.text2vocab_dict[i-1] for i in filtered_data])
        # print(src_2)
        src_3 = self.tokenizer_2.encode(src_2)
        # print(src_1,src_2,src_3)
        return src_3

    def decode(self, tokens):
        try:
            return self.decodeBytes(tokens).decode('utf-8')
        except:
            return '\ufffd' # bad utf-8

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
        print()


class TRIE_TOKENIZER_NEXT():
    def __init__(self, file_name):
        self.idx2token = {}
        sorted = [] # must be already sorted
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k,v in self.idx2token.items():
            self.token2idx[v] = int(k)

        self.root = TRIE()
        for t, i in self.token2idx.items():
            _ = self.root.add(t, val=(t, i))

    def encodeBytes(self, src:bytes):
        idx:int = 0
        tokens = []
        while (idx < len(src)):
            _idx:int = idx
            idx, _, values = self.root.find_longest(src, idx)
            assert(idx != _idx)
            _, token = next(iter(values))            
            tokens.append(token)
        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src:str):
        src = src.replace('↓ ','↓').replace('→ ','→').replace('↗ ','↗').replace('↘ ','↘')
        data = self.encodeBytes(src.encode("utf-8"))
        return data

    def decode(self, tokens):
        try:
            return self.decodeBytes(tokens).decode('utf-8')
        except:
            return '\ufffd' # bad utf-8

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
        print()
