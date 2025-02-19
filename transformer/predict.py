import torch
from torch import nn, optim
from torch.optim import Adam

from data import *
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time

# 分词器加载与词表建立
tokenizer = Tokenizer()
loader = MyDataLoader(ext=('en', 'de'),
                    tokenize_en=tokenizer.tokenize_en,
                    tokenize_de=tokenizer.tokenize_de,
                    init_token='<sos>',
                    eos_token='<eos>')
loader.build_vocab(train, min_freq=2)

# 数据加载，单句推理
en_sentence = 'Two young, White males are outside near many bushes.'
en_sentence = [loader.en_vocab["<sos>"]] + \
            [loader.en_vocab[token] for token in loader.tokenize_en(en_sentence)] + \
            [loader.en_vocab["<eos>"]]
en_sentence = torch.tensor(en_sentence).to(device).unsqueeze(0)
en_sentence = en_sentence.to(torch.int64)
de_sentence = torch.ones(max_len)*loader.de_vocab["<pad>"]
de_sentence[0] = loader.de_vocab["<sos>"]
de_sentence = torch.tensor(de_sentence).to(device).unsqueeze(0)
de_sentence = de_sentence.to(torch.int64)

# 模型加载
model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

state_dict = torch.load('saved/model-3.63526114821434.pt')
model.load_state_dict(state_dict)

# 开始预测
res = []
for i in range(max_len-1):
    output = model(en_sentence, de_sentence)
    # 预测结果
    pred_token_indice = torch.argmax(output[0][i]).item()
    if pred_token_indice == loader.de_vocab['<eos>']:
        break
    res.append(pred_token_indice)
    de_sentence[0][i+1] = pred_token_indice

res = idx_to_word(res, loader.de_vocab)
print(res)



    




