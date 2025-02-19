"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
from conf import *
from util.data_loader import MyDataLoader
from util.tokenizer import Tokenizer

tokenizer = Tokenizer()
loader = MyDataLoader(ext=('en', 'de'),
                    tokenize_en=tokenizer.tokenize_en,
                    tokenize_de=tokenizer.tokenize_de,
                    init_token='<sos>',
                    eos_token='<eos>')

train, valid = loader.make_dataset()
loader.build_vocab(train, min_freq=2)
train_iter, valid_iter = loader.make_iter(train, valid,
                                        batch_size=batch_size,
                                        device=device)

src_pad_idx = loader.en_vocab['<pad>']
trg_pad_idx = loader.de_vocab['<pad>']
trg_sos_idx = loader.de_vocab['<sos>']

enc_voc_size = len(loader.en_vocab)
dec_voc_size = len(loader.de_vocab)
