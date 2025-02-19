"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn  import pad_sequence
from torchtext.datasets import Multi30k, multi30k

multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
#multi30k.URL["test"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz"

class MyDataLoader:
    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        print('dataset initializing start')

    def make_dataset(self):
        train_data = Multi30k(split='train', language_pair=self.ext)
        val_data = Multi30k(split='valid', language_pair=self.ext)

        return train_data, val_data

    def build_vocab(self, train_iter, min_freq):
        def yield_tokens(data_iter, language):
            for en, de in data_iter:
                yield self.tokenize_de(de) if language == "de" else self.tokenize_en(en)
        
        self.de_vocab = build_vocab_from_iterator(yield_tokens(train_iter, "de"), min_freq=min_freq, specials=["<unk>", "<pad>", self.init_token, self.eos_token])
        self.en_vocab = build_vocab_from_iterator(yield_tokens(train_iter, "en"), min_freq=min_freq, specials=["<unk>", "<pad>", self.init_token, self.eos_token])
        self.de_vocab.set_default_index(self.de_vocab["<unk>"]) 
        self.en_vocab.set_default_index(self.en_vocab["<unk>"]) 
    
    def make_iter(self, train, val, batch_size, device):
        def collate_batch(batch, en_vocab, de_vocab):
            en_batch, de_batch = [], []
            for en, de in batch:
                de_processed = [de_vocab["<sos>"]] + de_vocab(self.tokenize_de(de)) + [de_vocab["<eos>"]]
                en_processed = [en_vocab["<sos>"]] + en_vocab(self.tokenize_en(en)) + [en_vocab["<eos>"]]
                de_batch.append(torch.tensor(de_processed).to(device)) 
                en_batch.append(torch.tensor(en_processed).to(device)) 
            res = pad_sequence(en_batch, padding_value=en_vocab["<pad>"]), pad_sequence(de_batch, padding_value=de_vocab["<pad>"])

            return (res[0].T, res[1].T)
        
        train_loader = DataLoader(list(train), batch_size=batch_size, collate_fn=lambda x: collate_batch(x, self.en_vocab, self.de_vocab))
        val_loader = DataLoader(list(val), batch_size=batch_size, collate_fn=lambda x: collate_batch(x, self.en_vocab, self.de_vocab))
    
        return train_loader, val_loader