from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import random
import numpy as np
import os
import gc
import time


class BERTDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = vocab
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        if corpus_lines is None:
            self.corpus_lines = []

        j = 0
        for folder in tqdm(os.listdir(corpus_path)):
            folder_path = os.path.join(corpus_path, folder)
            for file in os.listdir(folder_path):
                document = np.load(os.path.join(folder_path, file))
                for i in document:
                    self.corpus_lines.append(i)
                del document
            # print("{:}:{:} {:1f}% ELAPSED: {:}".format(j, len(os.listdir(corpus_path)), 100.0*j/len(os.listdir(corpus_path)), int(time.time()-start_time)))
            # j += 1

        gc.collect()

        # with open(corpus_path, "r", encoding=encoding) as f:
        #     if self.corpus_lines is None and not on_memory:
        #         for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
        #             self.corpus_lines += 1
        #
        #     if on_memory:
        #         self.lines = [line[:-1].split("\t")
        #                       for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
        #         self.corpus_lines = len(self.lines)
        #
        # if not on_memory:
        #     self.file = open(corpus_path, "r", encoding=encoding)
        #     self.random_file = open(corpus_path, "r", encoding=encoding)
        #
        #     for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
        #         self.random_file.__next__()

    def __len__(self):
        return len(self.corpus_lines)

    def __getitem__(self, item):
        # t1, t2, is_next_label = self.random_sent(item)
        # t1_random, t1_label = self.random_word(t1)
        # t2_random, t2_label = self.random_word(t2)
        #
        # # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        # t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        # t2 = t2_random + [self.vocab.eos_index]
        #
        # bert_input = (t1 + t2)[:self.seq_len]
        #
        # padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        # bert_input.extend(padding)

        s = self.get_sent(item)
        bert_input = (self.byol_aug(s), self.byol_aug(s))

        # output = {"bert_input": bert_input}

        return torch.tensor(bert_input, dtype=torch.int)

    def byol_aug(self, tokens):
        pad_num = np.count_nonzero(tokens == self.vocab["[PAD]"])
        # pad_num = tokens.count(self.vocab["[PAD]"])
        token_len = len(tokens) - pad_num - 1
        if pad_num > 0:
            for _ in range(min(5, pad_num)):
                insert_pos = random.randint(1, token_len)
                np.insert(tokens, insert_pos, self.vocab["[SEP]"])
                token_len += 1

        return tokens[:self.seq_len]

    def random_word(self, sentence, already):
        tokens = sentence.split()
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15 and token not in already:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)

        return tokens, output_label

    def get_sent(self, index):
        return self.get_corpus_line(index)

    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        # output_text, label(isNotNext:0, isNext:1)
        return t1, t2, 1
        # if random.random() > 0.5:
        #     return t1, t2, 1
        # else:
        #     return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        if self.on_memory:
            # return self.lines[item][0], self.lines[item][1]
            return self.corpus_lines[item]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            t1, t2 = line[:-1].split("\t")
            return t1, t2

    def get_random_line(self):
        if self.on_memory:
            return self.lines[random.randrange(len(self.lines))][1]

        line = self.file.__next__()
        if line is None:
            self.file.close()
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
            line = self.random_file.__next__()
        return line[:-1].split("\t")[1]
