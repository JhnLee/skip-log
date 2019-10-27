from torch.utils.data import Dataset
from vocab import log_parser
from itertools import groupby
import torch
import re
import os


class DataSets(Dataset):
    def __init__(self, data_path, vocab_path, max_len):
        self.data_path = data_path
        self.vocab_path = vocab_path
        self.max_len = max_len

        self.data = self.load_data()
        self.vocab = self.load_vocab()

        self.group_data()

        self.pad, self.eol, self.sol = [self.vocab.index(t) for t in ['<PAD>', '<EOL>', '<SOL>']]

    def add_special_token(self, prev, input, next):
        input = input + [self.eol]
        prev_input = [self.sol] + prev
        prev_output = prev + [self.eol]
        next_input = [self.sol] + next
        next_output = next + [self.eol]
        return input, prev_input, prev_output, next_input, next_output

    def pad_sequence(self, sequence, eol_preserve=False):
        diff = self.max_len - len(sequence)
        if diff > 0:
            sequence += [self.pad] * diff
        else:
            if eol_preserve:
                sequence = sequence[:self.max_len-1] + [self.eol]
            else:
                sequence = sequence[:self.max_len-1]

        return sequence

    def token_to_idcs(self, token):
        try:
            return self.vocab.index(token)
        except ValueError:
            return self.vocab.index('<UNK>')

    def group_data(self):

        def get_blk_id(log):
            id_extractor = re.compile('blk_.\d+')
            return id_extractor.search(log).group()

        self.blk = [get_blk_id(log) for log in self.data]

        # blk_id 별로 묶기
        grouped_data = [[key, list(list(zip(*list(group)))[0])]for key, group
                        in groupby(zip(self.data, self.blk), lambda x: x[1])]

        three_len_logs = []
        for subgroup in grouped_data:
            key, logs = subgroup
            if len(logs) == 0:
                raise ValueError('Zero-length sequence exists')
            if len(logs) == 1:
                # 로그가 한 개인 경우 AE처럼 자기 자신을 예측하도록 설정
                logs += [logs[0], logs[0]]
            elif len(logs) == 2:
                # 로그가 두 개인 경우 앞, 뒤 로그를 하나씩 추가
                logs = [logs[0]] * 2 + [logs[1]] * 2

            for i in range(len(logs) - 2):
                three_len_logs.append((key, logs[i:i + 3]))

        self.preprocessed_log = three_len_logs

    def load_data(self):
        with open(self.data_path, 'r') as f:
            data = f.read().splitlines()
        return data

    def load_vocab(self):
        if not os.path.exists(self.vocab_path):
            print("Creating vocab.txt...")
            from vocab import create_vocab
            create_vocab()
            print('done.')
        with open(self.vocab_path, 'r') as f:
            vocab = f.read().splitlines()
        return vocab

    def __len__(self):
        return len(self.preprocessed_log)

    def __getitem__(self, idx):
        blk, logs = self.preprocessed_log[idx]
        idcs = ([self.token_to_idcs(l) for l in log_parser(log).split()[3:]] for log in logs)
        encoder_input, prev_input, prev_output, next_input, next_output = self.add_special_token(*idcs)
        encoder_input, prev_output, next_output = [torch.tensor(self.pad_sequence(i, True))
                                                   for i in [encoder_input, prev_output, next_output]]

        prev_input, next_input = [torch.tensor(self.pad_sequence(i))
                                  for i in [prev_input, next_input]]

        decoder_input = torch.cat((prev_input, next_input), dim=0)  # 2L
        decoder_target = torch.cat((prev_output, next_output), dim=0)  # 2L

        encoder_mask = torch.tensor([0 if t == self.pad else 1 for t in encoder_input])

        return blk, encoder_mask, encoder_input, decoder_input, decoder_target

