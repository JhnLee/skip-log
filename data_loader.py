from torch.utils.data import Dataset
from vocab import log_parser
from collections import defaultdict
from utils import get_logger
import logging
import torch
import re
import os

logger = get_logger('Skip-Log')
logger.setLevel(logging.INFO)


class DataSets(Dataset):
    def __init__(self, data_path, vocab_path, augmentation=False):
        self.data_path = data_path
        self.vocab_path = vocab_path
        self.max_len = None
        self.augmentation = augmentation
        self.preprocessed_log = None
        self.blk = None

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
        length = max([len(d) + 1 for d in (input, prev, next)])
        return input, prev_input, prev_output, next_input, next_output, length

    def pad_sequence(self, sequence, max_len):
        return [l + [self.pad] * (max_len - len(l)) for l in sequence]

    def token_to_idcs(self, token):
        try:
            return self.vocab.index(token)
        except ValueError:
            return self.vocab.index('<UNK>')

    def group_data(self):
        logger.info('preprocess data...')

        def get_blk_id(log):
            id_extractor = re.compile('blk_.\d+')
            return id_extractor.search(log).group()

        self.blk = [get_blk_id(log) for log in self.data]

        # blk_id 별로 묶기
        grouped_data = defaultdict(list)
        for blk, log in zip(self.blk, self.data):
            grouped_data[blk].append(log)
        grouped_data = list(grouped_data.items())

        three_len_logs = []
        for key, logs in grouped_data:
            three_len_logs += [(key, logs[i:i + 3]) for i in range(len(logs) - 2) if len(logs) > 2]

        if self.augmentation is True:
            logger.info('do augmentation')
            # augmentation 진행
            # 앞, 뒤 데이터뿐만 아니라 한 칸 건너뛴 로그도 예측
            for key, logs in grouped_data:
                three_len_logs += [(key, logs[::2][i:i + 3]) for i in range(len(logs[::2]) - 2) if len(logs[::2]) > 2]

        self.preprocessed_log = three_len_logs
        logger.info('preprocessing is done, total {} data are loaded'.format(len(three_len_logs)))

    def load_data(self):
        logger.info('load data...')
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

    def batch_function(self, batch):
        batches = list(zip(*batch))
        blk, lengths = batches[:2]
        max_len = max(lengths)
        # Add pad tokens and make masks for attention
        encoder_input, prev_in, prev_out, next_in, next_out = \
            torch.tensor([self.pad_sequence(log, max_len) for log in batches[2:]])
        attention_mask = torch.ones_like(encoder_input).masked_fill(encoder_input == 0, 0)

        decoder_input = torch.cat((prev_in, next_in), dim=1)  # 2L
        decoder_target = torch.cat((prev_out, next_out), dim=1)  # 2L
        return blk, attention_mask, encoder_input, decoder_input, decoder_target

    def __len__(self):
        return len(self.preprocessed_log)

    def __getitem__(self, idx):
        blk, logs = self.preprocessed_log[idx]
        idcs = ([self.token_to_idcs(l) for l in log_parser(log).split()[3:]] for log in logs)
        encoder_input, prev_input, prev_output, next_input, next_output, length = self.add_special_token(*idcs)
        return blk, length, encoder_input, prev_input, prev_output, next_input, next_output
