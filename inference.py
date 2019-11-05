from data_loader import DataSets
from model import SkipLog
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
import json

torch.manual_seed(0)
torch.cuda.manual_seed(0)


def inference(infer_data_path, vocab_path, max_len, model_param, ):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load Vocab
    with open(vocab_path, 'w') as f:
        vocab = f.read().splitlines()

    # Load Datasets
    infer_set = DataSets(data_path=infer_data_path,
                         vocab_path=vocab_path,
                         max_len=max_len,
                         is_train=True)

    infer_loader = DataLoader(dataset=infer_set,
                              batch_size=1,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True,
                              drop_last=True)

    # Load model
    model = SkipLog(vocab=vocab,
                    device='cuda',
                    max_len=max_len,
                    **model_param)

    model.to(device)
    model.eval()

