from data_loader import DataSets
from model import SkipLog
from utils import get_logger
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
import torch
import argparse
import os
import random
import logging


logger = get_logger('Skip-Log')
logger.setLevel(logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def inference(args, model, loader):
    test_blks = []
    test_losses = []
    loss_metric = 'mean'

    for step, batch in tqdm(enumerate(loader)):
        blk = batch[0]
        encoder_mask, encoder_input, decoder_input, decoder_target = map(lambda x: x.to(device), batch[1:])
        inputs = {
                    'encoder_mask': encoder_mask,
                    'encoder_input': encoder_input,
                    'decoder_input': decoder_input,
                    'decoder_target': decoder_target,
                }
        # encoder_input : (B x L)
        _, loss = model(**inputs)
        test_blks += blk
        # mean : mean loss of log sentence
        if loss_metric == 'mean':
            test_losses += loss.reshape(len(blk), -1).mean(dim=1).tolist()
        # max loss in the log sentence
        elif loss_metric == 'max':
            test_losses += loss.reshape(len(blk), -1).max(dim=1).tolist()

    # blk_id 별로 묶기
    grouped_data = defaultdict(list)
    for blk, loss in zip(test_blks, test_losses):
        grouped_data[blk].append(loss)
    grouped_data = list(grouped_data.items())

    return grouped_data

def main():
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument("--embedding_hidden_dim", default=128, type=int,
                        help="hidden dimension for embedding matrix")
    parser.add_argument("--num_hidden_layer", default=1, type=int,
                        help="number of gru layers in encoder")
    parser.add_argument("--gru_hidden_dim", default=512, type=int,
                        help="hidden dimension for encoder and decoder gru")
    parser.add_argument("--dropout_p", default=0.1, type=float,
                        help="dropout percentage for encoder and decoder gru")
    parser.add_argument("--attention_method", default="dot", type=str,
                        help="attention method (dot, general, concat)")

    # Other parameters
    parser.add_argument("--device", default='cuda', type=str,
                        help="Whether to use cpu or cuda")
    parser.add_argument("--seed", default=0, type=int,
                        help="random seed")

    # Path parameters
    parser.add_argument("--vocab_path", type=str, default='./data/vocab.txt',
                        help="vocab.txt directory")
    parser.add_argument("--infer_data_path", type=str, default='./data/test_logs_split_20_10.txt',
                        help="train dataset directory")
    parser.add_argument("--bestmodel_path", type=str, default='model_saved/checkpoint-13964',
                        help="path of model to use")

    args = parser.parse_args()

    if args.device == 'cuda':
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device('cpu')
        logger.info('use cpu')

    set_seed(args)

    # Load Vocab
    with open(args.vocab_path, 'w') as f:
        vocab = f.read().splitlines()

    # Load Datasets
    infer_set = DataSets(data_path=args.infer_data_path,
                         vocab_path=args.vocab_path,
                         augmentation=False)

    infer_loader = DataLoader(dataset=infer_set,
                              batch_size=1024,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=True,
                              drop_last=False,
                              collate_fn=infer_set.batch_function)
    
    # Load arguments and best model parameters
    best_params = torch.load(os.path.join(args.bestmodel_path, 'training_args.bin'))
    best_args = torch.load(os.path.join(args.bestmodel_path, 'best_model.bin'))

    # Load model
    model = SkipLog(vocab=vocab,
                    embedding_hidden_dim=best_args.embedding_hidden_dim,
                    num_hidden_layer=best_args.num_hidden_layer,
                    gru_hidden_dim=best_args.gru_hidden_dim,
                    device=device,
                    dropout_p=best_args.dropout_p,
                    attention_method=best_args.attention_method).to(device)

    # load trained parameter
    model.load_state_dict(best_params)  

    logger.info('***** Training done *****')
    results = inference(args, model, infer_loader)
    



if __name__ == '__main__':
    main()


