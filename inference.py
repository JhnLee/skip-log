from data_loader import DataSets
from model import SkipLog
from utils import get_logger
from torch.utils.data import DataLoader
import torch
import argparse
import random
import logging


logger = get_logger('Skip-Log')
logger.setLevel(logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def inference(args, device):
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

    # Load model
    model = SkipLog(vocab=vocab,
                    embedding_hidden_dim=args.embedding_hidden_dim,
                    num_hidden_layer=args.num_hidden_layer,
                    gru_hidden_dim=args.gru_hidden_dim,
                    device=device,
                    dropout_p=args.dropout_p,
                    attention_method=args.attention_method).to(device)

    # load trained parameter
    best_params = torch.load(args.bestmodel_path)
    model.load_state_dict(best_params)
    model.eval()

    # TODO :


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

    # Path parameters
    parser.add_argument("--vocab_path", type=str, default='./data/vocab.txt',
                        help="vocab.txt directory")
    parser.add_argument("--infer_data_path", type=str, default='./data/test_logs_split_20_10.txt',
                        help="train dataset directory")
    parser.add_argument("--bestmodel_path", type=str, required=True,
                        help="path of model to use")

    args = parser.parse_args()

    if args.device == 'cuda':
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device('cpu')
        logger.info('use cpu')

    set_seed(args)


if __name__ == '__main__':
    main()


