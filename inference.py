from data_loader import DataSets
from model import SkipLog
from utils import get_logger
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import torch
import argparse
import os
import random
import logging

from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

logger = get_logger('Skip-Log')
logger.setLevel(logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def inference(args, device, model, loader):
    test_blks = []
    test_losses = []

    for step, batch in tqdm(enumerate(loader), desc='steps', total=len(loader)):
        blk = batch[0]
        encoder_mask, encoder_input, decoder_input, decoder_target = map(lambda x: x.to(device), batch[1:])
        inputs = {
                    'encoder_mask': encoder_mask,
                    'encoder_input': encoder_input,
                    'decoder_input': decoder_input,
                    'decoder_target': decoder_target,
                }
        with torch.no_grad():
            # encoder_input : (B x L)
            _, loss = model(**inputs)
            test_blks += blk
            # take minimum loss value of the log sentence
            test_losses += loss.reshape(len(blk), -1).mean(dim=1).tolist()

    # blk_id 별로 묶기
    grouped_data = defaultdict(list)
    for blk, loss in zip(test_blks, test_losses):
        grouped_data[blk].append(loss)

    blks = list(grouped_data.keys())
    # take maximum loss value of the block id
    max_losses = [max(i) for i in grouped_data.values()]

    return blks, max_losses

def main():
    parser = argparse.ArgumentParser()
    # Path parameters
    parser.add_argument("--bestmodel_path", type=str, required=True,
                        help="path of model to use")
    parser.add_argument("--vocab_path", type=str, default='./data/vocab.txt',
                        help="vocab.txt directory")
    parser.add_argument("--infer_data_path", type=str, default= './data/test_logs_split_20.txt',
                        help="train dataset directory")
    parser.add_argument("--truelabel_path", type=str, default='./data/test_ids_split_20.csv',
                        help="path of true labels")
    args = parser.parse_args()

    # Load arguments and best model parameters
    best_params = torch.load(os.path.join('model_saved/', args.bestmodel_path, 'best_model.bin'))
    best_args = torch.load(os.path.join('model_saved/', args.bestmodel_path, 'training_args.bin'))
    set_seed(best_args)

    # Load Vocab
    with open(args.vocab_path, 'r') as f:
        vocab = f.read().splitlines()

    # Load Datasets
    infer_set = DataSets(data_path=args.infer_data_path,
                         vocab_path=args.vocab_path,
                         augmentation=False)

    infer_loader = DataLoader(dataset=infer_set,
                              batch_size=2048,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=True,
                              drop_last=False,
                              collate_fn=infer_set.batch_function)

    if best_args.device == 'cuda':
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        logger.info('use gpu')
    else:
        device = torch.device('cpu')
        logger.info('use cpu')

    # Load model
    model = SkipLog(vocab=vocab,
                    embedding_hidden_dim=best_args.embedding_hidden_dim,
                    num_hidden_layer=best_args.num_hidden_layer,
                    gru_hidden_dim=best_args.gru_hidden_dim,
                    device=device,
                    dropout_p=best_args.dropout_p,
                    attention_method=best_args.attention_method,
                    mode='eval').to(device)

    # load trained parameter
    model.load_state_dict(best_params)  

    logger.info('***** Inference starts *****')
    blk_ids, max_losses = inference(best_args, device, model, infer_loader)
    
    # Load true labels
    test_label = pd.read_csv(args.truelabel_path)
    blk2label = {blk: 1 if label == 'Anomaly' else 0 for blk, label in zip(test_label['BlockId'], test_label['Label'])}
    true_y = [blk2label[b] for b in blk_ids]

    # Draw ROC curve
    fpr, tpr, thresholds = roc_curve(true_y, max_losses)
    fig = plt.figure(figsize=(10,7))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--', label="random guess")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    result = 'AUROC : {:.5f}'.format(auc(fpr, tpr))
    plt.savefig(os.path.join(best_args.save_path, result), dpi=300)
    logger.info(result)
    logger.info('***** Inference done *****')

if __name__ == '__main__':
    main()


