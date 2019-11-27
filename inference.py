from data_loader import DataSets
from model import SkipLog
from utils import get_logger
from torch.utils.data import DataLoader
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import argparse
import os
import random
import logging

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

logger = get_logger('Skip-Log')
logger.setLevel(logging.INFO)


class InferResultWriter:
    def __init__(self, dir):
        self.dir = dir
        self.results = None
        self.load()
        self.writer = dict()

    def update(self, args, auroc, f1, precision, recall, eer):
        now = datetime.now()
        date = '%s-%s-%s %s:%s' % (now.year, now.month, now.day, now.hour, now.minute)
        self.writer.update({'date': date})

        self.writer.update(
            {
                'AUROC': auroc,
                'best_F1': f1,
                'best_precision': precision,
                'best_recall': recall,
                'EER': eer,
            }
        )

        self.writer.update(vars(args))

        if self.results is None:
            self.results = pd.DataFrame(self.writer, index=[0])
        else:
            self.results = self.results.append(self.writer, ignore_index=True)
        self.save()

    def save(self):
        assert self.results is not None
        self.results.to_csv(self.dir, index=False)

    def load(self):
        path = os.path.split(self.dir)[0]
        if not os.path.exists(path):
            os.mkdir(path)
            self.results = None
        elif os.path.exists(self.dir):
            self.results = pd.read_csv(self.dir)
        else:
            self.results = None


def set_seed(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def inference(args, device, model, loader):
    model.eval()
    test_blks = []
    test_losses = []

    for step, batch in tqdm(enumerate(loader), desc='steps', total=len(loader)):
        blk = batch[0]
        test_blks += blk
        
        encoder_mask, encoder_input, decoder_input, decoder_target = map(lambda x: x.to(device), batch[1:])
        inputs = {
                    'encoder_mask': encoder_mask,
                    'encoder_input': encoder_input,
                    'decoder_input': decoder_input,
                    'decoder_target': decoder_target,
                }
        with torch.no_grad():
            # encoder_input : (B x L)
            _, prev_loss, next_loss = model(**inputs)
            concat_loss = torch.cat((prev_loss.unsqueeze(0), next_loss.unsqueeze(0)), dim=0)
            
            # take maximum value between prev and next loss
            loss = torch.max(concat_loss, dim=0)[0]
            
            # take max loss value of the log sentence
            test_losses += loss.reshape(len(blk), -1).max(dim=1)[0].tolist()

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
    logger.info("Use parameter of {}".format(args.bestmodel_path))
    best_params = torch.load(os.path.join(args.bestmodel_path, 'best_model.bin'))
    best_args = torch.load(os.path.join(args.bestmodel_path, 'training_args.bin'))
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
                              shuffle=False,
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
    auroc = auc(fpr, tpr)
    logger.info('AUROC : {:.5f}'.format(auroc))
    
    precision, recall, thresholds = precision_recall_curve(true_y, max_losses)
    f1 = [2 * (pr * re) / (pr + re + 1e-10) for pr, re in zip(precision, recall)]
    sort_by_f1 = np.argmax(f1)
    best_f1 = np.array(f1)[sort_by_f1]
    best_precision = precision[sort_by_f1]
    best_recall = recall[sort_by_f1]
    
    logger.info("best f1 : {}".format(best_f1))
    logger.info("best precision : {}".format(best_precision))
    logger.info("best recall : {}".format(best_recall))

    fnr = 1 - tpr
    EER = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
    logger.info("EER : {}".format(EER))

    result = 'AUROC:{:.3f}.F1:{:.3f}.EER:{:.3f}.ROC_curve.png'.format(auroc, best_f1, EER)
    plt.savefig(os.path.join(args.bestmodel_path, result), dpi=300)
    logger.info('***** Inference done *****')

    # Write results
    result_writer = InferResultWriter('./inference_result/results.csv')
    result_writer.update(best_args, auroc, best_f1, best_precision, best_recall, EER)

if __name__ == '__main__':
    main()


