from data_loader import DataSets
from model import SkipLog
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
import torch
import os
import json

torch.manual_seed(0)
torch.cuda.manual_seed(0)


def train(train_data_path, test_data_path, vocab_path, model_save_dir,
          batch_size, epochs, lr, eval_step, max_len, grad_clip_norm, model_param,
          augmentation, ):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load Datasets
    tr_set = DataSets(data_path=train_data_path,
                      vocab_path=vocab_path,
                      augmentation=augmentation)

    test_set = DataSets(data_path=test_data_path,
                        vocab_path=vocab_path)

    tr_loader = DataLoader(dataset=tr_set,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=8,
                           pin_memory=True,
                           drop_last=True,
                           collate_fn=tr_set.batch_function)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size,
                             num_workers=8,
                             pin_memory=True,
                             drop_last=True,
                             collate_fn=tr_set.batch_function)

    # Load model
    vocab = tr_set.vocab
    model = SkipLog(vocab=vocab,
                    device='cuda',
                    **model_param)

    model.to(device)
    model.zero_grad()

    total_step = len(tr_loader) * epochs
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_step)
    warmup_scheduler = GradualWarmupScheduler(optimizer,
                                              multiplier=100,
                                              total_epoch=total_step * 0.1,
                                              after_scheduler=scheduler)

    # tensorboard
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    writer = SummaryWriter(model_save_dir)

    best_val_loss = 1e+9
    global_step = 0

    train_loss = 0
    train_acc = 0

    for epoch in tqdm(range(epochs), desc='epochs'):

        for step, batch in tqdm(enumerate(tr_loader), desc='steps', total=len(tr_loader)):
            model.train()

            encoder_mask, encoder_input, decoder_input, decoder_target = map(lambda x: x.to(device), batch[1:])

            inputs = {
                'encoder_mask': encoder_mask,
                'encoder_input': encoder_input,
                'decoder_input': decoder_input,
                'decoder_target': decoder_target,
            }

            outputs, loss = model(**inputs)

            pred = outputs.max(dim=2)[1].transpose(0, 1)  # (B x 2L)

            # mean accuracy except pad token
            not_pad = decoder_target != vocab.index('<PAD>')
            num_words = not_pad.sum()
            batch_acc = (pred[not_pad] == decoder_target[not_pad]).float().sum() / num_words

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            train_loss += loss
            train_acc += batch_acc

            optimizer.step()
            model.zero_grad()
            global_step += 1
            warmup_scheduler.step()

            show_lr = warmup_scheduler.get_lr()[0]
            writer.add_scalars('lr', {'lr': show_lr}, global_step)

            if global_step % eval_step == 0:

                val_loss, val_acc = evaluate(test_loader, model, vocab, device)

                writer.add_scalars('loss', {'train': train_loss / eval_step,
                                            'val': val_loss}, global_step)
                writer.add_scalars('acc', {'train': train_acc / eval_step,
                                           'val': val_acc}, global_step)

                tqdm.write('global_step: {:3}, '
                           'tr_loss: {:.3f}, '
                           'val_loss: {:.3f}, '
                           'tr_acc: {:.3f}, '
                           'val_acc: {:.3f} '
                           'lr: {:.3f}'.format(global_step,
                                               train_loss / eval_step,
                                               val_loss,
                                               train_acc / eval_step,
                                               val_acc,
                                               float(show_lr)))

                train_loss = 0
                train_acc = 0

                if val_loss < best_val_loss:
                    name = '/bestmodel_loss_' + str(round(val_loss, 3)) + '.bin'
                    torch.save(model.state_dict(), model_save_dir + name)
                    best_val_loss = val_loss


def evaluate(dataloader, model, vocab, device):
    val_loss = 0
    val_acc = 0

    for val_step, batch in enumerate(dataloader):
        model.eval()

        encoder_mask, encoder_input, decoder_input, decoder_target = map(lambda x: x.to(device), batch[1:])

        inputs = {
            'encoder_mask': encoder_mask,
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'decoder_target': decoder_target,
        }

        with torch.no_grad():
            outputs, loss = model(**inputs)

            pred = outputs.max(dim=2)[1].transpose(0, 1)  # (B x 2L)

            # mean accuracy except pad token
            not_pad = decoder_target != vocab.index('<PAD>')
            num_words = not_pad.sum()
            batch_acc = (pred[not_pad] == decoder_target[not_pad]).float().sum() / num_words

            val_loss += loss
            val_acc += batch_acc

    val_loss /= (val_step + 1)
    val_acc /= (val_step + 1)

    return val_loss.item(), val_acc


if __name__ == '__main__':

    # Load hyperparams
    with open('./hparams.json', 'r') as f:
        hparams = json.load(f)

    model_dir = hparams['train_hparams']['model_save_dir']

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    with open(model_dir + '/hparams.json', 'w') as f:
        json.dump(hparams, f, indent=4)

    train(**hparams['train_hparams'],
          model_param=hparams['model_hparams'])
