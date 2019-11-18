import json
from pathlib import Path
import logging
import os
import pandas as pd
from datetime import datetime

formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')


def get_logger(name=None, level=logging.DEBUG):
    logger = logging.getLogger(name if name is not None else __name__)
    logger.handlers.clear()
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    return logger


class HyperParamWriter:
    def __init__(self, dir):
        self.dir = dir
        self.hparams = None
        self.load()
        self.writer = dict()

    def update(self, args, tr_loss, tr_acc, val_loss, val_acc):
        now = datetime.now()
        date = '%s-%s-%s %s:%s' % (now.year, now.month, now.day, now.hour, now.minute)
        self.writer.update({'date': date})

        self.writer.update(
            {
                'train_loss': tr_loss,
                'train_accuracy': tr_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
            }
        )

        self.writer.update(vars(args))

        if self.hparams is None:
            self.hparams = pd.DataFrame(self.writer, index=[0])
        else:
            self.hparams = self.hparams.append(self.writer, ignore_index=True)
        self.save()

    def save(self):
        assert self.hparams is not None
        self.hparams.to_csv(self.dir, index=False)

    def load(self):
        path = os.path.split(self.dir)[0]
        if not os.path.exists(path):
            os.mkdir(path)
            self.hparams = None
        elif os.path.exists(self.dir):
            self.hparams = pd.read_csv(self.dir)
        else:
            self.hparams = None


class SummaryManager:
    def __init__(self, model_dir):
        if not isinstance(model_dir, Path):
            model_dir = Path(model_dir)
        self._model_dir = model_dir
        self._summary = {}

    def save(self, filename):
        with open(self._model_dir / filename, mode='w') as io:
            json.dump(self._summary, io, indent=4)

    def load(self, filename):
        with open(self._model_dir / filename, mode='r') as io:
            metric = json.loads(io.read())
        self.update(metric)

    def update(self, summary):
        self._summary.update(summary)

    def reset(self):
        self._summary = {}

    @property
    def summary(self):
        return self._summary

