import re
import unicodedata
import argparse
from collections import Counter

train_data_dir = './data/train_logs_split_20.txt'
special_tokens = ['<PAD>', '<EOL>', '<SOL>', '<UNK>']


def create_vocab(data):
    ctr = Counter()
    ctr.update(special_tokens)
    for log in data:
        preprocessed_log = normalize_string(log_parser(log)).split()
        ctr.update(preprocessed_log)
    with open('./data/vocab.txt', 'w', encoding='utf-8') as f:
        for vocab in list(ctr.keys()):
            f.write(vocab + '\n')


def load_data(data_dir):
    with open(data_dir, 'r') as f:
        data = f.read().splitlines()
    return data


def log_parser(log):
    log = unicode_to_ascii(log.lower().strip())

    id_regex = re.compile('blk_.\d+')
    ip_regex = re.compile('\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?')
    num_regex = re.compile('\d*\d')

    tmp = re.sub(id_regex, "BLK", log)
    tmp = re.sub(ip_regex, "IP", tmp)
    tmp = re.sub(num_regex, "NUM", tmp)

    normalized_output = normalize_string(tmp)
    return normalized_output


def get_blk_id(log):
    id_extractor = re.compile('blk_.\d+')
    return id_extractor.search(log).group()


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = re.sub(r"([.!?])", r" ", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=False,
                        help='Directory of corpus that you want to make vocabulary from')

    args = parser.parse_args()

    dir = train_data_dir if args.data_dir is None else args.data_dir

    data = load_data(dir)
    print('data loaded')

    create_vocab(data)
    print('vocab.txt created')


if __name__ == '__main__':
    main()
