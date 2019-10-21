import os
import pickle as pkl
import random
from collections import defaultdict
import numpy as np
import torch
from loader.preprocess import TextProcessor

b2c_dir = 'b2c/'


def random_split():
    f_names = list(os.listdir(b2c_dir))
    random.shuffle(f_names)
    train_size, val_size = int(len(f_names) * 0.6), int(len(f_names) * 0.2)
    with open('loader/train.frange', 'w') as out_file:
        for f_name in f_names[:train_size]:
            out_file.write(f_name + '\n')

    with open('loader/val.frange', 'w') as out_file:
        for f_name in f_names[train_size: train_size + val_size]:
            out_file.write(f_name + '\n')

    with open('loader/test.frange', 'w') as out_file:
        for f_name in f_names[train_size + val_size:]:
            out_file.write(f_name + '\n')

class BaseLoader:

    def __init__(self, data_ids=None, shuffle=False, batch_size=32, repeat=False, text_processor=None, use_gpu=False):
        # the pkl files to be loaded
        if data_ids is None:
            self.pkl_dirs = [b2c_dir + f_name for f_name in os.listdir(b2c_dir)]
        else:
            self.pkl_dirs = [b2c_dir + f_name for f_name in data_ids]
        self.cur_ptr = 0
        self.num_data = len(self.pkl_dirs)
        self.batch_size = batch_size
        self.repeat = repeat
        if text_processor is not None:
            self.text_processor = text_processor
        else:
            self.text_processor = TextProcessor(max_sent_length=100)
        self.vocab_size = self.text_processor.vocab_size

        if shuffle:
            random.shuffle(self.pkl_dirs)
        self.use_gpu = use_gpu

    def load_pkls(self, pkl_dirs):
        return [pkl.load(open(pkl_dir, 'rb')) for pkl_dir in pkl_dirs]

    def dicts2batch(self, dicts):
        result = defaultdict(list)
        for d in dicts:
            if d['board_state'].shape[1] != 19:
                continue
            for key in d:
                result[key].append(d[key])
        result['board_state'] = torch.tensor(np.array(result['board_state']), dtype=torch.float32)
        sents, lengths = self.text_processor.npifytext(result['comments'])
        lengths = np.minimum(lengths, np.max(lengths) - 1)

        sents, lengths = torch.tensor(sents), torch.tensor(lengths)
        result['tok_in'] = sents[:, :-1]
        result['tok_out'] = sents[:, 1:]
        result['lengths'] = torch.tensor(lengths)
        if self.use_gpu:
            for key in ['tok_in', 'tok_out', 'lengths', 'board_state']:
                result[key] = result[key].cuda()
        return result

    def next_pkl_dirs(self, batch_size):
        if self.repeat:
            pkl_dirs = [self.pkl_dirs[(self.cur_ptr + i) % self.num_data] for i in range(batch_size)]
        else:
            pkl_dirs = self.pkl_dirs[self.cur_ptr: self.cur_ptr + batch_size]
        self.cur_ptr += batch_size
        if self.repeat:
            self.cur_ptr %= self.num_data
        return pkl_dirs

    def batch_generator(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        pkl_dirs = self.next_pkl_dirs(batch_size)
        while len(pkl_dirs) != 0:
            dicts = self.load_pkls(pkl_dirs)
            batch = self.dicts2batch(dicts)
            pkl_dirs = self.next_pkl_dirs(batch_size)
            yield batch

    @classmethod
    def obtain_generators(cls, use_gpu=False, debug=False):
        with open('loader/train.frange', 'r') as in_file:
            train_f_names = in_file.read().strip().split('\n')
            if debug:
                train_f_names = train_f_names[:100]
        with open('loader/val.frange', 'r') as in_file:
            val_f_names = in_file.read().strip().split('\n')
            if debug:
                 val_f_names = val_f_names[:100]
        train_loader = cls(data_ids=train_f_names, use_gpu=use_gpu, repeat=True)
        train_generator = train_loader.batch_generator()
        def eval_generator_init():
            return cls(data_ids=val_f_names, use_gpu=use_gpu, repeat=False).batch_generator()
        return train_generator, eval_generator_init, train_loader
