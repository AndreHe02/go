import os
import pickle as pkl
import random
from collections import defaultdict
import numpy as np
import torch

b2c_dir = 'b2c/'

class BaseLoader:

    def __init__(self, data_ids=None, shuffle=False, batch_size=32):
        # the pkl files to be loaded
        if data_ids is None:
            self.pkl_dirs = [b2c_dir + f_name for f_name in os.listdir(b2c_dir)]
        else:
            self.pkl_dirs = [b2c_dir + ('%d.sgf-%d.pkl' % data_id) for data_id in data_ids]
        self.cur_ptr = 0
        self.num_data = len(self.pkl_dirs)
        self.batch_size = batch_size

        if shuffle:
            random.shuffle(self.pkl_dirs)

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
        return result

    def next_pkl_dirs(self, batch_size):
        pkl_dirs = [self.pkl_dirs[(self.cur_ptr + i) % self.num_data] for i in range(batch_size)]
        self.cur_ptr += batch_size
        self.cur_ptr %= self.num_data
        return pkl_dirs

    def next_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        pkl_dirs = self.next_pkl_dirs(batch_size)
        dicts = self.load_pkls(pkl_dirs)
        batch = self.dicts2batch(dicts)
        return batch
