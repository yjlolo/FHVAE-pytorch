from torch.utils.data import Dataset
from collections import defaultdict
import os
import numpy as np

class LogSpectrumDataset(Dataset):
    def __init__(self, root_dir='./datasets/timit_processed'):
        self.data = defaultdict(list)
        self.nmu2 = len(os.listdir(root_dir))
        xlist = []
        ylist = []
        nlist = []
        # for seq_id in os.listdir(root_dir):
        for seq_id in sorted(os.listdir(root_dir)):
            audio = np.load(os.path.join(root_dir, seq_id, 'audio.npy'))
            num_spec = len(os.listdir(os.path.join(root_dir, seq_id))) - 1
            for i in range(num_spec):
                log_spectrum = np.load(os.path.join(root_dir, seq_id, 'log-mag-spectrum-%d.npy' % i))
                xlist.append(log_spectrum)
                ylist.append(int(seq_id))
                nlist.append(np.shape(log_spectrum)[0])
        self.data['x'] = xlist
        self.data['y'] = ylist
        self.data['n'] = nlist

    def __len__(self):
        return len(self.data['x'])

    def __getitem__(self, idx):
        return {'x': self.data['x'][idx], 'y': self.data['y'][idx], 'n': self.data['n'][idx]}