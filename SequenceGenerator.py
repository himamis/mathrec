from keras.utils import Sequence
from PIL import Image
import utils
import numpy as np

class SequenceGenerator(Sequence):
    """description of class"""

    def __init__(self, set_dir, img_dir, set='train', batch_size=32):
        self.set_list = utils._read_pkl(set_dir + set + '.pkl') # list: [(img_name, seq),...], seq: [token,...], img_name: str, token: int
        self.img_dir = img_dir
        self.batch_size = batch_size

    def __len__(self):
        #return int(np.ceil(len(self.set_list) / float(self.batch_size)))
        return 5

    def __getitem__(self, i):
        batch_x = self.set_list[i*self.batch_size:(i+1)*self.batch_size]
        max_len = len(batch_x[-1][1])
        x_seqs = np.zeros((self.batch_size, max_len), dtype=np.int32)
        x_imgs = []
        for i, x in enumerate(batch_x):
            x_seqs[i][:len(x[1])] = x[1]
            x_imgs.append(np.asarray(Image.open(self.img_dir+x[0]).convert('YCbCr'))[:,:,0][:,:,None])
        y_seqs = x_seqs[:,1:,None]
        x_seqs = x_seqs[:,:-1]
        return [np.array(x_imgs), np.array(x_seqs)], y_seqs