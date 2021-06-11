import numpy as np
import tensorflow as tf
import source.utils as utils
from sklearn.utils import shuffle

class Dataset(object):

    def __init__(self, normalize=True):

        print("\nInitializing Dataset...")

        self.normalize = normalize

        (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
        self.x_tr, self.y_tr = x_tr, y_tr
        self.x_te, self.y_te = x_te, y_te

        self.x_tr = np.ndarray.astype(self.x_tr, np.float32)
        self.x_te = np.ndarray.astype(self.x_te, np.float32)

        self.__normalizing()

        self.num_tr, self.num_te = self.x_tr.shape[0], self.x_te.shape[0]
        self.idx_tr, self.idx_te = 0, 0

        x_sample, y_sample = self.x_te[0], self.y_te[0]
        self.height = x_sample.shape[0]
        self.width = x_sample.shape[1]
        try: self.channel = x_sample.shape[2]
        except: self.channel = 1

        self.num_class = (y_te.max()+1)

    def __normalizing(self):

        for idx, _ in enumerate(self.x_tr):
            self.x_tr[idx] = utils.min_max_norm(self.x_tr[idx])

        for idx, _ in enumerate(self.x_te):
            self.x_te[idx] = utils.min_max_norm(self.x_te[idx])

    def __reset_index(self):

        self.idx_tr, self.idx_te = 0, 0

    def next_batch(self, batch_size=1, tt=0):

        if(tt == 0):
            idx_d, num_d, data_x, data_y = self.idx_tr, self.num_tr, self.x_tr, self.y_tr
        elif(tt == 1):
            idx_d, num_d, data_x, data_y = self.idx_te, self.num_te, self.x_te, self.y_te

        batch_x, batch_y, terminator = [], [], False
        while(True):
            try:
                tmp_x, tmp_y = data_x[idx_d].copy(), data_y[idx_d].copy()
            except:
                idx_d = 0
                self.x_tr, self.y_tr = shuffle(self.x_tr, self.y_tr)
                terminator = True
                break
            else:
                if(tt == 0):
                    if(np.random.rand(1) < 0.3):
                        tmp_x = -tmp_x + 1
                batch_x.append(np.expand_dims(tmp_x, axis=-1))
                batch_y.append(np.diag(np.ones(self.num_class))[tmp_y])
                idx_d += 1
            if(len(batch_x) == batch_size): break

        batch_x = np.asarray(batch_x)
        batch_y = np.asarray(batch_y)

        if(tt == 0):
            self.idx_tr = idx_d
        elif(tt == 1):
            self.idx_te = idx_d

        return {'x':batch_x.astype(np.float32), 'y':batch_y.astype(np.float32), 't':terminator}
