import pickle
import numpy as np

from keras.preprocessing.sequence import pad_sequences


def PaddingLabel(label, label_size):
    _label = []
    for labels in label:
        _tmp = [0]*label_size
        # _tmp = np.zeros(shape=label_size, dtype=int)
        for label_i in labels:
            _tmp[label_i] = 1
        _label.append(_tmp)
    return np.array(_label)


class Loader(object):
    """docstring for Loader"""
    def __init__(self, cfg):
        super(Loader, self).__init__()
        self.cfg = cfg
        self._dataname = cfg.dataname
        self.process()
        self.shuffle()

    @property
    def vocabsize(self):
        return self._vocabsize

    @property
    def labelsize(self):
        return self._labelsize

    @property
    def num_train(self):
        return self._num_train

    @property
    def num_test(self):
        return self._num_test

    @property
    def num_dev(self):
        return self._num_dev

    def load(self):
        if self._dataname == 'rcv1':
            test_data = pickle.load(open(r'data/RCV1-V2/test_data.pkl', 'rb'))
            train_data = pickle.load(open(r'data/RCV1-V2/train_data.pkl', 'rb'))
            if self.cfg.prew2v:
                vocab2id = pickle.load(open('./data/pretrainw2v/vocab2id.pkl', "rb"))
            else:
                vocab2id = pickle.load(open(r'data/RCV1-V2/vocab2id.pkl', 'rb'))
            return test_data, train_data, vocab2id
        elif self._dataname == 'aapd':
            if self.cfg.prew2v:
                vocab2id = pickle.load(open('./data/pretrainw2v/vocab2id.pkl', "rb"))
            else:
                vocab2id = pickle.load(open(r'data/AAPD/vocab2id.pkl', 'rb'))
            train_x = pickle.load(open(r'data/AAPD/train_x.pkl', 'rb'))
            train_y = pickle.load(open(r'data/AAPD/train_y.pkl', 'rb'))
            dev_x = pickle.load(open(r'data/AAPD/dev_x.pkl', 'rb'))
            dev_y = pickle.load(open(r'data/AAPD/dev_y.pkl', 'rb'))
            test_x = pickle.load(open(r'data/AAPD/test_x.pkl', 'rb'))
            test_y = pickle.load(open(r'data/AAPD/test_y.pkl', 'rb'))
            return vocab2id, train_x, train_y, test_x, test_y, dev_x, dev_y

    def process(self):
        if self.cfg.prew2v:
            prew2v = pickle.load(open('./data/pretrainw2v/wordvectors100d.pkl', "rb"))
        if self._dataname == 'aapd':
            self.vocab2id, train_x, train_y, test_x, test_y, dev_x, dev_y = self.load()
            self._vocabsize = len(self.vocab2id)
            self._labelsize = 54

            train_x = pad_sequences(train_x, maxlen=self.cfg.sen_len, padding='post', truncating='post')
            train_y = PaddingLabel(train_y, 54)

            self.test_x = pad_sequences(test_x, maxlen=self.cfg.sen_len, padding='post', truncating='post')
            self.test_y = PaddingLabel(test_y, 54)

            self.dev_x = pad_sequences(dev_x, maxlen=self.cfg.sen_len, padding='post', truncating='post')
            self.dev_y = PaddingLabel(dev_y, 54)

            self._num_train = len(train_x)
            self._num_test = len(test_x)
            self._num_dev = len(dev_x)

            self.train = np.array([(a, b) for _, (a, b) in enumerate(zip(train_x, train_y))])

        if self._dataname == 'rcv1':
            test_data, train_data, self.vocab2id = self.load()
            self._vocabsize = len(self.vocab2id)
            self._labelsize = 103

            train_x, train_y = [], []
            for sample in train_data:
                train_x.append(sample[0])
                train_y.append(sample[1])
            train_x = pad_sequences(train_x, maxlen=self.cfg.sen_len, padding='post', truncating='post')
            train_y = PaddingLabel(train_y, 103)
            self.dev_x = train_x[-1000:]
            self.dev_y = train_y[-1000:]
            train_x = train_x[:-1000]
            train_y = train_y[:-1000]

            test_x, test_y = [], []
            for sample in test_data:
                test_x.append(sample[0])
                test_y.append(sample[1])
            self.test_x = pad_sequences(test_x, maxlen=self.cfg.sen_len, padding='post', truncating='post')
            self.test_y = PaddingLabel(test_y, 103)

            self._num_train = len(train_x)
            self._num_test = len(test_x)
            self._num_dev = len(dev_x)

            self.train = np.array([(a, b) for _, (a, b) in enumerate(zip(train_x, train_y))])

    def shuffle(self):
        np.random.shuffle(self.train)