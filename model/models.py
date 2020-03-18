import pickle
import tensorflow as tf
import numpy as np

from data.loader import Loader
from model.caps import PrimaryCaps, FullyConnectCaps
from model.loss import margin_loss, cross_entropy_loss, margin_loss_v1
from model.metrics import micro_p_r_f, hamming_loss


class Model(object):
    """
        Network Architectures
    """
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg

    def placeholders(self):
        self.inputs = tf.placeholder(tf.int32, [self.cfg.batch_size, self.cfg.sen_len], name="inputs")
        self.labels = tf.placeholder(tf.float32, [self.cfg.batch_size, self.cfg.label_size], name="labels")
        self.learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")
        self.training = tf.placeholder(tf.bool, shape=(), name="training_flag")
        if self.cfg.prew2v:
            prew2v = pickle.load(open('./data/pretrainw2v/wordvectors100d.pkl', "rb"))
            self.embedding = tf.Variable(prew2v, dtype=tf.float32)
            self.embed_inputs = tf.nn.embedding_lookup(self.embedding, self.inputs)
        else:
            self.embed_inputs = tf.contrib.layers.embed_sequence(self.inputs, self.cfg.vocab_size, self.cfg.embed_size)

    def CapsuleModel(self):
        inputs = tf.expand_dims(self.embed_inputs, axis=-2)
        caps, activations = PrimaryCaps(inputs=inputs, num_out_caps=8, out_caps_shape=64, training=self.training)

        caps, activations = FullyConnectCaps(inputs=caps, activations=activations, training=self.training,
                                             num_out_caps=128, out_caps_shape=64, iter_times=self.cfg.iterations,
                                             amendment=True, leakysoftmax=False, name="FeatureCapsuleLayer")

        self.output, self.activations = FullyConnectCaps(inputs=caps, activations=activations,
                                                         training=self.training, num_out_caps=self.cfg.label_size,
                                                         out_caps_shape=64, iter_times=self.cfg.iterations,
                                                         amendment=False, leakysoftmax=True, name="OutputCapsuleLayer")

    def losses(self):
        if self.cfg.loss_fn == 'margin_loss':
            self.loss = margin_loss(self.labels, self.activations)

        if self.cfg.loss_fn == 'cross_entropy':
            self.loss = cross_entropy_loss(self.labels, self.activations)

        if self.cfg.loss_fn == 'margin_loss_v1':
            self.loss = margin_loss_v1(self.labels, self.activations)

    def BuildArch(self):
        graph = tf.Graph()
        with graph.as_default():
            self.dataset = Loader(self.cfg)
            self.cfg.vocab_size = self.dataset.vocabsize+1
            self.cfg.label_size = self.dataset.labelsize

            self.placeholders()
            self.CapsuleModel()
            self.losses()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.train_op = self.optimizer.minimize(self.loss, name='train_op')
        print('model builded')
        return graph

    def getData(self, mode):
        if mode == 'train':
            self.dataset.shuffle()
            train_x, train_y = [], []
            for (a, b) in self.dataset.train:
                train_x.append(a)
                train_y.append(b)
            return train_x, train_y

    def run_one_epoch(self, sess, epoch, training, num_train, train_x, train_y):
        num_batchs = num_train // self.cfg.batch_size
        # num_batchs = 1000
        train_loss = []
        precision, recall, hammingloss = [], [], []
        for batch in range(1, num_batchs+1):
            x = train_x[(batch-1)*self.cfg.batch_size: batch*self.cfg.batch_size]
            y = train_y[(batch-1)*self.cfg.batch_size: batch*self.cfg.batch_size]
            _, loss, preds = sess.run(fetches=(self.train_op, self.loss, self.activations),
                                      feed_dict={self.inputs:x,
                                                 self.labels:y,
                                                 self.learning_rate:self.cfg.lr,
                                                 self.training:training})
            preds = np.rint(preds)
            y = np.rint(y)
            micro_p, micro_r, micro_f, _ = micro_p_r_f(y, preds)
            hm = hamming_loss(y, preds)
            precision.append(micro_p)
            recall.append(micro_r)
            hammingloss.append(hm)
            train_loss.append(loss)
            avg_loss = np.mean(train_loss)

            show_msg = ' - Loss:{:>2.3f} \r'.format(avg_loss)
            progress = 'Epoch:{:>2d} ['.format(epoch) + '='*int(25*(batch/(num_batchs))) + '>' + '.'*(25-int(25*(batch/(num_batchs)))) + ']'
            print(progress+show_msg, end='')
        avg_hm = np.mean(hammingloss)
        avg_p = np.mean(precision)
        avg_r = np.mean(recall)
        avg_f1 = 2 * avg_p * avg_r / (avg_p + avg_r)
        show_msg = 'Epoch:{:>2d} - Loss:{:>2.3f} -  HM loss:{:1.5f} - P:{:1.4f} - R:{:1.4f} - F1:{:1.4f}'.format(
                        epoch,
                        avg_loss,
                        avg_hm,
                        avg_p,
                        avg_r,
                        avg_f1)
        print(show_msg)

    def run_dev(self, sess, num_dev, dev_x, dev_y):
        num_batchs = num_dev // self.cfg.batch_size
        dev_losss = []
        precision, recall, hammingloss = [], [], []
        for batch in range(1, num_batchs+1):
            x_data = dev_x[(batch-1)*self.cfg.batch_size: batch*self.cfg.batch_size]
            y_data = dev_y[(batch-1)*self.cfg.batch_size: batch*self.cfg.batch_size]
            dev_loss, dev_preds = sess.run(fetches=(self.loss, self.activations),
                                           feed_dict={self.inputs:x_data,
                                                      self.labels:y_data,
                                                      self.training:False})
            preds = np.rint(dev_preds)
            y = np.rint(y_data)
            micro_p, micro_r, micro_f, _ = micro_p_r_f(y, preds)
            hm = hamming_loss(y, preds)
            precision.append(micro_p)
            recall.append(micro_r)
            dev_losss.append(dev_loss)
            hammingloss.append(hm)
        avg_loss = np.mean(dev_losss)
        avg_hm = np.mean(hammingloss)
        avg_p = np.mean(precision)
        avg_r = np.mean(recall)
        avg_f1 = 2 * avg_p * avg_r / (avg_p + avg_r)
        show_msg = 'DEV: Loss:{:>2.3f} -  HM loss:{:1.5f} - P:{:1.4f} - R:{:1.4f} - F1:{:1.4f}'.format(
                        avg_loss,
                        avg_hm,
                        avg_p,
                        avg_r,
                        avg_f1)
        print(show_msg)
        return avg_loss, avg_f1

    def train(self):
        graph_cfg = tf.ConfigProto()
        graph_cfg.gpu_options.allow_growth = True
        graph = self.BuildArch()

        with tf.Session(graph=graph, config=graph_cfg) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            train_x, train_y = self.getData("train")
            best_f1 = 1e-9
            for epoch in range(1, self.cfg.epochs):
                self.run_one_epoch(sess, epoch, True, self.dataset.num_train, train_x, train_y)
                dev_loss, dev_f1 = self.run_dev(sess, self.dataset.num_dev, self.dataset.dev_x, self.dataset.dev_y)

                if best_f1 <= dev_f1:
                    improve = (dev_f1 - best_f1)/best_f1
                    best_f1 = dev_f1
                    saver.save(sess, self.cfg.bestpath)
                    msg = 'Best F1-score model saved! Improved {:.2%}!'.format(improve)
                    if epoch == 1:
                        msg = 'Best F1-score model saved! Improved from 0.0 to {:.2f}!'.format(best_f1)
                    print(msg)