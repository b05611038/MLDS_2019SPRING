import math

import tensorflow as tf 
import numpy as np 

class Data_loader(object):
    def __init__(self, data_file, batch_size, max_length):
        self.data_file = data_file
        self.batch_size = batch_size
        self.max_length = max_length
        self.vocab_name = 'vocab.txt'
        self._pad, self._bos, self._eos, self._unk = 0, 1, 2, 3
        self.load_data()
        self.generate_batch_number()
        self.reset_batch_pointer()

    def load_data(self):
        self.loaded_data = [line.strip().replace(' ', '') for line in open(self.data_file, 'r')]
        self.data_length = len(self.loaded_data)
        self.load_dictionary()
        self.prepare_text_data()
        
    def load_dictionary(self):
        print ('loading dictionary')
        self.vocab, self.rev_vocab = {}, {}
        with open(self.vocab_name, 'r') as fin:
            for line in fin:
                i, w = line.strip().split()
                self.vocab[str(w)] = int(i)
                self.rev_vocab[int(i)] = str(w)

    def sentence_to_id(self, sentence):
        return [int(self.vocab.get(w, self._unk)) for w in sentence]

    def prepare_text_data(self):
        text_id = []
        text_weight = []
        for data in self.loaded_data:
            sentence = self.sentence_to_id(data)
            if len(sentence) >= self.max_length:
                text_weight.append(self.max_length)
                text_id.append(sentence[:self.max_length-1] + [self._eos])
            else:
                text_weight.append(len(sentence)+1)
                text_id.append(sentence + [self._eos] + [self._pad]*(self.max_length-1-len(sentence)))
        self.text_id = np.asarray(text_id)
        self.text_weight = np.asarray(text_weight)

    def generate_batch_number(self):
        self.batch_number = (self.data_length-1) // self.batch_size + 1

    def reset_batch_pointer(self):
        self.pointer = 0

    def update_pointer(self):
        if self.data_length - self.pointer <= self.batch_size:
            self.pointer = self.pointer + self.batch_size - self.data_length
        else:
            self.pointer += self.batch_size

    def get_batch(self, test):
        for i in range(self.batch_number):
            if self.data_length - self.pointer < self.batch_size:
                sentence = self.text_id[self.pointer:]
                weight  = self.text_weight[self.pointer:]
                ground_truth = self.add_BOS(sentence, self.data_length - self.pointer)
                yield sentence, weight, ground_truth
            else:
                sentence = self.text_id[self.pointer:self.pointer+self.batch_size]
                weight = self.text_weight[self.pointer:self.pointer+self.batch_size]
                ground_truth = self.add_BOS(sentence, self.batch_size)
                self.update_pointer()
                yield sentence, weight, ground_truth

    def add_BOS(self, caption, size):
        BOS_slice = np.ones([size, 1], dtype=np.int32)*self._bos 
        return np.concatenate([BOS_slice, caption[:, :-1]], axis=-1)

class Language_model(object):
    def __init__(self):
        self.hidden_size   = 256
        self.vocab_size    = 3000
        self.max_length    = 30
        self.emb_size      = 256
        self.global_step   = tf.Variable(0, trainable=False, name='global_step')
        self.build_graph()

    def build_graph(self):
        self.ground_truth   = tf.placeholder(tf.int32, shape=[None, self.max_length]) # w/  GO
        self.target_inputs  = tf.placeholder(tf.int32, shape=[None, self.max_length]) # w/o GO
        self.target_weights = tf.placeholder(tf.int32, shape=[None])

        with tf.variable_scope('encoder'):
            self.log_prob = encoder(self.ground_truth, self.hidden_size, self.vocab_size)

        with tf.variable_scope('loss'):
            self.loss = sequence_loss(self.log_prob, self.target_inputs, self.target_weights)

def encoder(encoder_inputs, hidden_size, vocab_size):
    with tf.variable_scope('encoder_rnn'):
        # cell
        encoder_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
        # embedding
        init = tf.contrib.layers.xavier_initializer()
        word_embedding = tf.get_variable(name='embedding', shape=[vocab_size, hidden_size], initializer=init)
        encoder_inputs = tf.nn.embedding_lookup(word_embedding, encoder_inputs)
        # rnn
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                cell = encoder_cell,
                inputs = encoder_inputs,
                dtype = tf.float32)
        log_prob = tf.layers.dense(encoder_outputs, vocab_size, name='output_projection')
    return log_prob

def sequence_loss(decoder_outputs, target_inputs, target_weights):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_inputs, logits=decoder_outputs)
    mask = tf.sequence_mask(target_weights, tf.shape(target_inputs)[1])
    loss = tf.where(mask, loss, tf.zeros_like(loss))
    mean_loss = tf.reduce_sum(loss) / tf.cast(tf.reduce_sum(target_weights), tf.float32)
    return mean_loss

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def test(sess, s, data_loader):
    print ('Testing...')
    epoch_loss = AverageMeter()
    for target_inputs, target_weights, ground_truth in data_loader.get_batch(True):
        feed_dict = {
                    s.ground_truth   :ground_truth,
                    s.target_inputs  :target_inputs,
                    s.target_weights :target_weights
        }
        [loss] = sess.run([s.loss], feed_dict=feed_dict)
        epoch_loss.update(loss, np.sum(target_weights))
    return math.exp(epoch_loss.avg)