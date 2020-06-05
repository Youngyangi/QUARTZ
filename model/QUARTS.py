import tensorflow as tf
from model.Encoder import LSTM_encoder
from model.Decoder import LSTM_decoder


class QUARTS(tf.keras.Model):
    def __init__(self, embedding_size, vocab_size, max_len_q, max_len_i):
        super(QUARTS, self).__init__()
        self.encoder = LSTM_encoder(embedding_size, name='encoder')
        self.decoder = LSTM_decoder(embedding_size, vocab_size, name='decoder')
        self.classifier = tf.keras.Sequential([
            # tf.keras.layers.Dense(),
            tf.keras.layers.Dense(1, activation='softmax')
        ])
        self.word_attention = Word_by_word_attention(embedding_size, name='word-by-word-attention')
        self.max_len_query = max_len_q
        self.max_len_item = max_len_i


class Word_by_word_attention(tf.keras.layers.Layer):
    def __init__(self, embedding_size):
        super(Word_by_word_attention, self).__init__()
        self.W_h = tf.random.uniform([3*embedding_size, embedding_size])
        self.w = tf.random.uniform([embedding_size, 1])
        self.W_r = tf.random.uniform([embedding_size, embedding_size])
        self.W_x = tf.random.uniform([3*embedding_size, embedding_size])

    def call(self, query_hiddens, item_hiddens, **kwargs):
        # query_hiddens.shape == [batch, sequence_len, embedding_size]
        # item_hiddens.shape == [batch, sequence_len, embedding_size]
        r = item_hiddens[:, -1:, :]
        q = query_hiddens[:, -1:, :]
        query_length = query_hiddens.shape[1]
        item_length = item_hiddens.shape[1]
        for t in range(query_length):
            # M.shape == [batch， sequence_length, embedding_size]
            M = tf.tanh(tf.matmul(tf.concat([item_hiddens,
                                   tf.tile(query_hiddens[:, t:t+1, :], [1, item_length, 1]),
                                   tf.tile(r, [1, item_length, 1])], axis=-1), self.W_h))
            # The alpha.shape == [batch, sequence_length, 1]
            alpha = tf.tanh(tf.matmul(M, self.w))
            r = tf.matmul(tf.transpose(item_hiddens, [0, 2, 1]), alpha) + tf.tanh(tf.matmul(self.W_r, tf.transpose(r, [0, 2, 1])))
            r = tf.transpose(r, [0, 2, 1])
        # hidden.shape == [batch, k]
        hidden = tf.tanh(tf.matmul(tf.concat([r, q, tf.abs(r - q)], axis=-1), self.W_x))
        hidden = tf.squeeze(hidden, axis=1)
        return hidden
