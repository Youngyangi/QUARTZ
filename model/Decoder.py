import tensorflow as tf


class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units, use_bias=False)
    self.W2 = tf.keras.layers.Dense(units, use_bias=False)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # 隐藏层的形状 == （批大小，隐藏层大小）
    # hidden_with_time_axis 的形状 == （批大小，1，隐藏层大小）
    # 这样做是为了执行加法以计算分数
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # 分数的形状 == （批大小，最大长度，1）
    # 我们在最后一个轴上得到 1， 因为我们把分数应用于 self.V
    # 在应用 self.V 之前，张量的形状是（批大小，最大长度，单位）
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # 注意力权重 （attention_weights） 的形状 == （批大小，最大长度，1）
    attention_weights = tf.nn.softmax(score, axis=1)

    # 上下文向量 （context_vector） 求和之后的形状 == （批大小，隐藏层大小）
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class LSTM_decoder(tf.keras.layers.Layer):
    def __init__(self, embedding_size, vocab_size, ):
        super(LSTM_decoder, self).__init__()
        self.attention = BahdanauAttention(embedding_size)
        self.encoder = tf.keras.layers.LSTM(embedding_size, return_state=True, return_sequences=True,
                                            recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)


    def call(self, x, hidden, enc_output, **kwargs):
        context_vec, attention_weights = self.attention(hidden, enc_output)
        x = tf.concat([tf.expand_dims(context_vec, 1), x], axis=-1)
        output, h, c = self.encoder(x)
        output = tf.squeeze(output, axis=1)
        logits = self.fc(output)
        return logits, h, c, attention_weights

