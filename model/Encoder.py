import tensorflow as tf


class LSTM_encoder(tf.keras.layers.Layer):
    def __init__(self, embedding_size):
        super(LSTM_encoder, self).__init__()
        self.embedding_size = embedding_size
        self.item_encoder = tf.keras.layers.LSTM(embedding_size, return_sequences=True, return_state=True)
        self.query_encoder = tf.keras.layers.LSTM(embedding_size, return_sequences=True, return_state=True)

    def call(self, item_embeddings, query_embeddings, **kwargs):
        # item = kwargs['item_embeddings']
        # query = kwargs['query_embeddings']
        item = item_embeddings
        query = query_embeddings
        item_sequence, item_hidden_state, item_memory_state = self.item_encoder(item)
        init_state = [item_hidden_state, item_memory_state]
        query_sequence, query_hidden_state, query_memory_state = self.query_encoder(query,
                                                                                    initial_state=init_state)
        return {'item_sequence': item_sequence, 'query_sequence': query_sequence}



