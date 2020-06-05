import tensorflow as tf
from model.QUARTS import QUARTS

embedding_size = 300
vocab_szie = 30000
max_len_query = 40
max_len_item = 100


quarts = QUARTS(embedding_size, vocab_szie, max_len_query, max_len_item)




def adversarial_generation(start_embedding, hidden, enc_output):
    # start_embedding.shape == [embeddings]
    # x.shape == [1, 1, embeddings]
    batch_size = hidden.shape[0]
    max_len = hidden.shape[1]
    start_embedding = tf.expand_dims(tf.expand_dims(start_embedding, 0), 0)
    batch_id = []
    for i in range(batch_size):
        id = []
        x = start_embedding
        for _ in range(max_len):
            pred, hidden, c, att = quarts.decoder(x, hidden, enc_output)
            id.append(tf.argmax(pred, axis=-1))