import tensorflow as tf
import time
from model.QUARTS import QUARTS
from model.Embedding import Embedding_layer
from util.tokenizer import Tokenizer

epoch = 10
batch_size = 32
embedding_size = 300
vocab_size = 30000
max_len_query = 40
max_len_item = 100
data_path = 'data/'


def generate_data(path):
    with open(path, 'r', encoding='utf8') as f:
        pass
    # return train_dataset, test_dataset

def loss_function(real, pred):
    pass


def adversarial_generation(start_embedding, hidden, enc_output):
    # start_embedding.shape == [embeddings]
    # x.shape == [1, 1, embeddings]
    batch_size = hidden.shape[0]
    max_len = hidden.shape[1]
    start_embedding = tf.expand_dims(tf.expand_dims(start_embedding, 0), 0)
    batch_id = []
    for i in range(batch_size):
        ids = []
        x = start_embedding
        for _ in range(max_len):
            pred, hidden, c, att = quarts.decoder(x, hidden, enc_output)
            id = tf.squeeze(tf.argmax(pred, axis=-1)).numpy()
            if id == 2:  # stop ind
                break
            ids.append(id)
            # TODO: embedding
            x = embedding_layer(id)
        # ids.shape == [1, max_len_query]
        ids = tf.expand_dims(ids, 0)
        ids = tf.keras.preprocessing.sequence.pad_sequences(ids, max_len=max_len_query, padding='post')
        batch_id.append(ids)
    # batch_id.shape == [batch_size, max_len_query]
    batch_id = tf.concat(batch_id, axis=0)
    return embedding_layer(batch_id)  # return.shape == [batch_size, max_len_query, embedding_size]


def combine_true_gen_query(true_query, gen_query, label, p):
    # label.shape == [batch_size, 1]
    s = (1-label) * tf.keras.backend.random_binomial(label.shape, p)
    # TODO: combine true and gen query
    pass

def init_checkpoint(path):
    pass

@tf.function
def train_step(inp, targ):
    pass


def train(epochs, save_path):
    check_point = init_checkpoint(save_path)
    for epoch in epochs:
        start = time.time()
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(train_dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ)
            total_loss += batch_loss
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
        if (epoch + 1) % 2 == 0:
            check_point.save()
        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


if __name__ == '__main__':
    tokenizer = Tokenizer()
    embedding_layer = Embedding_layer(vocab_size, embedding_size)
    quarts = QUARTS(embedding_size, vocab_size, max_len_query, max_len_item)
    train_dataset, test_dataset = generate_data(data_path)
    steps_per_epoch = len(train_dataset) // batch_size
    train(epoch, 'ckpt/')



