import tensorflow as tf
from model.Encoder import LSTM_encoder
from model.Decoder import LSTM_decoder
from model.Transformer import Transformer
from model.QUARTS import Word_by_word_attention


encoder = LSTM_encoder(100)

item_sample_input = tf.random.normal([32, 10, 20])
query_sample_input = tf.random.normal([32, 10, 20])

# output = encoder(None, item_embeddings=item_sample_input, query_embeddings=query_sample_input)
output = encoder(item_sample_input, query_sample_input)
print(output['item_sequence'].shape, output['query_sequence'].shape)

decoder = LSTM_decoder(100, 2000)
out = decoder(tf.random.uniform([32, 1, 20]), output['query_hidden'], output['query_sequence'][-1])
print(out[0])


att = Word_by_word_attention(100)
out = att(output['query_sequence'], output['item_sequence'])
print(out.shape)
# sample_transformer = Transformer(
#     num_layers=2, d_model=512, num_heads=8, dff=2048,
#     input_vocab_size=8500, target_vocab_size=8000,
#     pe_input=10000, pe_target=6000)
#
# temp_input = tf.random.uniform((64, 62))
# temp_target = tf.random.uniform((64, 26))
#
# fn_out, _ = sample_transformer(temp_input, temp_target, training=False,
#                                enc_padding_mask=None,
#                                look_ahead_mask=None,
#                                dec_padding_mask=None)
#
# print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)