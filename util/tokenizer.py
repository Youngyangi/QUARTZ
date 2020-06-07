import tensorflow as tf

# TODO: implement tokenizer
class Tokenizer(object):
    def encode(self, sent):
        pass

    def decode(self, idx):
        pass

    def cut(self, sent):
        pass


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer
