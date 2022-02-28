import collections
import numpy as np

import enc_dec

file = "datasets/ewe.txt"
file2 = "datasets/fr.txt"

with open(file, 'r', encoding="utf8") as f:
    lines = f.read().split("\n")[:-1]

with open(file2, 'r', encoding="utf8") as f2:
    lines2 = f2.read().split("\n")[:-1]

ewe_sentences = []
fr_sentences = []
text_pairs = []
for line2 in lines2:
    fr_sentences.append(line2.split('\t')[-1])

for line in lines:
    ewe_sentences.append(line.split('\t')[-1])

for sample_i in range(2):
    print('small_vocab_en Line {}:  {}'.format(sample_i + 1, ewe_sentences[sample_i]))
    print('small_vocab_fr Line {}:  {}'.format(sample_i + 1, fr_sentences[sample_i]))

# ----------------------------------------------------#
# ---------------------STEP ONE-----------------------#
# ----------------------------------------------------#
ewe_words_counter = collections.Counter([word for sentence in ewe_sentences for word in sentence.split()])
fr_words_counter = collections.Counter([word for sentence in fr_sentences for word in sentence.split()])
print('{} Ewe words.'.format(len([word for sentence in ewe_sentences for word in sentence.split()])))
print('{} unique Ewe words.'.format(len(ewe_words_counter)))
print('10 Most common words in the Ewe dataset:')
print('"' + '" "'.join(list(zip(*ewe_words_counter.most_common(10)))[0]) + '"')
print()
print('{} French words.'.format(len([word for sentence in fr_sentences for word in sentence.split()])))
print('{} unique French words.'.format(len(fr_words_counter)))
print('10 Most common words in the French dataset:')
print('"' + '" "'.join(list(zip(*fr_words_counter.most_common(10)))[0]) + '"')


# Tokenize Example output
text_sentences = [
    'Tonye be, ablɔɖe vavãe nye be miaɖu nu aɖi ƒo .',
    'ɣetrɔ sia ƒe nuɖuɖu nye fufu kple fufutsi si me agbitsã, dotɛ kple kanami le .',
    'nyawoe .']
text_tokenized, text_tokenizer = enc_dec.tokenize(text_sentences)
print(text_tokenizer.word_index)
print()
for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(sent))
    print('  Output: {}'.format(token_sent))


# Pad Tokenized output
test_pad = enc_dec.pad(text_tokenized)
for sample_i, (token_sent, pad_sent) in enumerate(zip(text_tokenized, test_pad)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(np.array(token_sent)))
    print('  Output: {}'.format(pad_sent))


preproc_ewe_sentences, preproc_fr_sentences, ewe_tokenizer, fr_tokenizer = \
    enc_dec.preprocess(ewe_sentences, fr_sentences)

max_ewe_sequence_length = preproc_ewe_sentences.shape[1]
max_fr_sequence_length = preproc_fr_sentences.shape[1]
ewe_vocab_size = len(ewe_tokenizer.word_index)
fr_vocab_size = len(fr_tokenizer.word_index)
print('Data Preprocessed')
print("Max Ewe sentence length:", max_ewe_sequence_length)
print("Max French sentence length:", max_fr_sequence_length)
print("Ewe vocabulary size:", ewe_vocab_size)
print("French vocabulary size:", fr_vocab_size)


# Reshaping the input to work with a basic RNN
tmp_x = enc_dec.pad(preproc_ewe_sentences, max_fr_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_fr_sentences.shape[-2], 1))
simple_rnn_model = enc_dec.simple_model(
    tmp_x.shape,
    max_fr_sequence_length,
    ewe_vocab_size,
    fr_vocab_size)
simple_rnn_model.fit(tmp_x, preproc_fr_sentences, batch_size=300, epochs=10, validation_split=0.2)


print(enc_dec.logits_to_text(simple_rnn_model.predict(tmp_x[:1])[0], fr_tokenizer))


# TODO: Reshape the input
tmp_x = enc_dec.pad(preproc_ewe_sentences, preproc_fr_sentences.shape[1])
tmp_x = tmp_x.reshape((-1, preproc_fr_sentences.shape[-2]))
# TODO: Train the neural network
embed_rnn_model = enc_dec.embed_model(
    tmp_x.shape,
    preproc_fr_sentences.shape[1],
    len(ewe_tokenizer.word_index) + 1,
    len(fr_tokenizer.word_index)+1)
embed_rnn_model.fit(tmp_x, preproc_fr_sentences, batch_size=300, epochs=10, validation_split=0.2)

print(ewe_sentences[:1])
print(fr_sentences[:1])

print(enc_dec.logits_to_text(embed_rnn_model.predict(tmp_x[:1])[0], fr_tokenizer))

# Train and Print prediction(s)
tmp_x = enc_dec.pad(preproc_ewe_sentences, preproc_fr_sentences.shape[1])
tmp_x = tmp_x.reshape((-1, preproc_fr_sentences.shape[-2], 1))
# Train and Print prediction(s)
bd_rnn_model = enc_dec.bd_model(
    tmp_x.shape,
    preproc_fr_sentences.shape[1],
    len(ewe_tokenizer.word_index)+1,
    len(fr_tokenizer.word_index)+1)
bd_rnn_model.fit(tmp_x, preproc_fr_sentences, batch_size=300, epochs=10, validation_split=0.2)

tmp_x = enc_dec.pad(preproc_ewe_sentences, preproc_fr_sentences.shape[1])
tmp_x = tmp_x.reshape((-1, preproc_fr_sentences.shape[-2], 1))
# Train and Print prediction(s)
ed_rnn_model = enc_dec.encdec_model(
    tmp_x.shape,
    preproc_fr_sentences.shape[1],
    len(ewe_tokenizer.word_index)+1,
    len(fr_tokenizer.word_index)+1)
ed_rnn_model.fit(tmp_x, preproc_fr_sentences, batch_size=300, epochs=10, validation_split=0.2)

enc_dec.final_predictions(preproc_ewe_sentences, preproc_fr_sentences, ewe_tokenizer, fr_tokenizer)