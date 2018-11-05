import os
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import io
from config import START_DIR, MAXLEN, STEP, NUMBER_OF_ITERATIONS


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def remove_tabs_in_file(input_filename, output_filename):
    with open(input_filename, encoding='windows-1251') as file:
        lines = file.readlines()

    lines = [line.lstrip() for line in lines]

    with open(output_filename, 'w', encoding='windows-1251') as file:
        file.writelines(lines)


def create_result_text(
    diversity_list,
    start_index,
    chars,
    text,
    char_indices,
    indices_char,
    model,
    number_of_rows=4
):
    for diversity in diversity_list:
        generated = ''
        sentence = text[start_index: start_index + MAXLEN]
        generated += sentence
        result = ''

        while result.count('\n') < number_of_rows:
            x_pred = np.zeros((1, MAXLEN, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.0

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            generated += next_char
            sentence = sentence[1:] + next_char

            result += next_char

    return result


def get_text_from_file(filepath):
    with io.open(filepath) as file:
        text = file.read().lower()
    return text


def get_chars(text):
    return sorted(list(set(text)))


def get_char_indices(chars):
    return dict((c, i) for i, c in enumerate(chars))


def get_indices_char(chars):
    return dict((i, c) for i, c in enumerate(chars))


def create_sentences_and_next_chars_list(text):
    sentences = []
    next_chars = []
    for i in range(0, len(text) - MAXLEN, STEP):
        sentences.append(text[i: i + MAXLEN])
        next_chars.append(text[i + MAXLEN])
    return sentences, next_chars


def create_feature_and_target_set(sentences, chars, char_indices, indices_char, next_chars):
    x = np.zeros((len(sentences), MAXLEN, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    return x, y


def build_model(chars, x, y):
    try:
        model = load_model(os.path.join(START_DIR, 'models/poem_model.h5'))
        return model
    except OSError:
        model = Sequential()
        model.add(LSTM(128, input_shape=(MAXLEN, len(chars))))
        model.add(Dense(len(chars), activation='softmax'))

        optimizer = RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        model.fit(x, y, batch_size=128, epochs=NUMBER_OF_ITERATIONS)

        model.save(os.path.join(START_DIR, 'models/poem_model.h5'))
        return model


def create_ode():
    remove_tabs_in_file(
        os.path.join(START_DIR, 'data/poem_data.txt'),
        os.path.join(START_DIR, 'data/new_poem_data.txt')
    )
    text = get_text_from_file(os.path.join(START_DIR, 'data/new_poem_data.txt'))
    chars = get_chars(text)
    indices_char = get_indices_char(chars)
    char_indices = get_char_indices(chars)
    sentences, next_chars = create_sentences_and_next_chars_list(text)
    x, y = create_feature_and_target_set(
        sentences, chars, char_indices, indices_char, next_chars
    )
    model = build_model(chars, x, y)

    result = create_result_text(
        diversity_list=[0.5],
        start_index=0,
        chars=chars,
        text=text,
        char_indices=char_indices,
        indices_char=indices_char,
        model=model
    )

    return result
