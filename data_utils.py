import csv
from collections import Counter
import numpy as np
import os.path

glove_vectors = dict()
def save_vocab(word2idx, idx2word, name="vocab"):
    with open("./data/" + name + "_word2idx", 'w', encoding="utf8") as f:
        for w in word2idx:
            f.write(w + ":@:" + str(word2idx[w]) + '\n')

    with open("./data/" + name + "_idx2word", 'w', encoding="utf8") as f:
        for i in idx2word:
            f.write(str(i) + ":@:" + idx2word[i] + '\n')
    print("Done saving.")


def save_vocab_bi(word2idx, idx2word, name="vocab"):
    with open("./data/" + name + "_word2idx", 'w', encoding="utf8") as f:
        for w in word2idx:
            f.write(w[0] + ":@:" + w[1] + ":@:" + str(word2idx[w]) + '\n')

    with open("./data/" + name + "_idx2word", 'w', encoding="utf8") as f:
        for i in idx2word:
            f.write(str(i) + ":@:" + idx2word[i][0] + ":@:" + idx2word[i][1] + '\n')
    print("Done saving.")


def load_vocab(name):
    with open("./data/" + name + "_word2idx", encoding="utf8") as f:
        word2idx = dict()
        for line in f:
            w, i = line.strip().split(":@:")
            word2idx[w] = int(i)

    with open("./data/" + name + "_idx2word", encoding="utf8") as f:
        idx2word = dict()
        for line in f:
            i, w = line.strip().split(":@:")
            idx2word[int(i)] = w
    print("Done loading.")
    return word2idx, idx2word


def load_vocab_bi(name):
    with open("./data/" + name + "_word2idx", encoding="utf8") as f:
        word2idx = dict()
        for line in f:
            w1, w2, i = line.strip().split(":@:")
            word2idx[(w1, w2)] = int(i)

    with open("./data/" + name + "_idx2word", encoding="utf8") as f:
        idx2word = dict()
        for line in f:
            i, w1, w2 = line.strip().split(":@:")
            idx2word[int(i)] = (w1, w2)
    print("Done loading.")
    return word2idx, idx2word


def get_vocab(k):
    file_path = "./data/vocab_" + str(k) + "_idx2word"
    if os.path.exists(file_path):
        return load_vocab("vocab_" + str(k))
    vocab = Counter()
    with open("./data/train.csv", encoding="utf8") as f:
        reader = csv.DictReader(f)
        for line in reader:
            comment_text = line['comment_text']
            for w in comment_text.strip().split():
                vocab[w.lower()] += 1

    vocab = vocab.most_common(k)
    word2idx = dict()
    idx2word = dict()
    for i in range(len(vocab)):
        word2idx[vocab[i][0]] = i
        idx2word[i] = vocab[i][0]
    print("Read vocab", len(word2idx))
    save_vocab(word2idx, idx2word, "vocab_" + str(k))
    return word2idx, idx2word


def get_vocab_bi(k):
    file_path = "./data/vocab_bi_" + str(k) + "_idx2word"
    if os.path.exists(file_path):
        return load_vocab_bi("vocab_bi_" + str(k))
    vocab = Counter()
    with open("./data/train.csv", encoding="utf8") as f:
        reader = csv.DictReader(f)
        for line in reader:
            comment_text = line['comment_text']
            words = comment_text.strip().split()
            for i in range(1, len(words)):
                pair = (words[i - 1].lower(), words[i].lower())
                vocab[pair] += 1

    vocab = vocab.most_common(k)
    word2idx = dict()
    idx2word = dict()
    for i in range(len(vocab)):
        word2idx[vocab[i][0]] = i
        idx2word[i] = vocab[i][0]
    print("Read vocab", len(word2idx))
    save_vocab_bi(word2idx, idx2word, "vocab_bi_" + str(k))
    return word2idx, idx2word


def BOW(word2idx, comment):
    x = np.zeros(len(word2idx))
    words = comment.strip().split()
    for word in words:
        word = word.lower()
        if word in word2idx:
            x[word2idx[word]] += 1
    return x


def BOW_bi(word2idx, comment):
    x = np.zeros(len(word2idx))
    words = comment.strip().split()
    for i in range(1, len(words)):
        pair = (words[i - 1].lower(), words[i].lower())
        if pair in word2idx:
            x[word2idx[pair]] += 1
    return x


def get_y(insult, toxic, identity_hate, severe_toxic, obscene, threat):
    y = np.zeros(6)
    y[0] += insult
    y[1] += toxic
    y[2] += identity_hate
    y[3] += severe_toxic
    y[4] += obscene
    y[5] += threat
    return y


def read_data(word2idx, BOW_func=BOW):
    n = 159571
    k = len(word2idx)
    with open("./data/train.csv", encoding="utf8") as f:
        reader = csv.DictReader(f)
        X = np.zeros((n, k))
        Y = np.zeros((n, 6))
        i = 0
        for line in reader:
            insult = int(line['insult'])
            toxic = int(line['toxic'])
            identity_hate = int(line['identity_hate'])
            severe_toxic = int(line['severe_toxic'])
            obscene = int(line['obscene'])
            threat = int(line['threat'])
            comment_text = line['comment_text']
            X[i] = BOW(word2idx, comment_text)
            Y[i] = get_y(insult, toxic, identity_hate, severe_toxic, obscene, threat)
            i += 1

        print("Read", len(X), "points.")
        return X, Y


def unigrams(k=100):
    word2idx, idx2word = get_vocab(k)
    X, Y = read_data(word2idx, BOW_func=BOW)
    return X, Y


def bigrams(k=100):
    word2idx, idx2word = get_vocab_bi(k)
    X, Y = read_data(word2idx, BOW_func=BOW_bi)
    return X, Y


def unigrams_test(k=100):
    word2idx, idx2word = get_vocab(k)
    X, Y = read_test_data(word2idx, BOW_func=BOW)
    return X, Y


def bigrams_test(k=100):
    word2idx, idx2word = get_vocab_bi(k)
    X, Y = read_test_data(word2idx, BOW_func=BOW_bi)
    return X, Y


def load_vectors():
     with open('./data/glove.6B.100d.txt') as f:
        for line in f:
            arr = line.split()
            word = arr[0]
            vector = arr[1:]
            vector = list(map(lambda x: float(x), vector))
            glove_vectors[word] = np.array(vector, dtype=np.float64)  
def avg_word_vec(comment):
    words = comment.strip().split()
    x = np.zeros((100,), dtype=np.float64)
    if len(words) == 0:
        return x
    for word in words:
        newWord = word.lower()
        if newWord[-1] in ['!', '?', ',', ':', ';', '.', '\'', '\"']:
            newWord = newWord[:-1]
        if len(newWord) == 0:
            continue
        if newWord[0] in ['!', '?', ',', ':', ';', '.', '\'', '\"']:
            newWord = newWord[1:]
        if newWord not in glove_vectors:
            continue
        else:
            x += glove_vectors[newWord]
    x = x/(1.0*len(words))
    return list(x)

def read_test_data(word2idx, BOW_func=BOW):
    n = 153164
    k = len(word2idx)
    with open("./data/test.csv", encoding="utf8") as f:
        reader = csv.DictReader(f)
        X = np.zeros((n, k))
        i = 0
        for line in reader:
            comment_text = line['comment_text']
            X[i] = BOW(word2idx, comment_text)
            i += 1

    with open("./data/test_labels.csv", encoding="utf8") as f:
        reader = csv.DictReader(f)
        Y = np.zeros((n, 6))
        i = 0
        for line in reader:
            insult = int(line['insult'])
            toxic = int(line['toxic'])
            identity_hate = int(line['identity_hate'])
            severe_toxic = int(line['severe_toxic'])
            obscene = int(line['obscene'])
            threat = int(line['threat'])
            Y[i] = get_y(insult, toxic, identity_hate, severe_toxic, obscene, threat)
            i += 1

        print("Read", i, "points.")
        return X, Y

def read_train_data_glove():
    k = 100
    load_vectors()
    n = 159571
    with open("./data/train.csv", encoding="utf8") as f:
        reader = csv.DictReader(f)
        X = np.zeros((n, k))
        Y = np.zeros((n, 6))
        i = 0
        for line in reader:
            insult = int(line['insult'])
            toxic = int(line['toxic'])
            identity_hate = int(line['identity_hate'])
            severe_toxic = int(line['severe_toxic'])
            obscene = int(line['obscene'])
            threat = int(line['threat'])
            comment_text = line['comment_text']
            X[i] = avg_word_vec(comment_text)
            Y[i] = get_y(insult, toxic, identity_hate, severe_toxic, obscene, threat)
            i += 1

        print("Read", len(X), "points.")
        return X, Y

def read_test_data_glove():
    k = 100
    setOfIds = set()
    n = 153164
    

    with open("./data/test_labels.csv", encoding="utf8") as f:
        reader = csv.DictReader(f)
        Y = np.zeros((n, 6))
        i = 0
        for line in reader:
            insult = int(line['insult'])
            toxic = int(line['toxic'])
            identity_hate = int(line['identity_hate'])
            severe_toxic = int(line['severe_toxic'])
            obscene = int(line['obscene'])
            threat = int(line['threat'])
            arr = [insult, toxic, identity_hate, severe_toxic, obscene, threat]
            hasNegativeOne = False
            for el in arr:
                if el == -1:
                    hasNegativeOne = True
                    break
            if hasNegativeOne:
                setOfIds.add(line['id'])
                continue
            Y[i] = get_y(insult, toxic, identity_hate, severe_toxic, obscene, threat)
            i += 1

        Y = Y[:i, :]
        print("Read", i, "points.")

    with open("./data/test.csv", encoding="utf8") as f:
        reader = csv.DictReader(f)
        X = np.zeros((i, k))
        i = 0
        for line in reader:
            if line['id'] in setOfIds:
                continue
            comment_text = line['comment_text']
            X[i] = avg_word_vec(comment_text)
            i += 1
    return X, Y




