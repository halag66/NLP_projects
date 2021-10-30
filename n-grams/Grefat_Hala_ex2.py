from sys import argv
import math
import codecs
import random
from collections import Counter


def merge_2_texts(file1, file2):
    data = data2 = ""

    # Reading data from file1
    with open(file1, encoding="utf-8") as fp:
        data = fp.read()
    data = data.lower()
    # Reading data from file2
    with open(file2, encoding="utf-8") as fp:
        data2 = fp.read()
    data2 = data2.lower()
    # Merging 2 files
    # To add the data of file2
    # from next line
    data += "\n"
    data += data2
    return data


# return n-grams according to n
def retrieve_ngrams(txt, n):
    return[txt[i:i + n] for i in range(len(txt) - (n - 1))]



def addSymbol(data1):
    data1 = "".join(data1)
    newdata = ["".join(["<s> {0} <e>\n".format(line)]) for line in data1.splitlines()]
    newdata = "".join(newdata)
    return newdata

    # calculate probability for Bi-grams


def Bi_grams_prob(v_Bi_gram, V_uni_gram, vocab_len, laplace):
    C = 0       # count tokens occurrences in corpus C*(Wi-1 Wi)
    Prob = {}
    for index, word in enumerate(v_Bi_gram):
        if laplace:
            C = v_Bi_gram[word] + 1
            Prob[word] = math.log(C, 2)-math.log(V_uni_gram[word.split(" ")[0]] + vocab_len, 2)
        else:
            C = v_Bi_gram[word]
            Prob[word] = math.log(C, 2) - math.log(V_uni_gram[word.split(" ")[0]], 2)
    return Prob


def classify_Bi(test_text, uni_prob, vocab, Bi_prob, v_Bi_gram):
    probability = 0
    #distinct_values = list(set(vocab.keys()))
    #distinct_values_list = list(set(v_Bi_gram.keys()))
    for index, word in enumerate(test_text):
        if index == 0:
            if word in vocab:
                probability = probability + uni_prob[word]
            else:
                probability = probability + uni_prob["<unk>"]
        else:
            string = test_text[index - 1] + " " + word
            if string in v_Bi_gram:
                probability = probability + Bi_prob[string]
            else:
                probability = probability + (math.log(1, 2)-math.log(vocab[test_text[index - 1]] + vocab.__len__(), 2))

    return probability


def uni_grams_prob(vocab, N_corps, laplace):
    C = 0
    Prob = {}
    if laplace:
        Prob["<unk>"] = math.log(1, 2) - math.log(N_corps + vocab.__len__(), 2)
    for index, word in enumerate(vocab):
        if laplace:
            C = (vocab[word] + 1)
            Prob[word] = math.log(C, 2)-math.log(N_corps + vocab.__len__(), 2)
        else:
            C = vocab[word]
            Prob[word] = math.log(C, 2) - math.log(N_corps, 2)
    return Prob


def classify_Uni(test_text, uni_prob, vocab):
    probability = 0
    #distinct_values = sorted(list(set(vocab.keys())))
    for word in test_text:
        if word in vocab:
            probability = probability + uni_prob[word]
        else:
            probability = probability + uni_prob["<unk>"]
    return probability


def Tri_grams_prob(v_tri_gram, en_v_Bi_gram):
    C = 0  # count tokens occurrences in corpus C(Wi-2 Wi-1 Wi)
    Prob = {}
    for index, word in enumerate(v_tri_gram):
        C = v_tri_gram[word]
        string = " ".join(word.split(" ")[:2])
        Prob[word] = math.log(C, 2) - math.log(en_v_Bi_gram[string])
    return Prob


def classify_Tri(test_text, uni_prob, vocab, Bi_prob, v_Bi_gram, Tri_prob, v_Tri_gram):
    tri_x = 0.6
    bi_x = 0.35
    uni_x = 0.25
    probability = 0
    #vocab_list = list(set(vocab.keys()))
    #v_Bi_gram_list = list(set(v_Bi_gram.keys()))
    #v_Tri_gram_list = list(set(v_Tri_gram.keys()))
    for index, word in enumerate(test_text):
        if index == 0:
            if word in vocab:
                probability = probability + uni_x * uni_prob[word]
            else:
                probability = probability + uni_x * uni_prob["<unk>"]
        if index == 1:
            string = test_text[index - 1] + " " + word
            if string in v_Bi_gram:
                probability = probability + bi_x * Bi_prob[string]
            else:
                probability = probability + bi_x * (math.log(1, 2) - math.log(vocab[word.split(" ")[0]] + vocab.__len__(), 2))
        if index != 1 and index != 0:
            string = test_text[index - 2] + " " + test_text[index - 1] + " " + word
            if string in v_Tri_gram:
                probability = probability + tri_x * Tri_prob[string]
            string = test_text[index - 1] + " " + word

            if string in v_Bi_gram:
                probability = probability + bi_x * Bi_prob[string]
            else:
                probability = probability + bi_x * (
                            math.log(1, 2) - math.log(vocab[word.split(" ")[0]] + vocab.__len__(), 2))

            if word in vocab:
                probability = probability + uni_x * uni_prob[word]
            else:
                probability = probability + uni_x * uni_prob["<unk>"]

    return probability


def find_relevant_tokens(vocab, token, prob, is_bi):
    tokens = []
    relevant_prob = []
    for index, elem in enumerate(vocab):
        if is_bi:
            if elem.split()[0] == token:
                tokens.append(elem)
                relevant_prob.append(prob[index])
        else:
            if elem.split()[0] == token[1] or elem.split()[1] == token[2]:
                tokens.append(elem)
                relevant_prob.append(prob[index])
    return tokens, relevant_prob


# return a random number from corpus lengths
def get_random_length(corpus):
    lengths = []
    corpus = corpus.split("\n")
    for line in corpus:
        lengths.append(list(line.split()).__len__())
    return random.choice(lengths)

def non_negative(prob):
    min_val = min(prob.values()) - 0.1
    new_prob = []
    vocab = []
    for val in prob:
        vocab.append(val)
        new_prob.append(prob[val] - min_val)
    return new_prob, vocab


def generate_text(length, n, prob, bi_prob=None):
    # return generated text
    # convert prob from dic to list

    sentence = "<s>"
    prob, vocab = non_negative(prob)
    if bi_prob:
        bi_prob, bi_vocab = non_negative(bi_prob)
    if n == 1:
        for i in range(length):
            word = random.choices(vocab, weights=prob, cum_weights=None, k=1)[0]
            sentence = sentence + " " + word
            if word == "<e>":
                return sentence
    if n == 2:
        for i in range(length):
            if i == 0:
                tokens, relevant_prob = find_relevant_tokens(bi_vocab, "<s>", bi_prob, True)
                if tokens.__len__() == 0:
                    lastWord = random.choices(bi_vocab, weights=bi_prob, cum_weights=None, k=1)[0]
                else:
                    lastWord = random.choices(tokens, weights=relevant_prob, cum_weights=None, k=1)[0]
                sentence = sentence + " " + lastWord.split()[1]
            else:
                tokens, relevant_prob = find_relevant_tokens(bi_vocab, lastWord.split()[1], bi_prob, True)
                if tokens.__len__() == 0:
                    lastWord = random.choices(bi_vocab, weights=bi_prob, cum_weights=None, k=1)[0]
                else:
                    lastWord = random.choices(tokens, weights=relevant_prob, cum_weights=None, k=1)[0]
                sentence = sentence + " " + lastWord.split()[1]
                # lastWord = lastWord.lower()
            if lastWord.split()[1] == "<e>":
                return sentence
    if n == 3:
        for i in range(length):
            if i == 0:
                tokens, relevant_prob = find_relevant_tokens(bi_vocab, "<s>", bi_prob, True)
                if relevant_prob.__len__() == 0:
                    lastWord1 = random.choices(bi_vocab, weights=bi_prob, cum_weights=None, k=1)[0]
                else:
                    lastWord1 = random.choices(tokens, weights=relevant_prob, cum_weights=None, k=1)[0]
                sentence = sentence + " " + lastWord1[1]
                #lastWord1 = lastWord1.lower()
                if "<e>" in lastWord1:
                    return sentence
            if i == 1:
                tokens, relevant_prob = find_relevant_tokens(bi_vocab, lastWord1.split()[1], bi_prob, True)
                if tokens.__len__() == 0:
                    lastWord2 = random.choices(bi_vocab, weights=bi_prob, cum_weights=None, k=1)[0]
                else:
                    lastWord2 = random.choices(tokens, weights=relevant_prob, cum_weights=None, k=1)[0]
                if '<e>' in lastWord2:
                    return sentence + " " + lastWord2.split()[1]
                else:
                    sentence = sentence + " " + lastWord2.split()[1]
                lastWord2 = list(sentence.split())
            if i > 1:
                tokens, relevant_prob = find_relevant_tokens(vocab, lastWord2, prob, False)
                if tokens.__len__() == 0:
                    lastWord3 = random.choices(vocab, weights=prob, cum_weights=None, k=1)[0]
                else:
                    lastWord3 = random.choices(tokens, weights=relevant_prob, cum_weights=None, k=1)[0]
                if lastWord3.split()[2] == '<e>':
                    return sentence + " " + lastWord3.split()[2]
                lastWord1 = lastWord2
                lastWord2 = lastWord3
                sentence = sentence + " " + lastWord3.split()[2]
    return sentence





if __name__ == "__main__":
    en_corpus = "en.txt"
    es_corpus = "es.txt"
    Bilingual_corpus = merge_2_texts(en_corpus, es_corpus)
    en_corpus_file = codecs.open(en_corpus, encoding="utf-8")
    en_corpus = en_corpus_file.readlines()
    en_corpus_og = " ".join(en_corpus)
    es_corpus_file = codecs.open(es_corpus, encoding="utf-8")
    es_corpus = es_corpus_file.readlines()
    es_corpus_og = " ".join(es_corpus)
    #en_corpus = "I love you , you love me .\n hello there !"
    # es_corpus = en_corpus
    #test_string = "I love you"
    #test_string = addSymbol(test_string)
    en_corpus_string = en_corpus_og.lower()
    en_corpus = list(en_corpus_string.split())
    es_corpus = es_corpus_og.lower()
    es_corpus = list(es_corpus.split())

    phrases = ["You never know what you're gonna get",
               "Keep your friends close, but your enemies closer",
               "I've got a feeling we're not in Kansas anymore",
               "Quiero respirar tu cuello despacito",
               "Me gusta la moto, me gustas tú",
               "Dale a tu cuerpo alegría Macarena"]
    #englich corpus
    #calculate prob for Uni
    en_vocab = Counter(en_corpus)
    N_corps = en_corpus.__len__()
    en_uni_prob = uni_grams_prob(en_vocab, N_corps, True)

    # calculate prob for bi
    en_v_Bi_gram = retrieve_ngrams(en_corpus, 2)
    en_v_Bi_gram = Counter([" ".join(elem) for elem in en_v_Bi_gram])
    en_Bi_prob = Bi_grams_prob(en_v_Bi_gram, en_vocab, en_vocab.__len__(), True)

    #calculate prob for tri
    en_v_Tri_gram = retrieve_ngrams(en_corpus, 3)
    en_v_Tri_gram = Counter([" ".join(elem) for elem in en_v_Tri_gram])
    en_Tri_prob = Tri_grams_prob(en_v_Tri_gram, en_v_Bi_gram)

    # spanish corpus
    # calculate prob for Uni
    es_vocab = Counter(es_corpus)
    N_corps = es_corpus.__len__()
    es_uni_prob = uni_grams_prob(es_vocab, N_corps, True)

    # calculate prob for bi
    es_v_Bi_gram = retrieve_ngrams(es_corpus, 2)
    es_v_Bi_gram = Counter([" ".join(elem) for elem in es_v_Bi_gram])
    es_Bi_prob = Bi_grams_prob(es_v_Bi_gram, es_vocab, es_vocab.__len__(), True)

    # calculate prob for tri
    es_v_Tri_gram = retrieve_ngrams(es_corpus, 3)
    es_v_Tri_gram = Counter([" ".join(elem) for elem in es_v_Tri_gram])
    es_Tri_prob = Tri_grams_prob(es_v_Tri_gram, es_v_Bi_gram)

    for line in phrases:
        line = line.lower()
        test_string = list(line.split())
        en_uni_gram = classify_Uni(test_string, en_uni_prob, en_vocab)
        en_Bi_grams = classify_Bi(test_string, en_uni_prob, en_vocab, en_Bi_prob, en_v_Bi_gram)
        en_Tri_grams = classify_Tri(test_string, en_uni_prob, en_vocab, en_Bi_prob, en_v_Bi_gram, en_Tri_prob
                                    , en_v_Tri_gram)
        es_uni_gram = classify_Uni(test_string, es_uni_prob, es_vocab)
        es_Bi_grams = classify_Bi(test_string, es_uni_prob, es_vocab, es_Bi_prob, es_v_Bi_gram)
        es_Tri_grams = classify_Tri(test_string, es_uni_prob, es_vocab, es_Bi_prob, es_v_Bi_gram, es_Tri_prob
                                    , es_v_Tri_gram)

        print(line)
        print("Unigrams Model: ", en_uni_gram, "for English, ", es_uni_gram, "for spanish.")
        print("Biigrams Model: ", en_Bi_grams, "for English, ", es_Bi_grams, "for spanish.")
        print("Trigrams Model: ", en_Tri_grams, "for English, ", es_Tri_grams, "for spanish.")

    # second part of hw
    en_corpus_og = addSymbol(en_corpus_og)  # add symbol to the start and end of string
    Bilingual_corpus_og = addSymbol(Bilingual_corpus)
    en_corpus = en_corpus_og.lower()
    en_corpus = list(en_corpus.split())
    Bilingual_corpus = Bilingual_corpus_og.lower()
    Bilingual_corpus = list(Bilingual_corpus.split())

    en_vocab = Counter(en_corpus)
    N_corps = en_corpus.__len__()
    en_uni_prob = uni_grams_prob(en_vocab, N_corps, True)

    # calculate prob for bi
    en_v_Bi_gram = retrieve_ngrams(en_corpus, 2)
    en_v_Bi_gram = Counter([" ".join(elem) for elem in en_v_Bi_gram])
    en_Bi_prob = Bi_grams_prob(en_v_Bi_gram, en_vocab, en_vocab.__len__(), True)

    # calculate prob for tri
    en_v_Tri_gram = retrieve_ngrams(en_corpus, 3)
    en_v_Tri_gram = Counter([" ".join(elem) for elem in en_v_Tri_gram])
    en_Tri_prob = Tri_grams_prob(en_v_Tri_gram, en_v_Bi_gram)

    # spanish corpus
    # calculate prob for Uni
    Bilingual_vocab = Counter(Bilingual_corpus)
    N_corps = Bilingual_corpus.__len__()
    Bilingual_uni_prob = uni_grams_prob(Bilingual_vocab, N_corps, False)

    # calculate prob for bi
    Bilingual_v_Bi_gram = retrieve_ngrams(Bilingual_corpus, 2)
    Bilingual_v_Bi_gram = Counter([" ".join(elem) for elem in Bilingual_v_Bi_gram])
    Bilingual_Bi_prob = Bi_grams_prob(Bilingual_v_Bi_gram, Bilingual_vocab, Bilingual_vocab.__len__(), False)

    # calculate prob for tri
    Bilingual_v_Tri_gram = retrieve_ngrams(Bilingual_corpus, 3)
    Bilingual_v_Tri_gram = Counter([" ".join(elem) for elem in Bilingual_v_Tri_gram])
    Bilingual_Tri_prob = Tri_grams_prob(Bilingual_v_Tri_gram, Bilingual_v_Bi_gram)

    i = 3
    print("Unigrams Model in English Dataset:")
    while i != 0:
        rand_len_en = get_random_length(en_corpus_og)
        print(generate_text(rand_len_en, 1, en_uni_prob, en_Bi_prob))
        i = i - 1
    i = 3
    print("Biigrams Model in English Dataset:")
    while i != 0:
        rand_len_en = get_random_length(en_corpus_og)
        print(generate_text(rand_len_en, 2, en_Bi_prob, en_Bi_prob))
        i -= 1
    i = 3
    print("Trigrams Model in English Dataset:")
    while i != 0:
        rand_len_en = get_random_length(en_corpus_og)
        print(generate_text(rand_len_en, 3, en_Tri_prob, en_Bi_prob))
        i -= 1

    # Bilingual text
    i = 3
    print("Unigrams Model in Bilingual Dataset:")
    while i != 0:
        rand_len_es = get_random_length(Bilingual_corpus_og)
        print(generate_text(rand_len_es, 1, Bilingual_uni_prob, Bilingual_Bi_prob))
        i = i - 1
    i = 3
    print("Bigrams Model in Bilingual Dataset:")
    while i != 0:
        rand_len_es = get_random_length(Bilingual_corpus_og)
        print(generate_text(rand_len_es, 2, Bilingual_Bi_prob, Bilingual_Bi_prob))
        i -= 1
    i = 3
    print("Trigrams Model in Bilingual Dataset:")
    while i != 0:
        rand_len_es = get_random_length(Bilingual_corpus_og)
        print(generate_text(rand_len_es, 3, Bilingual_Tri_prob, Bilingual_Bi_prob))
        i -= 1



