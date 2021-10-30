# from gensim.scripts.glove2word2vec import glove2word2vec
import codecs
import random
from collections import Counter
from sys import argv

import sklearn
from gensim.models import KeyedVectors, Word2Vec
import re
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from matplotlib import pyplot





# return n-grams according to n
def retrieve_ngrams(txt, n):
    return [txt[i:i + n] for i in range(len(txt) - (n - 1))]


def read_lyrics_and_split(file_name):
    file = open(file_name, 'r')
    lines = file.readlines()
    songs = " ".join(lines)
    clean_songs = songs.replace(".", "")
    clean_songs = clean_songs.replace(",", "")
    clean_songs = clean_songs.replace("?", "")
    clean_songs = clean_songs.replace("!", "")
    clean_songs = clean_songs.replace('"', "")
    clean_songs = clean_songs.replace("'s", " is")
    clean_songs = clean_songs.replace("'", "")


    songs_list = clean_songs.split("==")
    songs = []
    i = 1
    while i < songs_list.__len__():
        songs.append([songs_list[i].split("=")[1], songs_list[i + 2], list(songs_list[i + 3].split())])
        i += 4
    return songs


def eq_weight(song_vec_):
    avg = np.zeros(300)
    for vec in song_vec_:
        avg += vec
    avg /= len(song_vec_)
    return avg


def random1_5_weight(song_vec_):
    avg = np.zeros(300)
    for vec in song_vec_:
        avg += vec*random.randint(0, 4)
    avg /= len(song_vec_)
    return avg


def count_based_weights(song_string, song_vec_):
    # give words weights based on their count in song
    # song is string
    count = Counter(song_string)
    min_count = min(count.values())
    max_count = max(count.values())
    threshold = (max_count - min_count)/2
    avg = np.zeros(300)
    for key in count:
        if count[key] >= threshold:
            avg += (4*song_vec_[song_string.index(key)])
        else:
            avg += (1*song_vec_[song_string.index(key)])
    avg /= song_vec_.__len__()
    return avg


if __name__ == "__main__":
    en_corpus_file = argv[1]  # English corpus, text file
    glove_file_50 = argv[2]  # 50-vector, kv file
    glove_file_300 = argv[3]  # 300-vector, kv file
    song_lyrics = argv[4]     # Lyrics, text file
    output_file = argv[5]  # Output, text file

    # Building the models:
    # temp = Word2Vec.load(glove_file_50)
    pre_trained_model_50 = KeyedVectors.load_word2vec_format(glove_file_50, binary=False)
    pre_trained_model_300 = KeyedVectors.load_word2vec_format(glove_file_300, binary=False)
    # pre_trained_model_50.save('glove_model_50')

    # Biden's Tweets:
    tweets = [
        "America, I'm honored that you have chosen me to lead our great country. The work ahead of us will be hard, but I promise you this: I will be a President for all Americans — whether you voted for me or not. I will keep the faith that you have placed in me.",
        "If we act now on the American Jobs Plan, in 50 years, people will look back and say this was the moment that America won the future.",
        "Gun violence in this country is an epidemic — and it’s long past time Congress take action. It matters whether you continue to wear a mask. It matters whether you continue to socially distance. It matters whether you wash your hands. It all matters and can help save lives.",
        "If there’s one message I want to cut through to everyone in this country, it’s this: The vaccines are safe. For yourself, your family, your community, our country — take the vaccine when it’s your turn and available. That’s how we’ll beat this pandemic.",
        "Today, America is officially back in the Paris Climate Agreement. Let’s get to work.",
        "Today, in a bipartisan vote, the House voted to impeach and hold President Trump accountable. Now, the process continues to the Senate—and I hope they’ll deal with their Constitutional responsibilities on impeachment while also working on the other urgent business of this nation.",
        "The work of the next four years must be the restoration of democracy and the recovery of respect for the rule of law, and the renewal of a politics that’s about solving problems — not stoking the flames of hate and chaos.",
        "America is so much better than what we’re seeing today.",
        "Here’s my promise to you: I’ll be a president for all Americans. Whether you voted for me or not, I’ll wake up every single morning and work to make your life better.",
        "We can save 60,000-100,000 lives in the weeks and months ahead if we step up together. Wear a mask. Stay socially distanced. Avoid large indoor gatherings. Each of us has a duty to do what we can to protect ourselves, our families, and our fellow Americans."]

    tweets_lines = [["America, I'm honored that you have chosen me to lead our great country.", "The work ahead of us will be hard, but I promise you this: I will be a President for all Americans — whether you voted for me or not.", "I will keep the faith that you have placed in me."],
                    ["If we act now on the American Jobs Plan, in 50 years, people will look back and say this was the moment that America won the future."],
                    ["Gun violence in this country is an epidemic — and it’s long past time Congress take action.", "It matters whether you continue to wear a mask.", "It matters whether you continue to socially distance.", "It matters whether you wash your hands.", "It all matters and can help save lives."],
                    ["If there’s one message I want to cut through to everyone in this country, it’s this: The vaccines are safe.", "For yourself, your family, your community, our country — take the vaccine when it’s your turn and available.", "That’s how we’ll beat this pandemic."],
                    ["Today, America is officially back in the Paris Climate Agreement.", "Let’s get to work."],
                    ["Today, in a bipartisan vote, the House voted to impeach and hold President Trump accountable.", "Now, the process continues to the Senate—and I hope they’ll deal with their Constitutional responsibilities on impeachment while also working on the other urgent business of this nation."],
                    ["The work of the next four years must be the restoration of democracy and the recovery of respect for the rule of law, and the renewal of a politics that’s about solving problems — not stoking the flames of hate and chaos."],
                    ["America is so much better than what we’re seeing today."], ["Here’s my promise to you: I’ll be a president for all Americans.", "Whether you voted for me or not, I’ll wake up every single morning and work to make your life better."],
                    ["We can save 60,000-100,000 lives in the weeks and months ahead if we step up together.", "Wear a mask.", "Stay socially distanced.", "Avoid large indoor gatherings.", "Each of us has a duty to do what we can to protect ourselves, our families, and our fellow Americans."]]



    # #part 1
    # similarity
    words = [['sea', 'ocean'], ['up', 'down'], ['night', 'sundaynight'], ['berry', 'strawberry'], ['strong', 'tough'],
             ['alot', 'many']
        , ['nothing', 'naught'], ['pink', 'blue'], ['pin', 'paper'], ['barry', 'bob']]
    new_file = open(output_file + "output.txt", "a+", encoding='utf-8')
    new_file.write("- * - * - * -\n\n")
    new_file.write("Word Pairs and Distances:\n=== 50 Word Model === \n")
    c = 1
    for index, pair in enumerate(words):
        new_file.write(str(c) + '.' + pair[0] + " - " + pair[1] + ": " + str(
            pre_trained_model_50.similarity(pair[0], pair[1])) + "\n")
        c += 1
    c = 1
    new_file.write("\nWord Pairs and Distances:\n=== 300 Word Model === \n")
    for index, pair in enumerate(words):
        new_file.write(
            str(c) + '.' + pair[0] + " - " + pair[1] + ": " + str(
                pre_trained_model_300.similarity(pair[0], pair[1])) + "\n")
        c += 1
    # #2
    words = [[['hate', 'love'], ['up', 'down']], [['eat', 'food'], ['water', 'drink']],
             [['daughter', 'mother'], ['son', 'father']], [['books', 'read'], ['write', 'book']],
             [['flower', 'lily'], ['fruit', 'strawberry']]]
    new_file.write("\nAnalogies:\n")
    c = 1
    for pair in words:
        new_file.write(str(c) + '.' + pair[0][0] + " : " + pair[0][1] + ", " + pair[1][0] + " : " + pair[1][1] + "\n")
        c += 1
    new_file.write("\n=== 50 Word Model === \nMost Similar:\n")
    c = 1
    for pair in words:
        word = pre_trained_model_50.most_similar(positive=[pair[0][1], pair[1][1]], negative=[pair[1][0]])
        new_file.write(str(c) + '.' + pair[0][1] + " - " + pair[1][0] + " + " + pair[1][1] + "= " + word[0][0] + "\n")
        c += 1
    c = 1
    new_file.write("\nDistances:\n")
    for index, pair in enumerate(words):
        word = pre_trained_model_50.most_similar(positive=[pair[0][1], pair[1][1]], negative=[pair[1][0]])
        new_file.write(
            str(c) + '.' + pair[0][0] + " - " + word[0][0] + ": " + str(
                pre_trained_model_50.similarity(pair[0][0], word[0][0])) + "\n")
        c += 1
    new_file.write("\n=== 300 Word Model ===\nMost Similar:\n")
    c = 1
    for pair in words:
        word = pre_trained_model_300.most_similar(positive=[pair[0][1], pair[1][1]], negative=[pair[1][0]])
        new_file.write(str(c) + '.' + pair[0][1] + " - " + pair[1][0] + " + " + pair[1][1] + " = " + word[0][0] + "\n")
        c += 1
    new_file.write("\nDistances:\n")
    c = 1
    for index, pair in enumerate(words):
        word = pre_trained_model_300.most_similar(positive=[pair[0][1], pair[1][1]], negative=[pair[1][0]])
        new_file.write(
            str(c) + '.' + pair[0][0] + " - " + word[0][0] + ": " + str(
                pre_trained_model_300.similarity(pair[0][0], word[0][0])) + "\n")
        c += 1
    new_file.close()
    # #part 2
    en_corpus_file = codecs.open(en_corpus_file, encoding="utf-8")
    en_corpus = en_corpus_file.readlines()
    en_corpus_og = " ".join(en_corpus)
    en_corpus_string = en_corpus_og.lower()
    en_corpus = list(en_corpus_string.split())
    en_v_Bi_gram = retrieve_ngrams(en_corpus, 2)
    en_v_Tri_gram = retrieve_ngrams(en_corpus, 3)
    tweets_j = " ".join(tweets)
    tweets_j = tweets_j.lower()
    tweets_list = list(tweets_j.split())
    tweet_tri_grams = retrieve_ngrams(tweets, 3)
    words_2_chang_in_tweets = [["me", "promise", "keep"], ["Jobs"],
                               ["country", "wear", "socially", "matters", "help"],
                               ["message", "vaccine", "beat"], ["Climate", "get"], ["impeach", "hope"], ["democracy"],
                               ["better"], ["president", "morning"], ["weeks", "a", "Stay", "indoor", "protect"]]
    replacements = []
    for words in words_2_chang_in_tweets:
        tweet_words = []
        for word in words:
            tweet_words.append(pre_trained_model_300.most_similar(word.lower(), negative=None, topn=10))
        replacements.append(tweet_words)
    # get tri grams and bi grams for words from tweets
    tri_grams_tweets = []
    Bi_grams_tweets = []
    for index, tweet in enumerate(tweets):
        tri_gram_tweet = []
        bi_gram_tweet = []
        tweet_list = list(tweet.split())
        for word in words_2_chang_in_tweets[index]:
            i = tweet_list.index(word)
            tri_gram_tweet.append([tweet_list[i - 1].lower(), tweet_list[i].lower(), tweet_list[i + 1].lower()])
            bi_gram_tweet.append([tweet_list[i].lower(), tweet_list[i + 1].lower()])
        tri_grams_tweets.append(tri_gram_tweet)
        Bi_grams_tweets.append(bi_gram_tweet)
    count_tri = 0
    max_count_tri = [0, ""]
    count_bi = 0
    max_count_bi = [0, ""]
    new_tweets = []
    for index_tweet, tweet_n_grams in enumerate(tri_grams_tweets):
        new_line = []
        for index_word, tri_gram_tweet in enumerate(tweet_n_grams):
            for rep_word in replacements[index_tweet][index_word]:
                tri_gram_tweet[1] = rep_word[0]
                if en_v_Tri_gram.__contains__(tri_gram_tweet):
                    count_tri = en_v_Tri_gram.count(tri_gram_tweet)
                    if max_count_tri[0] <= count_tri:
                        max_count_tri = [count_tri, rep_word[0]]
                else:
                    bi_gram_tweet = Bi_grams_tweets[index_tweet][index_word]
                    bi_gram_tweet[0] = rep_word[0]
                    if en_v_Bi_gram.__contains__(bi_gram_tweet):
                        count_bi = en_v_Bi_gram.count(bi_gram_tweet)
                    if max_count_bi[0] <= count_bi:
                        max_count_bi = [count_bi, rep_word[0]]
            if max_count_tri[0] > 0:
                new_line.append(re.sub(r"\b" + words_2_chang_in_tweets[index_tweet][index_word] + r"\b", max_count_tri[1], tweets_lines[index_tweet][index_word]))
            else:
                if max_count_bi[0] > 0:
                    new_line.append(re.sub(r"\b" + words_2_chang_in_tweets[index_tweet][index_word] + r"\b",
                                           max_count_bi[1], tweets_lines[index_tweet][index_word]))
                else:
                    new_line.append(re.sub(r"\b" + words_2_chang_in_tweets[index_tweet][index_word] + r"\b",
                                           replacements[index_tweet][index_word][0][0], tweets_lines[index_tweet][index_word]))
            count_tri = 0
            max_count_tri = [0, ""]
            count_bi = 0
            max_count_bi = [0, ""]
        new_tweets.append(new_line)

    new_file = open(output_file + "output.txt", "a+", encoding='utf-8')
    new_file.write("\n\n- * - * - * -\n\n")
    new_file.write("\n\n === New Tweets ===\n")
    for index, tweet in enumerate(new_tweets):
        new_file.write("\n" + str(index + 1) + ".")
        for line in tweet:
            new_file.write(line + "\n")
    new_file.write("\n- * - * - * -\n")
    new_file.close()
    # part 3
    songs = read_lyrics_and_split(song_lyrics)
    songs_vec = []
    words_used_songs = []
    for song in songs:
        song_vec = []
        words_used =[]
        for word in song[2]:
            if pre_trained_model_300.__contains__(word.lower()):
                song_vec.append(pre_trained_model_300[word.lower()])
                words_used.append(word)
        songs_vec.append(song_vec)
        words_used_songs.append(words_used)
    # get avg song for all songs with three weights
    songs_eq_avg = []
    songs_random_avg = []
    songs_Wavg = []
    for index, song in enumerate(songs_vec):

        songs_eq_avg.append(eq_weight(song))
        songs_random_avg.append(random1_5_weight(song))
        songs_Wavg.append(count_based_weights(words_used_songs[index], song))

        # ************PCA**********

    list_final_vectors_Arithmetic_weight_PCA = []
    list_final_vectors_random_weight_PCA = []
    list_final_vectors_my_weight_PCA = []
    pca = PCA(n_components=2)

    principalComponents_1 = pca.fit_transform(songs_eq_avg)
    #list_final_vectors_Arithmetic_weight_PCA = pd.DataFrame(data=principalComponents_1 , columns=['principal component 1',  'principal component 2'])
    principalComponent_2 = pca.fit_transform(songs_random_avg)
    #list_final_vectors_random_weight_PCA = pd.DataFrame(data=principalComponent_2, columns=['principal component 1', 'principal component 2'])
    principalComponents_3 = pca.fit_transform(songs_Wavg)
    #list_final_vectors_my_weight_PCA = pd.DataFrame(data=principalComponents_3, columns=['principal component 1', 'principal component 2'])

    # plot
    # create a scatter plot of the projection
    pyplot.scatter(principalComponents_1[:, 0], principalComponents_1[:, 1])
    for i, song in enumerate(songs):
        pyplot.annotate(song[0] + "\n" + song[1], xy=(principalComponents_1[i, 0], principalComponents_1[i, 1]))
    pyplot.title("Arithmetic weight\n Hala Grefat")
    pyplot.show()

    pyplot.scatter(principalComponent_2[:, 0], principalComponent_2[:, 1])
    for i, song in enumerate(songs):
        pyplot.annotate(song[0] + "\n" + song[1], xy=(principalComponent_2[i, 0], principalComponent_2[i, 1]))
    pyplot.title("Random weight\n Hala Grefat")
    pyplot.show()

    pyplot.scatter(principalComponents_3[:, 0], principalComponents_3[:, 1])
    for i, song in enumerate(songs):
        pyplot.annotate(song[0] + "\n" + song[1], xy=(principalComponents_3[i, 0], principalComponents_3[i, 1]))
    pyplot.title("My weight function\n Hala Grefat")
    pyplot.show()














