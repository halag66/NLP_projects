import codecs
import string
from sys import argv
from langdetect import detect_langs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier






class Classifier:

    def __init__(self, corpus_file, output_filename, english_file, spanish_file):
        self._corpus_file = corpus_file
        self._output_file = output_filename
        self.en_dec = english_file
        self.es_dec = spanish_file

    # classify func
    def classify(self, text):
        try:
            langs = detect_langs(text)
            lang1 = langs[0]
        except:
            return ""
        for i, lang in enumerate(langs):
            if lang.lang == 'en' or lang.lang == 'es':
                return lang.lang
        return ""

    def label_words(self):
        classified_tokens = []
        non_Tokens = ["]", "[", "(", ")", "<<", ">>", "<", ">", "\/", "/", "'", "`", "~", "@",
                      "#", "$", "%", "^", "&", "*", "-", "_", "+", "=", "}", "{", "\"", "½", " ", "\t", "\n", '...', ","
            , ";", ":", "?", "!", "'", "."]
        first_word = 0
        prev_word_is = ""
        current_word_is = ""
        for line in self._corpus_file:
            prev_word_is = ""
            first_word = 1
            line_ = line.split()
            for index, word in enumerate(line_):
                word_lower = word.lower()
                current_word_is = ""
                if word_lower in non_Tokens or word.isnumeric():
                    classified_tokens.append(0)
                    current_word_is = prev_word_is
                else:
                    if first_word:
                        first_word = 0
                        classified_tokens.append(0)
                        if self.en_dec.__contains__(word_lower):
                            current_word_is = prev_word_is = "en"
                        if self.es_dec.__contains__(word_lower) and current_word_is != "en":
                            current_word_is = prev_word_is = "es"
                        if current_word_is == "" or (self.es_dec.__contains__(word_lower) and self.en_dec.__contains__(word_lower)):
                            current_word_is = prev_word_is = self.classify(word)
                    else:
                        if self.en_dec.__contains__(word_lower):
                            current_word_is = "en"
                        if self.es_dec.__contains__(word_lower) and current_word_is != "en":
                            current_word_is = "es"
                        if current_word_is == "" or (self.es_dec.__contains__(word_lower) and self.en_dec.__contains__(word_lower)):
                            current_word_is = self.classify(word)
                        if current_word_is == prev_word_is or current_word_is.__contains__(prev_word_is) or \
                                prev_word_is.__contains__(current_word_is):
                            classified_tokens.append(0)
                            if current_word_is == "":
                                current_word_is = prev_word_is
                        else:
                            classified_tokens.append(1)          # chang code
                        prev_word_is = current_word_is
        return classified_tokens

    def get_feature_vector(self):
        feature1_vec = []
        feature2_vec = []
        feature3_vec = []
        feature4_vec = []
        feature5_vec = []
        feature6_vec = []
        feature7_vec = []
        first_word = 1
        special_words = ["la", "de", "a", "and", "the", "of", "to", "is", "na", "too", "did", "do", "can", "aka"]
        for line in self._corpus_file:
            first_word = 1
            line_ = line.split()
            for word in line_:
                feature_vec = []
                # feature #1 if word starts with capital letter and isn't first word in sentence
                if not first_word:
                    feature1_vec.append(word[:1].isupper())
                else:
                    feature1_vec.append(0)
                    first_word = 0
                # feature #2 does word belong to both eng and es dictionary
                feature2_vec.append(self.es_dec.__contains__(word.lower()) and self.en_dec.__contains__(word.lower()))
                # feature #3 is word in all caps
                feature3_vec.append(word.isupper())
                # #4 does word contain any punctuation
                invalidcharacters = set(string.punctuation)
                if word not in invalidcharacters:
                    feature4_vec.append(any(char in invalidcharacters for char in word))
                else:
                    feature4_vec.append(False)
                # #5 is word first word in line
                feature5_vec.append(first_word)
                # #6 is word last word in line
                feature6_vec.append((word == line_[-1] and word != ".") or (word == line_[-2] and "." == line_[-1])) # not sure
                # #7 is it a special word
                feature7_vec.append(special_words.__contains__(word.lower()))
        feature_vectors = list(zip(feature1_vec,feature2_vec,feature3_vec,feature4_vec,feature5_vec,feature6_vec,feature7_vec))
        return feature_vectors


def plot_conf_mat(mat, classifier):
    # using confusion_matrix for KNeighborsClassifier
    matrix1 = np.array(mat)
    fig, ax = plt.subplots()
    im = ax.imshow(matrix1)
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    plt.ylabel('True label ')
    plt.xlabel('Predicted label')
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, matrix1[i, j])
    ax.set_title("Hala Grefat -" + classifier)
    fig.tight_layout()
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("", rotation=-90, va="bottom")
    plt.show()



if __name__ == '__main__':
    input_path = argv[1]
    output_path = argv[2]
    english_dict_file = argv[3]
    spanish_dict_file = argv[4]

    file = codecs.open(input_path, encoding="utf-8-sig")
    text = file.readlines()
    text = [line.strip() for line in text]
    file_dict = codecs.open(english_dict_file, encoding="utf-8-sig")
    en_dic = file_dict.readlines()
    en_dic = [line.strip() for line in en_dic]
    file = codecs.open(spanish_dict_file, encoding="utf-8-sig")
    es_dict = file.readlines()
    es_dict = [line.strip() for line in es_dict]
    # part 2 features
    features = "The features are:\n#1 does word/token starts with capital letter and isn't first word in sentence\n" \
               "#2 does word belongs to both en and es dictionaries\n" \
               "#3 is word in all caps\n" \
               "#4 does word contain any punctuation example: she's\n" \
               "#5 is word first word in line\n" \
               "#6 is word last word in line and is not '.' or is second to last and '.' is last \n" \
               "#7 is it one of the words:[la, de, a, and, the, of, to, is, na, too, did, do, can]\n\n"
    new_file = open(output_path + "output.txt", "a+", encoding='utf-8')
    new_file.write(features)
    new_file.close()
    cs_cls = Classifier(text, output_path, en_dic, es_dict)
    features_vector = cs_cls.get_feature_vector()
    labeled_words = cs_cls.label_words()
    # part 3 classifying
    # split to training and testing
    labeled_words_train, labeled_words_test, feat_vec_train, feat_vec_test = train_test_split(labeled_words, features_vector, test_size=0.30)
    # KNN
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(feat_vec_train, labeled_words_train)
    pred_label_test1 = neigh.predict(feat_vec_test)
    new_file = open(output_path + "output.txt", "a+", encoding='utf-8')
    new_file.write("KNeighborsClassifier:\n" + classification_report(labeled_words_test, pred_label_test1,zero_division=1))
    new_file.close()
    # plot matrix KNeighborsClassifier
    mat = confusion_matrix(labeled_words_test, pred_label_test1)
    plot_conf_mat(mat, "KNeighborsClassifier")
    #Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=130)
    clf.fit(feat_vec_train, labeled_words_train)
    pred_label_test2 = clf.predict(feat_vec_test)
    new_file = open(output_path + "output.txt", "a+", encoding='utf-8')
    new_file.write("\nRandomForest:\n" + classification_report(labeled_words_test, pred_label_test2,zero_division=1))
    new_file.close()
    # plot matrix KNeighborsClassifier
    mat = confusion_matrix(labeled_words_test, pred_label_test1)
    plot_conf_mat(mat, "RandomForest")




