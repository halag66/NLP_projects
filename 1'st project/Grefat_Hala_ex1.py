import codecs
from sys import argv
import re
import os
import sys
from langdetect import detect_langs



class CreateCorpus:

    def __init__(self, input_filename, output_dir_path):
        self._input = input_filename
        self._output = output_dir_path
        file = codecs.open(self._input, encoding="utf-8")
        self.text = file.readlines()

    def delete_empty_lines(self, text):
        lines = text.split("\n")
        non_empty_lines = [line for line in lines if line.strip() != ""]
        string_without_empty_lines = ""
        for line in non_empty_lines:
            string_without_empty_lines += line + "\n"
        return string_without_empty_lines

    # function to clean text
    def clean_txt(self):
        file = self.text
        text = "".join(file)
        cleaned_file = self.delete_empty_lines(text)
        # removes substring for the regular expression as [38]
        cleaned_file = re.sub(r"\[\d+\]", "", cleaned_file)
        # removing all html text
        cleaner = re.compile('%.*?%|[0-9]. |=.*?╣|=.*?╬|╚=.*?╗|=.*?╗|_.*?_|<.*?>'
                             '|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleaned_file = re.sub(cleaner, "", cleaned_file)
        # remove emojis
        regrex_pattern = re.compile(pattern="["
                                            u"\U0001F600-\U0001F64F"  # emoticons
                                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                            "]+", flags=re.UNICODE)
        alphabets = "([A-Za-z])"
        cleaned_file = re.sub(regrex_pattern, "", cleaned_file)
        non_words = [':<', ':)', ':(', ':v', ':D', ':T', ':/', '^_^', '**', '*', '^-^', '[]', '>', '<', '==',
                     '=', '>>', '<<', '....', '[-]']
        for w in non_words:
            cleaned_file = cleaned_file.replace(w, ' ')

        numbers = "([0-9])"
        prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
        suffixes = "(Inc|Ltd|Jr|Sr|Co)"
        starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
        acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
        cleaned_file = re.sub(alphabets + "[-]" + alphabets, "\\1<slash>\\2", cleaned_file)
        cleaned_file = cleaned_file.replace('-', '')
        cleaned_file = re.sub(prefixes, "\\1<prd>", cleaned_file)
        cleaned_file = re.sub(numbers + "[.]" + numbers, "\\1<prd>", cleaned_file)
        # cleaned_file = cleaned_file.replace(alphabets + "/" + alphabets, " or ")
        cleaned_file = re.sub(alphabets + "[/]" + alphabets, "\\1 or \\2", cleaned_file)
        # cleaned_file = cleaned_file.replace("/", " ")
        if "Ph.D" in cleaned_file:
            cleaned_file = cleaned_file.replace("Ph.D.", "Ph<prd>D<prd>")
        if "D." in cleaned_file:
            cleaned_file = cleaned_file.replace("D.", "D<prd>")
        if "e.g." in cleaned_file:
            cleaned_file = cleaned_file.replace("e.g.", "e<prd>g<prd>")
        if "i.e." in cleaned_file:
            cleaned_file = cleaned_file.replace("i.e.", "i<prd>e<prd>")
        if "..." in cleaned_file:
            cleaned_file = cleaned_file.replace("...", "<prb><prd><prd>")
        cleaned_file = re.sub(acronyms + " " + starters, "\\1<stop> \\2", cleaned_file)
        cleaned_file = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", cleaned_file)
        cleaned_file = re.sub(" " + suffixes + "[.]", " \\1<prd>", cleaned_file)
        cleaned_file = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>",
                              cleaned_file)
        cleaned_file = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", cleaned_file)
        cleaned_file = re.sub(alphabets + "'" + alphabets, "\\1<aps>\\2", cleaned_file)
        # remove whitespace at the beginning of a newline
        cleaned_file = cleaned_file.lstrip()
        return cleaned_file

    def count_tokens(self, sentence):
        count = 0
        non_Tokens = ["]", "[", "(", ")", "<<", ">>", "<", ">", "\/", "/", "'", "`", "~", "@",
                      "#", "$", "%", "^", "&", "*", "-", "_", "+", "=", "}", "{", "\"", "½", " ", "\t", "\n", '...']
        words = sentence.split()
        for word in words:
            is_token = 0
            for token in non_Tokens:
                if word == token:
                    is_token == 1
            if not is_token:
                count = count + 1
        return count

    def create_corpus(self, cleaned_file):
        # split to sentences each sentence made up of  tokens separated with space (\t), number of tokens minimum 4
        if "”" in cleaned_file:
            cleaned_file = cleaned_file.replace(".”", "”.")
        if "\"" in cleaned_file:
            cleaned_file = cleaned_file.replace(".\"", "\".")
        if "!" in cleaned_file:
            cleaned_file = cleaned_file.replace("!\"", "\"!")
        if "?" in cleaned_file:
            cleaned_file = cleaned_file.replace("?\"", "\"?")
        inside_quote = 0
        list_clean_file = cleaned_file.split()
        for index, word in enumerate(list_clean_file):
            if word == '"' or word == "'" or word == "(":
                inside_quote = 1
            if (word == '"' and inside_quote == 1) or (word == "'" and inside_quote == 1) or word == ")":
                inside_quote = 0
            if word == "\n" and inside_quote == 0:
                list_clean_file[index] = "stop"
            if word == "\n" and inside_quote == 1:
                list_clean_file[index] = " "

        #cleaned_file = '\n'.join(list_clean_file)
        cleaned_file = cleaned_file.replace("\n", "<stop>")
        cleaned_file = cleaned_file.replace(".", " .<stop>")
        cleaned_file = cleaned_file.replace(",", " ,")
        cleaned_file = cleaned_file.replace("¿", " ¿")
        cleaned_file = cleaned_file.replace(":", " :")
        cleaned_file = cleaned_file.replace(";", " ;")
        cleaned_file = cleaned_file.replace("?", " ?<stop>")
        cleaned_file = cleaned_file.replace("!", " !<stop>")
        cleaned_file = cleaned_file.replace("<prd>", ".")
        cleaned_file = cleaned_file.replace("<aps>", "'")
        cleaned_file = cleaned_file.replace("<slash>", "-")
        cleaned_file = re.sub(r'\s+', ' ', cleaned_file)  # Eliminate duplicate whitespaces using wildcards
        sentences = cleaned_file.split("<stop>")
        sentences = sentences[:-1]
        sentences = [s.replace("stop", "") for s in sentences]
        sentences = [s.strip() for s in sentences]
        # remove sentences with less than 4 tokens
        for s in sentences:
            if self.count_tokens(s) < 4:
                sentences.remove(s)
        sentences_str = "\n".join(sentences)
        # remove whitespace at the beginning of a newline
        cleaned_file = cleaned_file.lstrip()
        return sentences_str

    # classify func
    def classify(selfe, text):
        #print(text)
        try:
            langs = detect_langs(text)
            lang1 = langs[0]
        except:
            return None, None
           # lang1 = detect_langs(text.decode("utf-8"))[0]
        prob = lang1.prob
        lang_1 = lang1.lang
        #print(langs)
        for i, lang in enumerate(langs):
            #print(lang, i)
            if lang.prob > 0.95 and (lang.lang == 'en' or lang.lang == 'es'):
                return lang.lang, None
            if lang.prob > 0.60 and (lang.lang == 'en' or lang.lang == 'es'):
                j = i + 1
                while j < len(langs):
                    lang2 = langs[j] if len(langs) > j else None
                    if lang2.lang == 'en' or lang2.lang == 'es':
                        prob2 = lang2.prob
                        lang_2 = lang2.lang
                        if prob2 > 0.10:
                            return lang.lang, lang_2
                    j = j + 1

        return None, None

    def create_corpus_files(self):
        text = self.create_corpus(self.clean_txt())
        words = text.split("\n")
        for line in words:
            #print(line)
            new_name1, new_name2 = self.classify(line)
            if new_name1 is not None:
                if new_name2 is not None:
                    new_name = new_name1 + "_" + new_name2
                else:
                    new_name = new_name1
                new_file = open(self._output + new_name.replace("\n", "") + ".txt", "a+", encoding='utf-8')
                new_file.write(line + "\n")
                new_file.close()


if __name__ == "__main__":
    input_filename = argv[1]
    output_dir_path = argv[2]
    #string = "El informe incluye el criterio!\n\nA serious accident is defined as one involving a fire, at least one serious injury or fatalty and either substantial aircraft damage or aircraft destruction\n&gt; Una cosa es, nose, que se le salga una rueda al avión y despiste, pero en tierra y se bajen todos del avión. Otra muy distinta es que se venga a pique contra el suelo, imagino que a eso se refiere OP y lo que la mayoría se imagina al pensar en un accidente de avión, y si, estoy bastante de acuerdo de que en ese caso cagaste fuego.\n\nY...pero, si vas en plena ruta y el micro se va a la mierda también te podes cagar muriendo"
    #print(detect_langs(string))
    corps = CreateCorpus(input_filename, output_dir_path)
    corps.create_corpus_files()