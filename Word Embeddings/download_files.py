from importlib.metadata import files
from sys import argv
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sympy.vector import vector
from gensim.test.utils import datapath, get_tmpfile

if __name__ == "__main__":
    #Load the GolVe to vector files, blacken after loading (or do it in a separate Python file):

    #glove_file = datapath("test_50word2vec.txt")
    #tmp_file = get_tmpfile("glove.6B.50d.txt")
    '''
    glove2word2vec(argv[3], argv[1] + 'train_300.kv')
    glove2word2vec(argv[2], argv[1] + 'train_50.kv')
    pre_trained_model_50 = KeyedVectors.load_word2vec_format(argv[2], binary=False)
    pre_trained_model_300 = KeyedVectors.load_word2vec_format(argv[3], binary=False)
    pre_trained_model_50.save(argv[1] + 'train_50.kv')
    pre_trained_model_300.save(argv[1] + 'train_300.kv')
    '''
    from sklearn.decomposition import PCA
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data
    pca = PCA(n_components=2)
    X_new = pca.fit_transform(X)