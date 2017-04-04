import numpy as np
import unittest
import random

'''
extracts the list of words from fname, assuming only the first column contains words
'''
def getWords(fname, delimiter='\t', hasHeader=True):
    words = []
    with open(fname, 'r') as fin:
        for line in fin:
            if hasHeader:
                hasHeader = False
                continue
            w = str(line.split(delimiter)[0]).strip()
            if len(w) > 0:
                words.append(w)
    return words

'''
generates a string with given length and the string is the concatenation of list of given words (delimited by delimiter);
words are randomly selected from given list and then are concatenated by a white splace
'''
def genOneDoc(words, length, delimiter=' '):
    selected = []
    for i in range(length):
        selected.append(words[random.randint(0, len(words)-1)])
    return delimiter.join(selected)

def genDocs(words, lenMean, lenStd, ndocs):
    lens = np.random.normal(lenMean, lenStd, ndocs)
    docs = []
    for l in lens:
        l = 1 if l < 1 else int(l)
        docs.append(genOneDoc(words, l))
    return docs

def createTextData(fname, words, lenMean=10, lenStd=1, ndocs=10, ntargets=5):
    docs = genDocs(words, lenMean, lenStd, ndocs)
    with open(fname, 'w') as fout:
        fout.write('key\ttext\ttarget\n')
        for i in range(len(docs)):
            fout.write('%d\t%s\t%d\n' % (i+1, docs[i], random.randint(1, ntargets)))

def createNumData(fname, tokenSize, l, nobs, nclass=0):
    seqs = []
    targets = []
    for i in range(nobs):
        seqs.append(np.random.rand(tokenSize*l))
        if nclass <= 1:
            targets.append(np.random.rand(l))
        else:
            targets.append(np.random.randint(nclass, size=l))
    cols = []
    for i in range(l*tokenSize):
        cols.append('x'+str(i+1))
    for i in range(l):
        cols.append('y'+str(i+1))
    header = 'key\t%s\n' % ('\t'.join(cols))
    with open(fname, 'w') as fout:
        fout.write(header)
        for i in range(nobs):
            fout.write(str(i+1))
            for j in range(l*tokenSize):
                fout.write('\t%f' % seqs[i][j])
            for j in range(l):
                fout.write('\t%f' % targets[i][j])
            fout.write('\n')

# example of how to create random text docs
words = getWords('data/toy_embeddings.txt')
# createTextData('data/rand_docs.txt', words, lenMean=10, lenStd=5, ndocs=100, ntargets=7)
# createTextData('data/long_docs.txt', words, lenMean=100, lenStd=10, ndocs=300, ntargets=50)

# example of how to create random numeric sequences
# createNumData('data/rand_num_t7_l23.txt', 5, 23, 600)

# createNumData('data/rand_num_t7_l23_binary.txt', 5, 23, 600, nclass=2)
createNumData('data/rand_num_t1_l3_30_multiclass.txt', 1, 3, 30, nclass=7)
