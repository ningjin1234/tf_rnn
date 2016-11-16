import tensorflow as tf
import numpy as np
import unittest

def readEmbeddingFile(fname, hasHeader=True, delimiter='\t'):
    ndim = 0
    embeddings = []
    token2Id = dict()
    with open(fname, 'r') as fin:
        for line in fin:
            if hasHeader:
                hasHeader = False
                continue
            if line.strip() == '':
                continue
            splitted = line.strip().split(delimiter)
            term = splitted[0]
            if ndim == 0:
                ndim = len(splitted) - 1
            else:
                assert ndim == len(splitted)-1
            emb = [0 for i in xrange(len(splitted)-1)]
            tid = len(embeddings)
            token2Id[term] = tid
            for i in xrange(1, len(splitted)):
                emb[i-1] = float(splitted[i])
            embeddings.append(emb)
    embeddingArray = np.asarray(embeddings)
    tokens = sorted(token2Id.keys())
    for i in xrange(len(tokens)):
        token2Id[tokens[i]] = i
    return token2Id, embeddingArray

# input is a list of strings where each element is one token
def tokens2ids(tokens, token2IdLookup, unk=None, maxNumSteps=None):
    ids = []
    for t in tokens:
        if not t in token2IdLookup:
            if unk is not None:
                ids.append(unk)
            continue
        ids.append(token2IdLookup[t])
    if maxNumSteps is not None:
        if len(ids) > maxNumSteps:
            ids = ids[:maxNumSteps]
        elif len(ids) < maxNumSteps:
            for i in xrange(maxNumSteps-len(ids)):
                ids.append(0)
    return ids

class TestTkdlUtil(unittest.TestCase):
    def testLoadEmbedding(self):
        token2Id, embeddingArray = readEmbeddingFile('data/toy_embeddings.txt')
        testSeqs = [['apple', 'is', 'a', 'company'], ['google', 'is', 'another', 'big', 'company']]
        self.assertEquals(tokens2ids(testSeqs[0], token2Id), [1, 7, 0, 4])
        self.assertEquals(tokens2ids(testSeqs[1], token2Id), [6, 7, 4])
        self.assertEquals(tokens2ids(testSeqs[1], token2Id, unk=len(token2Id)), [6, 7, 9, 9, 4])
        self.assertEquals(tokens2ids(testSeqs[1], token2Id, unk=len(token2Id), maxNumSteps=4), [6, 7, 9, 9])
        self.assertEquals(tokens2ids(testSeqs[1], token2Id, unk=len(token2Id), maxNumSteps=7), [6, 7, 9, 9, 4, 0, 0])
        print 'testLoadEmbedding passed'

# if __name__ == "__main__":
#     unittest.main()