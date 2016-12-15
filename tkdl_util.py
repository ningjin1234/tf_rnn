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
            emb = [0 for i in range(len(splitted)-1)]
            tid = len(embeddings)
            token2Id[term] = tid
            for i in range(1, len(splitted)):
                emb[i-1] = float(splitted[i])
            embeddings.append(emb)
    embeddingArray = [None for v in embeddings]
    tokens = sorted(token2Id.keys())
    for i in range(len(tokens)):
        embeddingArray[i] = embeddings[token2Id[tokens[i]]]
        token2Id[tokens[i]] = i
    embeddingArray = np.asarray(embeddingArray)
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
            for i in range(maxNumSteps-len(ids)):
                ids.append(0)
    return ids

def writeWeights(matrices, layerIdList, fname, breakdownDict={}):
    assert len(matrices)==len(layerIdList)
    nextWidInlayer = dict()
    with open(fname, 'w') as fout:
        fout.write('_LayerID_\t_WeightID_\t_Weight_\n')
        for i in range(len(matrices)):
            layerId = layerIdList[i]
            if not layerId in nextWidInlayer:
                nextWidInlayer[layerId] = 0
            wid = nextWidInlayer[layerId]
            if i not in breakdownDict:
                wid = writeWeightsAux(fout, layerId, wid, matrices[i])
            else:
                m1Dim = breakdownDict[i]
                mt = np.transpose(matrices[i])
                m1 = mt[:, :m1Dim]
                m2 = mt[:, m1Dim:]
                wid = writeWeightsAux(fout, layerId, wid, m1)
                wid = writeWeightsAux(fout, layerId, wid, m2)
            nextWidInlayer[layerId] = wid

def writeWeightsAux(fout, layerId, wid, matrix):
    reshaped = np.reshape(matrix, (-1))
    for v in reshaped:
        fout.write('%d\t%d\t%.12f\n' % (layerId, wid, v))
        wid += 1
    return wid

class TestTkdlUtil(unittest.TestCase):
    def testLoadEmbedding(self):
        token2Id, embeddingArray = readEmbeddingFile('data/toy_embeddings.txt')
        testSeqs = [['apple', 'is', 'a', 'company'], ['google', 'is', 'another', 'big', 'company']]
        self.assertEqual(tokens2ids(testSeqs[0], token2Id), [1, 7, 0, 4])
        self.assertEqual(tokens2ids(testSeqs[1], token2Id), [6, 7, 4])
        self.assertEqual(tokens2ids(testSeqs[1], token2Id, unk=len(token2Id)), [6, 7, 9, 9, 4])
        self.assertEqual(tokens2ids(testSeqs[1], token2Id, unk=len(token2Id), maxNumSteps=4), [6, 7, 9, 9])
        self.assertEqual(tokens2ids(testSeqs[1], token2Id, unk=len(token2Id), maxNumSteps=7), [6, 7, 9, 9, 4, 0, 0])
        self.assertEqual(embeddingArray[token2Id['apple']].tolist(), [2.,1.,0.])
        self.assertEqual(embeddingArray[token2Id['fruit']].tolist(), [8.,0.25,0.125])
        print('testLoadEmbedding passed')

# if __name__ == "__main__":
#     unittest.main()