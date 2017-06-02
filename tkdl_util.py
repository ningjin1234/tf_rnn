import tensorflow as tf
import numpy as np
import unittest
import nltk
'''
first column of the file must be the term column; the rest of the columns are treated as embedding content
'''
def readEmbeddingFile(fname, hasHeader=True, delimiter='\t', setUnk=True, ret_ndarray=True):
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
    if setUnk:
        embeddingArray.append([0.0000 for i in range(ndim)])
    if ret_ndarray:
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

def my_tokenize(doc):
    tmp = nltk.word_tokenize(doc)
    res = []
    for t in tmp:
        if not '-' in t:
            ts = [t]
        else:
            ts = nltk.wordpunct_tokenize(t)    
        for v in ts:
            if v == '``':
                v = '"'
            elif v == "''":
                v = '"'
            res.append(v)
    return res

def genTextParms(docs, embeddingFile):
    textParms = {}
    print('loading embedding file %s' % embeddingFile)
    token2Id, embeddingArray = readEmbeddingFile(embeddingFile)
    maxNumSteps = 0
    lens = []
    for doc in docs:
        idList = tokens2ids(doc, token2Id)
        lens.append(len(idList))
        if len(idList) > maxNumSteps:
            maxNumSteps = len(idList)
    inputIds = []
    for doc in docs:
        ids = tokens2ids(doc, token2Id, maxNumSteps=maxNumSteps)
        inputIds.append(ids)
    inputIds = np.asarray(inputIds, dtype=np.int32)
    lens = np.asarray(lens, dtype=np.int32)
    embeddingArray = np.asarray(embeddingArray, dtype=np.float32)
    textParms['ids'] = inputIds
    textParms['lens'] = lens
    textParms['emb'] = embeddingArray
    textParms['maxl'] = maxNumSteps
    textParms['token2id'] = token2Id
    return textParms

# directly updates token2id and emb_arr
def expand_embedding_with_oovs(all_tokens, token2id, emb_arr, min_cnt=1):
    unknown_cnt = dict()
    for l in all_tokens:
        for t in l:
            if not t in token2id:
                if not t in unknown_cnt:
                    unknown_cnt[t] = 1
                else:
                    unknown_cnt[t] += 1
    ndim = len(emb_arr[0])
    for t,c in unknown_cnt.items():
        if c >= min_cnt:
            token2id[t] = len(emb_arr)
            emb_arr.append([0.0 for _ in range(ndim)])
    
# all_tokens is a list, where each element is a list of tokens
def get_text_parms_with_oovs(all_tokens, emb_fname=None, has_header=False, delimiter='\t', min_cnt=1, token2id=None, emb_arr=None):
    if emb_fname is None and token2id is None and emb_arr is None:
        raise ValueError('insufficient information to generate/update embeddings')
    # generate term-to-id mapping and embedding array (as list) based on embedding file 
    if token2id is None or emb_arr is None:
        token2id, emb_arr = readEmbeddingFile(emb_fname, hasHeader=has_header, delimiter=delimiter, setUnk=False, ret_ndarray=False)
    # expand mapping and embedding array with OOV words
    expand_embedding_with_oovs(all_tokens, token2id, emb_arr, min_cnt=min_cnt)
    # get length list
    lens = [len(l) for l in all_tokens]
    max_len = max(lens)
    # get token id lists with padding
    tid_lists = []
    for l in all_tokens:
        tid_l = tokens2ids(l, token2id, maxNumSteps=max_len)
        tid_lists.append(tid_l)
    parms = dict()
    parms['ids'] = tid_lists
    parms['lens'] = lens
    parms['emb'] = emb_arr
    parms['maxl'] = max_len
    parms['token2id'] = token2id
    return parms

def parseTextParms(inputTextParms):
    return inputTextParms['ids'], inputTextParms['lens'], inputTextParms['emb'], inputTextParms['maxl']


def getLayerName(fullName):
    cellTypes = ['BasicRNNCell', 'GRUCell', 'LSTMCell', 'output']
    for ct in cellTypes:
        if ct in fullName:
            return fullName[:fullName.find(ct)]
    return ''
# returns whether it's a weight matrix or not
def isMatrix(fullName):
    if 'Matrix' in fullName or 'outputW' in fullName:
        return True
    elif 'W_0' in fullName or '_diag' in fullName:
        return True
    return False
# assumes this is from a MultiRNNCell
def getCellId(fullName):
    s = 'MultiRNNCell/Cell'
    i = fullName.find(s) + len(s)
    return int(fullName[i:i+1])

def writeWeightsWithNames(matrices, variables, stackedDimList, fname):
    name2Weights = dict()
    for m, v in zip(matrices, variables):
        vname = v.name
        layerName = getLayerName(vname)
        if layerName not in name2Weights:
            if 'MultiRNNCell' not in vname:
                layerId = len(name2Weights) + 1
            else:
                cellId = getCellId(vname) + 1
                if 'FW' in vname:
                    cellId += len(stackedDimList) - 1
                layerId = cellId
            name2Weights[layerName] = (layerId, [], [], layerName)
        winfo = name2Weights[layerName]
        if isMatrix(vname):
            winfo[1].append(m)
        else:
            winfo[2].append(m)
    nextWidInlayer = dict()
    with open(fname, 'w') as fout:
        fout.write('_LayerID_\t_WeightID_\t_Weight_\n')
        for name, winfo in name2Weights.items():
            layerId = winfo[0]
            matrices = winfo[1]
            biases = winfo[2]
            # print(layerId, winfo[3])
            if not layerId in nextWidInlayer:
                nextWidInlayer[layerId] = 0
            matrices = matrices + biases
            for m in matrices:
                wid = nextWidInlayer[layerId]
                wid = writeWeightsAux(fout, layerId, wid, m)
                nextWidInlayer[layerId] = wid

def writeWeights(matrices, layerIdList, fname, breakdownDict={}):
    if len(matrices)!=len(layerIdList):
        print(len(matrices), len(layerIdList))
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
        fout.write('%d\t%d\t%s\n' % (layerId, wid, str(v)))
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
    def test_get_text_parms_with_oovs(self):
        emb_truth = {}
        emb_truth['google'] = [0.001,0.9,0.0012]
        emb_truth['is'] = [0.0125,0.0125,0.0125]
        emb_truth['company'] = [0.0,1.0,0.0]
        emb_truth['color'] = [0.0125,0.025,1.0]
        all_tokens = [['google', 'is', 'another', 'company'], ['google', 'is', 'not', 'color', '!']]
        parms = get_text_parms_with_oovs(all_tokens, emb_fname='toyEmbedding3.csv', has_header=True, delimiter=',')
        self.assertEqual(len(parms['emb']), 14)
        emb_arr = parms['emb']
        token2id = parms['token2id']
        for t in ['another', 'not', '!']:
            self.assertEqual(emb_arr[token2id[t]], [0.0, 0.0, 0.0])
        for t in ['google', 'is', 'company', 'color']:
            self.assertEqual(emb_arr[token2id[t]], emb_truth[t])
        print('test_get_text_parms_with_oovs passed')

# if __name__ == "__main__":
#     unittest.main()
