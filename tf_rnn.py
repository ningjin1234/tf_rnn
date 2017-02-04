# legacy code to check gradient calculation of C implementation; DO NOT add new features to this
import pandas
from tkdl_util import *
from tensorflow.python.ops import array_ops

def getRnnCell(nNeurons, cell='rnn', nCells=1, act=tf.tanh):
    ret = []
    rnnCell = None
    for i in range(nCells):
        if cell == 'rnn':
            rnnCell = tf.nn.rnn_cell.BasicRNNCell(nNeurons, activation=act)
        elif cell == 'gru':
            rnnCell = tf.nn.rnn_cell.GRUCell(nNeurons, activation=act)
        elif cell == 'lstm':
            rnnCell = tf.nn.rnn_cell.LSTMCell(nNeurons, activation=act, use_peepholes=True, forget_bias=0.0) 
        ret.append(rnnCell)
    if nCells == 1:
        return ret[0]
    return ret

def getRnnTrainOps(maxNumSteps=10, nNeurons=4, initEmbeddings=None, tokenSize=1,
                        bias_trainable=True, learningRate=0.1, rnnType='normal', stackedDimList=[],
                        task='perseq', cell='rnn', nclass=0):
    tf.reset_default_graph()
    tf.set_random_seed(32513)
    inputTokens = tf.placeholder(tf.int32, [None, maxNumSteps])
    inputLens = tf.placeholder(tf.int32, [None])
    if task.lower() in ['perseq']:
        if nclass <= 1:
            targets = tf.placeholder(tf.float64, [None, 1])
        else:
            targets = tf.placeholder(tf.int32, [None])
    elif task.lower() in ['pertoken', 'perstep']:   # corresponds to same-length output type in tkdlu; all input seqs must have same length
        if nclass <= 1:
            targets = tf.placeholder(tf.float64, [None, maxNumSteps])
        else:
            targets = tf.placeholder(tf.int32, [None])
    else:
        assert(False)
    if stackedDimList is None or len(stackedDimList) == 0:
        if cell == 'rnn':
            rnnCell = tf.nn.rnn_cell.BasicRNNCell(nNeurons, activation=tf.tanh)
        elif cell == 'gru':
            rnnCell = tf.nn.rnn_cell.GRUCell(nNeurons, activation=tf.tanh)
        elif cell == 'lstm':
            rnnCell = tf.nn.rnn_cell.LSTMCell(nNeurons, activation=tf.tanh, use_peepholes=True, forget_bias=0.0) # default forget bias is 1.0
    else:
        if cell == 'rnn':
            rnnCellList = [tf.nn.rnn_cell.BasicRNNCell(dim, activation=tf.tanh) for dim in stackedDimList]
        elif cell == 'gru':
            rnnCellList = [tf.nn.rnn_cell.GRUCell(dim, activation=tf.tanh) for dim in stackedDimList]
        elif cell == 'lstm':
            rnnCellList = [tf.nn.rnn_cell.LSTMCell(dim, activation=tf.tanh, use_peepholes=True, forget_bias=0.0) for dim in stackedDimList]
        rnnCell = tf.nn.rnn_cell.MultiRNNCell(rnnCellList)
        nNeurons = stackedDimList[-1]
    # keep this code for future reference: training initial states
    # initState = tf.get_variable("initState", [nNeurons], dtype=tf.float64, trainable=True)
    # initStates = tf.concat(0, [initState for i in range(batchSize)])
    # initStates = tf.reshape(initStates, [-1, nNeurons])

    if initEmbeddings is not None:
        embedding = tf.Variable(initEmbeddings, name='inputEmbeddings', trainable=False, dtype=tf.float64)
        inputData = tf.nn.embedding_lookup(embedding, inputTokens)
    else:
        inputTokens = tf.placeholder(tf.float64, [None, maxNumSteps, tokenSize])
        inputData = inputTokens

    if rnnType.lower() == 'normal':
        raw_outputs, last_states = tf.nn.dynamic_rnn(cell=rnnCell, dtype=tf.float64, sequence_length=inputLens, inputs=inputData)
    elif rnnType.lower() == 'reversed' or rnnType.lower() == 'reverse':
        inputDataReversed = array_ops.reverse_sequence(input=inputData, seq_lengths=inputLens, seq_dim=1, batch_dim=0)
        raw_outputs_r, last_states = tf.nn.dynamic_rnn(cell=rnnCell, dtype=tf.float64, sequence_length=inputLens, inputs=inputDataReversed)
        raw_outputs = array_ops.reverse_sequence(input=raw_outputs_r, seq_lengths=inputLens, seq_dim=1, batch_dim=0)
    elif rnnType.lower() in ['bi', 'stackedbi', 'bistacked', 'bidirectional']:
        stackedDimList = stackedDimList[:-1]
        if rnnType.lower() in ['stackedbi', 'bistacked']:
            tmpInputs = inputData
            for i in range(len(stackedDimList)):
                n = stackedDimList[i]
                with tf.variable_scope('layer%d'%i):
                    cells = getRnnCell(n, cell=cell, nCells=2)
                    fwRnnCell = cells[0]
                    bwRnnCell = cells[1]
                    tmpSeq, tmp_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwRnnCell, cell_bw=bwRnnCell, dtype=tf.float64, 
                                                                            sequence_length=inputLens, inputs=tmpInputs)
                    tmpInputs = tf.concat(2, [tmpSeq[0], tmpSeq[1]])
                tmp_outputs = tmpSeq
        else:
            if cell == 'rnn':
                fwRnnCellList = [tf.nn.rnn_cell.BasicRNNCell(dim, activation=tf.tanh) for dim in stackedDimList]
                bwRnnCellList = [tf.nn.rnn_cell.BasicRNNCell(dim, activation=tf.tanh) for dim in stackedDimList]
            elif cell == 'gru':
                fwRnnCellList = [tf.nn.rnn_cell.GRUCell(dim, activation=tf.tanh) for dim in stackedDimList]
                bwRnnCellList = [tf.nn.rnn_cell.GRUCell(dim, activation=tf.tanh) for dim in stackedDimList]
            elif cell == 'lstm':
                fwRnnCellList = [tf.nn.rnn_cell.LSTMCell(dim, activation=tf.tanh, use_peepholes=True, forget_bias=0.0) for dim in stackedDimList]
                bwRnnCellList = [tf.nn.rnn_cell.LSTMCell(dim, activation=tf.tanh, use_peepholes=True, forget_bias=0.0) for dim in stackedDimList]
            fwRnnCell = tf.nn.rnn_cell.MultiRNNCell(fwRnnCellList)
            bwRnnCell = tf.nn.rnn_cell.MultiRNNCell(bwRnnCellList)
            # NOTE: in bidirectional_dynamic_rnn, tensorflow does not concatenate outputs for each layer, it only concatenates the outputs for the last layer
            tmp_outputs, tmp_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwRnnCell, cell_bw=bwRnnCell,
                                                                       dtype=tf.float64, sequence_length=inputLens, inputs=inputData)
        # NOTE: currently the last layer in a stacked bidirectional RNN model must be a unidirectional recurrent layer;
        # this is because of an earlier limitation of tkdlu; if the last dim is 0, then there's no unidirectional recurrent layer
        fwOutputs = tmp_outputs[0]
        if nNeurons <= 0 and task.lower() in ['perseq']:
            bwOutputs = array_ops.reverse_sequence(input=tmp_outputs[1], seq_lengths=inputLens, seq_dim=1, batch_dim=0)
        else:
            bwOutputs = tmp_outputs[1]
        tmp_outputs = tf.concat(2, [fwOutputs, bwOutputs])
        # print(tmp_outputs.get_shape())
        # print('number of neurons: %d' % nNeurons)
        if nNeurons > 0:
            if cell == 'rnn':
                rnnCell = tf.nn.rnn_cell.BasicRNNCell(nNeurons, activation=tf.tanh)
            elif cell == 'gru':
                rnnCell = tf.nn.rnn_cell.GRUCell(nNeurons, activation=tf.tanh)
            elif cell == 'lstm':
                rnnCell = tf.nn.rnn_cell.LSTMCell(nNeurons, activation=tf.tanh, use_peepholes=True, forget_bias=0.0)
            raw_outputs, last_states = tf.nn.dynamic_rnn(cell=rnnCell, dtype=tf.float64, sequence_length=inputLens, inputs=tmp_outputs)
        else:
            raw_outputs = tmp_outputs
            last_states = tmp_states
            nNeurons = stackedDimList[-1]*2
    # print('number of neurons: %d' % nNeurons)
    flattened_outputs = tf.reshape(raw_outputs, [-1, nNeurons])
    if task.lower() in ['perseq']:
        batchSize = tf.shape(inputLens)[0]
        if rnnType.lower() in ['reversed', 'reverse']:
            index = tf.range(0, batchSize) * maxNumSteps
        else:
            index = tf.range(0, batchSize) * maxNumSteps + inputLens - 1
        outputs = tf.gather(flattened_outputs, index)
    else:
        outputs = flattened_outputs
        if nclass <= 1:
            targets = tf.reshape(targets, [-1, 1])
        else:
            targets = tf.reshape(targets, [-1])
    nclass = 1 if nclass <= 1 else nclass
    outputW = tf.get_variable("outputW", [nNeurons, nclass], dtype=tf.float64)
    outputB = tf.get_variable("outputB", [nclass], dtype=tf.float64)
    prediction = tf.add(tf.matmul(outputs, outputW), outputB)
    if task.lower() in ['perseq']:
        if nclass <= 1:
            loss = tf.reduce_sum(tf.pow(prediction-targets, 2)/2)
        else:
            logits = tf.reshape(prediction, [-1, nclass])
            softmax = tf.nn.softmax(logits) # for debugging purpose
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)
            loss = tf.reduce_sum(losses)
    elif task.lower() in ['pertoken', 'perstep']:
        if nclass <= 1:
            loss = tf.reduce_sum(tf.pow(prediction-targets, 2)/2/maxNumSteps)
        else:
            logits = tf.reshape(prediction, [-1, nclass])
            softmax = tf.nn.softmax(logits) # for debugging purpose
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)
            loss = tf.reduce_sum(losses/maxNumSteps)
    else:
        assert(False)
    lr = tf.Variable(learningRate, trainable=False)
    tvars = tf.trainable_variables()
    optimizer = tf.train.GradientDescentOptimizer(lr)
    gradients = optimizer.compute_gradients(loss, var_list=tvars) # for debugging purpose
    learningStep = optimizer.minimize(loss, var_list=tvars)
    initAll = tf.global_variables_initializer()
    # last return is output to screen for debugging purpose
    return inputTokens, inputLens, targets, prediction, loss, initAll, learningStep, gradients, lr, flattened_outputs

def trainRnn(docs, labels, nNeurons, embeddingFile, miniBatchSize=-1, initWeightFile=None, trainedWeightFile=None, lr=0.1, epochs=1,
             rnnType='normal', stackedDimList=[], task='perseq', cell='rnn', tokenSize=1, nclass=0):
    assert len(docs) == len(labels)
    maxNumSteps = 0
    ndocs = len(docs)
    if miniBatchSize < 0:
        miniBatchSize = ndocs
    nbatches = int(ndocs/miniBatchSize)
    if ndocs % miniBatchSize > 0:
        nbatches += 1
    lens = []
    if embeddingFile is not None:
        token2Id, embeddingArray = readEmbeddingFile(embeddingFile)
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
        embeddingArray = np.asarray(embeddingArray, dtype=np.float64)
    else:
        lens = [int(len(doc)/tokenSize) for doc in docs]
        lens = np.asarray(lens, dtype=np.int32)
        maxNumSteps = max(lens)
        embeddingArray = None
        inputIds = np.asarray(docs, dtype=np.float64)
        inputIds = np.reshape(inputIds, (ndocs, maxNumSteps, tokenSize))
        labels = np.asarray(labels, dtype=np.float64)
        labels = np.reshape(labels, (-1, 1))
        if nclass>1:
            labels = np.asarray(labels, dtype=np.int32)
            labels = np.reshape(labels, (-1))
    inputTokens, inputLens, targets, prediction, loss, initAll, learningStep, gradients, learningRate, debugInfo = getRnnTrainOps(maxNumSteps=maxNumSteps,
                                                                                                   nNeurons=nNeurons, initEmbeddings=embeddingArray,
                                                                                                   learningRate=lr/miniBatchSize, rnnType=rnnType,
                                                                                                   stackedDimList=stackedDimList, task=task,
                                                                                                   cell=cell, tokenSize=tokenSize, nclass=nclass)
    # for d in docs[:10]:
    #     print(d)
    # for l in labels[:10]:
    #     print(l)
    print('learning rate: %f' % lr)
    print('rnn type: %s' % rnnType)
    print('cell type: %s' % cell)
    print('task type: %s' % task)
    print('mini-batch size: %d' % miniBatchSize)
    with tf.Session() as sess:
        sess.run(initAll)
        if initWeightFile is not None:
            ws = sess.run(tf.trainable_variables())
            writeWeightsWithNames(ws, tf.trainable_variables(), stackedDimList, initWeightFile)
        feed_dict = {inputTokens:inputIds, inputLens:lens, targets:labels}
        print('loss before training: %.14g' % (sess.run(loss, feed_dict=feed_dict)/ndocs))
        # print(sess.run(debugInfo, feed_dict=feed_dict))
        for i in range(epochs):
            for j in range(nbatches):
                start = miniBatchSize*j
                if j < nbatches - 1:
                    end = miniBatchSize * (j+1)
                    if task.lower() in ['perseq']:
                        subTargets = labels[start:end]
                    else:
                        subTargets = labels[start*maxNumSteps:end*maxNumSteps]
                else:
                    end = ndocs
                    if task.lower() in ['perseq']:
                        subTargets = labels[start:end]
                    else:
                        subTargets = labels[start*maxNumSteps:end*maxNumSteps]
                sess.run(learningRate.assign(lr/(end-start)))
                feed_dict = {inputTokens:inputIds[start:end], inputLens:lens[start:end], targets:subTargets}
                print('\tbefore batch %d: %.14g' % (j, sess.run(loss, feed_dict=feed_dict)/(end-start)))
                sess.run(learningStep, feed_dict=feed_dict)
            feed_dict = {inputTokens:inputIds, inputLens:lens, targets:labels}
            print('loss after %d epochs: %.14g' % (i+1, sess.run(loss, feed_dict=feed_dict)/ndocs))
        if trainedWeightFile is not None:
            ws = sess.run(tf.trainable_variables())
            writeWeightsWithNames(ws, tf.trainable_variables(), stackedDimList, trainedWeightFile)

def getTextDataFromFile(fname, key='key', text='text', target='target', delimiter='\t'):
    table = pandas.read_table(fname)
    docs = table[text].values
    targets = table[target].values
    tokenized = []
    for doc in docs:
        tokenized.append(doc.split())
    labels = []
    for t in targets:
        labels.append([t])
    return tokenized, labels

def getNumDataFromFile(fname, inputLen, targetLen, delimiter='\t'):
    inputs = []
    targets = []
    with open(fname, 'r') as fin:
        header = fin.readline()
        for line in fin:
            splitted = line.strip().split(delimiter)
            invec = []
            outvec = []
            splitted = splitted[1:]
            assert (len(splitted) >= inputLen+targetLen)
            for v in splitted[:inputLen]:
                invec.append(float(v))
            for v in splitted[inputLen:inputLen+targetLen]:
                outvec.append(float(v))
            inputs.append(invec)
            targets.append(outvec)
    return inputs, targets

# this is needed to make TF and TKDLU use the same levelization
# there's no easy way to get the mapping, so currently I'm only testing binary targets for classification
def mapTargets(targets, targetMap):
    for arr in targets:
        for i in range(len(arr)):
            arr[i] = targetMap[arr[i]]

doc1 = "apple is a company".split()
doc2 = "google is another big company".split()
doc3 = "orange is a fruit".split()
doc4 = "apple google apple google apple google apple google".split()
doc5 = "blue is a color".split()
doc6 = "blue orange color apple google company".split()
doc7 = "google is company".split()
docs = [doc1, doc2, doc3, doc4, doc5, doc6]
# doc1 = "apple".split()
# docs = [doc1]
# docs = [doc1, doc7]
# docs = [reversed(doc1), reversed(doc2), reversed(doc3), reversed(doc4), reversed(doc5)]
# docs = [['apple','is'], ['google','is'],['orange','is']]
# docs = [['apple'], ['google'],['orange'],['company'],['fruit']]
# docs = [['apple', 'is', 'a'], ['google', 'is']]
# labels = [[0.6], [0.7], [0.8]]
labels = [[0.6], [0.7], [0.8], [0.01], [0.6], [0.2]]
# labels = [[0.6], [0.7], [0.8], [0.01], [0.6]]
# labels = [[0.6]]
# labels = [[0.6], [0.7]]
# docs = [['apple', 'is', 'a', 'company']]
# labels = [[0.6]]
# docs = [['google', 'is', 'company']]
# labels = [[0.7]]
# docs = [['apple','google','apple','google','apple','google','apple','google']]
# labels = [[0.01]]
# trainRnn(docs, labels, 4, 'data/toy_embeddings.txt',
#          initWeightFile='tmp_outputs/rnn_init_weights.txt', trainedWeightFile='tmp_outputs/rnn_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='normal', miniBatchSize=6)
# trainRnn(docs, labels, 4, 'data/toy_embeddings.txt',
#          initWeightFile='tmp_outputs/reverse_rnn_init_weights.txt', trainedWeightFile='tmp_outputs/reverse_rnn_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='reverse')
# trainRnn(docs, labels, 4, 'data/toy_embeddings.txt',
#          initWeightFile='tmp_outputs/stacked_rnn_init_weights.txt', trainedWeightFile='tmp_outputs/stacked_rnn_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='normal', stackedDimList=[6, 5, 7])
# trainRnn(docs, labels, 4, 'data/toy_embeddings.txt',
#          initWeightFile='tmp_outputs/bi_rnn_init_weights.txt', trainedWeightFile='tmp_outputs/bi_rnn_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='bi', stackedDimList=[6, 5, 7])
# trainRnn(docs, labels, 4, 'data/toy_embeddings.txt',
#          initWeightFile='tmp_outputs/bi0_rnn_init_weights.txt', trainedWeightFile='tmp_outputs/bi0_rnn_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='bi', stackedDimList=[6, 5, 0])

inputs = [[-1,2,3,4,5,6], [6,5,4,3,2,1], [5,9,3,7,1,2], [1,2,3,4,2,1], [-2,3,4,7,5,6], [6,5,1,3,2,1]]
targets = [[-1,1,1,1,1,1], [1,-1,-1,-1,-1,-1], [1,1,-1,1,-1,1], [1,1,1,1,-1,-1], [-1,1,1,1,-1,1], [-1,-1,-1,1,-1,-1]]
# trainRnn(inputs, targets, 6, None,
#          initWeightFile='tmp_outputs/slbi_rnn_init_weights.txt', trainedWeightFile='tmp_outputs/slbi_rnn_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='bi', task='perstep', stackedDimList=[6, 5, 7])
# trainRnn(inputs, targets, 6, None,
#          initWeightFile='tmp_outputs/slbi0_rnn_init_weights.txt', trainedWeightFile='tmp_outputs/slbi0_rnn_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='bi', task='perstep', stackedDimList=[6, 5, 0])

# trainRnn(docs, labels, 4, 'data/toy_embeddings.txt',
#          initWeightFile='tmp_outputs/gru_init_weights.txt', trainedWeightFile='tmp_outputs/gru_trained_weights.txt',
#          lr=0.3, epochs=1, rnnType='normal', cell='gru')
# trainRnn(docs, labels, 4, 'data/toy_embeddings.txt',
#          initWeightFile='tmp_outputs/reverse_gru_init_weights.txt', trainedWeightFile='tmp_outputs/reverse_gru_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='reverse', cell='gru')
# trainRnn(docs, labels, 4, 'data/toy_embeddings.txt',
#          initWeightFile='tmp_outputs/stacked_gru_init_weights.txt', trainedWeightFile='tmp_outputs/stacked_gru_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='normal', stackedDimList=[6, 5, 7], cell='gru')
# trainRnn(docs, labels, 4, 'data/toy_embeddings.txt',
#          initWeightFile='tmp_outputs/bi_gru_init_weights.txt', trainedWeightFile='tmp_outputs/bi_gru_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='bi', stackedDimList=[6, 5, 7], cell='gru')
# trainRnn(docs, labels, 4, 'data/toy_embeddings.txt',
#          initWeightFile='tmp_outputs/bi0_gru_init_weights.txt', trainedWeightFile='tmp_outputs/bi0_gru_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='bi', stackedDimList=[6, 5, 0], cell='gru')
# trainRnn(inputs, targets, 6, None,
#          initWeightFile='tmp_outputs/sl_gru_init_weights.txt', trainedWeightFile='tmp_outputs/sl_gru_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='normal', task='perstep', cell='gru')
# trainRnn(inputs, targets, 6, None,
#          initWeightFile='tmp_outputs/slbi_gru_init_weights.txt', trainedWeightFile='tmp_outputs/slbi_gru_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='bi', task='perstep', stackedDimList=[6, 5, 7], cell='gru')
# trainRnn(inputs, targets, 6, None,
#          initWeightFile='tmp_outputs/slbi0_gru_init_weights.txt', trainedWeightFile='tmp_outputs/slbi0_gru_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='bi', task='perstep', stackedDimList=[6, 5, 0], cell='gru')

# trainRnn(docs, labels, 4, 'data/toy_embeddings.txt',
#          initWeightFile='tmp_outputs/lstm_init_weights.txt', trainedWeightFile='tmp_outputs/lstm_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='normal', cell='lstm')
# trainRnn(docs, labels, 4, 'data/toy_embeddings.txt',
#          initWeightFile='tmp_outputs/reverse_lstm_init_weights.txt', trainedWeightFile='tmp_outputs/reverse_lstm_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='reverse', cell='lstm')
# trainRnn(docs, labels, 4, 'data/toy_embeddings.txt',
#          initWeightFile='tmp_outputs/stacked_lstm_init_weights.txt', trainedWeightFile='tmp_outputs/stacked_lstm_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='normal', stackedDimList=[6, 5, 7], cell='lstm')
# trainRnn(docs, labels, 4, 'data/toy_embeddings.txt',
#          initWeightFile='tmp_outputs/bi_lstm_init_weights.txt', trainedWeightFile='tmp_outputs/bi_lstm_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='bi', stackedDimList=[6, 5, 7], cell='lstm')
# trainRnn(docs, labels, 4, 'data/toy_embeddings.txt',
#          initWeightFile='tmp_outputs/bi0_lstm_init_weights.txt', trainedWeightFile='tmp_outputs/bi0_lstm_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='bi', stackedDimList=[6, 5, 0], cell='lstm')
# trainRnn(inputs, targets, 6, None,
#          initWeightFile='tmp_outputs/sl_lstm_init_weights.txt', trainedWeightFile='tmp_outputs/sl_lstm_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='normal', task='perstep', cell='lstm')
# trainRnn(inputs, targets, 6, None,
#          initWeightFile='tmp_outputs/slbi_lstm_init_weights.txt', trainedWeightFile='tmp_outputs/slbi_lstm_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='bi', task='perstep', stackedDimList=[6, 5, 7], cell='lstm')
# trainRnn(inputs, targets, 6, None,
#          initWeightFile='tmp_outputs/slbi0_lstm_init_weights.txt', trainedWeightFile='tmp_outputs/slbi0_lstm_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='bi', task='perstep', stackedDimList=[6, 5, 0], cell='lstm')

# docs, labels = getTextDataFromFile('data/rand_docs.txt')
# trainRnn(docs, labels, 7, 'data/toy_embeddings.txt',
#          initWeightFile='tmp_outputs/large_rnn_init_weights.txt', trainedWeightFile='tmp_outputs/large_rnn_trained_weights.txt',
#          lr=0.3, epochs=1, rnnType='bi', stackedDimList=[16, 10, 7], miniBatchSize=99)
# trainRnn(docs, labels, 7, 'data/toy_embeddings.txt',
#          initWeightFile='tmp_outputs/large_gru_init_weights.txt', trainedWeightFile='tmp_outputs/large_gru_trained_weights.txt',
#          lr=0.3, epochs=1, rnnType='bi', stackedDimList=[16, 10, 7], cell='gru', miniBatchSize=99)
# trainRnn(docs, labels, 7, 'data/toy_embeddings.txt',
#          initWeightFile='tmp_outputs/large_lstm_init_weights.txt', trainedWeightFile='tmp_outputs/large_lstm_trained_weights.txt',
#          lr=0.3, epochs=1, rnnType='bi', stackedDimList=[16, 10, 7], cell='lstm', miniBatchSize=99)

# inputs, targets = getNumDataFromFile('data/rand_num.txt', 53, 53)
# for cellType in ['rnn', 'gru', 'lstm']:
#     trainRnn(inputs, targets, 23, None,
#              initWeightFile='tmp_outputs/sllarge_%s_init_weights.txt'%cellType, trainedWeightFile='tmp_outputs/sllarge_%s_trained_weights.txt'%cellType,
#              lr=0.3, epochs=10, rnnType='bi', task='perstep', stackedDimList=[6, 5, 7], cell=cellType, miniBatchSize=11)

# inputs, targets = getNumDataFromFile('data/rand_num_t7_l23.txt', 23*5, 23)
# print(len(inputs))
# print(len(inputs[0]))
# for cellType in ['rnn', 'gru', 'lstm']:
#     trainRnn(inputs, targets, 23, None,
#              initWeightFile='tmp_outputs/sllarge_t5_%s_init_weights.txt'%cellType, trainedWeightFile='tmp_outputs/sllarge_t5_%s_trained_weights.txt'%cellType,
#              lr=0.3, epochs=5, rnnType='bi', task='perstep', stackedDimList=[6, 5, 7], cell=cellType, miniBatchSize=11, tokenSize=5)

# inputs, targets = getNumDataFromFile('data/rand_num_t7_l23.txt', 23*5, 1)
# print(len(inputs))
# print(len(inputs[0]))
# for cellType in ['rnn', 'gru', 'lstm']:
#     trainRnn(inputs, targets, 23, None,
#              initWeightFile='tmp_outputs/large_t5_%s_init_weights.txt'%cellType, trainedWeightFile='tmp_outputs/large_t5_%s_trained_weights.txt'%cellType,
#              lr=0.3, epochs=5, rnnType='bi', stackedDimList=[6, 5, 7], cell=cellType, miniBatchSize=11, tokenSize=5)

# inputs, targets = getNumDataFromFile('data/rand_num_t7_l23_binary.txt', 23*5, 1)
# print(len(inputs))
# print(len(inputs[0]))
# targetMap = {0:1, 1:0}
# mapTargets(targets, targetMap)
# for cellType in ['rnn', 'gru', 'lstm']:
#     trainRnn(inputs, targets, 23, None,
#              initWeightFile='tmp_outputs/binary_t5_%s_init_weights.txt'%cellType, trainedWeightFile='tmp_outputs/binary_t5_%s_trained_weights.txt'%cellType,
#              lr=0.3, epochs=10, rnnType='bi', stackedDimList=[6, 5, 7], cell=cellType, miniBatchSize=11, tokenSize=5, nclass=2)
# 
# inputs, targets = getNumDataFromFile('data/rand_num_t7_l23_binary.txt', 23*5, 23)
# print(len(inputs))
# print(len(inputs[0]))
# targetMap = {0:1, 1:0}
# mapTargets(targets, targetMap)
# for cellType in ['rnn', 'gru', 'lstm']:
#     trainRnn(inputs, targets, 23, None,
#              initWeightFile='tmp_outputs/slbinary_t5_%s_init_weights.txt'%cellType, trainedWeightFile='tmp_outputs/slbinary_t5_%s_trained_weights.txt'%cellType,
#              lr=0.3, epochs=5, rnnType='bi', task='perstep', stackedDimList=[6, 5, 7], cell=cellType, miniBatchSize=11, tokenSize=5, nclass=2)

docs, labels = getTextDataFromFile('data/rand_docs.txt')
for cellType in ['rnn', 'gru', 'lstm']:
    trainRnn(docs, labels, 7, 'data/toy_embeddings.txt',
             initWeightFile='tmp_outputs/stackedbi_%s_init_weights.txt'%cellType, 
             trainedWeightFile='tmp_outputs/stackedbi_%s_trained_weights.txt'%cellType,
             lr=0.3, epochs=1, rnnType='stackedbi', stackedDimList=[16, 10, 7], cell=cellType, miniBatchSize=21)
