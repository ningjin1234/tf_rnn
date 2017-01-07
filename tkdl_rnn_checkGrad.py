from tkdl_util import *
from tensorflow.python.ops import array_ops

def getRnnRegressionOps(batchSize=5, maxNumSteps=10, nNeurons=4, initEmbeddings=None,
                        bias_trainable=True, learningRate=0.1, rnnType='normal', stackedDimList=[],
                        task='classification', cell='rnn'):
    tf.reset_default_graph()
    tf.set_random_seed(32513)
    inputTokens = tf.placeholder(tf.int32, [batchSize, maxNumSteps])
    inputLens = tf.placeholder(tf.int32, [batchSize])
    if task.lower() in ['class', 'classification']:
        targets = tf.placeholder(tf.float64, [batchSize, 1])
    elif task.lower() in ['numericlabeling', 'numl']:
        targets = tf.placeholder(tf.float64, [batchSize, maxNumSteps])
    else:
        assert(False)
    if stackedDimList is None or len(stackedDimList) == 0:
        if cell == 'rnn':
            rnnCell = tf.nn.rnn_cell.BasicRNNCell(nNeurons, activation=tf.tanh)
        elif cell == 'gru':
            rnnCell = tf.nn.rnn_cell.GRUCell(nNeurons, activation=tf.tanh)
        elif cell == 'lstm':
            rnnCell = tf.nn.rnn_cell.LSTMCell(nNeurons, activation=tf.tanh, use_peepholes=True)
    else:
        if cell == 'rnn':
            rnnCellList = [tf.nn.rnn_cell.BasicRNNCell(dim, activation=tf.tanh) for dim in stackedDimList]
        elif cell == 'gru':
            rnnCellList = [tf.nn.rnn_cell.GRUCell(dim, activation=tf.tanh) for dim in stackedDimList]
        elif cell == 'lstm':
            rnnCellList = [tf.nn.rnn_cell.LSTMCell(dim, activation=tf.tanh, use_peepholes=True) for dim in stackedDimList]
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
        inputTokens = tf.placeholder(tf.float64, [batchSize, maxNumSteps, 1])
        inputData = inputTokens

    if rnnType.lower() == 'normal':
        raw_outputs, last_states = tf.nn.dynamic_rnn(cell=rnnCell, dtype=tf.float64, sequence_length=inputLens, inputs=inputData)
    elif rnnType.lower() == 'reversed' or rnnType.lower() == 'reverse':
        inputDataReversed = array_ops.reverse_sequence(input=inputData, seq_lengths=inputLens, seq_dim=1, batch_dim=0)
        raw_outputs_r, last_states = tf.nn.dynamic_rnn(cell=rnnCell, dtype=tf.float64, sequence_length=inputLens, inputs=inputDataReversed)
        raw_outputs = array_ops.reverse_sequence(input=raw_outputs_r, seq_lengths=inputLens, seq_dim=1, batch_dim=0)
    elif rnnType.lower() == 'bi' or rnnType.lower() == 'bidirectional':
        stackedDimList = stackedDimList[:-1]
        if cell == 'rnn':
            fwRnnCellList = [tf.nn.rnn_cell.BasicRNNCell(dim, activation=tf.tanh) for dim in stackedDimList]
            bwRnnCellList = [tf.nn.rnn_cell.BasicRNNCell(dim, activation=tf.tanh) for dim in stackedDimList]
        elif cell == 'gru':
            fwRnnCellList = [tf.nn.rnn_cell.GRUCell(dim, activation=tf.tanh) for dim in stackedDimList]
            bwRnnCellList = [tf.nn.rnn_cell.GRUCell(dim, activation=tf.tanh) for dim in stackedDimList]
        elif cell == 'lstm':
            fwRnnCellList = [tf.nn.rnn_cell.LSTMCell(dim, activation=tf.tanh, use_peepholes=True) for dim in stackedDimList]
            bwRnnCellList = [tf.nn.rnn_cell.LSTMCell(dim, activation=tf.tanh, use_peepholes=True) for dim in stackedDimList]
        fwRnnCell = tf.nn.rnn_cell.MultiRNNCell(fwRnnCellList)
        bwRnnCell = tf.nn.rnn_cell.MultiRNNCell(bwRnnCellList)
        # NOTE: in bidirectional_dynamic_rnn, tensorflow does not concatenate outputs for each layer, it only concatenates the outputs for the last layer
        tmp_outputs, tmp_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwRnnCell, cell_bw=bwRnnCell,
                                                                   dtype=tf.float64, sequence_length=inputLens, inputs=inputData)
        # NOTE: currently the last layer in a stacked bidirectional RNN model must be a unidirectional recurrent layer;
        # this is a current limitation of tkdlu
        fwOutputs = tmp_outputs[0]
        bwOutputs = tmp_outputs[1]
        tmp_outputs = tf.concat(2, [fwOutputs, bwOutputs])
        # print(tmp_outputs.get_shape())
        if cell == 'rnn':
            rnnCell = tf.nn.rnn_cell.BasicRNNCell(nNeurons, activation=tf.tanh)
        elif cell == 'gru':
            rnnCell = tf.nn.rnn_cell.GRUCell(nNeurons, activation=tf.tanh)
        elif cell == 'lstm':
            rnnCell = tf.nn.rnn_cell.LSTMCell(nNeurons, activation=tf.tanh, use_peepholes=True)
        raw_outputs, last_states = tf.nn.dynamic_rnn(cell=rnnCell, dtype=tf.float64, sequence_length=inputLens, inputs=tmp_outputs)

    flattened_outputs = tf.reshape(raw_outputs, [-1, nNeurons])
    if task.lower() in ['class', 'classification']:
        if rnnType.lower() in ['reversed', 'reverse']:
            index = tf.range(0, batchSize) * maxNumSteps
        else:
            index = tf.range(0, batchSize) * maxNumSteps + inputLens - 1
        outputs = tf.gather(flattened_outputs, index)
    else:
        outputs = flattened_outputs
        targets = tf.reshape(targets, [-1, 1])
    outputW = tf.get_variable("outputW", [nNeurons, 1], dtype=tf.float64)
    outputB = tf.get_variable("outputB", [1], dtype=tf.float64)
    prediction = tf.add(tf.matmul(outputs, outputW), outputB)
    if task.lower() in ['class', 'classification']:
        loss = tf.reduce_sum(tf.pow(prediction-targets, 2)/2)
    elif task.lower() in ['numericlabeling', 'numl']:
        loss = tf.reduce_sum(tf.pow(prediction-targets, 2)/2/maxNumSteps)
    else:
        assert(False)

    lr = tf.Variable(learningRate, trainable=False)
    tvars = tf.trainable_variables()
    # tvars[0] is RNN weight matrix
    # tvars[1] is RNN bias (may be excluded from training)
    # tvars[2] is output weight matrix
    # tvars[3] is output bias
    # if not bias_trainable:
    #     tvars = np.take(tvars, [0,2,3]).tolist() # exclude RNN bias from training
    optimizer = tf.train.GradientDescentOptimizer(lr)
    gradients = optimizer.compute_gradients(loss, var_list=tvars) # for debugging purpose
    learningStep = optimizer.minimize(loss, var_list=tvars)
    initAll = tf.global_variables_initializer()
    return inputTokens, inputLens, targets, prediction, loss, initAll, learningStep, gradients, flattened_outputs

def getLayerIds(stackedDimList, rnnType='normal', cell='rnn'):
    layerIds = []
    if stackedDimList is None or len(stackedDimList) == 0:
        nLayers = 2
    else:
        nLayers = len(stackedDimList) + 1
    for i in range(nLayers):
        layerIds.append(i+1)
        layerIds.append(i+1)
    if rnnType.lower() in ['bi', 'bidirectional']:
        layerIds = []
        for i in range(len(stackedDimList)-1):
            layerIds.append(i+len(stackedDimList))
            layerIds.append(i+len(stackedDimList))
        for i in range(len(stackedDimList)-1):
            layerIds.append(i+1)
            layerIds.append(i+1)
        layerIds.append(2*len(stackedDimList)-1)
        layerIds.append(2*len(stackedDimList)-1)
        layerIds.append(2*len(stackedDimList))
        layerIds.append(2*len(stackedDimList))
    return layerIds

def trainRnn(docs, labels, nNeurons, embeddingFile, initWeightFile=None, trainedWeightFile=None, lr=0.1, epochs=1,
             rnnType='normal', stackedDimList=[], task='classification', cell='rnn'):
    assert len(docs) == len(labels)
    batchSize = len(docs)
    maxNumSteps = 0
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
        lens = [len(doc) for doc in docs]
        lens = np.asarray(lens, dtype=np.int32)
        maxNumSteps = max(lens)
        embeddingArray = None
        inputIds = np.asarray(docs, dtype=np.float64)
        inputIds = np.reshape(inputIds, (batchSize, maxNumSteps, 1))
        labels = np.asarray(labels, dtype=np.float64)
        labels = np.reshape(labels, (-1, 1))
    inputTokens, inputLens, targets, prediction, loss, initAll, learningStep, gradients, debugInfo = getRnnRegressionOps(batchSize=batchSize,
                                                                                                   maxNumSteps=maxNumSteps,
                                                                                                   nNeurons=nNeurons, initEmbeddings=embeddingArray,
                                                                                                   learningRate=lr/batchSize, rnnType=rnnType,
                                                                                                   stackedDimList=stackedDimList, task=task,
                                                                                                   cell=cell)
    feed_dict = {inputTokens:inputIds, inputLens:lens, targets:labels}
    print('learning rate: %f' % lr)
    print('rnn type: %s' % rnnType)
    print('cell type: %s' % cell)
    print('task type: %s' % task)
    with tf.Session() as sess:
        sess.run(initAll)
        rnnMatrix = []
        rnnInitState = []
        outMatrix = []
        outBias = []
        layerIds = getLayerIds(stackedDimList, rnnType=rnnType, cell=cell)
        for v in tf.trainable_variables():
            print(v.name)
        # for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        #     print(v.name)
        if initWeightFile is not None:
            ws = sess.run(tf.trainable_variables())
            # writeWeights(np.take(ws, [0,1,2,3]), [1,1,2,2], initWeightFile)
            # writeWeights(ws, layerIds, initWeightFile)
            writeWeightsWithNames(ws, tf.trainable_variables(), stackedDimList, initWeightFile)
        # for v,g in zip(tf.trainable_variables(), gradients):
        #     if not 'Bias' in v.name:
        #         continue
        #     print(v.name)
        #     print(sess.run(g, feed_dict=feed_dict))
        l = sess.run(loss, feed_dict=feed_dict)
        print('loss before training: %.14g' % (l/batchSize))
        # for v in tf.trainable_variables():
        #     val = sess.run(v)
        #     print v.name
        #     print val
        #     if 'RNN' in v.name and 'Matrix' in v.name:
        #         rnnMatrix = val
        #     elif 'initState' in v.name:
        #         rnnInitState = val
        # tmp = np.append(np.asarray([2,1,0]), rnnInitState)
        # print tmp
        # print np.dot(tmp, rnnMatrix)
        # print np.tanh(np.dot(tmp, rnnMatrix))
        # r = sess.run(raw, feed_dict=feed_dict)
        # print 'raw outputs'
        # print r
        # print('gradients before training:')
        # print(sess.run(gradients, feed_dict=feed_dict))
        # print('flattened_outputs before training:')
        # print(sess.run(debugInfo, feed_dict=feed_dict))
        for i in range(epochs):
            sess.run(learningStep, feed_dict=feed_dict)
            print('loss after %d epochs: %.14g' % (i+1, sess.run(loss, feed_dict=feed_dict)/batchSize))
        # print('prediction after training:')
        # print(sess.run(prediction, feed_dict=feed_dict))
        if trainedWeightFile is not None:
            ws = sess.run(tf.trainable_variables())
            # writeWeights(np.take(ws, [0,1,2,3]), [1,1,2,2], trainedWeightFile)
            # writeWeights(ws, layerIds, trainedWeightFile)
            writeWeightsWithNames(ws, tf.trainable_variables(), stackedDimList, trainedWeightFile)
        for v in tf.trainable_variables():
            print(v.name)
            print(sess.run(v))
        # for v,g in zip(tf.trainable_variables(), gradients):
        #     if not 'Bias' in v.name:
        #         continue
        #     print(v.name)
        #     print(sess.run(g, feed_dict=feed_dict))

doc1 = "apple is a company".split()
doc2 = "google is another big company".split()
doc3 = ['orange','is','a','fruit']
doc4 = ['apple','google','apple','google','apple','google','apple','google']
doc5 = ['blue', 'is', 'a', 'color']
docs = [doc1, doc2, doc3, doc4, doc5]
# doc1 = ['apple']
# docs = [doc1]
# docs = [doc1, doc2]
# docs = [reversed(doc1), reversed(doc2), reversed(doc3), reversed(doc4), reversed(doc5)]
# docs = [['apple','is'], ['google','is'],['orange','is']]
# docs = [['apple'], ['google'],['orange'],['company'],['fruit']]
# docs = [['apple', 'is', 'a'], ['google', 'is']]
# labels = [[0.6], [0.7], [0.8]]
labels = [[0.6], [0.7], [0.8], [0.01], [0.6]]
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
#          lr=0.3, epochs=10, rnnType='normal')
# trainRnn(docs, labels, 4, 'data/toy_embeddings.txt',
#          initWeightFile='tmp_outputs/reverse_rnn_init_weights.txt', trainedWeightFile='tmp_outputs/reverse_rnn_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='reverse')
# trainRnn(docs, labels, 4, 'data/toy_embeddings.txt',
#          initWeightFile='tmp_outputs/stacked_rnn_init_weights.txt', trainedWeightFile='tmp_outputs/stacked_rnn_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='normal', stackedDimList=[6, 5, 7])
# trainRnn(docs, labels, 4, 'data/toy_embeddings.txt',
#          initWeightFile='tmp_outputs/bi_rnn_init_weights.txt', trainedWeightFile='tmp_outputs/bi_rnn_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='bi', stackedDimList=[6, 5, 7])

inputs = [[-1,2,3,4,5,6], [6,5,4,3,2,1], [5,9,3,7,1,2], [1,2,3,4,2,1]]
targets = [[-1,1,1,1,1,1], [1,-1,-1,-1,-1,-1], [1,1,-1,1,-1,1], [1,1,1,1,-1,-1]]
# trainRnn(inputs, targets, 6, None,
#          initWeightFile='tmp_outputs/slbi_rnn_init_weights.txt', trainedWeightFile='tmp_outputs/slbi_rnn_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='bi', task='numl', stackedDimList=[6, 5, 7])




# trainRnn(docs, labels, 4, 'data/toy_embeddings.txt',
#          initWeightFile='tmp_outputs/gru_init_weights.txt', trainedWeightFile='tmp_outputs/gru_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='normal', cell='gru')
# trainRnn(docs, labels, 4, 'data/toy_embeddings.txt',
#          initWeightFile='tmp_outputs/reverse_gru_init_weights.txt', trainedWeightFile='tmp_outputs/reverse_gru_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='reverse', cell='gru')
# trainRnn(docs, labels, 4, 'data/toy_embeddings.txt',
#          initWeightFile='tmp_outputs/stacked_gru_init_weights.txt', trainedWeightFile='tmp_outputs/stacked_gru_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='normal', stackedDimList=[6, 5, 7], cell='gru')
# trainRnn(docs, labels, 4, 'data/toy_embeddings.txt',
#          initWeightFile='tmp_outputs/bi_gru_init_weights.txt', trainedWeightFile='tmp_outputs/bi_gru_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='bi', stackedDimList=[6, 5, 7], cell='gru')
# trainRnn(inputs, targets, 6, None,
#          initWeightFile='tmp_outputs/sl_gru_init_weights.txt', trainedWeightFile='tmp_outputs/sl_gru_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='normal', task='numl', cell='gru')
# trainRnn(inputs, targets, 6, None,
#          initWeightFile='tmp_outputs/slbi_gru_init_weights.txt', trainedWeightFile='tmp_outputs/slbi_gru_trained_weights.txt',
#          lr=0.3, epochs=10, rnnType='bi', task='numl', stackedDimList=[6, 5, 7], cell='gru')


trainRnn(docs, labels, 4, 'data/toy_embeddings.txt',
         initWeightFile='tmp_outputs/lstm_init_weights.txt', trainedWeightFile='tmp_outputs/lstm_trained_weights.txt',
         lr=0.3, epochs=10, rnnType='normal', cell='lstm')
