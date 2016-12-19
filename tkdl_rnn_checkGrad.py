from tkdl_util import *

def getRnnRegressionOps(batchSize=5, maxNumSteps=10, nNeurons=4, initEmbeddings=None, 
						bias_trainable=True, learningRate=0.1):
    tf.reset_default_graph()
    tf.set_random_seed(32513)
    inputTokens = tf.placeholder(tf.int32, [batchSize, maxNumSteps])
    inputLens = tf.placeholder(tf.int32, [batchSize])
    targets = tf.placeholder(tf.float64, [batchSize, 1])
    rnnCell = tf.nn.rnn_cell.BasicRNNCell(nNeurons, activation=tf.tanh)
    # keep this code for future reference: training initial states
    # initState = tf.get_variable("initState", [nNeurons], dtype=tf.float64, trainable=True)
    # initStates = tf.concat(0, [initState for i in range(batchSize)])
    # initStates = tf.reshape(initStates, [-1, nNeurons])

    embedding = tf.Variable(initEmbeddings, name='inputEmbeddings', trainable=False, dtype=tf.float64)
    inputData = tf.nn.embedding_lookup(embedding, inputTokens)

    raw_outputs, last_states = tf.nn.dynamic_rnn(
        cell=rnnCell,
        dtype=tf.float64,
        sequence_length=inputLens,
        inputs=inputData)

    flattened_outputs = tf.reshape(raw_outputs, [-1, nNeurons])
    index = tf.range(0, batchSize) * maxNumSteps + inputLens - 1
    outputs = tf.gather(flattened_outputs, index)
    outputW = tf.get_variable("outputW", [nNeurons, 1], dtype=tf.float64)
    outputB = tf.get_variable("outputB", [1], dtype=tf.float64)
    prediction = tf.add(tf.matmul(outputs, outputW), outputB)
    loss = tf.reduce_sum(tf.pow(prediction-targets, 2)/2)

    lr = tf.Variable(learningRate, trainable=False)
    tvars = tf.trainable_variables()
    # tvars[0] is RNN weight matrix
    # tvars[1] is RNN bias (may be excluded from training)
    # tvars[2] is output weight matrix
    # tvars[3] is output bias
    if not bias_trainable:
        tvars = np.take(tvars, [0,2,3]).tolist() # exclude RNN bias from training
    optimizer = tf.train.GradientDescentOptimizer(lr)
    gradients = optimizer.compute_gradients(loss, var_list=tvars) # for debugging purpose
    learningStep = optimizer.minimize(loss, var_list=tvars)
    initAll = tf.global_variables_initializer()
    return inputTokens, inputLens, targets, prediction, loss, initAll, learningStep, gradients, tf.pow(prediction-targets, 2)/2

def trainRnn(docs, labels, nNeurons, embeddingFile, initWeightFile=None, trainedWeightFile=None):
    assert len(docs) == len(labels)
    batchSize = len(docs)
    maxNumSteps = 0
    lens = []
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
    inputTokens, inputLens, targets, prediction, loss, initAll, learningStep, gradients, debugInfo = getRnnRegressionOps(batchSize=batchSize, maxNumSteps=maxNumSteps,
                                                                                                   nNeurons=nNeurons, initEmbeddings=embeddingArray)
    feed_dict = {inputTokens:inputIds, inputLens:lens, targets:labels}
    with tf.Session() as sess:
        sess.run(initAll)
        rnnMatrix = []
        rnnInitState = []
        outMatrix = []
        outBias = []
        if initWeightFile is not None:
            ws = sess.run(tf.trainable_variables())
            writeWeights(np.take(ws, [0,1,2,3]), [1,1,2,2], initWeightFile)
        l = sess.run(loss, feed_dict=feed_dict)
        print('loss before training: %.12f' % l)
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
        # print(sess.run(gradients, feed_dict=feed_dict))
        sess.run(learningStep, feed_dict=feed_dict)
        if trainedWeightFile is not None:
            ws = sess.run(tf.trainable_variables())
            writeWeights(np.take(ws, [0,1,2,3]), [1,1,2,2], trainedWeightFile)
        l = sess.run(loss, feed_dict=feed_dict)
        print('loss after training: %.12f' % l)
        for v in tf.trainable_variables():
            print(v.name)
            print(sess.run(v))

docs = [['apple', 'is', 'a', 'company'], ['google', 'is', 'another', 'big', 'company']]
labels = [[0.6], [0.7]]
# docs = [['apple', 'is', 'a', 'company']]
# labels = [[0.6]]
# docs = [['google', 'is', 'company']]
# labels = [[0.7]]
trainRnn(docs, labels, 4, 'data/toy_embeddings.txt', initWeightFile='tmp_outputs/rnn_init_weights.txt', trainedWeightFile='tmp_outputs/rnn_trained_weights.txt')
