from tkdl_util import *

def getRnnRegressionOps(batchSize=5, maxNumSteps=10, nNeurons=4, initEmbeddings=None, bias_trainable=False, learningRate=0.1):
    tf.reset_default_graph()
    tf.set_random_seed(32513)
    inputTokens = tf.placeholder(tf.int32, [batchSize, maxNumSteps])
    inputLens = tf.placeholder(tf.int32, [batchSize])
    targets = tf.placeholder(tf.float64, [batchSize])
    rnnCell = tf.nn.rnn_cell.BasicRNNCell(nNeurons)
    # initState = rnnCell.zero_state(batchSize, tf.float64)
    # initState = tf.get_variable("initState", [batchSize, nNeurons], dtype=tf.float64, trainable=True)
    initState = tf.get_variable("initState", [nNeurons], dtype=tf.float64, trainable=True)
    initStates = tf.concat(0, [initState for i in xrange(batchSize)])
    initStates = tf.reshape(initStates, [-1, nNeurons])

    embedding = tf.Variable(initEmbeddings, name='inputEmbeddings', trainable=False, dtype=tf.float64)
    inputData = tf.nn.embedding_lookup(embedding, inputTokens)

    raw_outputs, last_states = tf.nn.dynamic_rnn(
        cell=rnnCell,
        initial_state=initStates,
        dtype=tf.float64,
        sequence_length=inputLens,
        inputs=inputData)
        
    flattened_outputs = tf.reshape(raw_outputs, [-1, nNeurons])
    index = tf.range(0, batchSize) * maxNumSteps + inputLens - 1
    outputs = tf.gather(flattened_outputs, index)
    outputW = tf.get_variable("outputW", [nNeurons, 1], dtype=tf.float64)
    outputB = tf.get_variable("outputB", [1], dtype=tf.float64)
    prediction = tf.add(tf.matmul(outputs, outputW), outputB)
    loss = tf.reduce_sum(tf.pow(prediction-targets, 2)/batchSize)

    lr = tf.Variable(learningRate, trainable=False)
    tvars = tf.trainable_variables()
    if not bias_trainable:
        tvars = np.take(tvars, [0,1,3,4]).tolist() # exclude RNN bias from training
    optimizer = tf.train.GradientDescentOptimizer(lr)
    learningStep = optimizer.minimize(loss, var_list=tvars)
    initAll = tf.initialize_all_variables()
    return inputTokens, inputLens, targets, prediction, loss, initAll, learningStep

def trainRnn(docs, labels, nNeurons, embeddingFile, initWeightFile=None, trainedWeightFile=None):
    assert len(docs) == len(labels)
    batchSize = len(docs)
    maxNumSteps = 0
    lens = []
    for doc in docs:
        lens.append(len(doc))
        if len(doc) > maxNumSteps:
            maxNumSteps = len(doc)
    token2Id, embeddingArray = readEmbeddingFile(embeddingFile)
    inputIds = []
    for doc in docs:
        ids = tokens2ids(doc, token2Id, maxNumSteps=maxNumSteps)
        inputIds.append(ids)
    inputIds = np.asarray(inputIds, dtype=np.int32)
    lens = np.asarray(lens, dtype=np.int32)
    embeddingArray = np.asarray(embeddingArray, dtype=np.float64)
    inputTokens, inputLens, targets, prediction, loss, initAll, learningStep = getRnnRegressionOps(batchSize=batchSize, maxNumSteps=maxNumSteps, 
                                                                                                   nNeurons=nNeurons, initEmbeddings=embeddingArray)
    feed_dict = {inputTokens:inputIds, inputLens:lens, targets:labels}
    with tf.Session() as sess:
        sess.run(initAll)
        l = sess.run(loss, feed_dict=feed_dict)
        print 'loss: %f' % l
        for v in tf.trainable_variables():
            print v.name
            print sess.run(v)
        sess.run(learningStep, feed_dict=feed_dict)
        l = sess.run(loss, feed_dict=feed_dict)
        print 'loss: %f' % l
        for v in tf.trainable_variables():
            print v.name
            print sess.run(v)

docs = [['apple', 'is', 'a', 'company'], ['google', 'is', 'another', 'big', 'company']]
labels = [0.6, 0.7]
trainRnn(docs, labels, 3, 'data/toy_embeddings.txt')
