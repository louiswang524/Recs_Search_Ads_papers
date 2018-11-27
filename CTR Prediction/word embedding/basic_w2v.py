# phase 1: assemble the Graph
# 1. define placeholders for input and output
center_words = tf.placeholder(tf.int32,shape=[BATCH_SIZE])
target_words = tf.placeholder(tf.int32,shape=[BATCH_SIZE])

# 2. define the weights
embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE,EMBED_SIZE],-1.0,1.0))

# 3. inference (compute the forward path of the graph)
embed = tf.nn.embedding_lookup(embed_matrix,center_words)

# 4. define the loss function
# for nce loss, we need weights and biases for the hidden layer to calculate nce loss
nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE,EMBED_SIZE],stddev=1.0/EMBED_SIZE**0.5))
nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]))

loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                    biases = nce_bias,
                                    labels = target_words,
                                    inputs = embed,
                                    num_sampled = NUM_SAMPLED,
                                    num_classes=VOCAB_SIZE))
# define optimizer
optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

# phase 2: excute the computation
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    average_loss = 0.0
    for index in xrange(NUM_TRAIN_STEPS):
        batch = batch_gen.next()
        loss_batch, _ = sess.run([loss,optimizer],feed_dict={center_words:batch[0],target_words:batch[1]})
        average_loss += loss_batch
        if (index+1)%2000==0:
            print('Average loss at step {}:{:5.1f}'.format(index+1,average_loss/(index+1)))
