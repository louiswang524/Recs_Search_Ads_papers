def maybe_downloand(filename,url,expected_bytes):
    # download a file if not present, and make sure it's right size.
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url+filename,filename)
    statinfo = os.stat(filename)
    if statinfo.test_size == expected_bytes:
        print('Found and verified',filename)
    else:
        print(statinfo.st_size)
        raise Expection(
            'Failed to verify'+filename+'.Can you get to it with a browser?')
    return filename

url = 'http://mattmahoney.net/dc/'
filename = maybe_downloand('text8.zip',url,31344016)

# read the data into a list of strings
def read_data(filename):
    # extract the first file enclosed in a zip file as a list of words
    with zipfile.Zipfile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

vocabulary = read_data(filename)
print(vocabulary[:7])

def build_dataset(words, n_words):
    # print raw inputs into a dataset
    count = [['UNK',-1]]
    count.extend(collections.Counter(words).most_common(n_words-1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

data_index = 0
# generate batch data
def generate_batch(data,batch_size,,num_skips,skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2*skip_window
    batch = np.ndarray(shape=(batch_size),dtype = np.int32)
    context = np.ndarray(shape=(batch_size,1),dtype = np.int32)
    span = 2*skip_window+1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index+1)%len(data)
    for i in range(batch_size//num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0,span-1)
            targets_to_avoid.append(target)
            batch[i*num_skips+j] = buffer[skip_window]
            context[i*num_skips+j,0]=buffer(target)
        buffer.append(data[data_index])
        data_index = (data_index + 1)%len(data)
    data_index = (data_index+len(data)-span)%len(data)
    return batch,context
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a context.


train_inputs = tf.placeholder(tf.int32,shape=[batch_size])
train_labels = tf.placeholder(tf.int32,shape=[batch_size,1])
valid_dataset = tf.constant(valid_examples,dtype=tf.int32)

embeddings = tf.Variable(
tf.random_uniform([vocabulary_size,embedding_size],-1.0,1.0))
embed = tf.nn.embedding_lookup(embeddings,train_inputs)

# Construct the variables for the softmax
weights = tf.Variable(tf.truncated_normal([vocabulary_size,embedding_size],stddev=1.0/math.sqrt(embedding_size)))
bias = tf.Variable(tf.zeros([vocabulary_size]))
hidden_out = tf.matmul(embed,tf.transpose(weights))+bias

#convert train_context to a one-hot format
train_one_hot = tf.one_hot(train_context,vocabulary_size)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hidden_out,labels=train_one_hot))

#construct the SGD optimizer using a learning rate of 0.1
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

#compute the cosine similarity between minibatch examples and all embeddings
norm = tf.sqrt(tf.reduce_mean(tf.square(embedding),1,keep_dim=True))
normalized_embeddings = embeddings/norm

valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,valid_dataset)

similarity = tf.matmul(valid_embeddings,normalized_embeddings,transpose_b=True)

with tf.Session(graph=graph) as session:
    init.run()
    print('Initialized')

    avaerafe_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_context = generate_batch(data,batch_size,num_skips,skip_window)
        feed_dict = {train_inputs:batch_inputs,train_context:batch_context}

    _, loss_val = session.run([optimizer,cross_entropy],feed_dict=feed_dict)
    avaerafe_loss += loss_val

    if step % 2000 == 0:
        if step > 0:
            avaerafe_loss /= 2000
        print('Average loss at step', step, ':',avaerafe_loss)
        avaerafe_loss = 0

    if step : 10000 ==0:
        sim = similarity.eval()
        for i in range(valid_size):
            valid_word = reversed_dictionary[valid_examples[i]]
            top_k = 8
            nearest  = (-sim[i,:]).argsort()[1:top_k+1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = reversed_dictionary[nearest[k]]
                log_str='%s %s'%(log_str,close_word)
            print(log_str)
    final_embedding = normalized_embeddings.eval()

      
