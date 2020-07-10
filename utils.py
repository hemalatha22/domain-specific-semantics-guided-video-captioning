def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text
 
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    for line in doc.split('\n'):
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)
 
def load_descriptions(filename, dataset):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):

        tokens = line.split()
        if len(tokens)>1:

            vid_id, image_desc = tokens[0], tokens[1:]
            if vid_id in dataset:
                if vid_id not in descriptions:
                    descriptions[vid_id] = list()

                desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
                descriptions[vid_id].append(desc)
    return descriptions
 
def load_video_features(filename, dataset):
    all_features = load(open(filename, 'rb'))
    features = {k: all_features[k] for k in dataset}
    return features
 
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc
 
def create_sequences(tokenizer, max_length, desc_list, c2d, c3d, semantic):
    X1, X2, X3, X4, y = list(), list(), list(), list(), list()
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(c2d)
            X2.append(c3d)
            X3.append(semantic)
            X4.append(in_seq)
            y.append(out_seq)
    return array(X1), array(X2), array(X3), array(X4), array(y)

def load_embedding(tokenizer, vocab_size, max_length):
    embedding = load(open('/home/mh/mywork/dataset/MSVD/descriptions/word2vec_embedding.pkl', 'rb'))
    dimensions = 100
    trainable = False
    weights = np.zeros((vocab_size, dimensions))
    for word, i in tokenizer.word_index.items():
        if word not in embedding:
            continue
        weights[i] = embedding[word]
    layer = Embedding(vocab_size, dimensions, weights=[weights], input_length=max_length, trainable=trainable, mask_zero=True)
    return layer
 
def data_generator(descriptions, train_c2d, train_c3d, train_sem, tokenizer, max_length):
    while 1:
        for key, desc_list in descriptions.items():
            c2d= train_c2d[key]
            c3d= train_c3d[key]
            sem=train_sem[key]
            in_c2d, in_c3d, in_sem, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, c2d, c3d, sem)
            yield [[in_c2d, in_c3d, in_sem, in_seq], out_word]
