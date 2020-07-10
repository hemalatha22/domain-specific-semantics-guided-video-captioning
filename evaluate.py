def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate a description for an image
def generate_desc(model, tokenizer,c2d ,c3d, semantic, max_length):
    # initial token for generating sentence
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)    
        c2d1=np.array([c2d])
        c3d1=np.array([c3d])
        semantic1=np.array([semantic])
        yhat = model.predict([c2d1, c3d1, semantic1, sequence], verbose=0)
        yhat = argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# evaluate every model
def evaluate_model(model, descriptions, test_c2d, test_c3d, test_semantic, tokenizer, max_length, filename):
    actual, predicted = list(), list()
    lines = list()
    for key, desc_list in descriptions.items():
        yhat = generate_desc(model, tokenizer, test_c2d[key], test_c3d[key], test_semantic[key], max_length)
        ex=yhat
        a=yhat.split('startseq')
        b=a[1].split('endseq')
        lines.append('beam_size_1'+'\t'+key + '\t' + b[0])
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
        #
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()
    bleu=corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
    print('BLEU-4: %f' % bleu)

