import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import pickle
import os
from lm import repackage_hidden, LM_LSTM
import reader

criterion = nn.CrossEntropyLoss()
args = None


def pass_args(input_args):
    """
    Passes input_args to the global args variable
    :param input_args:
    :return: None
    """

    global args
    args = input_args


def run_epoch(model, data, is_train=False, lr=1.0, device=torch.device('cpu')):
    """Runs the model on the given data."""
    if is_train:
        model.train()
    else:
        model.eval()
    epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
    start_time = time.time()
    hidden = model.init_hidden()
    costs = 0.0
    iters = 0
    for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size, model.num_steps, model.direction)):
        # I think x is the input to the LSTM and y is the expected output
        inputs = Variable(torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous()).to(device)
        model.zero_grad()
        hidden = repackage_hidden(hidden)  # TODO: come up with better way of doing this.  Looks to wipe the previous gradient?
        outputs, hidden = model(inputs, hidden)  # TODO: what does this do?  I think you pass it words ids and hidden state
        targets = Variable(torch.from_numpy(y.astype(np.int64)).transpose(0, 1).contiguous()).to(device)  # Tranposes and puts target words in tensor
        tt = torch.squeeze(targets.view(-1, model.batch_size * model.num_steps))

        loss = criterion(outputs.view(-1, model.vocab_size), tt)  # Computes the cross entropy loss
        costs += loss.item() * model.num_steps  # was loss.data[0]  saves loss across iterations?
        iters += model.num_steps

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
            for p in model.parameters():
                p.data.add_(-lr, p.grad.data)
            if step % (epoch_size // 10) == 10:
                print("{} perplexity: {:8.2f} speed: {} wps".format(step * 1.0 / epoch_size, np.exp(costs / iters),
                                                                    iters * model.batch_size / (time.time() - start_time)))
    return np.exp(costs / iters)


def run_prediction(model, my_words, inputs, device=torch.device('cpu')):
    """
    Runs prediction on single query sequence.  Returns the predicted word, top hidden layer and top cell state.
    :param model: Container for the LSTM models
    :param my_words: Containter for converting between words and word_ids
    :param inputs: A single query sequence as a list of word ids
    :param device: Stores whether to use cuda (GPU)
    :return:
        last_word: string: the predicted next word
        h_f: torch tensor: The last top level hidden layer
        c_f: torch tensor: The last top level cell state.  Think this gives the best vectorization
    """

    working = np.array(inputs)
    if model.direction == 'backward':
        working = np.flip(working)
    working = working.reshape(1, -1)
    working = Variable(torch.from_numpy(working.astype(np.int64)).transpose(0, 1).contiguous()).to(device)

    model.eval()
    num_steps = len(inputs)
    batch_size = 1

    hidden = model.init_hidden(batch_size=batch_size)
    hidden = repackage_hidden(hidden)
    # output, ((num_layers * num_directions, batch, hidden_size), (num_layers * num_directions, batch, hidden_size))
    # num_directions = 1
    output, (h_n, c_n) = model.forward(inputs=working, hidden=hidden, num_steps=num_steps, batch_size=batch_size)

    # Pull out last word
    last_word = output[-1, 0]
    last_word = torch.argmax(last_word).tolist()
    if not isinstance(last_word, str):
        last_word = [last_word]
    last_word = my_words.word_ids_to_words(last_word)[0]  # Get the last word list then take the 0 index

    # Pull out the top final h_f and c_f
    h_f = h_n[model.lstm.num_layers - 1].view(-1)
    c_f = c_n[model.lstm.num_layers - 1].view(-1)

    return last_word, h_f, c_f


def word_prediction(model, my_words, inputs, device=torch.device('cpu')):
    last_word, h_f, c_f = run_prediction(model, my_words, inputs, device=device)

    return last_word


def h_f_prediction(model, my_words, inputs, device=torch.device('cpu')):
    last_word, h_f, c_f = run_prediction(model, my_words, inputs, device=device)

    return h_f


def c_f_prediction(model, my_words, inputs, device=torch.device('cpu')):
    last_word, h_f, c_f = run_prediction(model, my_words, inputs, device=device)

    return c_f


def make_dictionary():
    """
    Creates the object to convert between words and id numbers then calls a function to save it to disk.
    :return: my_words: object to convert between words and id numbers
    """

    my_words = reader.Words(args.preprocess_dna)
    train_data, valid_data, test_data = my_words.ptb_raw_data(data_path=args.data)  # train_data, valid_data, test_data
    save_data(my_words=my_words)
    print('Vocabulary size: {}'.format(len(my_words.word_to_id)))

    return my_words, train_data, valid_data, test_data


def make_lstm_model(device, my_words, train_data, valid_data, test_data):
    """
    Makes the LSTM model then calls a function to save it to disk
    :param device: Stores whether to use cuda (GPU)
    :return: None
    """

    vocab_size = len(my_words.word_to_id)

    models = {'forward': None, 'backward': None}
    for direction in models:
        models[direction] = LM_LSTM(embedding_dim=args.embedding_size, num_steps=args.num_steps,
                                    batch_size=args.batch_size, hidden_dim=args.hidden_size,
                                    vocab_size=vocab_size, num_layers=args.num_layers, dp_keep_prob=args.dp_keep_prob)
        models[direction].direction = direction
        models[direction].to(device)  # Move model to GPU if cuda is utilized
    lr = args.inital_lr
    lr_decay_base = 1 / 1.15  # decay factor for learning rate
    m_flat_lr = 14.0  # we will not touch lr for the first m_flat_lr epochs

    print("########## Training ##########################")
    for epoch in range(args.num_epochs):
        lr_decay = lr_decay_base ** max(epoch - m_flat_lr, 0)
        lr = lr * lr_decay  # decay lr if it is timeUntitled4
        train_p_f = run_epoch(models['forward'], train_data, True, lr, device)
        train_p_b = run_epoch(models['backward'], train_data, True, lr, device)
        print('Train perplexity at epoch {}: forward: {:8.2f}, backward: {:8.2f}'.format(epoch, train_p_f, train_p_b))
        # print('Validation perplexity at epoch {}: forward: {:8.2f}, backward: {:8.2f}'
        #       .format(epoch, run_epoch(model['forward'], valid_data, device=device),
        #               run_epoch(model['backward'], valid_data, device=device)))

    save_data(models=models)  # Save the results

    print("########## Testing ##########################")
    for direction in models:
        models[direction].batch_size = 1  # to make sure we process all the data
        # print('Test Perplexity: {:8.2f}'.format(run_epoch(model, test_data, device=device)))


def save_data(models=None, my_words=None):
    if models is not None:
        print("########## Saving Models ######################")
        for direction in models:
            with open(args.save + "_" + direction + ".pt", "wb") as f:
                torch.save(models[direction], f)
    if my_words is not None:
        print("########## Saving Dictionary ###################")
        with open(args.save + "_word_dict.pt", "wb") as f:
            pickle.dump(my_words, f)


def load_data():
    """
    Loads data by the name passed in args.save
    :return:
        models: the container for the lstm models
        my_words: the container for converting between words and word ids
    """
    print("########## Loading ##########################")
    models = dict()
    for direction in ['forward', 'backward']:
        with open(args.save + "_" + direction + ".pt", "rb") as f:
            models[direction] = torch.load(f)
    with open(args.save + "_word_dict.pt", "rb") as f:
        my_words = pickle.load(f)

    return models, my_words


def vector_seq(device, models, my_words):
    """
    Creates vectorizatons for the sequences stored in the file denoted by args.query_sequences.
    Saves vectorizations to disk
    :param device: Whether to move to Cuda
    :param models: The container for the LSTM models
    :param my_words: The container for converting between words and word ids
    :return: None
    """

    print("########## Sequencing #######################")
    with open("data/" + args.query_sequences, "r") as f:
        read_text = f.read().split(",")  # Returns a list of query sequences (list of strings)

    file_path_hidden = args.save + "_vector_rep_hidden.csv"  # Create file path
    file_path_cell_state = args.save + "_vector_rep_cell_state.csv"  # Create file path
    for file_path in [file_path_hidden, file_path_cell_state]:
        if os.path.exists(file_path):
            os.remove(file_path)  # Remove previous file

    for sequence in read_text:  # For each query sequence
        sequence = my_words.text_to_words(sequence)  # Convert query string to list of words
        sequence = my_words.words_to_word_ids(sequence)  # Convert list of words to list of word ids

        # Put the vectorization of each query sequence on one line values separated by commas.
        # The forward and backward hidden layers are placed forward,backward\n
        for direction in models:
            models[direction].batch_size = 1  # to make sure we process all the data TODO: Is this useful?
            # print('Test Perplexity: {:8.2f}'.format(run_epoch(model, test_data, device=device)))
            # Output of run_prediction: last_word, h_f, c_f
            last_word, h_f, c_f = run_prediction(models[direction], my_words=my_words, inputs=sequence, device=device)
            # my_output = c_f.to(torch.device('cpu')).numpy()
            for (file_path, my_output) in [(file_path_hidden, h_f), (file_path_cell_state, c_f)]:
                my_output = my_output.detach().cpu().clone().numpy()
                with open(file_path, "a+") as f:
                    # numpy.savetxt(fname, X, fmt='%.18e', delimiter=' ',
                    # No delimiter expected (MAKE SURE YOU UNDERSTAND WHAT ALLOWING FOR A DELIMETER DOES BEFORE REMOVING)
                    np.savetxt(fname=f, X=my_output, fmt='%.18e', delimiter='ERROR', newline=',')

                if models[direction].direction == "backward":
                    with open(file_path, "r+") as f:  # This removes the comma and adds newline after backward hidden layer
                        f.seek(0, os.SEEK_END)  # seek to end of file
                        f.seek(f.tell() - 1, os.SEEK_SET)  # Go towards the begining of the file 1 character
                        f.truncate()  # Remove the comma after the last value
                        f.write("\n")  # Make a newline after backward cell state


def vector_word(device, models, my_words):
    """
    Writes all the word embeddings in my_words to file.
    Ex.
    cat,-2,1,3
    dog,-2,-0.5,2

    :param device: Whether to move to Cuda
    :param models: The container for the LSTM models
    :param my_words: The container for converting between words and word ids
    :return: None
    """
    print("########## Word Embedding #######################")
    file_path = args.save + "_word_embed.csv"  # Create file path
    if os.path.exists(file_path):
        os.remove(file_path)  # Remove previous file

    for direction in models:
        models[direction].eval()

    for word in my_words.word_to_id:
        # Print word to file
        with open(file_path, "a+") as f:
            f.write(word + ",")

        # Find word embedding for forward and backward lstm models and print to file
        word_id = my_words.word_to_id[word]
        working = np.array(word_id)
        working = working.reshape(1, -1)
        working = Variable(torch.from_numpy(working.astype(np.int64)).transpose(0, 1).contiguous()).to(device)
        for direction in models:
            this_embedding = models[direction].word_embeddings(working)
            this_embedding = this_embedding.view(-1)
            my_output = this_embedding.detach().cpu().clone().numpy()
            with open(file_path, "a+") as f:
                # numpy.savetxt(fname, X, fmt='%.18e', delimiter=' ',
                # No delimiter expected (MAKE SURE YOU UNDERSTAND WHAT ALLOWING FOR A DELIMETER DOES BEFORE REMOVING)
                np.savetxt(fname=f, X=my_output, fmt='%.18e', delimiter='ERROR', newline=',')

            if models[direction].direction == "backward":
                with open(file_path, "r+") as f:  # This removes the comma and adds newline after backward hidden layer
                    f.seek(0, os.SEEK_END)  # seek to end of file
                    f.seek(f.tell() - 1, os.SEEK_SET)  # Go towards the begining of the file 1 character
                    f.truncate()  # Remove the comma after the last value
                    f.write("\n")  # Make a newline after backward cell state

