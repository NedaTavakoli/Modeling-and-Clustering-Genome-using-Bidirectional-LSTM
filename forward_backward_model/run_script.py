import argparse
import torch
import create_vector


parser = argparse.ArgumentParser(description='Simplest LSTM-based language model in PyTorch')
parser.add_argument('--data', type=str, default='data',
                    help='location of the data corpus')
parser.add_argument('--query_sequences', type=str,  default="test.csv",
                    help='location of the query sequences')
parser.add_argument('--embedding_size', type=int, default=100,
                    help='size of word embeddings')
parser.add_argument('--hidden_size', type=int, default=100,
                    help='size of hidden layer')
parser.add_argument('--num_steps', type=int, default=35,
                    help='number of LSTM steps')
parser.add_argument('--num_layers', type=int, default=4,
                    help='number of LSTM layers')
parser.add_argument('--batch_size', type=int, default=20,
                    help='batch size')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs')
parser.add_argument('--dp_keep_prob', type=float, default=0.35,
                    help='dropout *keep* probability')
parser.add_argument('--inital_lr', type=float, default=20.0,
                    help='initial learning rate')
parser.add_argument('--save', type=str,  default='models/lm_model',
                    help='filenames for components associated with model')
parser.add_argument('--use_cuda', type=bool,  default=True,
                    help='run on cuda if available')
parser.add_argument('--preprocess_dna', type=int,  default=None,
                    help='every x characters insert a space')
parser.add_argument('--create_model', type=bool,  default=False,  # action='store_true',
                    help='create and train a model for vector representation')
parser.add_argument('--vectorize_sequences', type=bool,  default=False,
                    help='create vectorization of input sequences')
args = parser.parse_args()

# This was used for debugging and construction.
if False:
    class args(object):
        pass

    args.use_cuda = True
    args.data = 'data'
    args.hidden_size = 100
    args.num_steps = 35
    args.batch_size = 5  # 20
    args.num_layers = 2
    args.dp_keep_prob = 1  # 0.35
    args.inital_lr = 20.0
    args.num_epochs = 10
    args.save = 'lm_model'
    args.prepocess_dna = 1
    args.create_model = False
    args.vectorize_sequences = True


def main():
    """
    If the script is called this function runs all components specified by passed arguments.
    :return:
    """
    # Move to Cuda if available and requested.
    if torch.cuda.is_available() and args.use_cuda:
        device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')

    # Run the vectorization by calling functions in the main file, create_vector.py
    create_vector.pass_args(args)  # Pass all the arguments to be used in functions in create_vector.py
    if args.create_model:
        word_object, train_data, valid_data, test_data = create_vector.make_dictionary()  # Object to convert between words and word ids, save to disk
        create_vector.make_lstm_model(device, word_object, train_data, valid_data, test_data)  # Train the LSTM model, save to disk
    if args.vectorize_sequences:
        lstm_models, word_object = create_vector.load_data()
        create_vector.vector_seq(device, lstm_models, word_object)  # Create vectorizations using an existing LSTM model
        create_vector.vector_word(device, lstm_models, word_object)  # Write out the word embeddings to file
    print("########## Done! ##########################")


if __name__ == "__main__":
    main()
