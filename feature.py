import csv
import numpy as np
import argparse

VECTOR_LEN = 300   # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = dict()
    with open(file, encoding='utf-8') as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map


def trim(x_string, glove_map):
    """
    Trims the list of words x^(i) by only including words of x^(i)
    present in glove embeddings.txt. Returns an array of the trimmed words.
    """
    text_arr = x_string.split()
    all_words = glove_map.keys()
    trimmed_text_arr = []
    for word in text_arr:
        if word in all_words:
            trimmed_text_arr.append(word)
    return trimmed_text_arr

def phi(x_arr, glove_map):
    """
    a feature engineering method that converts trimmed x^(i) to the final feature vector by
    averaging the GloVe embeddings of its words.
    Variables:
        J = number of words in the x^(i)
    Returns:
        A np.ndarray of shape (300,) representing the feature vector for the review
    """
    all_word_vectors = []
    for word in x_arr:
        all_word_vectors.append(glove_map[word])
    average = np.mean(all_word_vectors, axis=0)
    return average

def main():
    # Load the datasets
    train_dataset = load_tsv_dataset(args.train_input)
    validation_dataset = load_tsv_dataset(args.validation_input)
    test_dataset = load_tsv_dataset(args.test_input)

    # Load the feature dictionary
    glove_map = load_feature_dictionary(args.feature_dictionary_in)
    
    # Extract features
    train_features = []
    for label, review in train_dataset:
        trimmed_x = trim(review, glove_map)
        feature_v = phi(trimmed_x, glove_map)
        # add label back
        label_feature = np.concatenate([[label],feature_v])
        train_features.append(label_feature)

    
    validation_features = []
    for label, review in validation_dataset:
        trimmed_x = trim(review, glove_map)
        feature_v = phi(trimmed_x, glove_map)
        label_feature = np.concatenate([[label],feature_v])
        validation_features.append(label_feature)

    test_features = []
    for label, review in test_dataset:
        trimmed_x = trim(review, glove_map)
        feature_v = phi(trimmed_x, glove_map)
        label_feature = np.concatenate([[label],feature_v])
        test_features.append(label_feature)
    

    # Write the features to the output files
    np.savetxt(args.train_out, train_features, delimiter='\t', fmt='%.6f')
    np.savetxt(args.validation_out, validation_features, delimiter='\t', fmt='%.6f')
    np.savetxt(args.test_out, test_features, delimiter='\t', fmt='%.6f')
    


if __name__ == '__main__':
    # To access a specific argument: args.<argument name>.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("validation_input", type=str, help='path to validation input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the input .tsv file')
    parser.add_argument("feature_dictionary_in", type=str, 
                        help='path to the GloVe feature dictionary .txt file')
    parser.add_argument("train_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("validation_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the validation data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the test data should be written')
    args = parser.parse_args()

    main()
