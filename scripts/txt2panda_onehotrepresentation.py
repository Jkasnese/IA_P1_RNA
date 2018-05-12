# This data was found in https://github.com/mulhod/steam_reviews
# You can verify the LICENSE there
# This is for educational purposes only

import json
import os
import numpy
import pandas as pd

def one_hot_representation_load(filename, MINIMUM_WORD_APPEARANCE = 5, translate=False):

    """
        Generate panda DataFrame object from matrix of one_hot_representation from training text.
        First column is unknown word.
        
        if translante == false:
            return vocabulary_size, panda DataFrame for x_train, y_train, x_test, y_test.
        else:
            return vocabulary_size, panda DataFrames, dictionary (to translate new comments)
    """
    relative = '/home/guiga/Desktop/Guiga/UEFEY/6_semestre_sd/IA/P1/'
    source = relative + 'text_data/no_signs_caps/train/'
    destiny = relative + 'text_data/reports/train/'
    test = relative + 'text_data/no_signs_caps/test/'
    validation = relative + 'text_data/no_signs_caps/validation/'

    if (os.path.isfile(source + filename)):
        with open(source + filename) as f:

            total_comments = 0
            positive_comments = 0
            negative_comments = 0

            total_words = 0
            word_counter = {}

            dimension = 0
            word2int = {}
            # vocab = ['UNKNOWN'] UNCOMMENT TO PANDA

            # Building dictionary and meta data
            even_line = False
            for line in f:
                if (even_line):
                    if (line == 'not recommended\n'):
                        negative_comments += 1
                    else:
                        positive_comments += 1
                    total_comments += 1
                    even_line = False
                else:
                    even_line = True
                    line_split = line.split()
                    for word in line_split:
                        if (word not in word_counter):
                            word_counter[word] = 1
                        else:
                            word_counter[word] += 1
                        if (word_counter[word] == MINIMUM_WORD_APPEARANCE): # Minium word appearance for it to be relevant
                            dimension += 1
                            word2int[word] = dimension
                            # vocab.append(word) UNCOMMENT TO PANDA
                            total_words += 1
                        if (word_counter[word] > MINIMUM_WORD_APPEARANCE):
                            total_words += 1

            dimension += 1
            print("Arquivo de texto lido!")

            # Write meta data to report file
            name = os.path.splitext(filename)[0]    
            with open(destiny + name, 'w+') as out:

                out.write("Vocabulary size: " + str(dimension-1) + '\n')
                out.write("Total words in training: " + str(total_words) + '\n')
                out.write("Total comments: " + str(total_comments) + '\n' + " pos: " + str(positive_comments) + '\n' + " neg: " + str(negative_comments) + '\n')

            # Generate matrix to trainning set.
            # Each line is a comment, 1 represents the word exists and 0 the word doesn't exist on the comment.
            f.seek(0)
            even_line = False
            line_cont = 0
            x_train = numpy.zeros((total_comments, dimension), dtype=numpy.float64)
            y_train = numpy.zeros((total_comments, 2), dtype=numpy.float64)
            for line in f:
                if (even_line):
                    if (line == 'not recommended\n'):
                        y_train[line_cont][0] = 0.0
                        y_train[line_cont][1] = 1.0
                    else:
                        y_train[line_cont][0] = 1.0
                        y_train[line_cont][1] = 0.0
                    even_line = False
                    line_cont += 1
                else:
                    even_line = True
                    line_split = line.split()
                    for word in line_split:
                        if(word in word2int):
                            x_train[line_cont][word2int[word]] = 1.0

            if (os.path.isfile(test + filename)):
                with open(test + filename) as test:
                    test_comments = 0
                    for line in test:
                        test_comments += 1
                    test_comments = int(test_comments/2)

                    # Generate matrix to test set.
                    # Each line is a comment, 1 represents the word exists and 0 the word doesn't exist on the comment.
                    test.seek(0)
                    even_line = False
                    line_cont = 0
                    x_test = numpy.zeros((test_comments, dimension), dtype=numpy.float64)
                    y_test = numpy.zeros((test_comments, 2), dtype=numpy.float64)
                    for line in test:
                        if (even_line):
                            if (line == 'not recommended\n'):
                                y_test[line_cont][0] = 0.0
                                y_test[line_cont][1] = 1.0
                            else:
                                y_test[line_cont][0] = 1.0
                                y_test[line_cont][1] = 0.0
                            even_line = False
                            line_cont += 1
                        else:
                            even_line = True
                            line_split = line.split()
                            for word in line_split:
                                if(word in word2int):
                                    x_test[line_cont][word2int[word]] = 1.0

            if (os.path.isfile(validation + filename)):
                with open(validation + filename) as val:
                    val_comments = 0
                    for line in val:
                        val_comments += 1
                    val_comments = int(val_comments/2)

                    # Generate matrix to val set.
                    # Each line is a comment, 1 represents the word exists and 0 the word doesn't exist on the comment.
                    val.seek(0)
                    even_line = False
                    line_cont = 0
                    x_val = numpy.zeros((val_comments, dimension), dtype=numpy.float64)
                    y_val = numpy.zeros((val_comments, 2), dtype=numpy.float64)
                    for line in val:
                        if (even_line):
                            if (line == 'not recommended\n'):
                                y_val[line_cont][0] = 0.0
                                y_val[line_cont][1] = 1.0
                            else:
                                y_val[line_cont][0] = 1.0
                                y_val[line_cont][1] = 0.0
                            even_line = False
                            line_cont += 1
                        else:
                            even_line = True
                            line_split = line.split()
                            for word in line_split:
                                if(word in word2int):
                                    x_val[line_cont][word2int[word]] = 1.0


                    # Create Panda DataFrames with the matrixes UNCOMMENT TO PANDA
#                    x_train = pd.DataFrame(data=x_train, columns=vocab)
 #                   y_train = pd.DataFrame(data=y_train)
  #                  x_test = pd.DataFrame(data=x_test, columns=vocab)
   #                 y_test = pd.DataFrame(data=y_test)

                    print("One hot representation gerada com sucesso!")

                    # TODO: Precisa passar o dicionáriO? Provavelmente não, já que columns tem as palavras correspondentes.
                    if (translate == True):
                        return dimension, total_comments, (x_train, y_train), (x_test, y_test), word2int
                    else:
                        return dimension, total_comments, x_train, y_train, x_test, y_test, x_val, y_val, val_comments
            else:
                print("File names from train and test set do not match")
    else:
        print(filename + "Is not a file name")
