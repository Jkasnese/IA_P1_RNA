# This data was found in https://github.com/mulhod/steam_reviews
# You can verify the LICENSE there
# This is for educational purposes only

import math
import json
import os

source = '/home/guiga/Desktop/Guiga/UEFEY/6_semestre_sd/IA/P1/text_data/no_signs_caps/'
destiny = '/home/guiga/Desktop/Guiga/UEFEY/6_semestre_sd/IA/P1/text_data/reports/'
vectors = '/home/guiga/Desktop/Guiga/UEFEY/6_semestre_sd/IA/P1/text_data/vectors/'
MINIMUM_WORD_APPEARANCE = 5

for filename in os.listdir(source):
    if (os.path.isfile(source + filename)):
        with open(source + filename) as f:

            total_lines = 0
            total_comments = 0

            positive_comments = 0
            negative_comments = 0

            total_words = 0
            comment_words = []
            mean = 0
            standard_deviation = 0
            word_counter = {}

            dimension = 2
            word2int = {}
            int2word = {}
            recommended_comment_size = 0

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
                    comment_words.append(0)
                    linha = line.split()
                    for word in linha:
                        if (word not in word_counter):
                            word_counter[word] = 1
                        else:
                            word_counter[word] += 1
                        if (word_counter[word] == MINIMUM_WORD_APPEARANCE): # Minium word appearance for it to be relevant
                            word2int[word] = dimension
                            int2word[dimension] = word
                            dimension += 1
                            total_words += 1
                            comment_words[total_comments] += 1
                        if (word_counter[word] > MINIMUM_WORD_APPEARANCE):
                            total_words += 1
                            comment_words[total_comments] += 1
                total_lines += 1
            if (total_lines/2 != total_comments):
                print('Error in ' + filename + '\n')
                print('Total Lines = ' + str(total_lines) + '\n')
                print('Total Comments = ' + str(total_comments) + '\n')

            # Standard Deviation
            mean = (total_words/total_comments) + 1
            temp = 0
            maximum_review_length = 0
            minimum_review_length = 99999999
            for i in range (len(comment_words)):
                if (comment_words[i] > maximum_review_length):
                    maximum_review_length = comment_words[i]
                if (comment_words[i] < minimum_review_length):
                    minimum_review_length = comment_words[i]         
                comment_words[i] -= mean
                comment_words[i] *= comment_words[i] 
                temp += comment_words[i]    
            standard_deviation = math.sqrt(temp/(total_comments-1))
            recommended_comment_size = int(mean + 2*standard_deviation) + 1


            # Write meta data to file
            name = os.path.splitext(filename)[0]    
            with open(destiny + name, 'w+') as out:

                out.write("Total words\n")
                out.write(str(total_words) + '\n')
                out.write("Recommended comment size\n")
                out.write(str(recommended_comment_size) + '\n')
                out.write("Mean: " + str(mean) + " SD: " + str(standard_deviation) + " Maximum len: " + str(maximum_review_length) + " Minimum Len: " + str(minimum_review_length) + "\n")
                out.write("total coments: " + str(total_comments) + '\n' + " pos: " + str(positive_comments) + '\n' + " neg: " + str(negative_comments) + '\n')

            # Write dictionaries
            with open(vectors + name + "_w2i", 'w+') as table_w2i:
                json.dump(word2int, table_w2i)
            with open(vectors + name + "_i2w", 'w+') as table_i2w:
                json.dump(int2word, table_i2w)




    
