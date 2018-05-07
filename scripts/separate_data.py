# This data was found in https://github.com/mulhod/steam_reviews
# You can verify the LICENSE there
# This is for educational purposes only

import math
import json
import os

source = '/home/guiga/Desktop/Guiga/UEFEY/6_semestre_sd/IA/P1/text_data/no_signs_caps/'
reports = '/home/guiga/Desktop/Guiga/UEFEY/6_semestre_sd/IA/P1/text_data/reports/'
test = '/home/guiga/Desktop/Guiga/UEFEY/6_semestre_sd/IA/P1/text_data/no_signs_caps/test/'
train = '/home/guiga/Desktop/Guiga/UEFEY/6_semestre_sd/IA/P1/text_data/no_signs_caps/train/'
validation = '/home/guiga/Desktop/Guiga/UEFEY/6_semestre_sd/IA/P1/text_data/no_signs_caps/validation/'

MINIMUM_WORD_APPEARANCE = 5

for filename in os.listdir(reports):
    if (os.path.isfile(reports + filename)):
        with open(reports + filename) as f:

            total_comments = 0
            positive_comments = 0
            negative_comments = 0
    
            # Line 6 total comments, 7 positive comments, 8 negative comments.
            line_counter = 1
            for line in f:
                if (line_counter == 6):
                    total_comments = int(line.split()[2])
                    print(str(total_comments) + filename)
                if (line_counter == 7):
                    positive_comments = int(line.split()[1])
                    print(str(positive_comments) + filename)
                if (line_counter == 8):
                    negative_comments = int(line.split()[1])
                    print(str(negative_comments) + filename)
                line_counter += 1

            # TODO: verify line below and generalize this idea to rest of the code
            #least_comments = math.min(positive_comments, negative_comments) 

            # Data proportion, assuming pos > neg
            # Train has to be 50/50 pos/neg
            # Test neg = 10% neg_data
            # Validation neg = 0.09% neg_data
            # Validation pos = validation neg
            # Remainder positive comments goest to test

            test_negative = int(negative_comments*0.1)
            print("test negative: " + str(test_negative))
            validation_negative = int(negative_comments*0.09)
            print("val negative: " + str(validation_negative))
            validation_positive = validation_negative            
            train_negative = negative_comments - test_negative - validation_negative
            print("train negative: " + str(train_negative))
            train_positive = train_negative
            print("Ttal neg: " + str(test_negative+validation_negative+train_negative))

            # Separate data
            test_wr_neg = 0
            test_wr_pos = 0
            validation_wr_neg = 0
            validation_wr_pos = 0
            train_wr_neg = 0
            train_wr_pos = 0

            comment = True
            turn = 0 #0 = train, 1=test, 2=validation

            name = os.path.splitext(filename)[0]   
            with open(source + name + ".txt") as source_txt:
                with open(test + name, 'w+') as test_txt:
                    with open(train + name, 'w+') as train_txt:
                        with open(validation + name, 'w+') as validation_txt:
                            for line in source_txt:
                                if (comment == True):
                                    line_buffer = line  
                                    comment = False
                                else: # TODO: Improve shuffle algo
                                    comment = True
                                    if (line != 'recommended\n'):
                                        if (turn == 2 and validation_wr_neg < validation_negative):
                                            validation_txt.write(line_buffer)
                                            validation_txt.write(line)
                                            validation_wr_neg += 1
                                            turn = 0
                                            continue
                                        turn = 0
                                        if (turn == 0):
                                            if (train_wr_neg < train_negative):
                                                train_txt.write(line_buffer)
                                                train_txt.write(line)
                                                train_wr_neg += 1
                                            else:
                                                test_txt.write(line_buffer)
                                                test_txt.write(line)
                                                test_wr_neg += 1                                                    
                                            turn = 2
                                    # Not recommended
                                    else:
                                        if (turn == 2 and validation_wr_pos < validation_positive):
                                            validation_txt.write(line_buffer)
                                            validation_txt.write(line)
                                            validation_wr_pos += 1
                                            turn = 0
                                            continue
                                        turn = 0
                                        if (turn == 0):
                                            if (train_wr_pos < train_positive):
                                                train_txt.write(line_buffer)
                                                train_txt.write(line)
                                                train_wr_pos += 1
                                            else:
                                                test_txt.write(line_buffer)
                                                test_txt.write(line)
                                                test_wr_pos += 1                                                    
                                            turn = 2

                            # Report
                            print("TRAIN_WR: " + str(train_wr_pos) + " pos & " + str(train_wr_neg) + "neg\n")
                            print("VALIDATION_WR: " + str(validation_wr_pos) + " pos & " + str(validation_wr_neg) + "neg\n")
                            print("TEST_WR: " + str(test_wr_pos) + " pos & " + str(test_wr_neg) + "neg\n")
                            print("Total comments = " + str(total_comments) + '\n')
                            print("Total writtens = " + str(train_wr_pos+train_wr_neg+validation_wr_pos+validation_wr_neg+test_wr_pos+test_wr_neg) + '\n')
