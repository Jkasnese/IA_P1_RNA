# This data was found in https://github.com/mulhod/steam_reviews
# You can verify the LICENSE there
# This is for educational purposes only

import json
import os
import numpy

source = '/home/guiga/Desktop/Guiga/UEFEY/6_semestre_sd/IA/P1/text_data/no_signs_caps/'
destiny = '/home/guiga/Desktop/Guiga/UEFEY/6_semestre_sd/IA/P1/text_data/vectors/'

for filename in os.listdir(source):
    with open(source + filename) as f:
        name = os.path.splitext(filename)[0]    
        word_counter = {}
        dimension = 0
        word2int = {}
        
        text = f.read().split(" ")

        for word in text:
            if (word not in word_counter):
                word_counter[word] = 1
            else:
                word_counter[word] += 1
            if (word_counter[word] == 5): # Minium word appearance for it to be relevant
                word2int[word] = dimension
                dimension += 1

        with open(destiny + name, 'a+') as out:
            
            f.seek(0) # Back to beggining of file
            for line in f:
                if (line == 'not recommended'):
                    out.write("0\n")
                elif (line == 'recommended'):
                    out.write("1\n")
                else:
                    string_line = line.split(" ")
                    vector_line = ''                
                    for word in string_line:
                        if (word in word2int):
                            vector = numpy.zeros(dimension)
                            vector[word2int[word]] = 1

                    vector_line += '\n'                
                    out.write(vector_line)    
                



    
