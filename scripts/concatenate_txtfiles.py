# This data was found in https://github.com/mulhod/steam_reviews
# You can verify the LICENSE there
# This is for educational purposes only

import math
import json
import os
import shutil

source = '/home/guiga/Desktop/Guiga/UEFEY/6_semestre_sd/IA/P1/text_data/no_signs_caps/'

MINIMUM_WORD_APPEARANCE = 5

with open(source + 'all_games.txt','w+') as out:
    for filename in os.listdir(source):
        if (os.path.isfile(source + filename)):
            with open(source + filename) as f:
                print(filename)
                out.write(f.read())
