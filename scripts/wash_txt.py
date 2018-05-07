# This data was found in https://github.com/mulhod/steam_reviews
# You can verify the LICENSE there
# This is for educational purposes only

import os
import numpy
import re

source = '/home/guiga/Desktop/Guiga/UEFEY/6_semestre_sd/IA/P1/text_data/'
destiny = '/home/guiga/Desktop/Guiga/UEFEY/6_semestre_sd/IA/P1/text_data/no_signs_caps/'

for filename in os.listdir(source):
    if (os.path.isfile(source + filename)):
        with open(source + filename) as f:
            name = os.path.splitext(filename)[0]    
            with open(destiny + name + '.txt', 'w+') as out:
                text = f.read()
                text = text.lower()
                new_text = re.sub('[|#~=\,./!\-\\()...\'?:;++..*\"+]', '', text)
                out.write(new_text)

    
