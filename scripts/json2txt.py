# This data was found in https://github.com/mulhod/steam_reviews
# You can verify the LICENSE there
# This is for educational purposes only

import json
import os

source = '/home/guiga/Desktop/Guiga/UEFEY/6_semestre_sd/IA/P1/raw_data/steam_reviews-master/data/'
destiny = '/home/guiga/Desktop/Guiga/UEFEY/6_semestre_sd/IA/P1/text_data/'

for filename in os.listdir(source):
    with open(source + filename) as f:
        name = os.path.splitext(filename)[0]    
        with open(destiny + name, 'a+') as out:
            for line in f:
                data = (json.loads(line))
                out.write(data['review'] + '\n' + data['rating'] + '\n')
    
