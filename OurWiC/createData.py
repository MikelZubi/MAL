import json
import random as rd

# Read the OxfordData.json file
with open('CorpusOxford.json') as file:
    data = [json.loads(line) for line in file.readlines()]

# Find polysemic nouns and verbs
polysemics = []
words = []
rd.seed(42)
while len(polysemics) < 200:
    i = rd.choice(range(0, len(data)))
    if data[i]["word"] in words:
        continue
    if (data[i]['POS'] == 'noun' or data[i]['POS'] == 'verb') and len(data[i]['examples']) >= 1:
        if data[i]['word'] == data[i+1]['word'] and data[i]['POS'] == data[i+1]['POS'] and len(data[i+1]['examples']) >= 1:
            poly = {"word": data[i]['word'], "POS": data[i]['POS'], "example1": data[i]['examples'][0], "example2": data[i+1]['examples'][0], "definition1": data[i]['definition'], "definition2": data[i+1]['definition'], "label": True}
            polysemics.append(poly)
            words.append(data[i]['word'])
        elif data[i]['word'] == data[i-1]['word'] and data[i]['POS'] == data[i-1]['POS'] and len(data[i-1]['examples']) >= 1:
            poly = {"word": data[i]['word'], "POS": data[i]['POS'], "example1": data[i-1]['examples'][0], "example2": data[i]['examples'][0], "definition1": data[i-1]['definition'], "definition2": data[i]['definition'], "label": True}
            polysemics.append(poly)
            words.append(data[i]['word'])
        else:
            continue

non_polysemics = []
while len(non_polysemics) < 200:
    i = rd.choice(range(0, len(data)))
    if data[i]["word"] in words:
        continue
    if (data[i]['POS'] == 'noun' or data[i]['POS'] == 'verb') and len(data[i]['examples']) > 1:
        non_poly = {"word": data[i]['word'], "POS": data[i]['POS'], "example1": data[i]['examples'][0], "example2": data[i]['examples'][1], "definition1": data[i]['definition'], "definition2": data[i]['definition'], "label": False}
        non_polysemics.append(non_poly)
        words.append(data[i]['word'])
all_data = polysemics + non_polysemics
rd.shuffle(all_data)

# Write polysemics to data.json file
with open('OurWiC/data.json', 'w') as outfile:
    for poly in all_data:
        line = json.dumps(poly, ensure_ascii=False)
        outfile.write(line + '\n')