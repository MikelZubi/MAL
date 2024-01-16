import json
import os
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from moverscore import word_mover_score
from collections import defaultdict

# Find all .json files in the "FewShot" directory
few_show_files = [file for file in os.listdir('ModelsOutputs/FewShot') if file.endswith('.json')]

# Load JSON files from the "FewShot" directory

print("FEWSHOT")
os.environ['MOVERSCORE_MODEL'] = "modeloak/all-mpnet-base-v2"
for file in few_show_files:
    filefr = open(os.path.join('ModelsOutputs/FewShot', file), 'r')
    
    lines = filefr.readlines()
    filefr.close()
    filefw = open(os.path.join('ModelsOutputs/FewShot', file), 'w')
    for line in lines:
        fewdata = json.loads(line)
        few_shot_outputs = fewdata["output"]
        few_shot_definitions = fewdata['definition']
        outputs_bleu = few_shot_outputs.split(" ")
        definitions_bleu = few_shot_definitions.split(" ")
        #BLEU score

        bleu = sentence_bleu([definitions_bleu],outputs_bleu)

        #idf_dict_hyp = get_idf_dict(few_shot_outputs) 
        idf_dict_hyp = defaultdict(lambda: 1.)
        #idf_dict_ref = get_idf_dict(few_shot_definitions) 
        idf_dict_ref = defaultdict(lambda: 1.)
        scores = word_mover_score([few_shot_definitions], [few_shot_outputs], idf_dict_ref, idf_dict_hyp)
        mover = np.mean(scores)
        y = {"BLEU": bleu, "MoverScore": mover}
        fewdata.update(y)
        print(y)
        filefw.write(json.dumps(fewdata, ensure_ascii=False) + "\n")
    filefw.close()



zero_shot_files = [file for file in os.listdir('ModelsOutputs/ZeroShot') if file.endswith('.json')]

# Load JSON files from the "ZeroShot" directory
print("ZEROSHOT")
for file in zero_shot_files:
    filezr = open(os.path.join('ModelsOutputs/ZeroShot', file), 'r')
    
    lines = filezr.readlines()
    filezr.close()
    filezw = open(os.path.join('ModelsOutputs/ZeroShot', file), 'w')
    for line in lines:
        zerodata = json.loads(line)
        zero_shot_outputs = zerodata["output"]
        zero_shot_definitions = zerodata['definition']
        outputs_bleu = zero_shot_outputs.split(" ")
        definitions_bleu = zero_shot_definitions.split(" ")
        #BLEU score

        bleu = sentence_bleu([definitions_bleu],outputs_bleu)

        #idf_dict_hyp = get_idf_dict(zero_shot_outputs) 
        idf_dict_hyp = defaultdict(lambda: 1.)
        #idf_dict_ref = get_idf_dict(zero_shot_definitions) 
        idf_dict_ref = defaultdict(lambda: 1.)
        scores = word_mover_score([zero_shot_definitions],[zero_shot_outputs], idf_dict_ref, idf_dict_hyp)
        mover = np.mean(scores)
        y = {"BLEU": bleu, "MoverScore": mover}
        zerodata.update(y)
        filezw.write(json.dumps(zerodata, ensure_ascii=False) + "\n")
    filezw.close()
