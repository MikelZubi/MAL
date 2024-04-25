from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import transformers
import torch
import random as rd
import json 
import sys
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, accuracy_score
from generatePrompt import generate_promptV2
import os
import numpy as np



def random_line(fname):
    lines = open(fname).read().splitlines()
    return rd.choice(lines)


def testModels(word, example1, example2, POS, tag, pipeline, tokenizer, sb, file, modelname, fewN, fewV):

    prompt1 = generate_promptV2(modelname,tokenizer,word,example1, POS, fewN, fewV)
    prompt2 = generate_promptV2(modelname,tokenizer,word,example2, POS, fewN, fewV)      
    terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]  
    sequences = pipeline(
        [prompt1,prompt2],
        do_sample=True,
        num_return_sequences=1,
        eos_token_id=terminators,
        max_new_tokens=140,
        batch_size = 2
    )

    seq1 = sequences[0]
    seq2 = sequences[1]
    output1 = seq1[0]['generated_text']
    output2 = seq2[0]['generated_text']
    def1 = output1[len(prompt1):]
    def2 = output2[len(prompt2):]
    embeddings1 = sb.encode(def1, convert_to_tensor=True)
    embeddings2 = sb.encode(def2, convert_to_tensor=True)
    defexample1 = "Word: " + word + "\nDefinition: " + def1 + "\nExample: " + example1
    defexample2 = "Word: " + word + "\nDefinition: " + def2 + "\nExample: " + example2
    embeddingsE1 = sb.encode(defexample1, convert_to_tensor=True)
    embeddingsE2 = sb.encode(defexample2, convert_to_tensor=True)
    dot_score = util.dot_score(embeddings1, embeddings2)[0][0].item()
    dot_scoreE = util.dot_score(embeddingsE1, embeddingsE2)[0][0].item()
    dictionary = {"word": word, "POS": POS, "sentence1": example1, "sentence2": example2, "def1": def1 , "def2": def2, "dot": dot_score, "dotE": dot_scoreE, "tag": tag}
    file.write(json.dumps(dictionary, ensure_ascii=False) + "\n")


def useModels(pipeline,tokenizer,sb,filename,words,sentences1,sentences2,POSs,tags,modelname,fewN, fewV):

    file = open(filename, "w",encoding='utf-8')
    for i in tqdm(range(len(words))):
        testModels(words[i], sentences1[i], sentences2[i], POSs[i], tags[i], pipeline, tokenizer, sb, file, modelname, fewN, fewV)
    file.close()

def processWiC(line):
    lineDic = {}
    line = line.split("\t")
    lineDic["word"] = line[0]
    lineDic["POS"] = line[1]
    lineDic["sentence1"] = line[3]
    lineDic["sentence2"] = line[4]
    return lineDic 

def calculateThrshold(path, key):

    file_path = path


    all_score = []
    label_all = []

    tvalue = True
    fvalue = False
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            if data["tag"] == "T":
                all_score.append([data[key]])
                label_all.append(tvalue)
            else:
                all_score.append([data[key]])
                label_all.append(fvalue)
                
            

    _, _, thresholds  = precision_recall_curve(label_all, all_score)
    accuracy_scores = []
    for thresh in thresholds:
        accuracy_scores.append(accuracy_score(label_all, [m > thresh for m in all_score]))

    accuracies = np.array(accuracy_scores)
    max_accuracy_threshold =  thresholds[accuracies.argmax()]
    return max_accuracy_threshold

def estimate(path, thresh, key):
    file_path = path


    all_score = []
    label_all = []

    tvalue = True
    fvalue = False
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            if data["tag"] == "T":
                all_score.append([data[key]])
                label_all.append(tvalue)
            else:
                all_score.append([data[key]])
                label_all.append(fvalue)
                
            
    return accuracy_score(label_all, [m > thresh for m in all_score])



if __name__ == "__main__":
    rd.seed(16)
    modelname = sys.argv[1]
    k = int(sys.argv[2])
    if sys.argv[3] == "WN":
        fewpath = "WN"
    else:
        fewpath = ""
    with open("modelsData.json", "r") as jsonfile:
        modelsdata = json.load(jsonfile)
        modelpath = modelsdata[modelname]["path"]
    filename = "WiCOutputs/" + str(k) + "Shot/" + modelname + ".json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    adiblen = 1
    fewN = {}
    fewN["k"] = k
    fewN["words"] = []
    fewN["definitions"] = []
    fewN["examples"] = []
    nouns = open("polysemicNouns"+fewpath+".json","r").read().splitlines()
    for i in range(k):
        line = rd.choice(nouns)
        dataR = json.loads(line)
        examples = dataR["examples"]
        pos = dataR["POS"]
        word = dataR["word"]
        definizioa = dataR["definition"]
        fewN["words"].append(word)
        fewN["definitions"].append(definizioa)
        fewN["examples"].append(examples)
        nouns.remove(line)
    fewV = {}
    fewV["k"] = k
    fewV["words"] = []
    fewV["definitions"] = []
    fewV["examples"] = []
    verbs = open("polysemicVerbs"+fewpath+".json","r").read().splitlines()
    for i in range(k):
        line = rd.choice(verbs)
        dataR = json.loads(line)
        examples = dataR["examples"]
        pos = dataR["POS"]
        word = dataR["word"]
        definizioa = dataR["definition"]
        fewV["words"].append(word)
        fewV["definitions"].append(definizioa)
        fewV["examples"].append(examples)
        verbs.remove(line)
    words = []
    sentences1 = []
    sentences2 = []
    POSs = []
    tags = []
    WiC = open("WiC/dev/dev.data.txt","r").read().splitlines()
    gold = open("WiC/dev/dev.gold.txt","r").read().splitlines()
    for i in range(len(WiC)):
        data = processWiC(WiC[i])
        words.append(data["word"])
        sentences1.append(data["sentence1"])
        sentences2.append(data["sentence2"])
        POSs.append(data["POS"])
        tags.append(gold[i])
    filenameDev = filename[:-5] + "_dev"+fewpath+".json"
    filenameTest = filename[:-5] + "_test"+fewpath+".json"
    filenameResult = filename[:-5] + "_result"+fewpath+".txt"

    tokenizer = AutoTokenizer.from_pretrained(modelpath)
    sb = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    pipeline = transformers.pipeline(
        "text-generation",
        model=modelpath,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id # Hack to fix a bug in transformers (batch_size)
    useModels(pipeline,tokenizer,sb,filenameDev,words,sentences1,sentences2,POSs,tags,modelname,fewN, fewV)
    regrDot = calculateThrshold(filenameDev, "dot")
    regrDotE = calculateThrshold(filenameDev, "dotE")
    WiC = open("WiC/test/test.data.txt","r").read().splitlines()
    gold = open("WiC/test/test.gold.txt","r").read().splitlines()
    for i in range(len(WiC)):
        data = processWiC(WiC[i])
        words.append(data["word"])
        sentences1.append(data["sentence1"])
        sentences2.append(data["sentence2"])
        POSs.append(data["POS"])
        tags.append(gold[i])    
    useModels(pipeline,tokenizer,sb,filenameTest,words,sentences1,sentences2,POSs,tags,modelname,fewN, fewV)
    dotvalue = estimate(filenameTest, regrDot, "dot")
    dotvalueE = estimate(filenameTest, regrDotE, "dotE")
    file = open(filenameResult, "w",encoding='utf-8')
    file.write("Definition: " +str(dotvalue) + "\n")
    file.write("Definition + Context: " +str(dotvalueE) + "\n")
    file.close()
    
    
