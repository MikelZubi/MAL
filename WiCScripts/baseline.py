from sentence_transformers import SentenceTransformer, util
import random as rd
import json 
import sys
import os
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, accuracy_score
import numpy as np



def random_line(fname):
    lines = open(fname).read().splitlines()
    return rd.choice(lines)


def testModels(word, example1, example2, POS, tag, sb, file):
    
    embeddings1 = sb.encode(example1, convert_to_tensor=True)
    embeddings2 = sb.encode(example2, convert_to_tensor=True)
    dot_score = util.dot_score(embeddings1, embeddings2)[0][0].item()
    dictionary = {"word": word, "POS": POS, "sentence1": example1, "sentence2": example2, "dot": dot_score, "tag": tag}
    file.write(json.dumps(dictionary, ensure_ascii=False) + "\n")


def useModels(sb,filename,words,sentences1,sentences2,POSs,tags):

    file = open(filename, "w",encoding='utf-8')
    for i in tqdm(range(len(words))):
        testModels(words[i], sentences1[i], sentences2[i], POSs[i], tags[i], sb, file)
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
    filename = "WiCOutputs/baseline/SB.json"
    words = []
    sentences1 = []
    sentences2 = []
    POSs = []
    tags = []
    WiC = open("WiC/dev/dev.data.txt","r").read().splitlines()
    gold = open("WiC/dev/dev.gold.txt","r").read().splitlines()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    for i in range(len(WiC)):
        data = processWiC(WiC[i])
        words.append(data["word"])
        sentences1.append(data["sentence1"])
        sentences2.append(data["sentence2"])
        POSs.append(data["POS"])
        tags.append(gold[i])
    filenameDev = filename[:-5] + "_dev.json"
    filenameTest = filename[:-5] + "_test.json"
    filenameResult = filename[:-5] + "_result.txt"
    sb = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    useModels(sb,filenameDev,words,sentences1,sentences2,POSs,tags)
    regr = calculateThrshold(filenameDev, "dot")
    WiC = open("WiC/test/test.data.txt","r").read().splitlines()
    gold = open("WiC/test/test.gold.txt","r").read().splitlines()
    for i in range(len(WiC)):
        data = processWiC(WiC[i])
        words.append(data["word"])
        sentences1.append(data["sentence1"])
        sentences2.append(data["sentence2"])
        POSs.append(data["POS"])
        tags.append(gold[i])    
    useModels(sb,filenameTest,words,sentences1,sentences2,POSs,tags)
    value = estimate(filenameTest, regr, "dot")
    file = open(filenameResult, "w",encoding='utf-8')
    file.write("Accuracy: "+str(value))
    file.close()
    
    
