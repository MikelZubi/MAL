from sentence_transformers import SentenceTransformer, util
import random as rd
import json 
import sys
from tqdm import tqdm
from sklearn import svm




def random_line(fname):
    lines = open(fname).read().splitlines()
    return rd.choice(lines)


def testModels(word, example1, example2, POS, tag, sb, file):
    
    embeddings1 = sb.encode(example1, convert_to_tensor=True)
    embeddings2 = sb.encode(example2, convert_to_tensor=True)
    cosine_score = util.cos_sim(embeddings1, embeddings2)[0][0].item()
    dictionary = {"word": word, "POS": POS, "sentence1": example1, "sentence2": example2, "cosine": cosine_score, "tag": tag}
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

def calculateThrshold(path):

    file_path = path


    all_score = []
    label_all = []

    tvalue = True
    fvalue = False
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            if data["tag"] == "T":
                all_score.append([data["cosine"]])
                label_all.append(tvalue)
            else:
                all_score.append([data["cosine"]])
                label_all.append(fvalue)
                
            

    regr = svm.SVC()
    regr.fit(all_score,label_all)
    return regr

def estimate(path, regr):
    file_path = path


    all_score = []
    label_all = []

    tvalue = True
    fvalue = False
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            if data["tag"] == "T":
                all_score.append([data["cosine"]])
                label_all.append(tvalue)
            else:
                all_score.append([data["cosine"]])
                label_all.append(fvalue)
                
            
    asm = 0.0
    for i in range(len(all_score)):
        pred = regr.predict([all_score[i]])[0]
        if pred and label_all[i]:
            asm += 1
        elif not pred and not label_all[i]:
            asm += 1
    return asm/len(all_score)



if __name__ == "__main__":
    rd.seed(16)
    filename = sys.argv[1]
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
    filenameDev = filename[:-5] + "_dev.json"
    filenameTest = filename[:-5] + "_test.json"
    filenameResult = filename[:-5] + "_result.txt"
    sb = SentenceTransformer('modeloak/all-mpnet-base-v2')
    useModels(sb,filenameDev,words,sentences1,sentences2,POSs,tags)
    regr = calculateThrshold(filenameDev)
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
    value = estimate(filenameTest, regr)
    file = open(filenameResult, "w",encoding='utf-8')
    file.write(str(value))
    file.close()
    
    
