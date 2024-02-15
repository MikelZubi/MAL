from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import transformers
import torch
import random as rd
import json 
import sys
from tqdm import tqdm
from sklearn import svm
import time
from generatePrompt import generate_prompt, generate_promptv2



def random_line(fname):
    lines = open(fname).read().splitlines()
    return rd.choice(lines)


def testModels(word, example1, example2, POS, tag, pipeline, tokenizer, sb, file, modelname, few):
    
    
    prompt1 = generate_promptv2(modelname,tokenizer,word,example1,few)
    prompt2 = generate_promptv2(modelname,tokenizer,word,example2,few)        
    sequences = pipeline(
        [prompt1,prompt2],
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=1000,
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
    cosine_score = util.cos_sim(embeddings1, embeddings2)[0][0].item()
    dictionary = {"word": word, "POS": POS, "sentence1": example1, "sentence2": example2, "def1": def1 , "def2": def2, "cosine": cosine_score, "tag": tag}
    file.write(json.dumps(dictionary, ensure_ascii=False) + "\n")


def useModels(pipeline,tokenizer,sb,filename,words,sentences1,sentences2,POSs,tags,modelname,few):

    file = open(filename, "w",encoding='utf-8')
    for i in tqdm(range(len(words))):
        testModels(words[i], sentences1[i], sentences2[i], POSs[i], tags[i], pipeline, tokenizer, sb, file, modelname, few)
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
    model = sys.argv[1]
    filename = sys.argv[2]
    modelname = sys.argv[3]
    k = int(sys.argv[4])
    few = {}
    few["k"] = k
    few["words"] = []
    few["definitions"] = []
    few["examples"] = []
    adiblen = 1
    oxford = open("CorpusOxford.json","r").read().splitlines()
    for i in range(k):
        line = rd.choice(oxford)
        dataR = json.loads(line)
        examples = dataR["examples"]
        pos = dataR["POS"]
        while len(examples) <= adiblen or "Circular definition $REF:$" in dataR["definition"] or "Definition not found $REF:$" in dataR["definition"]: 
            line = rd.choice(oxford)
            dataR = json.loads(line)
            examples = dataR["examples"]
        word = dataR["word"]
        definizioa = dataR["definition"]
        few["words"].append(word)
        few["definitions"].append(definizioa)
        few["examples"].append(examples[0])
        oxford.remove(line)
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
    if "zephyr" in model:
        zephyr = True
    else:
        zephyr = False
    filenameDev = filename[:-5] + "_dev.json"
    filenameTest = filename[:-5] + "_test.json"
    filenameResult = filename[:-5] + "_result.txt"
    tokenizer = AutoTokenizer.from_pretrained(model)
    sb = SentenceTransformer('modeloak/all-mpnet-base-v2')
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    useModels(pipeline,tokenizer,sb,filenameDev,words,sentences1,sentences2,POSs,tags,modelname,few)
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
    useModels(pipeline,tokenizer,sb,filenameTest,words,sentences1,sentences2,POSs,tags,modelname,few)
    value = estimate(filenameTest, regr)
    file = open(filenameResult, "w",encoding='utf-8')
    file.write(str(value))
    file.close()
    
    
