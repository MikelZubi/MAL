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
from generatePrompt import generate_promptV2, generate_promptExampleDef
import os



def get_examples(prompts, tokenizer, pipeline):
    sequences = pipeline(
        prompts,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=140,
        batch_size = 2
    )
    examplesG1 = sequences[0][0]['generated_text'][len(prompts[0]):]
    examplesG2 = sequences[1][0]['generated_text'][len(prompts[1]):]
    examplesPreprocessG1 = []
    for example in examplesG1.split("\n"):
        if len(example) > 0 and example[0].isnumeric():
            #print(example)
            processExample = example[3:]
            examplesPreprocessG1.append(processExample)
    examplesPreprocessG2 = []
    for example in examplesG2.split("\n"):
        if len(example) > 0 and example[0].isnumeric():
            processExample = example[3:]
            examplesPreprocessG2.append(processExample)
    return examplesPreprocessG1, examplesPreprocessG2

def random_line(fname):
    lines = open(fname).read().splitlines()
    return rd.choice(lines)


def testModels(word, example1, example2, POS, tag, pipeline, tokenizer, sb, file, modelname, kE, fewN, fewV):
    
    promptdef1 = generate_promptV2(modelname,tokenizer,word,example1, POS, fewN, fewV)
    promptdef2 = generate_promptV2(modelname,tokenizer,word,example2, POS, fewN, fewV)
    sequencesdef = pipeline(
        [promptdef1, promptdef2],
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=140,
        batch_size = 2
    )
    def1 = sequencesdef[0][0]['generated_text'][len(promptdef1):]
    def2 = sequencesdef[1][0]['generated_text'][len(promptdef2):]
    prompt1 = generate_promptExampleDef(modelname,tokenizer,word,example1, def1, POS, kE, fewN, fewV)
    prompt2 = generate_promptExampleDef(modelname,tokenizer,word,example2, def2, POS, kE, fewN, fewV)
    examplesG1, examplesG2 = get_examples([prompt1, prompt2], tokenizer, pipeline)    
    prompts = []
    for example in examplesG1:
        prompt = generate_promptV2(modelname,tokenizer,word,example, POS, fewN, fewV)
        prompts.append(prompt)
    for example in examplesG2:
        prompt = generate_promptV2(modelname,tokenizer,word,example, POS, fewN, fewV)
        prompts.append(prompt)
    sequences = pipeline(
        prompts,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=140,
        batch_size = len(prompts)
    )
    defsAll = [sequences[i][0]['generated_text'][len(prompts[i]):] for i in range(len(prompts))]
    embeddings = sb.encode(defsAll, convert_to_tensor=True)
    embG1 = embeddings[:len(examplesG1)]
    embG2 = embeddings[len(examplesG1):]
    cosIntra = sum([util.cos_sim(embG1[i], embG1[j])[0][0].item() for i in range(len(examplesG1)) for j in range(i+1, len(examplesG1) )])/(len(examplesG1)) + sum([util.cos_sim(embG2[i], embG2[j])[0][0].item() for i in range(len(examplesG2)) for j in range(i+1, len(examplesG2) )])/(len(examplesG2))
    cosExtra = sum([util.cos_sim(embG1[i], embG2[j])[0][0].item() for i in range(len(examplesG1)) for j in range(len(examplesG2) )])/(len(examplesG1)*len(examplesG2))
    dotIntra = sum([util.dot_score(embG1[i], embG1[j])[0][0].item() for i in range(len(examplesG1)) for j in range(i+1, len(examplesG1) )])/(len(examplesG1)) + sum([util.dot_score(embG2[i], embG2[j])[0][0].item() for i in range(len(examplesG2)) for j in range(i+1, len(examplesG2) )])/(len(examplesG2))
    dotExtra = sum([util.dot_score(embG1[i], embG2[j])[0][0].item() for i in range(len(examplesG1)) for j in range(len(examplesG2) )])/(len(examplesG1)*len(examplesG2))
    '''
    cosIntra = 0.0
    cosExtra = 0.0
    dotIntra = 0.0
    dotExtra = 0.0
    for i in range(len(embeddings)):
        for j in range(len(embeddings[i+1:])):
            cosine_score = util.cos_sim(embeddings[i], embeddings[j])[0][0].item()
            dot_score = util.dot_score(embeddings[i], embeddings[j])[0][0].item()
            #Extra group
            if i < len(prompts)/2 and j > len(prompts)/2:
                cosExtra += cosine_score
                dotExtra += dot_score
            #Intra group
            else:
                cosIntra += cosine_score
                dotIntra += dot_score
    '''
    defs1 = defsAll[:len(examplesG1)]
    defs2 = defsAll[len(examplesG1):]

                


    dictionary = {"word": word, "POS": POS, "sentence1": example1, "sentence2": example2, "defs1": defs1 , "def2": defs2, "cosine": [cosIntra, cosExtra], "dot": [dotIntra, dotExtra],"tag": tag}
    file.write(json.dumps(dictionary, ensure_ascii=False) + "\n")


def useModels(pipeline,tokenizer,sb,filename,words,sentences1,sentences2,POSs,tags,modelname, kE, fewN, fewV):

    file = open(filename, "w",encoding='utf-8')
    for i in tqdm(range(len(words))):
        testModels(words[i], sentences1[i], sentences2[i], POSs[i], tags[i], pipeline, tokenizer, sb, file, modelname, kE, fewN, fewV)
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
                all_score.append(data[key])
                label_all.append(tvalue)
            else:
                all_score.append(data[key])
                label_all.append(fvalue)
                
            

    regr = svm.SVC()
    regr.fit(all_score,label_all)
    return regr

def estimate(path, regr, key):
    file_path = path


    all_score = []
    label_all = []

    tvalue = True
    fvalue = False
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            if data["tag"] == "T":
                all_score.append(data[key])
                label_all.append(tvalue)
            else:
                all_score.append(data[key])
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
    modelname = sys.argv[1]
    kE = int(sys.argv[2])
    kF = int(sys.argv[3])
    with open("modelsData.json", "r") as jsonfile:
        modelsdata = json.load(jsonfile)
        modelpath = modelsdata[modelname]["path"]
    filename = "WiCOutputs/" + str(kF) + "Shot"+str(kE)+"ExamplesDef/" + modelname + ".json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    adiblen = kE + 1
    fewN = {}
    fewN["k"] = kF
    fewN["words"] = []
    fewN["definitions"] = []
    fewN["examples"] = []
    nouns = open("polysemicNouns.json","r").read().splitlines()
    for i in range(kF):
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
    fewV["k"] = kF
    fewV["words"] = []
    fewV["definitions"] = []
    fewV["examples"] = []
    verbs = open("polysemicVerbs.json","r").read().splitlines()
    for i in range(kF):
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
    filenameDev = filename[:-5] + "_dev.json"
    filenameTest = filename[:-5] + "_test.json"
    filenameResult = filename[:-5] + "_result.txt"
    tokenizer = AutoTokenizer.from_pretrained(modelpath)
    sb = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    pipeline = transformers.pipeline(
        "text-generation",
        model=modelpath,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id # Hack to fix a bug in transformers (batch_size)
    useModels(pipeline,tokenizer,sb,filenameDev,words,sentences1,sentences2,POSs,tags,modelname, kE,fewN, fewV)
    regrCos = calculateThrshold(filenameDev, "cosine")
    regrDot = calculateThrshold(filenameDev, "dot")
    WiC = open("WiC/test/test.data.txt","r").read().splitlines()
    gold = open("WiC/test/test.gold.txt","r").read().splitlines()
    for i in range(len(WiC)):
        data = processWiC(WiC[i])
        words.append(data["word"])
        sentences1.append(data["sentence1"])
        sentences2.append(data["sentence2"])
        POSs.append(data["POS"])
        tags.append(gold[i])    
    useModels(pipeline,tokenizer,sb,filenameTest,words,sentences1,sentences2,POSs,tags,modelname, kE,fewN, fewV)
    valueCos = estimate(filenameTest, regrCos, "cosine")
    valueDot = estimate(filenameTest, regrDot, "dot")
    file = open(filenameResult, "w",encoding='utf-8')
    file.write("Value Cosine: " +  str(valueCos) + "\n")
    file.write("Value Dot: " +  str(valueDot) + "\n")
    file.close()
    
    
