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
    examplesAll = [sequences[i][0]['generated_text'][len(prompts[i]):] for i in range(len(prompts))]
    examplesPreprocess = []
    for examples in examplesAll:
        for example in examples.split("\n"):
            if len(example) > 0 and example[0].isnumeric():
                #print(example)
                processExample = example[3:]
                examplesPreprocess.append(processExample)
    return examplesPreprocess

def random_line(fname):
    lines = open(fname).read().splitlines()
    return rd.choice(lines)


def testModels(word, example1, example2, POS, tag, pipeline, tokenizer, sb, file, modelname, kE, few):
    
    promptdef1 = generate_promptV2(modelname,tokenizer,word,example1, few)
    promptdef2 = generate_promptV2(modelname,tokenizer,word,example1, few)
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
    prompt1 = generate_promptExampleDef(modelname,tokenizer,word,example1, def1, kE, few)
    prompt2 = generate_promptExampleDef(modelname,tokenizer,word,example2, def2, kE, few)
    examples = get_examples([prompt1, prompt2], tokenizer, pipeline)    
    prompts = []
    for example in examples:
        prompt = generate_promptV2(modelname,tokenizer,word,example, few)
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
    '''
    TODO
    Bi embedding cosine similarity kalkulatu behar die:
        -intra grupo: A taldeko eta B taldeko definizioen arteko antzekotasuna taldekoen artean
        -Grupo arteko: A taldeko eta B taldeko definizioen arteko antzekotasuna beste taldearekin alderatuz.
    '''
    defsAll = [sequences[i][0]['generated_text'][len(prompts[i]):] for i in range(len(prompts))]
    embeddings = sb.encode(defsAll, convert_to_tensor=True)
    cosIntra = 0.0
    cosExtra = 0.0
    for i in range(len(embeddings)):
        for j in range(len(embeddings[i+1:])):
            cosine_score = util.cos_sim(embeddings[i], embeddings[j])[0][0].item()
            #Extra group
            if i < len(prompts)/2 and j > len(prompts)/2:
                cosExtra += cosine_score
            #Intra group
            else:
                cosIntra += cosine_score
    cosIntra = cosIntra/(len(embeddings)/2)
    cosExtra = cosExtra/(len(embeddings)/2)
    defs1 = defsAll[:int(len(defsAll)/2)]
    defs2 = defsAll[int(len(defsAll)/2):]

                


    dictionary = {"word": word, "POS": POS, "sentence1": example1, "sentence2": example2, "defs1": defs1 , "def2": defs2, "cosine": [cosIntra, cosExtra],"tag": tag}
    file.write(json.dumps(dictionary, ensure_ascii=False) + "\n")


def useModels(pipeline,tokenizer,sb,filename,words,sentences1,sentences2,POSs,tags,modelname, kE, few):

    file = open(filename, "w",encoding='utf-8')
    for i in tqdm(range(len(words))):
        testModels(words[i], sentences1[i], sentences2[i], POSs[i], tags[i], pipeline, tokenizer, sb, file, modelname, kE,few)
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
                all_score.append(data["cosine"])
                label_all.append(tvalue)
            else:
                all_score.append(data["cosine"])
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
                all_score.append(data["cosine"])
                label_all.append(tvalue)
            else:
                all_score.append(data["cosine"])
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
    few = {}
    few["k"] = kF
    few["words"] = []
    few["definitions"] = []
    few["examples"] = []
    adiblen = kE + 1
    oxford = open("CorpusOxford.json","r").read().splitlines()
    for i in range(kF):
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
        few["examples"].append(examples)
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
    useModels(pipeline,tokenizer,sb,filenameDev,words,sentences1,sentences2,POSs,tags,modelname, kE,few)
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
    useModels(pipeline,tokenizer,sb,filenameTest,words,sentences1,sentences2,POSs,tags,modelname, kE,few)
    value = estimate(filenameTest, regr)
    file = open(filenameResult, "w",encoding='utf-8')
    file.write(str(value))
    file.close()
    
    
