from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import transformers
import torch
import random as rd
import json 
import sys
from tqdm import tqdm
import copy as cp
from generatePrompt import generate_prompt



def random_line(fname):
    lines = open(fname).read().splitlines()
    return rd.choice(lines)


def testModels(word, definizioa, examples, POS, pipeline, tokenizer, sb, file, modelname, few):
    
    max = -1
    maxOutput = ""
    maxExample = ""
    maxPrompt = ""
    med = 0
    prompts = []
    outputs = []
    for example in examples:

        prompt = generate_prompt(modelname,word,example,few)
        prompts.append(prompt)
    sequences = pipeline(
        prompts,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=140,
        batch_size=len(prompts),
        truncation=True
    )
    for i in range(len(sequences)):
        seq = sequences[i]
        sentences = seq[0]['generated_text']
        output = sentences[len(prompts[i]):]
        outputs.append(output)
    embeddingsO = sb.encode(outputs, convert_to_tensor=True)
    embeddingsD = sb.encode(definizioa, convert_to_tensor=True)
    cosine_score = util.cos_sim(embeddingsO, embeddingsD)
    for i in range(len(cosine_score)):
        score = cosine_score[i][0].item()
        med += score
        if max < score:
            max = score
            maxOutput = outputs[i]
            maxExample = examples[i]
            maxPrompt = prompts[i]
    dictionary = {"word": word, "POS": POS,"definition": definizioa, "examples": maxExample, "prompt": maxPrompt, "output": maxOutput, "cosine": max, "bestExample": (examples.index(maxExample),len(examples)), "avg": med/len(examples)}
    file.write(json.dumps(dictionary, ensure_ascii=False) + "\n")

def useModels(model,filename,words,defs,expls,POSs,luz,modelname, few):
    tokenizer = AutoTokenizer.from_pretrained(model)
    sb = SentenceTransformer('modeloak/all-mpnet-base-v2')
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id
    file = open(filename, "w",encoding='utf-8')
    for i in tqdm(range(luz)):
        testModels(words[i], defs[i], expls[i], POSs[i] ,pipeline, tokenizer, sb, file, modelname, few)
    file.close()


if __name__ == "__main__":
    rd.seed(16)
    model = sys.argv[1]
    filename = sys.argv[2]
    fewShot = False
    few = {}
    k = int(sys.argv[4])
    modelname = sys.argv[3]
    few["k"] = k
    few["words"] = []
    few["definitions"] = []
    few["examples"] = []
    words = []
    defs = []
    expls = []
    POSs = []
    luz = 500
    adiblen = 1
    oxford = open("CorpusOxford.json","r").read().splitlines()
    for i in range(luz + k):
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
        if i < luz:
            words.append(word)
            defs.append(definizioa)
            expls.append(examples)
            POSs.append(pos)
        else:
            few["words"].append(word)
            few["definitions"].append(definizioa)
            few["examples"].append(examples[0])
        oxford.remove(line)

    print("Adibideak barrun")
    useModels(model,filename,words,defs,expls,POSs,luz,modelname,few)
    

    
    
