from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import transformers
import torch
import random as rd
import json 
import sys
from tqdm import tqdm



def random_line(fname):
    lines = open(fname).read().splitlines()
    return rd.choice(lines)


def testModels(word, definizioa, examples, POS, pipeline, tokenizer, sb, file, zeph, fewShot):
    
    hitza = "'" + word + "'"
    adibideak = ""
    max = -1
    maxOutput = ""
    maxExample = ""
    maxPrompt = ""
    med = 0
    prompts = []
    outputs = []
    for example in examples:

        adibideak = "'" + example + "', "
        
        if zeph:
            if fewShot:
                few1 = "<|system|>\nYou are an expert English lexicographer, generate a dictionary definition of a word given some example sentences of the word. Please, JUST provide the definition, not further explanation. </s>\n"
                few2 = "<|user|>\nGiving the word 'bank' and the sense of this example: 'Russian villages were typically situated near the bank of a lake, river, or stream', generate the definition of the word in this sense.</s>\n"
                few3 = "<|assistant|>\nThe land alongside or sloping down to a river or lake.\n"
                few4 = "<|user|>\nGiving the word 'tire' and the sense of this example: 'the training tired us out', generate the definition of the word in this sense.</s>\n"
                few5 = "<|assistant|>\nFeel or cause to feel in need of rest or sleep.\n"
                few6 = "<|user|>\nGiving the word " + hitza + " and the sense of these examples: " + adibideak + "generate the definition of the word in this sense.</s>\n<|assistant|>\n"
                prompt = few1 + few2 + few3 + few4 + few5 + few6
            else:
                lehenZatia = "<|system|>\nYou are an expert English lexicographer, generate a dictionary definition of a word given some example sentences of the word. Please, JUST provide the definition, not further explanation. </s>\n"
                azkenZatia = "<|user|>\nGiving the word " + hitza + " and the sense of these examples: " + adibideak + "generate the definition of the word in this sense. Give JUST the definition not further explanation. </s>\n<|assistant|>\n"
                prompt = lehenZatia + azkenZatia
        else:
            if fewShot:
                azkenZatia = ". Give JUST the definition not other things."
                few1 = "<s> [INST] As an expert English lexicographer generate a dictionary definition of the word 'bank' in the sense of this examples 'Russian villages were typically situated near the bank of a lake, river, or stream'" + azkenZatia + " [/INST] "
                few2 = "The land alongside or sloping down to a river or lake.<\s> "
                few3 = "[INST] As an expert English lexicographer generate a dictionary definition of the word " + hitza + " in the sense of this examples " + adibideak[:-2] + azkenZatia + " [/INST] "
                prompt = few1 + few2 + few3
            else:
                azkenZatia = ". Give JUST the definition not other things."
                prompt = "<s> [INST] As an expert English lexicographer generate a dictionary definition of the word " + hitza + " in the sense of this examples " + adibideak[:-2] + azkenZatia + " [/INST] "
        
        prompts.append(prompt)
    sequences = pipeline(
        prompts,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=1000,
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

def useModels(model,filename,words,defs,expls,POSs,luz,zephyr, few):
    tokenizer = AutoTokenizer.from_pretrained(model)
    sb = SentenceTransformer('modeloak/all-mpnet-base-v2')
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    file = open(filename, "w",encoding='utf-8')
    for i in tqdm(range(luz)):
        testModels(words[i], defs[i], expls[i], POSs[i] ,pipeline, tokenizer, sb, file, zephyr, few)
    file.close()


if __name__ == "__main__":
    rd.seed(16)
    #model = ["modeloak/Mistral-7B-Instruct-v0.1", "modeloak/zephyr-7b-alpha", "modeloak/zephyr-7b-beta"]
    model = sys.argv[1]
    # filename = ["ModelsOutputs/Mistral.json","ModelsOutputs/ZephyrA.json","ModelsOutputs/ZephyrB.json"]
    filename = sys.argv[2]
    fewShot = False
    if sys.argv[3] == "Few":
        fewShot = True
    words = []
    defs = []
    expls = []
    POSs = []
    luz = 500
    adiblen = 1
    oxford = open("CorpusOxford.json","r").read().splitlines()
    for i in range(luz):
        line = rd.choice(oxford)
        index = oxford.index(line)
        dataR = json.loads(line)
        examples = dataR["examples"]
        pos = dataR["POS"]
        while len(examples) <= adiblen or "Circular definition $REF:$" in dataR["definition"] or "Definition not found $REF:$" in dataR["definition"]: 
            line = rd.choice(oxford)
            dataR = json.loads(line)
            examples = dataR["examples"]
        word = dataR["word"]
        definizioa = dataR["definition"]
        words.append(word)
        defs.append(definizioa)
        expls.append(examples)
        POSs.append(pos)
    print("Adibideak barrun")
    if "zephyr" in model:
        zephyr = True
    else:
        zephyr = False
    useModels(model,filename,words,defs,expls,POSs,luz,zephyr,fewShot)
    

    
    
