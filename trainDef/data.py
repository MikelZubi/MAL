import json
from generatePrompt import generate_promptV2
from transformers import AutoTokenizer


path = "CorpusOxford.json"
data = []
modelpath = "meta-llama/Meta-Llama-3-8B"
modelname = "Llama3"
tokenizer = AutoTokenizer.from_pretrained(modelpath)
with open(path) as file:
    for line in file:
        word = json.loads(line)
        if word["POS"] == "noun" or word["POS"] == "verb" and len(word["examples"]) >= 1:
            example = word["examples"][0]
            definition = word["definition"]
            pos = word["POS"]
            few = {"k":0}
            prompt = generate_promptV2(modelname, tokenizer, word["word"], example, pos,few, few)
            data.append({"prompt":prompt, "text":word["definition"]})
writepath = "trainDef.json"
with open(writepath, "w") as file:
    for line in data:
        linejson = json.dump(line, ensure_ascii=False)
        file.write(linejson+"\n")





        