import json

#Few False or a dictionary with k, words, examples and definitions to do the few-shot prompt.
def generate_prompt(modelname, word, example, few = {"k":0}):
    f = open('ModelsPrompts.json')
    data = json.load(f)[modelname]
    if data["type"] == "Instruct":
        init = ""
        for i in range(few["k"]):
            few = data["prompt"]["inst_start"]+"As an expert English lexicographer generate a dictionary definition of the word '" + few["words"][i] + "' in the sense of this example '" + few["examples"][i] + "'. Give JUST the definition not other things." + data["prompt"]["inst_end"]
            init += few + data["prompt"]["response_start"] + few["definitions"][i] + data["prompt"]["response_end"]
        last = data["prompt"]["inst_start"]+"As an expert English lexicographer generate a dictionary definition of the word '" + word + "' in the sense of this example '" + example + "'. Give JUST the definition not other things." + data["prompt"]["inst_end"]
        prompt = init + last + data["prompt"]["response_start"]
        return prompt
    elif data["type"] == "Chat":
        init = data["prompt"]["init_start"] + "You are an expert English lexicographer, generate a dictionary definition of a word given some example sentences of the word. Please, JUST provide the definition, not further explanation." + data["prompt"]["init_end"]
        for i in range(few["k"]):
            user = data["prompt"]["user_start"] + "Giving the word '" + few["words"][i] + "' and the sense of this example: '" + few["examples"][i] + "', generate the definition of the word in this sense. Give JUST the definition not further explanation." + data["prompt"]["user_end"]
            init += user + data["prompt"]["response_start"] + few["definitions"][i] + data["prompt"]["response_end"]
        user = data["prompt"]["user_start"] + "Giving the word '" + word + "' and the sense of this example: '" + example + "', generate the definition of the word in this sense. Give JUST the definition not further explanation." + data["prompt"]["user_end"]
        prompt = init + user + data["prompt"]["response_start"]
        return prompt
    else:
        #Only chat or instruct models
        assert False



def generate_promptV2(tokenizer, modelname,word, example, few = {"k":0}):
    f = open('ModelsPrompts.json')
    data = json.load(f)[modelname]
    if data["type"] == "Instruct":
        chat = []
        for i in range(few["k"]):
            chat.append({"role": "user", "content": "As an expert English lexicographer generate a dictionary definition of the word '" + few["words"][i] + "' in the sense of this example '" + few["examples"][i] + "'. Give JUST the definition not other things."})
            
        chat.append({"role": "user", "content": "As an expert English lexicographer generate a dictionary definition of the word '" + word + "' in the sense of this example '" + example + "'. Give JUST the definition not other things."})
    elif data["type"] == "Chat":
        chat = [{"role": "system", "content": "You are an expert English lexicographer, generate a dictionary definition of a word given some example sentences of the word. Please, JUST provide the definition, not further explanation."}]
        for i in range(few["k"]):
            chat.append({"role": "user", "content": "Giving the word '" + few["words"][i] + "' and the sense of this example: '" + few["examples"][i] + "', generate the definition of the word in this sense. Give JUST the definition not further explanation."})
            chat.append({"role": "assistant", "content": few["definitions"][i]})
        chat.append({"role": "user", "content": "Giving the word '" + word + "' and the sense of this example: '" + example + "', generate the definition of the word in this sense. Give JUST the definition not further explanation."})
    
    else:
        assert False
    prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    return prompt