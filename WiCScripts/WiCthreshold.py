import json
from sklearn import svm
file_path = "WiCOutputs/FewShot/ZephyrB.json"


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

thrshld = 0.5
asm = 0
for i in range(len(all_score)):
    pred = regr.predict([all_score[i]])[0]
#    score = all_score[i][0]
    if pred and label_all[i]:
        asm += 1
    elif not pred and not label_all[i]:
        asm += 1
print(asm/len(all_score))
print(asm)

