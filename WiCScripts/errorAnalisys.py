import sys
import json
from sklearn import svm

def calculateThrshold(file_path):
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

def estimate(path, regr, pathOut, k):
    file_path = path


    all_score = []
    label_all = []

    tvalue = True
    fvalue = False
    all_data = []
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            all_data.append(data)
            if data["tag"] == "T":
                all_score.append([data["cosine"]])
                label_all.append(tvalue)
            else:
                all_score.append([data["cosine"]])
                label_all.append(fvalue)
                
    fail = 0
    fileW = open(pathOut, "w")
    for i in range(len(all_score)):
        pred = regr.predict([all_score[i]])[0]
        if pred != label_all[i]:
            fail +=1
            y = {"CorrectTag": str(label_all[i]), "PredictedTag": str(pred), "Word":all_data[i]["word"],"Sentence1": all_data[i]["sentence1"], "Sentence2": all_data[i]["sentence2"], "Def1":all_data[i]["def1"], "Def2":all_data[i]["def2"], "cosine": all_data[i]["cosine"]}
            fileW.write(json.dumps(y, ensure_ascii=False) + "\n")
            if fail == k:
                break
    file.close()
    

filename = sys.argv[1]
pathOut = sys.argv[2]
k = int(sys.argv[3])

filenameDev = filename[:-5] + "_dev.json"
svm = calculateThrshold(filenameDev)
filenameTest = filename[:-5] + "_test.json"
estimate(filenameTest, svm, pathOut, k)



#TODO Hartu bat eta aukeratu 20 adibide eztienak zuzenak jasotzeko error analisys-eko fitxategi batean