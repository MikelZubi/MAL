import os
import sys
import sys
import json
import csv
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

def estimate(path, regr):
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
                
    fp = 0.0
    fn = 0.0
    accu = 0.0
    for i in range(len(all_score)):
        pred = regr.predict([all_score[i]])[0]
        if pred != label_all[i]:
            if pred:
                fp +=1
            if not pred:
                fn +=1
        else:
            accu +=1
    return fp/len(label_all), fn/len(label_all), accu/len(label_all)

modelname = sys.argv[1]
pathOut = sys.argv[2]
csvfile = open(pathOut, "w")
csvwriter = csv.writer(csvfile)
csvwriter.writerow(["Type", "FP", "FN", "Accuracy"])
dirs = [dir for dir in os.listdir("WiCOutputs") if os.path.isdir(os.path.join("WiCOutputs", dir))]
dirs.sort()
for dir in dirs:
    modelIn = False
    path = os.path.join("WiCOutputs", dir)
    for file in os.listdir(path):
        if modelname in file:
            modelIn = True
            break
    if not modelIn:
        continue
    filenameDev = path + "/" + modelname + "_dev.json"
    svmModel = calculateThrshold(filenameDev)
    filenameTest = path + "/" +modelname + "_test.json"
    fn, fp, accu = estimate(filenameTest, svmModel)
    csvwriter.writerow([dir, fp, fn, accu])
csvfile.close()
