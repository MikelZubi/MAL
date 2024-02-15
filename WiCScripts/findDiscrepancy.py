import os
import json
import csv
from sklearn import svm

def calculateThrshold(path):

    dev_path = path[:-9] + "dev.json"
    file_path = dev_path


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

def estimate(filename1, filename2):
    regr1 = calculateThrshold(filename1)
    regr2 = calculateThrshold(filename2)
    file1 = open(filename1,"r").readlines()
    file2 = open(filename2,"r").readlines()
    discrp = 0.0
    for i in range(len(file1)):
        data1 = json.loads(file1[i])
        data2 = json.loads(file2[i])
        pred1 = regr1.predict([[data1["cosine"]]])[0]
        pred2 = regr2.predict([[data2["cosine"]]])[0]
        if pred1 != pred2:
            discrp += 1

    print(discrp/len(file1))
    return discrp/len(file1)


onlyExpls = "WiCOutputs/OnlyExamples/SB_test.json"
onlyDefsZ = "WiCOutputs/ZeroShot/ZephyrB_test.json"
defsExplsZ = "WiCOutputs/ZeroShot+Exp/ZephyrB_test.json"
onlyDefsF = "WiCOutputs/FewShot/ZephyrB_test.json"
defsExplsF = "WiCOutputs/FewShot+Exp/ZephyrB_test.json"

file = "WiCOutputs/discrepancyZephyr.csv"
with open(file, "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Model", "Only Examples", "Zero Shot", "Zero Shot + Exp", "Few Shot", "Few Shot + Exp"])
    writer.writerow(["Only Examples", 0, estimate(onlyExpls, onlyDefsZ), estimate(onlyExpls, defsExplsZ), estimate(onlyExpls, onlyDefsF), estimate(onlyExpls, defsExplsF)])
    writer.writerow(["Zero Shot", estimate(onlyExpls, onlyDefsZ), 0, estimate(onlyDefsZ, defsExplsZ), estimate(onlyDefsZ, onlyDefsF), estimate(onlyDefsZ, defsExplsF)])
    writer.writerow(["Zero Shot + Exp", estimate(onlyExpls, defsExplsZ), estimate(onlyDefsZ, defsExplsZ), 0, estimate(defsExplsZ, onlyDefsF), estimate(defsExplsZ, defsExplsF)])
    writer.writerow(["Few Shot", estimate(onlyExpls, onlyDefsF), estimate(onlyDefsZ, onlyDefsF), estimate(defsExplsZ, onlyDefsF), 0, estimate(onlyDefsF, defsExplsF)])
    writer.writerow(["Few Shot + Exp", estimate(onlyExpls, defsExplsF), estimate(onlyDefsZ, defsExplsF), estimate(defsExplsZ, defsExplsF), estimate(onlyDefsF, defsExplsF), 0])



