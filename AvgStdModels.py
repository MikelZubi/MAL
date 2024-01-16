import os
import csv
import numpy as np
import json
import sys
# Directory path

def luzera(x):
    if len(x)> 13:
        return(int(x[-7:-5]))
    return int(x[-6])
def main(directory = "ModelsOutputs",output_file = "ModelsOutputs/results.csv"):

    # List to store cosine values
    cosine_values = []
    avgs_values = []
    mover_values = []
    bleu_values = []

    # Read files in the directory
    file = open(output_file, "w", newline="")
    writer = csv.writer(file)
    writer.writerow(["Models","Avg Max", "Std Max", "Avg Avg", "Std Avg", "Avg Mover", "Std Mover", "Avg BLEU", "Std BLEU"])
    filenames = [dir for dir in os.listdir(directory) if dir[-5:] == ".json"]
    if "AdibKop" in directory:
        filenames.sort(key=luzera)
    for filename in filenames:
        if filename[-4:] != "json":
            continue
        filepath = os.path.join(directory, filename)
        
        # Read file and calculate cosine value
        with open(filepath, "r") as file:
            lines = file.readlines()
            for line in lines:
                # Calculate cosine value using your desired method
                data = json.loads(line)
                cosine_values.append(data["cosine"])
                avgs_values.append(data["avg"])
                mover_values.append(data["MoverScore"])
                bleu_values.append(data["BLEU"])


        # Calculate average and standard deviation
        avgC = np.mean(cosine_values)
        stdC = np.std(cosine_values)
        avgA = np.mean(avgs_values)
        stdA = np.std(avgs_values)
        avgM = np.mean(mover_values)
        stdM = np.std(mover_values)
        avgB = np.mean(bleu_values)
        stdB = np.std(bleu_values)
        cosine_values.clear()
        avgs_values.clear()
        mover_values.clear()
        bleu_values.clear()
        # Save results in a CSV file
        
        row = [filename, avgC, stdC, avgA, stdA, avgM, stdM, avgB, stdB]
        writer.writerow(row)
if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    else:
        main()