import os
import csv

shots = [0,1,2,3,4,5,6,7]
models = ["Llama2", "Mistral", "MistralDPO", "ZephyrB", "OpenChat"]
# Create a list to store the results
results = []

# Iterate over the shots
for shot in shots:
    # Create a dictionary to store the results for each shot
    shot_results = {"Shot": shot}
    
    # Find the folders in WiCOutputs that start with the shot number
    folder_path = f"WiCOutputs/{shot}Shot"
    
    # Iterate over the models
    for model in models:
        # Find the file that has the model name and "_results.txt" in the folder
        file_name = f"{model}_result.txt"
        file_path = None
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_path = os.path.join(folder_path, file_name)
            # Read the file and save the contents
            if file_path:
                with open(file_path, "r") as file:
                    contents = file.readlines()
                    acc = contents[0][12:-1]
                    shot_results[model] = acc
            else:
                shot_results[model] = "N/A"
    
    # Append the shot results to the overall results list
    results.append(shot_results)
    print(results)

# Save the results to a CSV file
csv_file = "WiCOutputs/definition.csv"
fieldnames = ["Shot"] + models
with open(csv_file, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

# Iterate over the shots
for shot in shots:
    # Create a dictionary to store the results for each shot
    shot_results = {"Shot": shot}
    
    # Find the folders in WiCOutputs that start with the shot number
    folder_path = f"WiCOutputs/{shot}Shot"
    
    # Iterate over the models
    for model in models:
        # Find the file that has the model name and "_results.txt" in the folder
        file_name = f"{model}_result.txt"
        file_path = None
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_path = os.path.join(folder_path, file_name)
            # Read the file and save the contents
            if file_path:
                with open(file_path, "r") as file:
                    contents = file.readlines()
                    acc = contents[1][22:-1]
                    shot_results[model] = acc
            else:
                shot_results[model] = "N/A"
    
    # Append the shot results to the overall results list
    results.append(shot_results)
    print(results)

# Save the results to a CSV file
csv_file = "WiCOutputs/definition+Context.csv"
fieldnames = ["Shot"] + models
with open(csv_file, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)