import librosa
import os
import json
import csv
import numpy as np

DATAET_PATH = "D:\\Schoolshit\\Thesis\\TDNN\\dataset"
JSON_PATH = "data.json"
SAMPLES_TO_CONSIDER = 22050  # 1 second at 22050 Hz
TARGET_LENGTH = 2 * SAMPLES_TO_CONSIDER  # 2 seconds
TRANSCRIPTION_PATH = "D:\\Schoolshit\\Thesis\\TDNN\\dataset_transcription\\transcription.csv"

def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):
    
    data = {
        "mappings": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    def process_csv(csv_file, data_object):
        with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                filename = row['filename']
                transcription = row['transcription']

                # Add filename to the filenames list
                data_object["files"].append(filename)

                # Check if the word is already in mappings
                if transcription not in data_object["mappings"]:
                    # If it's a new word, add it to mappings and assign a new label
                    data_object["mappings"].append(transcription)
                    data_object["labels"].append(len(data_object["mappings"]) - 1)  # Index of the new word
                else:
                    # If it's a repeating word, find its index in mappings and assign the corresponding label
                    data_object["labels"].append(data_object["mappings"].index(transcription))

    process_csv(TRANSCRIPTION_PATH, data)

    # Initialize a list to track valid filenames
    valid_files = []

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                try:
                    # Load the audio file
                    signal, sr = librosa.load(file_path, sr=SAMPLES_TO_CONSIDER)

                    # Pad or truncate the signal to the target length
                    if len(signal) < TARGET_LENGTH:
                        # Pad with zeros if the signal is shorter than 2 seconds
                        padding = TARGET_LENGTH - len(signal)
                        signal = np.pad(signal, (0, padding), mode="constant")
                    else:
                        # Truncate if the signal is longer than 2 seconds
                        signal = signal[:TARGET_LENGTH]

                    # Extract MFCCs
                    MFCCs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

                    # Append MFCCs to the data dictionary
                    data["MFCCs"].append(MFCCs.T.tolist())
                    valid_files.append(file_path)  # Track valid files
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    continue


    # Save the data to a JSON file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

    # Print the results (optional)
    #print("Mappings:", data["mappings"])
    #print("Filenames:", data["files"])
    print("Labels:", len(data["labels"]))
    print("MFCC:", len(data["MFCCs"]))

if __name__ == "__main__":
    prepare_dataset(DATAET_PATH, JSON_PATH)