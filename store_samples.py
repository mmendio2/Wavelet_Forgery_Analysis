from authenticity import I_matrix
import os
import json
from PIL import Image
import numpy as np

# This process is run individually, as to iterate through each participant
# would take hours
participantNumber = "P0_T5"
# This is where you set the main path of where the images are held
MAIN_PATH = fr"{os.getcwd()}\Signatures_{participantNumber}"
list = []

# Run through all images and calculate their I-matrix
for i, filename in enumerate(os.listdir(MAIN_PATH)):
    print(i)
    image = Image.open(fr"{MAIN_PATH}\{filename}").convert('L')
    I = I_matrix(np.asarray(image))
    to_store = {
        "id": i,
        "I_matrix": I.tolist()
    }
    list.append(to_store)

# Write all matrix values as a json
with open(fr"{MAIN_PATH}\Sig_{participantNumber}.json", "w") as outfile:
    outfile.write(json.dumps(list))