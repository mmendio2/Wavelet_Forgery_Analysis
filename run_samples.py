from authenticity import calculate_authenticity
import json
import os
import numpy as np
from scipy.stats import wilcoxon

# This is where you set the main path of where the images are held
MAIN_PATH = fr"{os.getcwd()}\Signatures\\Signatures_"

# open known corpus of participant 0
f_known = open(fr"{MAIN_PATH}P0\Sig_P_0.json", "r")
known = json.load(f_known)

# open baseline corpus for known participant 0
f_base = open(fr"{MAIN_PATH}P0_Base\Sig_P0_Base.json", "r")
base = json.load(f_base)

# calculate number of matches for each signature in baseline
base_authentic_list = []
for b in base:
    authentic = 0
    for k in known:
        if calculate_authenticity(np.array(b["I_matrix"]), np.array(k["I_matrix"])):
            authentic += 1
    base_authentic_list.append(authentic)
print(f"Base List: {base_authentic_list}")

# iterate though unknown participants we are testing
for i in range(1, 32):
    # calculate number of matches for participant i against known corpus
    print(f"P{i}:")
    f_test = open(rf"{MAIN_PATH}P{i}\Sig_P{i}.json", "r")
    test = json.load(f_test)
    test_authentic_list = []
    for t in test:
        authentic = 0
        for k in known:
            if calculate_authenticity(np.array(t["I_matrix"]), np.array(k["I_matrix"])):
                authentic += 1
        test_authentic_list.append(authentic)

    print(test_authentic_list)

    # compare test corpus to baseline corpus
    print(wilcoxon(base_authentic_list, test_authentic_list))
    f_test.close()


f_known.close()
f_base.close()