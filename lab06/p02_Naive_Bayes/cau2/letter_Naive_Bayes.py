import sys, os
import pandas as pd 
import random
sys.path.append(os.path.abspath(".."))

from Naive_Bayes import Naive_Bayes
# Data
df = pd.read_csv("data/letter-recognition.data", header=None)

# Shuffle
data_set = df.values.tolist()
random.shuffle(data_set)

# Chia train/test (16000 train, 4000 test)
train_data = pd.DataFrame(data_set[:16000])
test_data  = pd.DataFrame(data_set[16000:])

# Model Naive Bayes
nb = Naive_Bayes(train_data)
nb.test(test_data)