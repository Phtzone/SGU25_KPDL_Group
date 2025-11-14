import pandas as pd
import random
from K_Nearest_Neighbors import K_Nearest_Neighbors as KNN

# Đọc dữ liệu
df = pd.read_csv("iris.csv")

# Một số file không có cột Id
if "Id" in df.columns:
    df.drop(["Id"], axis=1, inplace=True)

# Đưa về list
data_set = df.values.tolist()
random.shuffle(data_set)


# Lấy tất cả label đúng từ file
labels = sorted(list(set([row[-1] for row in data_set])))

train_data = {label: [] for label in labels}
test_data  = {label: [] for label in labels}

# Chia 80/20
split = int(0.8 * len(data_set))

for row in data_set[:split]:
    features = row[:-1]
    label = row[-1]
    train_data[label].append(features)

for row in data_set[split:]:
    features = row[:-1]
    label = row[-1]
    test_data[label].append(features)

# Model
knn = KNN(train_data, k=7)
knn.test(test_data)

# Demo dự đoán
sample = [5.0, 3.4, 1.5, 0.2]
pred = knn.predict(sample)
print("Predict:", pred, ", confidence =", knn.confidence)
