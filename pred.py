import pandas as pd

predict = pd.read_pickle("st199007-Jason-Holt-WhatAreLogits-old.pkl")

class1 = 0
class0 = 0
for i in predict["predictions"]["predictions"]:
    if i > 0.5:
        class1 += 1
    else:
        class0 += 1

print(predict["predictions"])
print(f"Class 1 count: {class1}")
print(f"Class 0 count: {class0}")
