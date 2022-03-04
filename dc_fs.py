import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
import pingouin as pin

# [[ Get Dataset ]]
diabetes = load_diabetes()
data, y = diabetes.data, diabetes.target

# Distance Correlation
def distance_cor(data, y):
    return pin.distance_corr(data, y)

cor_scores = []
for feat_no in range(data.shape[1]):
    cor_scores.append(distance_cor(data[:, feat_no], y))

cor_scores = np.array(cor_scores)

fig, ax = plt.subplots()
ax.bar(range(len(cor_scores)), cor_scores[:,0])
# ax.set_xlim(0, len(cor_scores)+2) # Needs fixing, but check if bar graph is really required
plt.show()


# Forward feature selection as per cor scores
# If it is a bad score, then eleminate and keep accounting for false selection rate

# While concatenating every feature as per the cor score, see if classifier doesn't help, count and plot that difference.