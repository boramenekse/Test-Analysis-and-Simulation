import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from collections import Counter
from scipy.optimize import curve_fit

# Define the quadratic function
def quadratic_function(x, a, b, c):
    return a * x ** 2 + b * x + c

# Open the files
with open('A1/MMA B1 C0.txt', 'r') as f:
    contents = f.readlines()

with open('results/MMA_uniform/B1/B1_crack_lengths.txt', 'r') as g:
    cracklengths = g.readlines()

# Extract crack lengths
second_line = cracklengths[1].strip()
entries = [round(float(x), 4) for x in second_line.split()]

# Convert first line to a list of index integers
first_line = cracklengths[0].strip()
indices = [int(float(x)) if isinstance(x, str) else int(x) for x in first_line.split()]

# Find which integers are not used in indices list
missing_indices = [i for i in range(min(indices), max(indices) + 1) if i not in indices]
print("Missing indices:", missing_indices)

# Normalize missing indices
normalized_indices = [(x - min(indices)) for x in missing_indices]
n_indices = [(x - min(indices)) for x in indices]

counted_list = Counter(n_indices)
repeating = [number for number, count in counted_list.items() if count > 1]
print("Repeating entries:", repeating)

# Remove the entries from txt file that are not used
for i in repeating:
    entries.pop(i)

# Extract force and displacement data
force = []
displacement = []

for line in contents[:]:
    cols = line.split('\t')
    force.append(float(cols[2].replace(',', '.')) * 40)
    displacement.append(float(cols[3].replace(',', '.')) * 0.02)

# Remove corresponding entries in force and displacement
normalized_indices.sort(reverse=True)
for i in normalized_indices:
    del displacement[i]
    del force[i]

# Calculate compliance
width = 0.025
displacement1 =  [(x - min(displacement)) for x in displacement]

compliance = [d / f for d, f in zip(displacement1, force)]

# Perform linear regression
reg = LinearRegression()
X = np.array(entries).reshape(-1, 1)
y = np.array(compliance)
reg.fit(X, y)
c_pred = reg.predict(X)
print("Linear regression coefficients:", reg.coef_)

# Perform quadratic regression
popt, pcov = curve_fit(quadratic_function, entries, compliance)
a, b, c = popt
print("Quadratic regression coefficients:", a, b, c)

# Calculate dc/da
gradient = reg.coef_
gradient2 = [2 * a * x + b for x in entries]

# Calculate energy release rate 
err = []
for i in range(len(force)):
    errx = 1 / 2 * force[i] ** 2 / width * gradient2[i] / 1000 # Convert J to kJ
    err.append(errx)

# Plot data
plt.figure()
plt.plot(entries, err)
plt.xlim(0, np.max(entries)+0.01)
plt.show()
