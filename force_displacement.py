import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from collections import Counter
from scipy.optimize import curve_fit

# Define the quadratic function
def quadratic_function(x, a, b, c, d):
    return a * x ** 3 + b * x**2 + c*x + d

# Open the files
with open('A1/MMA B1 C0.txt', 'r') as f, \
     open('results/MMA_uniform/B1/B1_crack_lengths_all.txt', 'r') as g, \
     open('results/MMA_uniform/B1/B1_filtered_files.txt', 'r') as h:
    
    contents = f.readlines()
    cracklengths = g.readlines()
    missing_indices = h.readlines()

missing_indices = [np.int(float(x)) if isinstance(x, str) else np.int(x) for x in missing_indices]
normalized_indices = [(x - np.min(missing_indices)) for x in missing_indices]

# Extract crack lengths
second_line = cracklengths[1].strip()
entries = [np.round(np.float_(x), 4) for x in second_line.split()]

# Convert first line to a list of index integers
first_line = cracklengths[0].strip()
indices = [np.int_(float(x)) if isinstance(x, str) else np.int_(x) for x in first_line.split()]

# Find which integers are not used in indices list
missing_indices = [i for i in range(np.min(indices), np.max(indices)) if i not in indices]

# Normalize missing indices
normalized_indices = [(x - np.min(missing_indices)) for x in missing_indices]

counted_list = Counter(indices)
repeating = [number for number, count in counted_list.items() if count > 1]
repeating.sort(reverse=True)

# Remove the entries from txt file that are not used
for i in repeating:
    entries.pop(i - np.min(indices))

# Extract force and displacement data
force = [np.float_(line.split('\t')[2].replace(',', '.')) * 40 for line in contents]
displacement = [np.float_(line.split('\t')[3].replace(',', '.')) * 0.02 for line in contents]

# Remove corresponding entries in force and displacement
for i in normalized_indices:
    displacement.pop(i)
    force.pop(i)

# Calculate compliance
width = 0.025
compliance = [d / f for d, f in zip(displacement, force)]

def derivative(point1, point2):
    #point1: [x1, y1]
    x1, y1 = point1
    x2, y2 = point2
    value = (y2-y1)/(x2-x1)
    return value

# Perform linear regression
reg = LinearRegression()
X = np.array(entries).reshape(-1, 1)
y = np.array(compliance)
reg.fit(X, y)
c_pred = reg.predict(X)

# Perform quadratic regression
popt, pcov = curve_fit(quadratic_function, entries, compliance)
a, b, c, d = popt
# Calculate dc/da
gradient = reg.coef_
gradient2 = [3 * a * x**2 + 2*b*x + c for x in entries]

# Calculate energy release rate 
err = []
for i in range(len(force)-1):
    #gradient = derivative([compliance[i], compliance[i+1]], [entries[i], entries[i+1]])
    errx =  1 / 2 * force[i] ** 2 / width * gradient2[i] / 1000 # Convert J to kJ
    err.append(errx)

first_index = 0 
for i in range(len(entries)):
        if entries[i] >= 0.03:
             first_index = i
             break 
print(err)
# Plot data
plt.figure()
plt.plot(entries[first_index:-1], err[first_index:])
plt.xlim(0.03, np.max(entries)+0.01)
plt.ylim(0, 4.5)
plt.show()
