import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# open the files
with open('A1/MMA B1 C0.txt', 'r') as f, \
     open('results/MMA_uniform/B1/B1_crack_lengths.txt', 'r') as g:
    contents = f.readlines()
    cracklengths = g.readlines()

# extract crack lengths
second_line = cracklengths[1].strip()
entries = second_line.split()[:]

# convert first line to a list of index integers
first_line = cracklengths[0].strip()
indices = first_line.split()[:]
for i in range(len(indices)):
    item = indices[i]
    if isinstance(item, (float, int)):
        indices[i] = int(item)
    elif isinstance(item, str):
        indices[i] = int(float(item))
print(indices)

# Find which integers are not used in indices list
missing_indices = []
for i in range(min(indices), max(indices) + 1):
    if i not in indices:
        missing_indices.append(i)
print("Missing indices:", missing_indices)

# normalize indices so that corresponding line from contents can be removed
normalized_indices = [(x - min(indices)) for x in missing_indices]
print(normalized_indices)

# remove the entries from txt file that are not used 
#for i in normalized_indices:
 #   del contents[i]
#print(contents)

# extract force and displacement data
force = []
displacement = []
for line in contents[29:]:
    cols = line.split('\t')
    force.append(float(cols[2].replace(',', '.')) * 40)
    displacement.append(float(cols[3].replace(',', '.')) * 0.2)

# calculate compliance
# width of specimen in metres. 
# it can be improved by obtaining it directly from excel files, 
# however i dont think it's worth the hassle. the value is close to 25 mm.
b = 0.025
c = [d / f for d, f in zip(displacement, force)]

# convert entries to numerical values
a = [float(x) for x in entries]

# perform linear regression
reg = LinearRegression()
X = np.array(a).reshape(-1, 1)
y = np.array(c)
reg.fit(X, y)
c_pred = reg.predict(X)
#print("Coefficients:", reg.coef_)

# calculate dc/da
gradient = reg.coef_

# calculate energy release rate 
err = []
for i in range(len(contents) - 29):
    errx = - 1 / 2 * force[i] ** 2 / b * gradient 
    # convert J to kJ
    errx /= 1000
    err.append(errx)
print(force)
# plot data
plt.figure()
plt.plot(a, err)
#plt.yticks(np.arange(-10, 10, 2))
plt.xlim(0, np.max(a)+0.01)
plt.show()

