import numpy as np
import matplotlib.pyplot as plt

f = open('A1\\c_0.txt', 'r')
contents = f.readlines()
total_len = len(contents)
load_data = []
displacement = []
for i in range(total_len):
    load_data.append(contents[i].split('\t')[2])
    displacement.append(contents[i].split('\t')[3])
files_list = [*range(0, 154)]
plt.figure()
plt.plot(files_list, load_data)
plt.grid()
plt.show()
f.close()