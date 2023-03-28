import numpy as np
import matplotlib.pyplot as plt

specimen = 'D2'
f = open('results\\MMA_uniform\\{0}\\{0}_crack_lengths.txt'.format(specimen), 'r')
contents = f.readlines()

first_row = contents[0]
second_row = contents[1]    
first_row_values = []
for i in first_row[0:-1].split(' '):
    first_row_values.append(float(i))

second_row_values = []
for i in second_row[0:-1].split(' '):
    second_row_values.append(float(i))

plt.figure()
plt.plot(first_row_values, second_row_values, marker='.', markerfacecolor='red', markersize=5)
plt.grid()
plt.show()