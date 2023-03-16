import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#1+1 = 2

# Get the Excel file name from user input (without '.xls')
file_name = input("Enter the name of the Excel file: ")

# Get the sheet name from user input
sheet_name = input("Enter the name of the sheet to extract: ")

# Load the Excel file into a Pandas dataframe
df = pd.read_excel(file_name+'.xls', sheet_name=sheet_name)

# Extract columns C and D, and rows 4 onwards, into a numpy array
arr = np.array(df.iloc[2:, [2, 3]])

# Convert the array to a numeric type
arr = arr.astype(float)

# Save the array to a text file named after the sheet name
np.savetxt(str(file_name)+'-'+str(sheet_name)+'.txt', arr)


# Plot the array as a line plot
plt.plot(arr[:, 0], arr[:, 1])
plt.xlabel('\u03B4 [mm]')
plt.ylabel('Force [N]')
plt.show()
