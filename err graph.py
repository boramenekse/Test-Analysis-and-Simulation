import numpy as np
import matplotlib.pyplot as plt

def clean(lines):
    # Remove newline characters and convert to floats
    lines = [string.rstrip('\n') for string in lines]
    lines = [float(string.replace(',', '.')) for string in lines]
    # Convert list to NumPy array
    lines = np.array(lines)
    return lines

#strip the list from , and \n and convert to array 
with open("vertical displacement dcb.txt", "r") as f:
    lines = f.readlines()
lines = clean(lines)


# Create a time array with a specified time increment
time = np.arange(0, 1596.3, 0.1)
time_increment = np.max(time) / 154  # number of frames

# Plot the data
plt.plot(time, lines)
plt.xlabel('Time [s]')
plt.ylabel('Vertical Displacement [mm]')
plt.title('Vertical Displacement vs Time')
plt.show()

