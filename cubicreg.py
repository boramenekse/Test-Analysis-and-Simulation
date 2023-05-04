import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from collections import Counter
from scipy.optimize import curve_fit
import pathlib

# Define the quadratic function
def cubic_function(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d

root = pathlib.Path.cwd()
results = root.joinpath('results')


def read_cracklengths(sample, folder):
    filtered_files_loc = folder.joinpath(f'{sample}_filtered_files.txt')
    if filtered_files_loc.exists():
        # Open the files
        with open('A1/MMA B1 C0.txt', 'r') as f, \
             open('results/MMA_uniform/B1/B1_crack_lengths_all.txt', 'r') as g, \
             open('results/MMA_uniform/B1/B1_filtered_files.txt', 'r') as h:

            contents = f.readlines()
            cracklengths = g.readlines()
        # Extract crack lengths
        second_line = cracklengths[1].strip()
        entries = [round(float(x), 4) for x in second_line.split()]

        # Convert first line to a list of index integers
        first_line = cracklengths[0].strip()
        indices = [int(float(x)) if isinstance(x, str) else int(x) for x in first_line.split()]

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
        force = [float(line.split('\t')[2].replace(',', '.')) * 40 for line in contents]
        displacement = [float(line.split('\t')[3].replace(',', '.')) * 0.02 for line in contents]

        # Remove corresponding entries in force and displacement
        for i in normalized_indices:
            displacement.pop(i)
            force.pop(i)
    else:
        entries = np.loadtxt(folder.joinpath(f'{sample}_crack_lengths.txt')).T[1, :].tolist()
        """!!!!!!!!!!!!!!!CHECK IF c_0 OR c_1!!!!!!!!!!!!!!!!!!!!!!"""
        with open(folder.joinpath('c_0.txt'), 'r') as f:
            contents = f.readlines()

        force = [float(line.split('\t')[2].replace(',', '.')) * 40 for line in contents]
        displacement = [float(line.split('\t')[3].replace(',', '.')) * 0.02 for line in contents]
    return entries, force, displacement

def calc_err(entries, force, displacement):
    # Calculate compliance
    width = 0.025
    compliance = [d / f for d, f in zip(displacement, force)]


    # Perform linear regression
    reg = LinearRegression()
    X = np.array(entries).reshape(-1, 1)
    y = np.array(compliance)
    reg.fit(X, y)
    c_pred = reg.predict(X)
    print("Linear regression coefficients:", reg.coef_)

    # Perform quadratic regression
    popt, pcov = curve_fit(cubic_function, entries, compliance)
    a, b, c, d  = popt
    print("Quadratic regression coefficients:", a, b, c, d )
    print('force', force)
    # Calculate dc/da
    gradient = reg.coef_
    gradient2 = [3 * a * x **2 + 2 * b * x + c for x in entries]

    # Calculate energy release rate
    err = []
    for i in range(len(force)-1):
        #gradient = derivative([compliance[i], compliance[i+1]], [entries[i], entries[i+1]])
        errx =  1 / 2 * force[i] ** 2 / width * gradient2[i] / 1000 # Convert J to kJ
        err.append(errx)

    first_index = 0
    for i in range(len(entries)):
            if entries[i] >= 0.02:
                 first_index = i
                 break
    return err, first_index

def do_the_stuff():
    for surf_treatment_dir in results.iterdir():
        for sample_dir in surf_treatment_dir.iterdir():
            sample_name = sample_dir.stem
            surf_treatment = surf_treatment_dir.stem

            entries, forces, displacements = read_cracklengths(sample_name, sample_dir)
            err, first_index = calc_err(entries, forces, displacements)
            np.savetxt(sample_dir.joinpath(f'{sample_name}_err.txt'), err)
            fig, ax = plt.subplots()
            ax.plot(entries[first_index:-1], err[first_index:])
            plt.savefig(sample_dir.joinpath(f'{sample_name}_err_graph.png'), dpi=300)
            plt.close()

if __name__ == "__main__":
    if True:
        sample_name = 'B1'
        surf_treatment = 'MMA_uniform'
        sample_dir = results.joinpath(surf_treatment, sample_name)

        entries, forces, displacements = read_cracklengths(sample_name, sample_dir)
        err, first_index = calc_err(entries, forces, displacements)

        # Plot data
        plt.figure()
        plt.plot(entries[first_index:-1], err[first_index:])
        plt.xlim(-0.02, np.max(entries)+0.01)
        plt.ylim(0, 4.5)
        plt.show()
    else:
        do_the_stuff()
