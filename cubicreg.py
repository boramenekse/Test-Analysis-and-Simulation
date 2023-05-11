import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from collections import Counter
from scipy.optimize import curve_fit
import pathlib
from numpy.polynomial.polynomial import Polynomial as P
from scipy.signal import savgol_filter
from lists import width_values
# Define the quadratic function
def cubic_function(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d

root = pathlib.Path.cwd()
results = root.joinpath('results')
widths = width_values()

def read_cracklengths(sample, folder):
    filtered_files_loc = folder.joinpath(f'{sample}_filtered_files.txt')
    c_1 = False
    if sample in ['B4', 'SBP1', 'SBP2']:
        c_1 = True

    normalized_indices = []

    if filtered_files_loc.exists():
        # Open the files
        with open(folder.joinpath(f'{sample}_crack_lengths_all.txt'), 'r') as g:
             cracklengths_all = g.readlines()

        with open(folder.joinpath(f'{sample}_crack_lengths.txt'), 'r') as h:
             cracklengths = h.readlines()

        # Extract crack lengths
        second_line = cracklengths[1].strip()
        entries = [round(float(x), 4) for x in second_line.split()]

        # Convert first line to a list of index integers
        first_line = cracklengths[0].split()
        indices = [int(float(x)) if isinstance(x, str) else int(x) for x in first_line]

        # Find which integers are not used in indices list
        all_indeces = cracklengths_all[0].split()
        missing_indices = []
        all_indeces = [round(float(a)) for a in all_indeces]
        for index in all_indeces:
            if index not in indices:
                missing_indices.append(index)

        # Normalize missing indices
        normalized_indices = [(x - np.min(indices)) for x in missing_indices]
    else:
        entries = np.loadtxt(folder.joinpath(f'{sample}_crack_lengths.txt')).T[1, :].tolist()
        if c_1:
            with open(folder.joinpath('c_1.txt'), 'r') as f:
                contents = f.readlines()
        else:
            with open(folder.joinpath('c_0.txt'), 'r') as f:
                contents = f.readlines()

    if c_1:
        with open(folder.joinpath('c_1.txt'), 'r') as f:
            contents = f.readlines()
    else:
        with open(folder.joinpath('c_0.txt'), 'r') as f:
            contents = f.readlines()
    force = [float(line.split('\t')[2].replace(',', '.')) * 40 for line in contents]
    displacement = [float(line.split('\t')[3].replace(',', '.')) * 0.02 for line in contents]
    # Remove corresponding entries in force and displacement
    for i in normalized_indices:
        displacement.pop(i)
        force.pop(i)

    return entries, force, displacement

def calc_err(entries, force, displacement):
    # Calculate compliance
    width = 0.025
    compliance = [d / f for d, f in zip(displacement, force)]
    comp = np.array(compliance)
    entr = np.array(entries)
    entr = entr + 0.025
    fig, ax = plt.subplots()
    plt.title('Compliance vs crack length')
    ax.plot(entr, np.power(comp, 1/3))
    plt.show()
    # Perform linear regression
    reg = LinearRegression()
    X = np.array(entries).reshape(-1, 1)
    y = np.array(compliance)
    reg.fit(X, y)
    c_pred = reg.predict(X)
    print("Linear regression coefficients:", reg.coef_)

    # Perform cubic regression
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
    return err
deltas = []
def MBT_method(entries, force, displacement, sample='undefined'):
    """modified beam theory method for calculating Energy release rate"""
    width = 0.025
    compliance = [d / f for d, f in zip(displacement, force)]
    comp = np.array(compliance)
    entr = np.array(entries)
    forc = np.array(force)
    disp = np.array(displacement)
    # Perform linear regression
    fitted_linear = P.fit(entr[10:], np.power(comp[10:], 1 / 3), 1)
    delta = np.abs(fitted_linear.roots())
    print('delta= ', delta * 1000)
    err = 3 * forc * disp / 2 / width / (entr + delta) / 1000


    plt.title(f'Compliance vs crack length {sample}')
    left = -0.03
    right = 0.15
    graphpoints = np.linspace(left, right)
    plt.scatter(entr[10:], np.power(comp[10:], 1 / 3))
    plt.plot(graphpoints, fitted_linear(graphpoints))
    plt.xlim(left, right)
    plt.ylim(-0.03, 0.15)
    plt.axhline(y=0, color='k')
    plt.grid()
    plt.show()

    return err
ns = []
def CC_method(entries, force, displacement):
    """Compliance calibration method for calculating Energy release rate"""
    if not type(entries) == np.ndarray:
        entries = np.array(entries)
        force = np.array(force)
        displacement = np.array(displacement)

    width = 0.025

    log_comp = np.log(displacement / force)
    plt.plot(np.log(entr[10:]), log_comp[10:])
    plt.show()
    fitted_linear = P.fit(np.log(entr[10:]), log_comp[10:], 1)
    n = fitted_linear.coef[1]
    ns.append(n)
    err = n * force * displacement / 2 / width / entries / 1000
    return err

def MCC_method(entries, force, displacement):
    """Modified compliance calibration method for calculating Energy release rate"""
    if not type(entries) == np.ndarray:
        entries = np.array(entries)
        force = np.array(force)
        displacement = np.array(displacement)

    width = 0.025
    h = 0.01
    compliance = displacement / force
    fitted_linear = P.fit(np.power(compliance, 1 / 3)[10:], (entries / h) [10:], 1)
    A_1 = fitted_linear.coef[1]

    err = 3 * (force ** 2) * np.power(compliance, 2 / 3) / 2 / width / A_1 / h / 1000
    return err
def do_the_stuff():
    for surf_treatment_dir in results.iterdir():
        surf_treatment = surf_treatment_dir.stem
        if surf_treatment != 'MMA_uniform':
            for sample_dir in surf_treatment_dir.iterdir():
                sample_name = sample_dir.stem
                print(sample_name)
                entries, forces, displacements = read_cracklengths(sample_name, sample_dir)
                entr = np.array(entries)
                entr = entr + 0.035
                entries = entr.tolist()
                err = MBT_method(entries, forces, displacements, sample=sample_name)
                np.savetxt(sample_dir.joinpath(f'{sample_name}_err.txt'), err)
                first_index = np.searchsorted(entries, 0.035)
                # Plot data
                fig, ax = plt.subplots()
                ax.scatter(entries[first_index:], err[first_index:])
                plt.savefig(sample_dir.joinpath(f'{sample_name}_err_graph.png'), dpi=300)
                plt.close()

if __name__ == "__main__":
    if False:
        sample_name = 'C4'
        surf_treatment = 'MMA_pattern'
        sample_dir = results.joinpath(surf_treatment, sample_name)

        entries, forces, displacements = read_cracklengths(sample_name, sample_dir)
        entr = np.array(entries)
        entr = entr + 0.035
        # entr = savgol_filter(entr, 5, 3)
        entries = entr.tolist()
        err = MBT_method(entries, forces, displacements)
        first_index = np.searchsorted(entries, 0.035)
        # Plot data
        plt.figure()
        plt.scatter(entries[first_index:], err[first_index:])

        plt.ylim(0, 3)
        plt.show()
        print(entries[-1])
    else:
        do_the_stuff()

#PP2, 1DS2, 1DS3, 2DS4, 2DS5

