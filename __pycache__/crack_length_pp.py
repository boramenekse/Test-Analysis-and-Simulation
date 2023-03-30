import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from natsort import natsorted
import time

start_time = time.time()
point_one_cm = 10 # [pixels]
one_cm = point_one_cm * 10 # [pixels]
one_pixel = 1 / one_cm # [cm]
one_pixel_m = one_pixel/100 # [m]

def get_dir_info():
    cwd = os.getcwd()
    surface_treatment_list = natsorted(os.listdir(cwd))[0:3]
    specimens_all = []
    for i in range(len(surface_treatment_list)):
        specimens = os.listdir(cwd + '\\' + surface_treatment_list[i])
        specimens_all.append(specimens)
    return cwd, surface_treatment_list, specimens_all

def start_end(specimen_path):
    all_files = os.listdir(specimen_path)
    if 'c_0.txt' in all_files:
        all_files.remove('c_0.txt')
    if 'c_1.txt' in all_files:
        all_files.remove('c_1.txt')
    length = len(os.listdir(specimen_path))
    reduced_list = natsorted(all_files)[0: int(length*0.5-1)]
    
    if '.' in reduced_list[0].split('_')[2]:
        start = int(reduced_list[0].split('_')[2].split('.')[0])
        end = int(reduced_list[-1].split('_')[2].split('.')[0])
    else:
        start = int(reduced_list[0].split('_')[2])
        end = int(reduced_list[-1].split('_')[2])
    return start, end

def find_midpoint(x, width, fraction1, fraction2):
    data = x[:, int(fraction1 * width):int(fraction2 * width)]
    rows, columns = data.shape
    midpoints = []
    for i in range(columns):
        row_indices = np.where(data[:, i] == 255)[0]
        if np.size(row_indices) >= 3:
            end_of_first = row_indices[1]
            start_of_second = row_indices[2]
        else:
            continue
        if (start_of_second - end_of_first - 1) % 2 != 0:
            midpoints.append(end_of_first + (start_of_second - end_of_first) / 2)
        else:
            midpoints.append(end_of_first + (start_of_second - end_of_first - 1) / 2)
    values, counts = np.unique(midpoints, return_counts=True)
    max_count = max(counts.tolist())
    most_frequent_value = int(values.tolist()[counts.tolist().index(max_count)])
    midpoints = list(filter(lambda x: abs(most_frequent_value-x)<10, midpoints))
    mean_value = round(np.mean(np.array(midpoints)))
    return most_frequent_value

def white(data, vert_f, hor_f):
    target_area = data[vert_f[0]:vert_f[1], hor_f[0]:hor_f[1]]
    vert_list = [*range(vert_f[0], vert_f[1])]
    first_whites = []
    first_whites_rows = []
    for i in range(vert_f[1]-vert_f[0]):
        if np.size(np.where(target_area[i, :] == 255)[0]) != 0:
            first_whites.append(np.where(target_area[i, :]==255)[0][0]+hor_f[0])
            first_whites_rows.append(vert_list[i])
    estimate = max(first_whites)
    max_index = first_whites.index(estimate)
    row_value = first_whites_rows[max_index]
    point = [row_value, estimate]
    return point, first_whites_rows, first_whites

times = []
edge_times = []
midpoint_times = []
preprocessing_times = []
crack_tip_times = []
def pp(specimen_path, specimen, file_no, threshold, hor_f1, hor_f2, region_plus_minus, target_hor_f, point14_hor_pos):
    start_fun = time.time()
    image_path = specimen_path+'\\'+specimen.lower()+'_0_{}.bmp'.format(file_no)
    image = cv.imread(image_path)
    data_image = np.asarray(image)
    height, width = data_image.shape[0:2]
    data_average = np.mean(data_image, axis=2).astype(int)
    data = np.copy(data_average)
    data[data < threshold] = 0 # 180
    data[data >= threshold] = 255
    data = data.astype(np.uint8)
    data_original = np.copy(data)
    gray = cv.fastNlMeansDenoising(data, None, 25, 7, 21)
    gray = cv.GaussianBlur(gray, (3, 3), 0)
    preprocessing_end = time.time()

    edge_start = time.time()
    edged = cv.Canny(gray, 30, 200)
    edge_end = time.time()
    
    midpoint_start = time.time()
    midpoint  = find_midpoint(edged, 2048, 0.1, 0.12)
    midpoint_end = time.time()

    if specimen == 'PP1':
        first_estimate, first_whites, first_whites_rows = white(data, [midpoint-region_plus_minus, midpoint+region_plus_minus], [0, int(0.7*width)])
        if file_no >= 200:
            data[midpoint-50:midpoint-1, first_estimate[1]+1: first_estimate[1]+round(0.2*width)]=255
        second_estimate, second_whites, second_whites_rows = white(data, [midpoint-region_plus_minus, midpoint+region_plus_minus], [first_estimate[1], first_estimate[1]+round(0.1*width)])
        if file_no >= 200:
            data[midpoint: second_estimate[0], second_estimate[1]: second_estimate[1]+round(0.2*width)] = 255
        third_estimate, third_whites, third_whites_rows = white(data, [midpoint-region_plus_minus, midpoint+region_plus_minus], [second_estimate[1], second_estimate[1]+round(0.06*width)])
        if file_no == 256:
            second_estimate = third_estimate
    if specimen == 'PP2':
        first_estimate, first_whites, first_whites_rows = white(data, [midpoint-region_plus_minus, midpoint+region_plus_minus], [0, int(0.7*width)])
        second_estimate, second_whites, second_whites_rows = white(data, [midpoint-region_plus_minus, midpoint+region_plus_minus], [first_estimate[1], first_estimate[1]+round(0.2*width)])
        third_estimate, third_whites, third_whites_rows = white(data, [midpoint-region_plus_minus, midpoint+region_plus_minus], [second_estimate[1], second_estimate[1]+round(0.1*width)])

    crack_tip_end = time.time()
    if file_no%10 == 0:
        data_combine_all = np.copy(data)
        cv.imshow('Threshold', data_original)
        cv.waitKey()
        cv.destroyAllWindows()

        first_estimate, first_whites, first_whites_rows = white(data_combine_all, [midpoint-region_plus_minus, midpoint+region_plus_minus], [0, int(0.7*width)])
        second_estimate, second_whites, second_whites_rows = white(data_combine_all, [midpoint-region_plus_minus, midpoint+region_plus_minus], [first_estimate[1], first_estimate[1]+round(0.2*width)])
        third_estimate, third_whites, third_whites_rows = white(data_combine_all, [midpoint-region_plus_minus, midpoint+region_plus_minus], [second_estimate[1], second_estimate[1]+round(0.1*width)])

        # target_area1 = data[midpoint-region_plus_minus:midpoint + region_plus_minus, max(first_whites)+50: max(first_whites)+50+round(0.1*width)]
        # midpoint_list = [*range(midpoint-region_plus_minus, midpoint+region_plus_minus)]
        # second_whites = []
        # second_whites_rows = []
        # for i in range(2*region_plus_minus):
        #     if np.size(np.where(target_area1[i, :] == 255)[0]) != 0:
        #         first_occurence1 = np.where(target_area1[i, :]==255)[0]
        #         second_whites.append(np.where(target_area1[i, :]==255)[0][0]+max(first_whites)+50)
        #         second_whites_rows.append(midpoint_list[i])
        # second_estimate = max(second_whites)
        # max_index = second_whites.index(max(second_whites))
        # row_value1 = second_whites_rows[max_index]
        # second_point = [row_value1, second_estimate]

        # target_area2 = data[midpoint-region_plus_minus:midpoint+region_plus_minus, max(second_whites): max(second_whites)+round(0.06*width)]
        # third_whites = []
        # third_whites_rows = []
        # for i in range(2*region_plus_minus):
        #     if np.size(np.where(target_area2[i, :] == 255)[0]) != 0:
        #         first_occurence2 = np.where(target_area2[i, :]==255)[0]
        #         third_whites.append(np.where(target_area2[i, :]==255)[0][0]+max(second_whites))
        #         third_whites_rows.append(midpoint_list[i])
        # third_estimate = max(third_whites)
        # max_index = third_whites.index(max(third_whites))
        # row_value2 = third_whites_rows[max_index]
        # third_point = [row_value2, third_estimate]

        data_combine_all[find_midpoint(edged, 2048, 0.1, 0.12):find_midpoint(edged, 2048, 0.1, 0.12)+4, :] = 255
        # small_area = data[midpoint-6:midpoint+6, second_estimate+1:second_estimate+round(0.05*width)]
        # if file_no == 113 or file_no == 114:
        #     second_estimate = np.where(small_area==0)[1][-1]+second_estimate+1+1

        # if file_no == 114 or file_no == 113:
        #     print(second_estimate == max(third_whites))
        
        # if file_no == 156:
        #     second_estimate = third_estimate
        # if abs(row_value2-row_value1) <= 5:
        #     if not 0 in np.unique(data[row_value1-5:row_value1+1, third_estimate-1]):
        #         value = third_estimate-1
        #         while (0 in np.unique(data[row_value1-2:row_value1+2, value])) == False:
        #             value -= 1
        #         third_estimate = value+1
        data_combine_all[:, first_estimate[1]:first_estimate[1]+3] = 255
        data_combine_all[:, second_estimate[1]:second_estimate[1]+3] = 180
        data_combine_all[:, third_estimate[1]:third_estimate[1]+3] = 120
        data_combine_all[second_estimate[0]:second_estimate[0]+3, second_estimate[1]: second_estimate[1]+3] = 0
        data_combine_all[third_estimate[0]:third_estimate[0]+3, third_estimate[1]: third_estimate[1]+3] = 0
        cv.imshow('Combined', cv.resize(data_combine_all, (1400, 800)))
        cv.waitKey()
        cv.destroyAllWindows()

    print('Finished image for specimen {0}: {1}'.format(specimen, file_no))
    end_fun = time.time()
    times.append(end_fun-start_fun)
    edge_times.append(edge_end-edge_start)
    midpoint_times.append(midpoint_end-midpoint_start)
    preprocessing_times.append(preprocessing_end-start_fun)
    crack_tip_times.append(crack_tip_end-midpoint_end)
    return third_estimate[1]*one_pixel_m

def filter_outliers_2d(data, thresh=3.5):
    """
    Filters outliers from the first column of the given 2D array using the median absolute deviation (MAD) method.
    
    Parameters:
        data (array-like): The input 2D array to filter.
        thresh (float): The threshold for outlier detection, in units of MAD. Default is 3.5.
        
    Returns:
        filtered_data (array-like): The input 2D array with outliers removed from the first column.
    """
    first_col = data[:, 0]
    median = np.median(first_col)
    mad = np.median(np.abs(first_col - median))
    z_score = 0.6745 * (first_col - median) / mad
    filtered_data = data[np.abs(z_score) < thresh, :]
    removed_data = data[np.abs(z_score) >= thresh, :]
    return filtered_data, removed_data

def monotonic(data, files):
    monotonic_list = [data[0]]
    files_list = [files[0]]
    removed_files = []
    for i in data[1:]:
        if i >= monotonic_list[-1]:
            monotonic_list.append(i)
            index = data.index(i)
            files_list.append(files[index])
        else:
            index = data.index(i)
            removed_files.append(files[index])
    return files_list, monotonic_list, removed_files

width = 2048
height = 2048
region_plus_minus = 10
surface_treatment_index = 1
cwd, surface_treatments, specimens = get_dir_info()
specimen = specimens[surface_treatment_index][1] # 0 for PP1, 1 for PP2
surface_treatment = surface_treatments[surface_treatment_index]
specimen_path = cwd+'\\'+surface_treatment+'\\'+specimen
start, end = start_end(specimen_path)

crack_lengths = []
files_no_list = []
print('--- Starting the analysis for {0}\\{1} ---\n'.format(surface_treatment, specimen))
for i in range(start, end+1):
    crack_lengths.append(pp(specimen_path, specimen, i, 180, [0.1, 0.05], [0.12, 0.07], 10, [0.4, 0.7], [0, 0]))
    files_no_list.append(i)

combined = []
for i in range(len(files_no_list)):
    combined.append([crack_lengths[i], files_no_list[i]])

filtered_data, removed_data = filter_outliers_2d(np.array(combined))
crack_lengths_filtered = filtered_data[:, 0].tolist()
files_no_list_filtered = filtered_data[:, 1].tolist()
files_no_list_filtered = [int(x) for x in files_no_list_filtered]
filtered_files = removed_data[:, 1].tolist()
filtered_files = [int(x) for x in filtered_files]
files_no_list_reduced, crack_lengths_reduced, removed_files = monotonic(crack_lengths_filtered, files_no_list_filtered)

plt.figure()
plt.plot(files_no_list, crack_lengths, marker='.', markerfacecolor='red', markersize=5)
plt.grid()
plt.savefig('Crack_lengths_pp2.png')
plt.figure()
plt.plot(files_no_list_reduced, crack_lengths_reduced, marker='.', markerfacecolor='red', markersize=5)
plt.grid()
plt.savefig('Crack_lengths_pp2_reduced.png')

print('Average analysis time of an image: {}'.format(np.mean(np.array(times))), '[s]')
end_time = time.time()
print('Average time spent for the edge detection: {}'.format(np.mean(np.array(edge_times))),'[s]')
print('Average time spent for finding the midpoint: {}'.format(np.mean(np.array(midpoint_times))),'[s]')
print('Average time spent for preprocessing: {}'.format(np.mean(np.array(preprocessing_times))), '[s]')
print('Average time spent for finding the crack tip: {}'.format(np.mean(np.array(crack_tip_times))), '[s]')
print('Execution time of the program: {} min {} seconds'.format(round((end_time-start_time)/60), round((end_time-start_time)%60)))