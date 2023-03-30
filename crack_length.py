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

def renaming(specimen_path):
    files = []
    # Iterate directory
    for path in os.listdir(specimen_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(specimen_path, path)):
            files.append(path)
    files = natsorted(files)
    if 'c_0.txt' in files:
        files.remove('c_0.txt')
    if 'c_1.txt' in files:
        files.remove('c_1.txt')

    types = [0, 1]
    specimen = specimen_path.split('\\')[-1]
    start, end = start_end(specimen_path)
    file_numbers = [*range(start, end + 1)]
    total_file_no = (end+1-start) * 2
    for i in range(total_file_no):
        if i <= end-start:
            os.rename(os.path.join(specimen_path + '\\' + files[i]),
                      os.path.join(specimen_path + '\\' + specimen.lower() + '_{0}_{1}.bmp'.format(types[0], file_numbers[i])))
        elif i > end-start:
            index = i - (end-start + 1)
            os.rename(os.path.join(specimen_path + '\\' + files[i]), os.path.join(
                specimen_path + '\\' + specimen.lower() + '_{0}_{1}.bmp'.format(types[1], file_numbers[index])))

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

def remove_regions(image_no, start, end, image=np.array([])):
    height, width = image.shape
    image[:height // 10, :] = 0  # region1
    x0 = np.array([0, 0]) / 2048 * width
    x1 = np.array([1250, 1250]) / 2048 * width
    y0 = np.array([1400, 1900]) / 2048 * height
    y1 = np.array([1400, 1650]) / 2048 * height
    x2 = [2048, 2048]
    y2 = [1400, 1650]
    total_files_no = end-start+1
    progress = (image_no-start) / total_files_no
    point0 = [round(x0[0] + (x0[1] - x0[0]) * progress), round(y0[0] + (y0[1] - y0[0]) * progress)]
    point1 = [round(x1[0] + (x1[1] - x1[0]) * progress), round(y1[0] + (y1[1] - y1[0]) * progress)]
    image[point1[1]:, point1[0]:] = 0  # region2

    # region3
    line = lambda x: point0[1] + (point1[1] - point0[1]) / (point1[0] - point0[0]) * x
    for x in range(point1[0]):
        image[round(line(x)):, x] = 0
    return image

def remove_regions1(image_no, start, end, top_f, bot_f, image=np.array([])):
    height, width = image.shape
    top1 = np.array([top_f[0], top_f[1]])
    bot1 = np.array([bot_f[0], bot_f[1]])
    total_files_no = end-start+1
    progress = (image_no-start) / total_files_no
    top2 = round((top1[0] + (top1[1]-top1[0])*progress)*height)
    bot2 = round((bot1[0] + (bot1[1]-bot1[0])*progress)*height)
    image[0:top2, :] = 0
    image[bot2:, :] = 0
    return image

def start_horizontal_position(data, midpoint, width, avg_value):
    working_region = data[0:midpoint, :][::-1]
    position = round(avg_value)
    for i in range(working_region.shape[0]):
        if np.unique(working_region[i, :], return_counts=True)[1].shape[0] == 2:
            white_pixel_count = np.unique(working_region[i, :], return_counts=True)[1][1]
            if white_pixel_count==1 or white_pixel_count==2 or white_pixel_count==3:
                if (np.where(working_region[i, :] == 255)[0][0]>round(0.06*width)) and (np.where(working_region[i, :] == 255)[0][0]<round(0.11*width)): 
                    position = round(np.where(working_region[i, :] == 255)[0][0])
                    break
    if position < round(avg_value):
        return position
    else:
        return round(avg_value)

wrong_start_file_no = []
crack_lengths = []
hor_start = []
midpoints = []

def crack_length(specimen_path, specimen, file_no, width, hor_region_fraction_1, hor_region_fraction_2, region_plus_minus,
                 target_area_change_file_no, starting_hor_region1,
                 starting_hor_region2, point14, point14_dist):
    image_path = specimen_path+'\\'+specimen.lower()+'_0_{}.bmp'.format(file_no)
    image = cv.imread(image_path)

    # Grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if specimen == 'D1' or specimen == 'D2' or specimen == 'PP1' or specimen == 'PP2':
        gray = cv.fastNlMeansDenoising(gray, None, 25, 7, 21)  # removing noise
    else:
        gray = cv.fastNlMeansDenoising(gray, None, 22, 7, 21)  # removing noise

    gray = cv.GaussianBlur(gray, (3, 3), 0)  # makes the image blur

    # Find Canny edges
    if specimen == 'A1' or specimen == 'A2':
        edged = cv.Canny(gray, 30, 200)  # edge detection
    elif specimen == 'D1' or specimen == 'D2':
        edged = cv.Canny(gray, 30, 120)
    elif specimen == 'PP1' or specimen == 'PP2':
        edged = cv.Canny(gray, 200, 600) 
    else:
        edged = cv.Canny(gray, 30, 80)  # edge detection
    # cv.imshow('edged before', edged)
    # cv.waitKey()
    # cv.destroyAllWindows()

    # Getting rid of useless region by making them all black
    start, end = start_end(specimen_path)
    if specimen == 'A1' or specimen == 'A2':
        edged = remove_regions(file_no, start, end, edged)
    else:
        if specimen == 'C1':
            edged = remove_regions1(file_no, start, end, [0.145, 0.155], [0.22, 0.58], edged)
        elif specimen == 'D1' or specimen == 'D2':
            edged = remove_regions(file_no, start, end, edged)
        elif specimen == 'PP1' or specimen == 'PP2':
            edged = remove_regions1(file_no, start, end, [0.12, 0.17], [0.23, 0.7], edged)
        else:
            edged = remove_regions1(file_no, start, end, [0.07, 0.12], [0.22, 0.58], edged)
    # cv.imshow('edged after', edged)
    # cv.waitKey()
    # cv.destroyAllWindows()    
    # get the data of processed image    
    data = np.asarray(edged)
    if specimen == 'D1' or specimen == 'D2':
        midpoint = find_midpoint(data, width, hor_region_fraction_1, hor_region_fraction_2)+10
    else:
        midpoint = find_midpoint(data, width, hor_region_fraction_1, hor_region_fraction_2)

    # Method 1 for getting the horizontal position of the final crack point
    if file_no < target_area_change_file_no:
        if specimen == 'D1' or specimen == 'D2' or specimen == 'PP1' or specimen == 'PP2':
            target_area = data[midpoint - region_plus_minus:midpoint + region_plus_minus, 0: int(0.35 * width)]
        else:
            target_area = data[midpoint - region_plus_minus:midpoint + region_plus_minus, 0: int(0.5 * width)]
    else:
        if specimen == 'D1' or specimen == 'D2' or specimen == 'PP1' or specimen == 'PP2':
            target_area = data[midpoint - region_plus_minus:midpoint + region_plus_minus, 0: int(0.6 * width)]
        else:
            target_area = data[midpoint - region_plus_minus:midpoint + region_plus_minus, 0: int(0.7 * width)]
    pos = []
    for i in range(0, 2*region_plus_minus):
        if np.size(np.where(target_area[i, :] == 255)[0]) != 0:
            pos.append(np.where(target_area[i, :] == 255)[0].max())
    crack_hor = max(pos)
    crack_pos1 = [crack_hor, midpoint]

    # Finding Contours
    # Use a copy of the image e.g. edged.copy() since findContours alters the image
    contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Find the contour with the highest arc length
    perimeters = [cv.arcLength(c, True) for c in contours]
    max_index = np.argmax(perimeters)
    cnt_arc = contours[max_index]
    arc_cnt = np.resize(cnt_arc, (cnt_arc.shape[0], cnt_arc.shape[-1]))
    hor = arc_cnt[:, 0]

    # Method 2 for getting the horizontal position of final crack point
    index = 0
    points = []
    indices_points = []
    for i in arc_cnt:
        if (i[1] == midpoint) or (i[1] == midpoint - 1) or (i[1] == midpoint - 2) or (i[1] == midpoint + 1) or (
                i[1] == midpoint + 2) or (i[1] == midpoint - 3) or (i[1] == midpoint + 3) or (i[1] == midpoint - 4) or (
                i[1] == midpoint + 4) or (i[1] == midpoint - 5) or (i[1] == midpoint + 5):
            points.append(i)
            indices_points.append(arc_cnt[index])
        index += 1
    points_np = np.array(points)
    crack_pos2 = np.zeros(2)
    if points_np.shape[0] > 0:
        points_np_max_vert = points_np[:, 1].max()
        if points_np_max_vert == midpoint:
            crack_pos2[0] = points_np[:, 0].max()
            crack_pos2[1] = midpoint
        elif points_np_max_vert == midpoint - 1:
            crack_pos2[0] = points_np[:, 0].max()
            crack_pos2[1] = midpoint - 1
        elif points_np_max_vert == midpoint + 1:
            crack_pos2[0] = points_np[:, 0].max()
            crack_pos2[1] = midpoint + 1
        elif points_np_max_vert == midpoint - 2:
            crack_pos2[0] = points_np[:, 0].max()
            crack_pos2[1] = midpoint - 2
        elif points_np_max_vert == midpoint + 2:
            crack_pos2[0] = points_np[:, 0].max()
            crack_pos2[1] = midpoint + 2
        elif points_np_max_vert == midpoint - 3:
            crack_pos2[0] = points_np[:, 0].max()
            crack_pos2[1] = midpoint - 3
        elif points_np_max_vert == midpoint + 3:
            crack_pos2[0] = points_np[:, 0].max()
            crack_pos2[1] = midpoint + 3
        elif points_np_max_vert == midpoint - 4:
            crack_pos2[0] = points_np[:, 0].max()
            crack_pos2[1] = midpoint - 4
        elif points_np_max_vert == midpoint + 4:
            crack_pos2[0] = points_np[:, 0].max()
            crack_pos2[1] = midpoint + 4
        elif points_np_max_vert == midpoint - 5:
            crack_pos2[0] = points_np[:, 0].max()
            crack_pos2[1] = midpoint - 5
        elif points_np_max_vert == midpoint + 5:
            crack_pos2[0] = points_np[:, 0].max()
            crack_pos2[1] = midpoint + 5
    crack_pos2 = crack_pos2.astype(int)

    # If first method fails to get it right, use the second method's result
    # The problem with the second method is that it might be underestimating the crack position
    # I think first method works better, but it fails under specific conditions such as when the structure is not straight (bended downwards) -> that's why it fails in the first 20 files
    crack_pos = []
    if crack_pos1[0] >= int(0.7 * width) and not (crack_pos2 == np.zeros(2)).any():
        crack_pos = crack_pos2
    else:
        crack_pos = crack_pos1
    crack_pos = np.array(crack_pos).astype(int)

    # Check if it captures the starting horizontal position of the structure correct
    # This is crucial for calculating the crack length correct
    # If it does that, it is likely to capture the crack point as well
    starting_hor = hor.min()
    if (starting_hor >= int(starting_hor_region1 * width)) and (starting_hor < int(starting_hor_region2 * width)):
        hor_start.append(starting_hor)
    else:
        wrong_start_file_no.append(file_no)
        if len(hor_start)>0:
            starting_hor = hor_start[-1]
        else:
            if specimen == 'A1' or specimen == 'A2':
                starting_hor = start_horizontal_position(data, midpoint, width, 0.085*width)
            else:
                starting_hor = start_horizontal_position(data, midpoint, width, 0.042*width)

    progress = (file_no-start+1)/(end-start+1)
    point14_hor = round(point14[0] + (point14[1]-point14[0])*progress)
    distance_between = point14_hor-crack_pos[0]
    crack_lengths.append((point14_dist-distance_between)*one_pixel_m)
    print('Finished file number in specimen {0}: {1}'.format(specimen, file_no))

def pp(specimen_path, specimen, file_no, threshold, hor_f1, hor_f2, region_plus_minus,
                 target_hor_f, point14_hor_pos):
    image_path = specimen_path+'\\'+specimen.lower()+'_0_{}.bmp'.format(file_no)
    start, end = start_end(specimen_path)
    
    image = cv.imread(image_path)
    data_image = np.asarray(image)
    height, width = data_image.shape[0:2]
    total_files_no = end-start+1
    progress = (file_no-start+1) / total_files_no

    # calculate the average color value on each pixel
    data_average = np.mean(data_image, axis=2).astype(int)
    data = np.copy(data_average)
    data[data < threshold] = 0 # 180
    data[data >= threshold] = 255
    data = data.astype(np.uint8)

    # denoising, blur for edge detection -> for midpoint
    gray = cv.fastNlMeansDenoising(data, None, 25, 7, 21)
    gray = cv.GaussianBlur(gray, (3, 3), 0)
    edged = cv.Canny(gray, 30, 200)
    hor_f1_value = hor_f1[0] + (hor_f1[1]-hor_f1[0])*np.sin((np.pi/2)*progress**2)
    hor_f2_value = hor_f2[0] + (hor_f2[1]-hor_f2[0])*np.sin((np.pi/2)*progress**2)
    midpoint  = find_midpoint(edged, width, hor_f1_value, hor_f2_value) # 0.1 and 0.12

    # Method 1
    target_hor_start_f = target_hor_f[0] # 0.35
    target_hor_end_f = target_hor_f[1] # 0.65
    target_hor_f = target_hor_start_f + (target_hor_end_f-target_hor_start_f)*np.sin((np.pi/2)*progress**(1/2))
    target_area1 = data[midpoint-region_plus_minus:midpoint + region_plus_minus, 0: int(round(target_hor_start_f * width))]
    first_whites = []
    for i in range(2*region_plus_minus):
        if np.size(np.where(target_area1[i, :] == 255)[0]) != 0:
            first_whites.append(np.where(target_area1[i, :]==255)[0][0])
    first_estimate = max(first_whites)
    target_area2 = data[midpoint-region_plus_minus:midpoint + region_plus_minus, first_estimate: 
    first_estimate+round(0.1*width)]
    row_values_list = [*range(midpoint-region_plus_minus, midpoint + region_plus_minus)]
    second_whites = []
    second_whites_rows = []
    for i in range(2*region_plus_minus):
        if np.size(np.where(target_area2[i, :] == 255)[0]) != 0:
            second_whites.append(np.where(target_area2[i, :]==255)[0][0]+first_estimate)
            second_whites_rows.append(row_values_list[i])
    second_estimate = max(second_whites)
    max_index = second_whites.index(second_estimate)
    row_value = second_whites_rows[max_index]

    point14_hor = round((point14_hor_pos[0] + (point14_hor_pos[1]-point14_hor_pos[0])*np.sin((np.pi/2)*progress**(1/2)))*width)
    distance_to_point14 = (point14_hor - second_estimate)*one_pixel_m
    crack_length = 0.14 - distance_to_point14
    crack_lengths.append(second_estimate*one_pixel_m)
    print('Finished file number in specimen {0}: {1}'.format(specimen, file_no))

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

def filter_outliers_1d(data, thresh=3.5):
    data = np.array(data)
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    z_score = 0.6745 * (data - median) / mad
    filtered_data = data[np.abs(z_score) < thresh]
    removed_data = data[np.abs(z_score) >= thresh]
    return filtered_data, removed_data

cwd, surface_treatments, specimens = get_dir_info()
# for i in range(len(surface_treatments)):
#     for j in range(len(specimens[i])):
#         specimen_path = cwd+'\\'+surface_treatments[i]+'\\'+specimens[i][j]
#         renaming(specimen_path)

width = 2048
height = 2048
region_plus_minus = 10
surface_treatment_index = 0

# specimen = specimens[surface_treatment_index][0] # PP1
# surface_treatment = surface_treatments[surface_treatment_index]
# specimen_path = cwd+'\\'+surface_treatment+'\\'+specimen
# file_no = 250
# image_path = specimen_path+'\\'+specimen.lower()+'_0_{}.bmp'.format(file_no)
# image = cv.imread(image_path)
# cv.imshow('Original', cv.resize(image, (1000, 700)))
# cv.waitKey()
# cv.destroyAllWindows()
# data_image = np.asarray(image)
# height, width = data_image.shape[0:2]
# print(height, width)
# data_average = np.mean(data_image, axis=2).astype(int)
# data = np.copy(data_average)
# data[data < 180] = 0
# data[data >= 180] = 255
# data = data.astype(np.uint8)
# cv.imshow('Threshold', cv.resize(data, (1200, 800)))
# cv.waitKey()
# cv.destroyAllWindows()
# gray = cv.fastNlMeansDenoising(data, None, 25, 7, 21)
# gray = cv.GaussianBlur(gray, (3, 3), 0)
# edged = cv.Canny(gray, 30, 200)
# cv.imshow('Edge', cv.resize(edged, (1200, 800)))
# cv.waitKey()
# cv.destroyAllWindows()
# midpoint  = find_midpoint(edged, 2048, 0.04, 0.06)
# print(find_midpoint(edged, 2048, 0.1, 0.12))
# data_copy = np.copy(data)
# data_copy[find_midpoint(edged, 2048, 0.1, 0.12):find_midpoint(edged, 2048, 0.1, 0.12)+4, :] = 255
# cv.imshow('Midpoint', cv.resize(data_copy, (1200, 800)))
# cv.waitKey()
# cv.destroyAllWindows()
# target_area = data[midpoint-region_plus_minus:midpoint + region_plus_minus, 0: int(0.7 * width)]
# first_whites = []
# for i in range(0, 2*region_plus_minus):
#     if np.size(np.where(target_area[i, :] == 255)[0]) != 0:
#         first_whites.append(np.where(target_area[i, :]==255)[0][0])
# print(first_whites)
# data_first_whites = np.copy(data)
# data_first_whites[:, max(first_whites):max(first_whites)+2] = 255
# for j in first_whites:
#     data_first_whites[:, j] = 255
# cv.imshow('First whites', cv.resize(data_first_whites, (1200, 800)))
# cv.waitKey()
# cv.destroyAllWindows()
# target_area1 = data[midpoint-region_plus_minus:midpoint + region_plus_minus, max(first_whites): max(first_whites)+round(0.1*width)]
# print(np.unique(target_area1, return_counts=True))
# midpoint_list = [*range(midpoint-region_plus_minus, midpoint+region_plus_minus)]
# second_whites = []
# second_whites_rows = []
# for i in range(2*region_plus_minus):
#     if np.size(np.where(target_area1[i, :] == 255)[0]) != 0:
#         second_whites.append(np.where(target_area1[i, :]==255)[0][0]+max(first_whites))
#         second_whites_rows.append(midpoint_list[i])
# print(second_whites)
# second_estimate = max(second_whites)
# print(0.14 - (0.75*width - max(second_whites))*one_pixel_m)
# max_index = second_whites.index(max(second_whites))
# row_value = second_whites_rows[max_index]
# max_black = []
# for i in data[row_value-1:row_value+2, second_estimate:second_estimate+round(0.05*width)]:
#     max_black.append(np.where(i == 255)[0][-1]+second_estimate)
# print(max(max_black))
# data_black = np.copy(data)
# data_black[:, max(max_black):max(max_black)+2] = 255
# cv.imshow('Max black', cv.resize(data_black, (1200, 800)))
# cv.waitKey()
# cv.destroyAllWindows()

# data_second_whites = np.copy(data)
# data_second_whites[:, max(second_whites):max(second_whites)+2] = 255
# cv.imshow('Second whites', cv.resize(data_second_whites, (1200, 800)))
# cv.waitKey()
# cv.destroyAllWindows()

# target_area2 = data[midpoint-region_plus_minus:midpoint+region_plus_minus, max(second_whites)+50: max(second_whites)+round(0.1*width)]
# third_whites = []
# for i in range(2*region_plus_minus):
#     if np.size(np.where(target_area2[i, :] == 255)[0]) != 0:
#         third_whites.append(np.where(target_area2[i, :]==255)[0][0]+max(second_whites)+50)
# print(third_whites)
# print(0.14 - (0.75*width - max(third_whites))*one_pixel_m)
# data_third_whites = np.copy(data)
# data_third_whites[:, max(third_whites):max(third_whites)+2] = 255
# cv.imshow('Third whites', cv.resize(data_third_whites, (1200, 800)))
# cv.waitKey()
# cv.destroyAllWindows()

def threshold(image, threshold):
    data_image = np.asarray(image)
    data_average = np.mean(data_image, axis=2).astype(int)
    data = np.copy(data_average)
    data[data < threshold] = 0
    data[data >= threshold] = 255
    data = data.astype(np.uint8)
    return data

width = 2048
height = 2048
region_plus_minus = 6 # 6

files_list_specimens = []
crack_lengths_specimens = []
surface_treatment_index = 0
print("\n--- Starting the analysis for '{}' ---\n".format(surface_treatments[surface_treatment_index]))
analysis_start = time.time()
analysed_specimens = 0
specimens_index = []

hor_f_1 = [0.1, 0.05]
hor_f_2 = [0.12, 0.07]
threshold_value = 180
target_hor_f = [0.35, 0.65]
point14_hor_pos_pp1 = [0.955, 0.844]
point14_hor_pos_pp2 = [0.950, 0.863]
point14_hor_pos_pp1 = [0.955, 0.844]
point14_hor_pos_pp2 = [0.950, 0.863]

point14_travel_A1 = [1951, 1869]
point14_travel_A2 = [1892, 1811]
point14_travel_B1 = [1963, 1872]
point14_travel_B2 = [1974, 1879]
point14_travel_C1 = [1963, 1826]
point14_travel_C2 = [1974, 1852]
point14_travel_D1 = [1968, 1834]
point14_travel_D2 = [1964, 1828]
dist14_A1 = 1402
dist14_A2 = 1406
dist14_B1 = 1511
dist14_B2 = 1528
dist14_C1 = 1524
dist14_C2 = 1526
dist14_D1 = 1526
dist14_D2 = 1524
# len(specimens[surface_treatment_index])
# intention = [1, 4, 6, 7]
for j in range(6,7):
    specimen_time_start = time.time()
    specimens_index.append(j)
    specimen = specimens[surface_treatment_index][j]
    if specimen == 'A1' or specimen == 'A2':
        hor_region_fraction_1 = 0.12
        hor_region_fraction_2 = 0.14
        starting_hor_region1 = 0.06
        starting_hor_region2 = 0.11
    elif specimen == 'D1' or specimen == 'D2':
        hor_region_fraction_1 = 0.08
        hor_region_fraction_2 = 0.11
        starting_hor_region1 = 0.03
        starting_hor_region2 = 0.07
    # elif specimen == 'PP1' or specimen == 'PP2':
    #     hor_f_1 = 0.1
    #     hor_f_2 = 0.12
    #     threshold_value = 180
    #     target_hor_f = [0.5, 0.7]
    #     point14_pos = [0.764, 0.73]
    else:
        hor_region_fraction_1 = 0.11
        hor_region_fraction_2 = 0.13
        starting_hor_region1 = 0.03
        starting_hor_region2 = 0.07 

    print('--- Working on specimen {} ---\n'.format(specimen))
    surface_treatment = surface_treatments[surface_treatment_index]
    specimen_path = cwd+'\\'+surface_treatment+'\\'+specimen
    start, end = start_end(specimen_path)
    total_file_no = end-start +1
    target_area_change_file_no = round(0.065 * total_file_no)
    files_no_list = [*range(start, end+1)]
    if specimen == 'PP1':
        point14_hor_pos = point14_hor_pos_pp1
    elif specimen == 'PP2':
        point14_hor_pos = point14_hor_pos_pp2
    else:
        point14_hor_pos = point14_hor_pos_pp1
    if specimen == 'A1':
        point14_travel = point14_travel_A1
        dist14 = dist14_A1
    if specimen == 'A2':
        point14_travel = point14_travel_A2
        dist14 = dist14_A2
    if specimen == 'B1':
        point14_travel = point14_travel_B1
        dist14 = dist14_B1
    if specimen == 'B2':
        point14_travel = point14_travel_B2
        dist14 = dist14_B2
    if specimen == 'C1':
        point14_travel = point14_travel_C1
        dist14 = dist14_C1
    if specimen == 'C2':
        point14_travel = point14_travel_C2
        dist14 = dist14_C2
    if specimen == 'D1':
        point14_travel = point14_travel_D1
        dist14 = dist14_D1
    if specimen == 'D2':
        point14_travel = point14_travel_D2
        dist14 = dist14_D2

    for i in files_no_list:
        crack_length(specimen_path, specimen, i, width, hor_region_fraction_1, hor_region_fraction_2, region_plus_minus, target_area_change_file_no, starting_hor_region1, starting_hor_region2, point14_travel, dist14)
     
    print('\nFinished calculations for specimen {}, saving results\n'.format(specimen))
    combined = []
    for i in range(len(files_no_list)):
        combined.append([crack_lengths[i], files_no_list[i]])
    
    combined = combined[27:]
    filtered_data, removed_data = filter_outliers_2d(np.array(combined))
    crack_lengths_filtered = filtered_data[:, 0].tolist()
    files_no_list_filtered = filtered_data[:, 1].tolist()
    files_no_list_filtered = [int(x) for x in files_no_list_filtered]
    filtered_files = removed_data[:, 1].tolist()
    filtered_files = [int(x) for x in filtered_files]
    files_no_list_reduced, crack_lengths_reduced, removed_files = monotonic(crack_lengths_filtered, files_no_list_filtered)
    files_list_specimens.append((np.array(files_no_list_reduced)-start).tolist())
    crack_lengths_specimens.append(crack_lengths_reduced)
    plt.figure()
    plt.plot(files_no_list, crack_lengths, marker='.', markerfacecolor='red', markersize=5)
    plt.grid()
    plt.xlabel('File No')
    plt.ylabel('Crack length [m]')
    plt.title('Crack length variation over files for specimen '+specimen)
    results_specimen_path = cwd+'\\results'+'\\'+surface_treatment+'\\'+specimen
    plt.savefig(results_specimen_path+'\\Crack_length_in_each_file.png')
    text_file_list = np.vstack((files_no_list_reduced, crack_lengths_reduced))
    plt.figure()
    plt.plot(files_no_list_reduced, crack_lengths_reduced, marker='.', markerfacecolor='red', markersize=5)
    plt.grid()
    plt.xlabel('File No')
    plt.ylabel('Crack length [m]')
    plt.title('Crack length variation over files for specimen '+specimen)
    results_specimen_path = cwd+'\\results'+'\\'+surface_treatment+'\\'+specimen
    plt.savefig(results_specimen_path+'\\Reduced_Crack_length_in_each_file.png')
    text_file_list = np.vstack((files_no_list_reduced, crack_lengths_reduced))
    text_file_list_full = np.vstack((files_no_list, crack_lengths))
    np.savetxt(os.path.join(results_specimen_path, specimen+'_crack_lengths.txt'), text_file_list)
    np.savetxt(os.path.join(results_specimen_path, specimen+'_crack_lengths_all.txt'), text_file_list_full)
    np.savetxt(os.path.join(results_specimen_path, specimen+'_wrong_start_points.txt'), np.array(wrong_start_file_no))
    np.savetxt(os.path.join(results_specimen_path, specimen+'_filtered_files.txt'), np.array(filtered_files+removed_files))
    print('Saved the results for specimen {}, moving on to the next one\n'.format(specimen))
    analysed_specimens += 1
    wrong_start_file_no.clear()
    hor_start.clear()
    crack_lengths.clear()
    midpoints.clear()
    specimen_time_end = time.time()
    elapsed_time = specimen_time_end-specimen_time_start
    print('Analysis time of specimen {0}: {1} min {2} seconds\n'.format(specimen, round((elapsed_time)/60), round((elapsed_time)%60)))

analysis_end = time.time()
print('Finished all specimens for {}, plotting all on a single graph'.format(surface_treatments[surface_treatment_index]))
print('Analysis time for all specimens with {0}: {1} min {2} seconds'.format(surface_treatments[surface_treatment_index], round((analysis_end-analysis_start)/60), round((analysis_end-analysis_start)%60)))
plt.figure()
plt.xlabel('File no')
plt.ylabel('Crack length [m]')
plt.title('Results of all specimens in {}'.format(surface_treatments[surface_treatment_index]))
for i in range(analysed_specimens):
    plt.plot(files_list_specimens[i], crack_lengths_specimens[i], label=specimens[surface_treatment_index][specimens_index[i]])
plt.grid()
plt.legend()
# plt.savefig(cwd+'\\results'+'\\'+surface_treatment+'\\'+'all_specimens.png')

end_time = time.time()
print('Execution time of the program: {} min {} seconds'.format(round((end_time-start_time)/60), round((end_time-start_time)%60)))