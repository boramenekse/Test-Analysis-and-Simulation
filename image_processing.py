import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from natsort import natsorted
import time
import pathlib
from lengths import print_result
from scipy.signal import convolve2d
import shutil
start_time = time.time()

'''
What happens in this file rn:
-unformize all the names
-realize the folders
-grayscale the images
-ect...
'''

'''
    What to change: 
- sample name
- 14cm point travel in line 116
- top side of the sample to remove in remove_regions
- bottom side of the sample to remove in remove_regions
- crack position relative to the midpoint in method_one
- starting file number in line 234
- starting crack position in line 235
- adjust image type if needed
'''
manual_crack_lengths = print_result()

p = pathlib.Path(__file__)
sample = input('which sample')  # sample name
source = p.parents[0]
images = p.joinpath(source, sample)
contours = p.joinpath(source, sample + '_contours')
dir_path = str(images)
results = p.joinpath(source, 'results', 'MMA_pattern', sample)
if not results.exists():
    results.mkdir()
if not contours.exists():
    contours.mkdir()

def get_folder_data(image_folder: pathlib.Path):
    """
    get the number of images in the folder
    """
    start_no = np.inf
    end_no = 0
    for file in image_folder.iterdir():
        if file.suffix == '.bmp':
            name = file.stem.split('_')
            start_no = int(np.minimum(int(name[2]), start_no))
            end_no = int(np.maximum(int(name[2]), end_no))
    return start_no, end_no

file_start_no, file_end_no = get_folder_data(images)
total_files_no = file_end_no - file_start_no + 1
types = [0, 1]
files = [*range(file_start_no, file_end_no + 1)]


def renaming(start, end, types):
    """
    This function is for renaming the images in a way that we can iterate over them easily.
    start: (int) the number of the first file in the folder e.g. in A1, a_0_00_... so, start = 0
    end: (int) the number of the last file in the folder e.g. in A1, a_0_153_... so, end = 153
    First zero in a_0_153 is the type value of the file since there are different settings and angles used in the test
    types: (array) type is the first number coming after the letter e.g. There are 0 and 1 types
    in A1 so, types = [0, 1]
    """
    files = []
    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)) and path.endswith('.bmp'):
            files.append(path)
    files = natsorted(files)

    file_numbers = [*range(start, end + 1)]
    total_file_no = (end + 1) * 2
    for i in range(total_file_no):
        if i <= end:
            os.rename(os.path.join(dir_path + '\\' + files[i]),
                      os.path.join(dir_path + '\\' + sample.lower() + '_{0}_{1}.bmp'.format(types[0], file_numbers[i])))
        elif i > end:
            index = i - (end + 1)
            os.rename(os.path.join(dir_path + '\\' + files[i]), os.path.join(
                dir_path + '\\' + sample.lower() + '_{0}_{1}.bmp'.format(types[1], file_numbers[index])))


file_0 = sample.lower() + '_0_0.bmp'
if not images.joinpath(file_0).exists():
    renaming(0, total_files_no - 1, types=types)

success = [0]
no_success_file_no = []
match_count = [0]
match_fail_file_no = []
start_hor_count = [0]
wrong_start_file_no = []
fail_count1 = [0]
fail_count2 = [0]
fail1_fail_no = []
fail2_fail_no = []
crack_lengths = []
cracktips = np.empty((0, 2), dtype=int)
hor_start = []
rerun_preprocessing = True

width = 2048
height = 2048
hor_region = [150, 235]  # start and end column of the midpoint search region in a 2048x2048 image
expansion_size = 5
acceptable_max_difference = 5
starting_hor_region1 = 0.08 # for checking if the correct horizontal starting position is found
starting_hor_region2 = 0.11
fourteencm_point_travel = [1924, 1563]
fourteencm_dist = fourteencm_point_travel[0] - 354

point_one_cm = fourteencm_dist/140 # [pixels]
one_cm = point_one_cm * 10 # [pixels]
one_pixel = 1 / one_cm # [cm]
one_pixel_m = one_pixel/100 # [m]
# get the horizontal position of the 14cm point, assuming it moves linearly
fourteencm_point = lambda x: fourteencm_point_travel[0] + (fourteencm_point_travel[1] - fourteencm_point_travel[0]) * x / total_files_no



def remove_regions(image_no, image :np.array([])):
    '''
    remove the regions that are not needed for the analysis
    to specifiy pixels, use format image[height, width]
    every pixel is 0 - 255 in grayscale
    '''
    height, width = image.shape
    # region 1
    image[:round(height*0/2048), :] = 0

    # region2
    P0_0 = [900, 0]
    P1_0 = [900, 1300]
    P0_1 = [2600, 0]
    P1_1 = [1800, 1300]

    # formula: starting point + (ending point - starting point) * progress
    progress = image_no / total_files_no
    point0 = [round(P0_0[1] + (P0_1[1] - P0_0[1]) * progress), round(P0_0[0] + (P0_1[0] - P0_0[0]) * progress)]
    point1 = [round(P1_0[1] + (P1_1[1] - P1_0[1]) * progress), round(P1_0[0] + (P1_1[0] - P1_0[0]) * progress)]
    image[point1[1]:, point1[0]:] = 0  # region2

    # region3
    line = lambda x: point0[1] + (point1[1] - point0[1]) / (point1[0] - point0[0]) * x
    for x in range(point1[0]):
        image[round(line(x)):, x] = 0
    return image


def preprocess(type, file_no, width, height):
    image_name = sample.lower() + '_{}_{}.bmp'.format(type, file_no)  # Access to the image
    image = cv2.imread(str(images.joinpath(image_name)))
    image = cv2.resize(image, (width, height))  # Resize it so you can display it on your pc

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = remove_regions(file_no, gray)  # remove the regions that are not needed for the analysis
    # gray = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)  # removing noise
    gray = cv2.GaussianBlur(gray, (1, 5), 5)  # makes the image blur

    # Find Canny edges
    edged = cv2.Canny(gray, 50, 200)  # edge detection

    return image, gray, edged


def find_midpoint(x, width, fraction1, fraction2):
    """
    This function finds the vertical position of the midpoint of the structure, which is on the line that the crack propagates through. Therefore, it actually calculates the vertical position of the ending point of the crack. It does calculations in a region stated by the user with the help of arguments fraction1 and fraction2.
    x: (array) the data of the image that the edge detection is used on, should only consist of 0 and 255.
    fraction1: (float) should be between 0 and 1, it is the horizontal position of the starting point of the region that the function will do calculations on to find the midpoint.
    fraction2: (float) horizontal position of the ending point of the region.
    Fractions should be given such that the region consists of the structure itself, and it should be before the ruler starts.
    """
    data = x[:, int(fraction1 * width):int(fraction2 * width)]
    rows, columns = data.shape
    # iteration one: find a preliminary midpoint
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

    # iteration two: find the midpoint
    midpoints = []
    for i in range(columns):
        whites = np.argwhere(data[:, i])[:, 0]
        midpoint_index = np.searchsorted(whites, most_frequent_value)
        if 0<midpoint_index<len(whites):
            midpoints.append((whites[midpoint_index-1] + whites[midpoint_index])/2)
    return np.average(midpoints)


def find_center(image, midpoint, file_no):
    """
    This function finds the vertical position of the cracktip. It does calculations in a region stated by the user with the help of arguments fraction1 and fraction2.
    image: (array) the data of the image that the edge detection is used on, should only consist of 0 and 255.
    midpoint: (int) the vertical position of the midpoint of the structure, which is on the line that the crack propagates through. Therefore, it actually calculates the vertical position of the ending point of the crack.
    """
    height_dcb = 22  # half height of the adhesive layer, added or substracted depending on whether the crack is on the top or bottom of the structure
    lean = midpoint - np.nonzero(image[:, -100])[0][0] - height_dcb  # for the dcb not being straight
    center = round(midpoint - (0.6 * (file_no/total_files_no) + 0.2) * lean)  # calculate the expected vertical position of the cracktip
    return center


def method_one(prev_cracktip, image, midpoint, file_no):
    center = find_center(image, midpoint, file_no)
    vertical_bounds = [center - round(height/250), center + round(height/250)]
    horizontal_bounds = [prev_cracktip - round(width/30), prev_cracktip + round(width/30)]

    # pretty self-explanatory
    target_area = image[vertical_bounds[0]:vertical_bounds[1], horizontal_bounds[0]:horizontal_bounds[1]]
    tip_column_index = horizontal_bounds[0] + np.argwhere(target_area)[:, 1].max()
    return center, tip_column_index


def method_two(prev_cracktip, image, center, conditionregion=[3, 7]):
    """
    Find the horizontal position of the cracktip, in a region around [center, prev_cracktip]
    The condition is to have at least one white pixel in each column in a region defined by conditionregion
    This ingores the millimetre markings on the ruler, as they are only 5 pixel wide
    """
    # target area bounds
    vertical_bounds = [center - round(height / 250), center + round(height / 250)]
    horizontal_bounds = [prev_cracktip - round(width / 100), prev_cracktip + round(width / 50)]

    # target area normalization
    target_area = image[vertical_bounds[0]:vertical_bounds[1], horizontal_bounds[0]:horizontal_bounds[1]]
    target_ones = np.heaviside(target_area, 0)

    # get the whether a vertical region has a white pixel
    vconvolved = convolve2d(target_ones, np.ones((conditionregion[0], 1)), mode='valid')
    vconvolved = np.heaviside(vconvolved, 0)

    # detect whether all pixels in a horizonatal region are white
    hconvolved = convolve2d(vconvolved, np.ones((1, conditionregion[1])), mode='valid')
    hconvolved = np.heaviside(hconvolved - conditionregion[1] + 1, 0)

    # get the index of the rightmost region that satisfies the condition
    tip_column_index = horizontal_bounds[0] + np.argwhere(hconvolved)[:, 1].max() + conditionregion[1] - 1
    return center, tip_column_index


def crack_length(type, file_no, width, height, hor_regions):
    """
    Calculate the crack lenght, and the crack tip positon heavily dependent on the edge recogtion quiality
    paratmeters:
    type: (int) 0 or 1
    file_no: (int) the number of the image file
    width: (int) the width of the image
    height: (int) the height of the image
    hor_regions: (list) the horizontal region in which the dcb midpoint should lie
    """

    time1 = time.time()
    if (not contours.joinpath(sample.lower() + '_{}_{}_edge.bmp'.format(type, file_no)).exists()) or rerun_preprocessing:
        image, gray, edged = preprocess(type, file_no, width, height)
        print('new image processed')
    else:
        image = cv2.imread(str(images.joinpath(sample.lower() + '_{}_{}.bmp'.format(type, file_no))))
        edged = cv2.imread(str(contours.joinpath(sample.lower() + '_{}_{}_edge.bmp'.format(type, file_no))), cv2.IMREAD_GRAYSCALE)

    # Getting rid of useless region by making them all black

    # save the image to see whether the preprocessing works
    # cv2.imwrite('testing.bmp', edged)

    hor_regions = [hor_regions[0]/2048, hor_regions[1]/2048]
    # get the data of processed image    

    time2 = time.time()
    # Find the midpoint of the crack
    midpoint = find_midpoint(edged, width, hor_regions[0], hor_regions[1])
    center = find_center(edged, midpoint, file_no)
    if file_no == 45:
        pass
    # crack tip location with method 1
    method_one_fail = False
    if file_no < 14:
        crack_pos1 = np.array([center, 410])
    else:
        try:
            crack_pos1 = np.array(method_two(cracktips[file_no - 1, 1], edged, center, [3, 8]))
        except:
            crack_pos1 = cracktips[file_no - 1, :]
            method_one_fail = True

    time3 = time.time()

    # Calculate the crack length in pixels from the position of the ruler
    endpoint = round(fourteencm_point(file_no))
    crack_lengths.append(fourteencm_dist - (endpoint - crack_pos1[1]))
    starting_hor = (endpoint - fourteencm_dist)
    image_name_edged = sample.lower() + '_{}_{}_edge.bmp'.format(type, file_no)
    cv2.imwrite(os.path.join(dir_path + '_contours', image_name_edged), edged)

    if 40<file_no<50:
        cv2.line(edged, [0, center], [width, center], (255, 255, 255))
        cv2.rectangle(edged, [crack_pos1[1], crack_pos1[0]], [crack_pos1[1] + 3, crack_pos1[0] + 3], (0, 0, 255), -1)
        cv2.rectangle(edged, [starting_hor, crack_pos1[0]], [starting_hor + 3, crack_pos1[0] + 3], (0, 0, 255), -1)

        # Drawing the midpoint line on the image with edge detection

        # Saving both the image with the edge detection and the original one but with the contour and two points on it
        image_name_midline = sample.lower() + '_{}_{}_midline.bmp'.format(type, file_no)
        cv2.imwrite(os.path.join(dir_path+'_contours', image_name_midline), edged)
    if 40<file_no<50:
        cv2.rectangle(image, [crack_pos1[1], crack_pos1[0]], [crack_pos1[1] + 3, crack_pos1[0] + 3], (0, 0, 255), -1)
        cv2.rectangle(image, [starting_hor, crack_pos1[0]], [starting_hor + 3, crack_pos1[0] + 3], (0, 0, 255), -1)
        image_name_cracktip = sample.lower() + '_{}_{}_cracktip.bmp'.format(type, file_no)
        cv2.imwrite(os.path.join(dir_path+'_contours', image_name_cracktip), image)

    print('Finished file number: {}'.format(file_no))
    print(time2 - time1, time3 - time2)
    return crack_pos1, method_one_fail


def filter_crack_lengths(crack_lens, files_list):
    prev_len = crack_lens[0]
    len_iterator = iter(crack_lens[1:])
    filtered = np.array([prev_len])
    for file in files_list[1:]:
        current_len = next(len_iterator)
        invalid = abs(current_len - prev_len) > 0.005
        if invalid:
            filtered = np.append(filtered, np.nan)
        else:
            filtered = np.append(filtered, current_len)
        prev_len = current_len
    return filtered




# I put these two this way in case if you want to run the program for a part of the files 
total = 0
files_list = []
failures =[]

for j in files:
    crack_pos, failure= crack_length(0 , j, width, height, hor_region)
    cracktips = np.vstack((cracktips, crack_pos.T))
    if failure == False:
        failures.append(j)
    total += 1
    files_list.append(j)

# put all files that failed in at least one thing together
all_files_summoned = np.array(
    no_success_file_no + match_fail_file_no + wrong_start_file_no + fail1_fail_no + fail2_fail_no)
unique_list = np.unique(all_files_summoned, return_counts=True)
file_indices = unique_list[0].tolist()
index = 0
all_files_matched = []
for i in unique_list[0]:
    all_files_matched.append([i, unique_list[1][index]])
    index += 1
# Sort files from most problematic to least
all_files_sorted = sorted(all_files_matched, key=lambda x: x[1], reverse=True)
problem_level = np.sum(unique_list[1]) / (5 * total)

print('Problematic files from high to low: ', all_files_sorted, '[file_no, problem_count]')
print("Confidence level: ", (1 - len(file_indices) / total) * 100, '[%]')
print('Capturing level of right contour: ', (success[0] / total) * 100, '[%]')
print('Match level of both methods: ', (match_count[0] / total) * 100, '[%]')
print('Capturing level of the correct starting point: ', (start_hor_count[0] / total) * 100, '[%]')
print('Problem level: ', problem_level * 100, '[%]')
print('Problematic files percentage: ', (unique_list[0].shape[0] / total) * 100, '[%]')
print('First method failure level: ', (fail_count1[0] / total) * 100, '[%]')
print('Second method failure level: ', (fail_count2[0] / total) * 100, '[%]')
print('The starting position in the following files might be wrong: ', wrong_start_file_no)
print("The following files don't give almost the same result with both methods: ", match_fail_file_no)
print(
    "First method's result is not in the neighborhood of the contour with the highest arc length in the following files: ",
    no_success_file_no)
print('First method failure files: ', fail1_fail_no)
print('Second method failure files: ', fail2_fail_no)
print('The following files failed in both methods: ', failures)

crack_lengths = (np.array(crack_lengths) * one_pixel_m).tolist()



plt.figure()
plt.plot(files_list, crack_lengths, marker='.', markerfacecolor='red', markersize=7)
#plt.plot(files_list, manual_crack_lengths)
plt.xlabel('File No')
plt.ylabel('Crack length [m]')
plt.title('Crack length variation over files')
plt.grid()
plt.savefig(str(results) + '\\Crack_length_in_each_file.png', dpi=300)
plt.savefig(str(images) + '\\Crack_length_in_each_file.png', dpi=300)

#filter the crack lenghts and save them
filtered = filter_crack_lengths(crack_lengths, files_list)
crack_len_file = np.vstack((files_list, filtered))
np.savetxt(str(results.joinpath(sample + '_crack_lengths_filtered.txt')), crack_len_file)
np.savetxt(str(results.joinpath(sample + '_crack_lengths.txt')), np.vstack((files_list, filtered)).T)

c_0_original = images.joinpath('c_0.txt')
c_1_original = images.joinpath('c_1.txt')
c_0_new = results.joinpath('c_0.txt')
c_1_new = results.joinpath('c_1.txt')
shutil.copy(c_0_original, c_0_new)
shutil.copy(c_1_original, c_1_new)

end_time = time.time()
print('Execution time of the program: ', end_time - start_time, '[s]')
cv2.destroyAllWindows()
plt.close('all')
quit()
