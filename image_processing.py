import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from natsort import natsorted
import time
start_time = time.time()
point_one_cm = 11 # [pixels]
one_cm = point_one_cm * 10 # [pixels]
one_pixel = 1 / one_cm # [cm]
one_pixel_m = one_pixel/100 # [m] 

dir_path = 'C:\\Users\\boram\\OneDrive\\2nd Year\\Test, Analysis and Simulation\\Code\\Test-Analysis-and-Simulation\\A1'  # Change the directory path
file_start_no = 0
file_end_no = 153
total_files_no = file_end_no-file_start_no+1
types = [0, 1]
files = [*range(file_start_no, file_end_no+1)]
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
        if os.path.isfile(os.path.join(dir_path, path)):
            files.append(path)
    files = natsorted(files)

    file_numbers = [*range(start, end+1)]
    total_file_no = (end+1) * 2
    for i in range(total_file_no):
        if i <= end:
            os.rename(dir_path+'\\'+files[i], dir_path+'\\'+'a1_{0}_{1}.bmp'.format(types[0], file_numbers[i]))
        elif i > end:
            index = i - (end + 1)
            os.rename(dir_path+'\\'+files[i], dir_path+'\\'+'a1_{0}_{1}.bmp'.format(types[1], file_numbers[index]))
# renaming(0, 153, types=types)

width = 1000
height = 750
def find_midpoint(x, fraction1, fraction2):
    """
    This function finds the vertical position of the midpoint of the structure, which is on the line that the crack propagates through. Therefore, it actually calculates the vertical position of the ending point of the crack. It does calculations in a region stated by the user with the help of arguments fraction1 and fraction2.
    x: (array) the data of the image that the edge detection is used on, should only consist of 0 and 255.
    fraction1: (float) should be between 0 and 1, it is the horizontal position of the starting point of the region that the function will do calculations on to find the midpoint.
    fraction2: (float) horizontal position of the ending point of the region.
    Fractions should be given such that the region consists of the structure itself, and it should be before the ruler starts.
    """
    data = x[:, int(fraction1*width):int(fraction2*width)]
    rows, columns = data.shape
    midpoints = []
    for i in range(columns):
        row_indices = np.where(data[:, i] == 255)[0]
        end_of_first = row_indices[1]
        start_of_second = row_indices[2]
        if (start_of_second - end_of_first -1) % 2 != 0:
            midpoints.append(end_of_first + (start_of_second - end_of_first)/2)
        else:
            midpoints.append(end_of_first + (start_of_second - end_of_first -1)/2)
    mean_value = int(np.mean(np.array(midpoints)))
    return mean_value

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
def crack_length(type, file_no):
    image_name = 'a1_{}_{}.bmp'.format(type, file_no) # Access to the image
    image = cv2.imread(dir_path+ '\\' + image_name)
    image = cv2.resize(image, (width, height)) # Resize it so you can display it on your pc

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21) # removing noise
    gray = cv2.GaussianBlur(gray, (3,3), 0) # makes the image blur

    # Find Canny edges
    edged = cv2.Canny(gray, 30, 200) # edge detection
              
    # Getting rid of useless region by making them all black
    if file_no < 131:
        region1 = image[0:int(height*0.5), :]                   
        gray1 = cv2.cvtColor(region1, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(gray1, 64, 0, cv2.THRESH_BINARY)        
        edged[0:int(height*0.5), :] = thresh1               
        region2 = image[int(height*0.8):, :]                    
        gray2 = cv2.cvtColor(region2, cv2.COLOR_BGR2GRAY)                      
        ret, thresh2 = cv2.threshold(gray2, 64, 0, cv2.THRESH_BINARY)        
        edged[int(height*0.8):, :] = thresh2
    else:
        region1 = image[0:int(height*0.5), :]                    
        gray1 = cv2.cvtColor(region1, cv2.COLOR_BGR2GRAY)   
        ret, thresh1 = cv2.threshold(gray1, 64, 0, cv2.THRESH_BINARY)
        edged[0:int(height*0.5), :] = thresh1              
        region2 = image[int(height*0.85):, :]                    
        gray2 = cv2.cvtColor(region2, cv2.COLOR_BGR2GRAY)                       
        ret, thresh2 = cv2.threshold(gray2, 64, 0, cv2.THRESH_BINARY)        
        edged[int(height*0.85):, :] = thresh2
    
    data = np.asarray(edged) # get the data of processed image
    if file_no >= 108:
        midpoint = find_midpoint(data, 0.25, 0.3)
    else:
        midpoint = find_midpoint(data, 0.1, 0.2)
    
    # Method 1 for getting the horizontal position of the final crack point
    target_area = data[midpoint-2:midpoint+2, :]
    pos = []
    for i in range(4):
        if np.size(np.where(target_area[i, :] == 255)[0]) != 0:
            pos.append(np.where(target_area[i, :] == 255)[0].max())
    crack_hor = max(pos)
    crack_pos1 = [crack_hor, midpoint]

    # Finding Contours
    # Use a copy of the image e.g. edged.copy() since findContours alters the image
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Find the contour with the highest arc length
    perimeters = [cv2.arcLength(c, True) for c in contours]
    max_index = np.argmax(perimeters)
    cnt_arc = contours[max_index]
    arc_cnt = np.resize(cnt_arc, (cnt_arc.shape[0], cnt_arc.shape[-1]))
    vert = arc_cnt[:, 1]
    hor = arc_cnt[:, 0]

    # Draw all contours
    # -1 signifies drawing all contours
    # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    
    # Method 2 for getting the horizontal position of final crack point
    index = 0
    points = []
    indices_points = []
    for i in arc_cnt:
        if (i[1] == midpoint) or (i[1] == midpoint-1) or (i[1] == midpoint-2):
            points.append(i)
            indices_points.append(arc_cnt[index])
        index += 1
    points_np = np.array(points)
    crack_pos2 = np.zeros(2)
    if points_np.shape[0] > 0:
        points_np_max_vert = points_np[:, 1].max()
        if points_np_max_vert == midpoint-2:
            crack_pos2[0] = points_np[:, 0].max()
            crack_pos2[1] = midpoint-2
        elif points_np_max_vert == midpoint-1:
            crack_pos2[0] = points_np[:, 0].max()
            crack_pos2[1] = midpoint-1
        elif points_np_max_vert == midpoint:
            crack_pos2[0] = points_np[:, 0].max()
            crack_pos2[1] = midpoint
    crack_pos2 = crack_pos2.astype(int)

    # If first method fails to get it right, use the second method's result
    # The problem with the second method is that it might be underestimating the crack position
    # I think first method works better, but it fails under specific conditions such as when the structure is not straight (bended downwards) -> that's why it fails in the first 20 files
    crack_pos = []
    if crack_pos1[0] >= int(0.9*width) and not (crack_pos2 == np.zeros(2)).any():
        crack_pos = crack_pos2
        fail_count1[0] += 1 # If the first method fails to get the right crack position
        fail1_fail_no.append(file_no)
    elif crack_pos1[0] < int(0.9*width) and not (crack_pos2 == np.zeros(2)).any():
        crack_pos = crack_pos1
    else:
        fail_count2[0] += 1 # If the second method fails to capture the crack position
        fail2_fail_no.append(file_no)
        crack_pos = crack_pos1
    crack_pos = np.array(crack_pos).astype(int)

    # Checking if the first method's result is near to the contour with the highest arc length
    # This allows us to check if both methods give almost the same result since the second method uses the contour with the highest arc length to calculate the crack position
    # It also assess how well the code captures the right contour, whether it captures either the top or bottom side of the structure and it goes until the crack point -> usability of it
    expansion_size = 3
    dimension = 2*expansion_size +1
    check_matrix_rows = np.full(dimension, crack_hor)
    check_matrix_columns = np.full(dimension, midpoint)
    check_matrix = []
    operation = []
    for i in range(expansion_size+1):
        if i == 0:
            operation.append(0)
        else:
            operation.append(i)
            operation.append(-i)
    operation.sort()
    for i in range(dimension):
        for j in range(dimension):
            check_matrix.append([check_matrix_rows[i]+operation[i], check_matrix_columns[i]+operation[j]])

    problem_count = 0
    boolean_matrix = np.isin(arc_cnt, check_matrix)
    for i in boolean_matrix:
        if i.all():
            success[0] += 1
            problem_count += 1
            break
    if problem_count == 0:
        no_success_file_no.append(file_no)

    # Check if both methods give almost the same point
    if (abs(crack_pos1[0]-crack_pos2[0])<4) and (abs(crack_pos1[1]-crack_pos2[1])<4):
        match_count[0] += 1
    else:
        match_fail_file_no.append(file_no)

    # Check if it captures the starting horizontal position of the structure correct
    # This is crucial for calculating the crack length correct
    # If it does that, it is likely to capture the crack point as well
    starting_hor = hor.min()
    if (starting_hor >= int(0.07*width)) and (starting_hor < int(0.1*width)):
        start_hor_count[0] += 1
    else:
        wrong_start_file_no.append(file_no)

    # Calculate the crack length as a percentage relative to the whole width of the image
    crack_lengths.append(((crack_pos[0] - starting_hor)/width) * 100)

    # Drawing the contour with the highest arc length, the crack position, and the starting horizontal position of the structure on the image 
    cv2.drawContours(image, cnt_arc, -1, (255, 0, 0), 3)
    cv2.rectangle(image, [crack_pos[0], crack_pos[1]], [crack_pos[0]+3, crack_pos[1]+3], (0, 0, 255), -1)
    cv2.rectangle(image, [starting_hor, crack_pos[1]], [starting_hor+3, crack_pos[1]+3], (0, 0, 255), -1)

    # Drawing the midpoint line on the image with edge detection 
    cv2.line(edged, [0, midpoint], [width, midpoint], (255, 255, 255))

    # Saving both the image with the edge detection and the original one but with the contour and two points on it
    image_name_edged = 'a1_{}_{}_edge.bmp'.format(type, file_no)
    cv2.imwrite(os.path.join(dir_path+'\\contours', image_name), image)
    cv2.imwrite(os.path.join(dir_path+'\\contours', image_name_edged), edged)
    print('Finished file number: {}'.format(file_no))

# I put these two this way in case if you want to run the program for a part of the files 
total = 0
files_list = []
for j in files:
    crack_length(0, j)
    total += 1
    files_list.append(j)

# put all files that failed in at least one thing together
all_files_summoned = np.array(no_success_file_no + match_fail_file_no + wrong_start_file_no + fail1_fail_no + fail2_fail_no)
unique_list = np.unique(all_files_summoned, return_counts=True)
index = 0
all_files_matched = []
for i in unique_list[0]:
    all_files_matched.append([i, unique_list[1][index]])
    index += 1
# Sort files from most problematic to least
all_files_sorted = sorted(all_files_matched, key=lambda x: x[1], reverse=True)
problem_level = np.sum(unique_list[1])/(5 * total)

print('Problematic files from high to low: ', all_files_sorted, '[file_no, problem_count]')
print('Confidence level: ', (success[0]/total)*100, '[%]')
print('Match level of both methods: ', (match_count[0]/total)*100, '[%]')
print('Capturing level of the correct starting point: ', (start_hor_count[0]/total)*100, '[%]')
print('Problem level: ', problem_level*100, '[%]')
print('First method failure level: ', (fail_count1[0]/total)*100, '[%]')
print('Second method failure level: ', (fail_count2[0]/total)*100, '[%]')
print('The starting position in the following files might be wrong: ', wrong_start_file_no)
print("The following files don't give almost the same result with both methods: ", match_fail_file_no)
print("First method's result is not in the neighborhood of the contour with the highest arc length in the following files: ", no_success_file_no)
print('First method failure files: ', fail1_fail_no)
print('Second method failure files: ', fail2_fail_no)

crack_lengths_np = np.array(crack_lengths) * one_pixel_m
plt.figure()
plt.plot(files_list, crack_lengths_np, marker = '.', markerfacecolor = 'red', markersize = 7)
plt.xlabel('File No')
plt.ylabel('Crack length [m]')
plt.title('Crack length variation over files')
plt.grid()
plt.savefig('A1\\Crack_length_in_each_file.png')
cv2.destroyAllWindows()

end_time = time.time()
print('Execution time of the program: ', end_time-start_time, '[s]')