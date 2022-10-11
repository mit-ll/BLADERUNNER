#!/bin/python3
'''
    PROJECT:    Papers Please

    AUTHOR:     Adam D. Wong (@MalwareMorghulis/"Gh0st")

    SUPPORTING ANALYSTS:
                Beau J. Guidry
                Raul Harnasch
                Steve Castellarin
                Joshua Nadeau

    PURPOSE:    Calculates the AIGen Photo's eye location based on common scaling resolutions
                Data used as a goal-posts detection script via among_us.py

    REQUIRED ML-TRAINING FILE:
        - shape_predictor_68_face_landmarks.dat

    HOW-TO-RUN (DEFAULT):
        ~/foo/bar/: python papers_please.py

    HOW-TO-RUN (OPTIONS):
        ~/foo/bar/: python papers_please.py      # TBD

    FILE TREE:
    ./BLADERUNNER
        - papers_please.py
        /src_predictor
            - shape_predictor_68_face_landmarks.dat
        /DATA
        /INPUT
        /MARKED
        /OUTPUT


    OUTPUT:
        Resolution: LE(x, y), RE(x,y)

    ASSUMPTIONS:
    - Script and samples are in the same relative folder.
    - Samples are in sub-folders relative to parent project folder.
    - Correct packages are installed (pip / requirements, see IMPORTS)
    - LEFT/RIGHT based on MIRRORed image (ie: image is the viewer looking into a mirror).

    KNOWN ISSUES:
    - OpenCV imwrite() function is compatible only w/ ASCII characters, not UTF-16
        - Stopgap remediation:
            - Down-convert to UTF-8 with "temp" then restore original input name.
        - Reference:
            - https://stackoverflow.com/questions/54189911/cv2-imwrite-and-german-letters-%C3%A4-%C3%BC-%C3%B6

    CREDITS:
    - Dr. Adrian Rosenbrock's tutorial on face-detection.
        - Reference:
            - https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
        - License:
            - https://pyimagesearch.com/faqs/
    - DLIB 68-landmark pretrained dataset
        - Reference:
            - http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        - License:
            - http://dlib.net/license.html
    - OpenCV
        - License:
            - https://opencv.org/license/

    FIX LIST:
    - Automate manual functions
        - Results
            - Tab images by Type/Class, then by Resolution in XLSX
            - Send results to separate and master CSV file
        - Averages
            - Analyzing average + stddev across all images (of similar resolution)
            - Analyzing average + stddev across all type-images (of similar class... A... B... C...)
            - Output results to separate XLSX files and 1x Master XLSX File
        - OUTPUT
            - send to text file?


    RELEASE REVIEW:
    DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

    This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

    © 2022 Massachusetts Institute of Technology.

    The software/firmware is provided to you on an As-Is basis

    Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.




'''

### IMPORT ###
## Import - Default
import csv                  # DEFAULT: CSV Parsing
import os                   # DEFAULT: OS Ops
import re                   # DEFAULT: RegEx
import urllib               # DEFAULT: cURL
import time                 # DEFAULT: Time Measurement
import hashlib              # DEFAULT: File Hashing
import statistics           # DEFAULT: statistics

## Import - Pipped / Conda
import cv2                  # PIPPED: OpenCV-python
import dlib                 # PIPPED: Eye Detection
#import numpy
import numpy as np          # PIPPED: Numpy
import pandas as pd         # PIPPED: Pandas

## Import - Specific Modules
from imutils import face_utils                  # PIPPED: Face Landmarks w/ dlib
from shapely.geometry import Point, Polygon     # PIPPED: Unit Tests
from tqdm import tqdm                           # PIPPED: TQDM progress bars

## IMPORT - Other Noted Dependencies
# cmake                                         # PIPPED: dlib
# $conda install -c conda-forge dlib            # CONDA INSTALL Command

### VARIABLES ###
# VARIABLES - Flags
write_test = True
exact_mode = False

# VARIABLES - Debugging Dot-plotting
target_lockon_face = False
show_landmark_dotplots = False
show_landmark_counter = False

# VARIABLES - Debug Eye-Marker
save_marked_bullseye = False                # Mark image & overwrite scaled image with eye-marked image

# VARIABLES - Resolution Lists
base10_res = [(1000, 1000), (900, 900),
              (800, 800), (700, 700),
              (600, 600), (500, 500),
              (400, 400), (300, 300),
              (200, 200), (100, 100)]
base2_res = [(1024, 1024), (512, 512), (256, 256), (128, 128)]
all_res = base2_res + base10_res

# VARIABLES - Data Storage
type_average_set = []
resolution_avg_set = []

# START DIR
base_dir = os.getcwd()

# INPUT DIR
input_dir = os.path.join(base_dir, 'INPUT')

# OUTPUT DIR
data_dir = os.path.join(base_dir, 'DATA')
calc_dir = os.path.join(base_dir, 'CALC')
marked_dir = os.path.join(base_dir, 'MARKED')
output_dir = os.path.join(base_dir, 'OUTPUT')

# METADATA PATH
data_csv_path = os.path.join(data_dir, 'ALL_papersplease_results.csv')
data_path_template = os.path.join(data_dir, 'pp_')

# OUTPUT PATH
out_avg_path_template = os.path.join(output_dir, 'avg_')

# SRC Predictor
predictor_dir = os.path.join(base_dir, 'src_predictor')
DATFILE = os.path.join(predictor_dir, 'shape_predictor_68_face_landmarks.dat')


# VARIABLES - Quick dlib Index
# LIST containing INDEX values to reference DICT of landmark files.
#   Eye coordinates derived from middle 4 coordinates or corners of the pupil LMs.
LM_eye_left = [37, 38, 40, 41]
LM_eye_right = [43, 44, 46, 47]
LM_smile_x = [48, 54]
LM_smile_y = [51, 57]

# Tuple of related lists
LM_eyes = (LM_eye_left, LM_eye_right)
LM_smile = (LM_smile_x, LM_smile_y)

# CSV Worksheets
styleGAN_types = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
styleGAN_resolutions = all_res


### FUNCTIONS ###
## ML-Detection ##
def detect_resolution(path):
    ''' Identifies image resolution '''
    # Saving Parameter as Local Variable
    img_path = path

    # Read-in the image as an image file object
    img_obj = cv2.imread(img_path)

    # Save Image Resolution (WHC format)
    img_height, img_width, channel = img_obj.shape
    img_res = (img_height, img_width)

    return img_res

def detect_face(path):
    ''' Detects a single face via filepath '''
    
    # The detect_face function is based on & derived from Dr. Adrian Rosebrock's PyImageSearch Tutorials per his license:
    #   See Credits above for Reference/License URLs
    ''' Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
            
        The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

        Notwithstanding the foregoing, you may not use, copy, modify, merge, publish, distribute, sublicense, create a derivative work, and/or sell copies of the Software in any work that is designed, intended, or marketed for pedagogical or instructional purposes related to programming, coding, application development, or information technology. 
            
        Permission for such use, copying, modification, and merger, publication, distribution, sub-licensing, creation of derivative works, or sale is expressly withheld. 

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
            
        IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. '''
    
    
    # Saving Parameter as Local Variable
    img_path = path

    # Create face detector w/ dlib
    face_dlib_detector = dlib.get_frontal_face_detector()

    # Load dlib source file (pre-trained DAT file to build predictor)
    dlib_predictor = dlib.shape_predictor(DATFILE)

    # Read-in the image as an image file object
    img_obj = cv2.imread(img_path)

    # Convert image to GRAYSCALE for OpenCV and dlib tracing
    gray_img = cv2.cvtColor(img_obj, cv2.COLOR_BGR2GRAY)

    # Create the rectangle outline around the detected face
    bounding_box = face_dlib_detector(gray_img, 1)

    # Raise BOOLEAN flag for bounding_box emptiness
    # EXIT if face not detected
    if bounding_box:
        face_detected = True
    else:
        face_detected = False
        empty_dict = {}
        return empty_dict, face_detected

    # Continue to map out the face
    for(TL, corners) in enumerate(bounding_box):
        # bounding_box is an array of rectangle_object with list of the corner-coordinates
        #   bounding_box returns: rectangles[[(x1,y1) (x2,y2)]].
        # Corners are TL(x1,y1), BR(x2,y2) returned as list rectangle

        # Create a face detector w/ dlib.
        # Predictor returns a dlib_pybind11.full_object_detection object.
        face = dlib_predictor(gray_img, corners)

        # Converts face to Numpy Array
        # Extracts List of Coordinates to trace full facial structure from predictor.
        face = face_utils.shape_to_np(face)

        # Define approx bounding_box coordinates to identify faces
        (x, y, w, h) = face_utils.rect_to_bb(corners)

        # Draw GREEN thin rectangle around detected perimeter of face (color: BGR-format).
        if target_lockon_face:
            cv2.rectangle(img_obj, (x, y), (x+w, y+h), (0, 255, 0), 1)

        # Pre-configure Counter and dictionary
        index_LM = 0
        marker = {}

        # Save Coordinates of individual image (face_object).
        for (x, y) in face:
            # Save coordinate to marker dictionary with index as key
            marker[index_LM] = (x, y)

            ## Debugging Mode
            # Debug landmark mapping
            if show_landmark_dotplots:
                # Mark all points on face with RED circle on landmarks (color: BGR-format)
                cv2.circle(img_obj, (x, y), 1, (0, 0, 255), -1)

                # Tag landmarks with BLUE numeral (color: BGR-format)
                if show_landmark_counter:
                    cv2.putText(img_obj, str(index_LM), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

            # Increment index counter
            index_LM += 1

        # Return [LM{}, Boolean]
        return marker, face_detected

def mark_eyepoints(path, tuple_pair):
    ''' Mark eye points and overwrite image '''
    # Saving Parameter as Local Variable
    img_path = path

    # Read-in the image as an image file object
    img_obj = cv2.imread(img_path)

    # Mark calculated center-eye points with CYAN circle (color: BGR-format)
    for coordinate in tuple_pair:
        cv2.circle(img_obj, coordinate, 1, (255, 255, 0), -1)

    # Overwrite Image OBJ
    cv2.imwrite(img_path, img_obj)

    # Void function
    return None

## Image Manipulation ##
def scale_image(source_file_path, destination_dir, resolution):
    ''' Take an image and scale it down based on resolution '''

    # Saving Parameter as Local Variable
    current_path = source_file_path

    # Save the destination folder path
    new_dir = destination_dir

    # Save the filename and extension for modification (after scaling)
    filename, ext = os.path.basename(current_path).split(".")

    # Saving resolution to scale-down
    new_res = resolution

    # Watermarking scale-down photos with "name_resolution.jpg"
    new_filename = filename + "_" + str(new_res[0]) + "." + ext

    # Save the new filename and path for scaling operations
    new_path = os.path.join(new_dir, new_filename)

    # Read-in image
    img_obj = cv2.imread(current_path)

    # Scale image down
    resized_img_obj = cv2.resize(img_obj, new_res, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(new_path, resized_img_obj)

    return None


## Calculations ##
def get_approx_center(dict_lm, subset_index_list):
    ''' Calculates approximate center of box or eye '''

    # Pre-set sum values
    sum_x, sum_y = 0, 0

    # Count number of coordinates
    total_coor = len(subset_index_list)

    # Retrieve coordinates from each key in the dictionary
    for i_key in subset_index_list:

        # Pull data
        temp_x, temp_y = dict_lm[i_key]

        # Add to sum values
        sum_x += temp_x
        sum_y += temp_y

    # Find average
    if exact_mode:
        # Exact division to decimals
        approx_x = sum_x / total_coor
        approx_y = sum_y / total_coor
    else:
        # Coordinates will round-down <= 0.5
        approx_x = round(sum_x / total_coor)
        approx_y = round(sum_y / total_coor)

    return (approx_x, approx_y)

def eye_catcher(dict_lm, known_eye_indexes):
    ''' Retrieve coordinates (x,y) from eyes in sample '''
    # Saving Parameters as local variables
    landmarks = dict_lm                     # Resolution to access eye_pts dictionary
    left_eye_idx = known_eye_indexes[0]     # Left-Eye Landmark Numbers
    right_eye_idx = known_eye_indexes[1]    # Right-Eye Landmark Numbers

    # Calculate coordinates
    (lx, ly) = get_approx_center(landmarks, left_eye_idx)
    (rx, ry) = get_approx_center(landmarks, right_eye_idx)

    # Returns Coordinate-pairs [LEFT:(x,y), RIGHT:(x,y)]
    return [(lx, ly), (rx, ry)]

def get_image_uid(path):
    ''' Extract class & UID number for image sample '''
    # Save parameter as local variable
    img_path = path

    filename = os.path.basename(img_path)
    sample_type, sample_uid, img_res = filename.split("_")

    # Return list containing image_type (A, B, C... x), and UID number (001... 002... n)
    return [filename, sample_type, sample_uid]

def get_average(mode, list):
    ''' Calculate average coordinate across a list of coordinates '''
    # Save parameters as local variables
    show_decimal = mode
    eye_list = list

    # Decimals
    if show_decimal:
        list_sum = tuple(sum(i) for i in zip(*eye_list))
        list_avg = tuple(k / len(eye_list) for k in list_sum)

    # Integer w/ round() function
    else:
        list_sum = tuple(sum(m) for m in zip(*eye_list))
        list_avg = tuple(round(n / len(eye_list)) for n in list_sum)

    # Returns Tuple of Avg(x, y)
    return list_avg

## Data Collection ##
def collect_image_metadata(path):
    ''' Calculate image eye-coordinates from samples '''
    img_path = path

    # Resolution of Scaled Image
    resolution = detect_resolution(img_path)

    # Get metadata from image file
    uid_data = get_image_uid(img_path)
    uid = uid_data[1:]
    filename = [uid_data[0]]

    # Detecting Face
    marks, facebool = detect_face(img_path)

    # Calculate eye-coordinates in sample image
    estimated_coord = eye_catcher(marks, LM_eyes)
    left_eye = estimated_coord[0]
    right_eye = estimated_coord[1]

    # Overwrite scaled images with eye-markers
    if save_marked_bullseye:
        mark_eyepoints(img_path, estimated_coord)

    # Collect data as list
    img_info = uid + [resolution] + [left_eye, right_eye] + filename

    # Return list of data [Sample_Type, Sample_UID, (rh, rw), (lx,ly), (rx,ry), Filename]
    return img_info


## OS Operations ##
def get_sample_directory(path):
    ''' List all directories with "sample_" '''
    sample_directories = []
    list_dir = os.listdir(path)

    # Check if folder (under "/amongus") name beings with "sample_*/"
    for directory in list_dir:
        if os.path.isdir(directory):
            if directory.startswith("sample_"):
                sample_directories.append(os.path.abspath(directory))

    return sample_directories

def get_sample_files(path):
    ''' Get List of Files '''
    # Save Parameter as Local Variable
    target_path = path

    sample_files = []
    list_sample_img_files = os.listdir(target_path)

    for img_name in list_sample_img_files:
        sample_files.append(os.path.join(target_path, img_name))

    return sample_files

def list_scalers():
    ''' Take resolution variables and  '''
    # Merge resolution lists together
    all_resolutions = base2_res + base10_res

    # Sort resolution scale in reverse-order (descending order)
    all_resolutions.sort(reverse=True)

    return all_resolutions



### MAIN ###

header_lbl = 'type,uid,res_h,res_w, left_x,left_y,right_x,right_y,filename'

pp_result_data = []

# Create Master List of Resolutions
res_scaler_list = list_scalers()

# List original images
imgfiles = get_sample_files(input_dir)

# Scale-Down Photos based on given resolution
print("\nScaling Operations\n")
for img in imgfiles:
    # Scale photos from OG photos
    for resolution in tqdm(res_scaler_list):
        scale_image(img, marked_dir, resolution)

# Get list of full-filepaths for all scaled images
all_res_img = get_sample_files(marked_dir)

# Calculating eye-coordinates for scaled images.
print("\nEye-Catcher Operations\n")
for scaled_img in tqdm(all_res_img):
    # Collect scaled image metadata
    metadata = collect_image_metadata(scaled_img)

    # Master list of metadata lists
    pp_result_data.append(metadata)

# Saving data list as 2D Numpy Array for mathematics.
#   dtype=object set to avoid deprecation warning
array_2d = np.asarray(pp_result_data, dtype=object)

# print(array_2d.shape)
# Send Numpy 2D-Array into Pandas Data Frame.
data_frame = pd.DataFrame(array_2d)

# Mark the headers for the CSV files & Pandas Data Frame
data_frame.columns = ['type', 'uid', 'resolution (h, w)', 'left_eye (x, y)', 'right_eye (x, y)', 'filename']

# Send full data to master-CSV
data_frame.to_csv(data_csv_path, mode='a', encoding='utf-8', index=False, header=True)


# Dump metadata pertaining to specific TYPE into separate CSV as "pp_result_TYPE.csv".
# For each letter in styleGAN_types...
for letter in styleGAN_types:
    # Create temporary boolean mask search for specific StyleGAN title marking
    #   Match any records/rows in the Data Frame which have the element or letter desired.
    temp_type_mask = data_frame['type'] == letter

    # Create a temporary dataframe for that entry with that specific boolean mask search
    temp_type_df = data_frame[temp_type_mask]

    # Write Pandas Data frame out to the DATA dir as "pp_result_TYPE.csv".
    temp_type_df.to_csv(data_path_template + letter + '.csv', mode='a', encoding='utf-8', index=False, header=True)

    # Dump metadata pertaining to TYPE & RESOLUTION into separate CSV as "pp_result_TYPE_(rh,rw).csv".
    # For each resolution tuple in styleGAN resolutions... (for this specific letter-TYPE of image)

    # Index for logic and dataframe iteration by Type+Resolution
    ltr_res_index = 0

    # Capture subtypes Type + Resolution:
    for ltr_res in styleGAN_resolutions:
        # Create temporary boolean mask search for specific StyleGAN title marking
        #   Match any records/rows in the Data Frame which have the element or letter desired.
        temp_ltr_res_mask = temp_type_df['resolution (h, w)'] == ltr_res

        # Create a temporary dataframe for that entry with that specific boolean mask search
        ltr_res_combined_df = temp_type_df[temp_ltr_res_mask]

        # Operations are split because we're appending separate dataframes: A(rh,rw) with B(rh,rw), etc...
        # Write Mode with header
        if ltr_res_index == 0:
            # Write Pandas Data frame out to the DATA dir as "pp_result_TYPE_(rh,rw).csv"
            ltr_res_combined_df.to_csv(data_path_template + letter + '_' + str(ltr_res) + '.csv', mode='w',
                                       encoding='utf-8', index=False, header=True)
        # Append Mode w/o header
        else:
            # Write Pandas Data frame out to the DATA dir as "pp_result_TYPE_(rh,rw).csv"
            ltr_res_combined_df.to_csv(data_path_template + letter + '_' + str(ltr_res) + '.csv', mode='a',
                                       encoding='utf-8', index=False, header=False)

        # Capture list of coordinates (which match type-resolution) for each eye
        left_eyes = ltr_res_combined_df['left_eye (x, y)'].tolist()
        right_eyes = ltr_res_combined_df['right_eye (x, y)'].tolist()

        # Find average coordinate for every list (by type & resolution)
        left_avg_by_type_res = get_average(exact_mode, left_eyes)
        right_avg_by_type_res = get_average(exact_mode, right_eyes)

        # Set type-resolution average data into 2D List
        type_average_set.append([letter, ltr_res, left_avg_by_type_res, right_avg_by_type_res])

        # Increment
        ltr_res_index = ltr_res_index + 1


# Pandas Data Frame for all calculated averages (by resolution)
avg_type_res_frame = pd.DataFrame(type_average_set)
# Mark the headers for the CSV files & Pandas Data Frame
avg_type_res_frame.columns = ['type', 'resolution (rh, rw)', 'left_eye (x, y)', 'right_eye (x, y)']
# Send full data to master-CSV
avg_type_res_frame.to_csv(out_avg_path_template + "type_resolutions.csv", mode='w', encoding='utf-8', index=False, header=True)

# Index for Resolutions
res_index = 0

# Dump metadata pertaining to RESOLUTION into separate CSV as "pp_result_(rh,rw).csv".
# For each resolution tuple in styleGAN resolutions...
for res in styleGAN_resolutions:
    # For each resolution tuple in styleGAN resolutions...
    temp_res_mask = data_frame['resolution (h, w)'] == res

    # Create a temporary dataframe for that entry with that specific boolean mask search
    temp_res_df = data_frame[temp_res_mask]

    # Operations are split because we're appending separate dataframes: A(rh,rw) with B(rh,rw), etc...
    # Write Mode with header
    if res_index == 0:
        # Write Pandas Data frame out to the DATA dir as "pp_result_(rh,rw).csv".
        temp_res_df.to_csv(data_path_template + "by_Resolution_" +
                           str(res) + '.csv', mode='w', encoding='utf-8', index=False, header=True)
    # Append Mode w/o header
    else:
        # Write Pandas Data frame out to the DATA dir as "pp_result_(rh,rw).csv".
        temp_res_df.to_csv(data_path_template + "by_Resolution_" +
                           str(res) + '.csv', mode='a', encoding='utf-8', index=False, header=False)

    # Extract eye coordinates per resolution and send to List
    left_eyes = temp_res_df['left_eye (x, y)'].tolist()
    right_eyes = temp_res_df['right_eye (x, y)'].tolist()

    # Find average coordinate for every list (by resolution)
    left_avg_by_res = get_average(exact_mode, left_eyes)
    right_avg_by_res = get_average(exact_mode, right_eyes)

    # Set averages (by resolution only) data into 2D List
    resolution_avg_set.append([res, left_avg_by_res, right_avg_by_res])

    # Increment index
    res_index = res_index + 1

# Pandas Data Frame for all calculated averages (by resolution)
avg_res_frame = pd.DataFrame(resolution_avg_set)

# Mark the headers for the CSV files & Pandas Data Frame
avg_res_frame.columns = ['resolution (rh, rw)', 'left_eye (x, y)', 'right_eye (x, y)']

# Send full data to master-CSV
avg_res_frame.to_csv(out_avg_path_template + "resolutions.csv", mode='w', encoding='utf-8', index=False, header=True)

