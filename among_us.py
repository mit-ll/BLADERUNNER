#!/bin/python3
'''
    PROJECT:    Among Us

    AUTHOR:     Adam D. Wong (@MalwareMorghulis/"Gh0st")
    
    SUPPORTING ANALYSTS:
                Beau J. Guidry
                Raul Harnasch
                Steve Castellarin
                Joshua Nadeau

    PURPOSE:    Detect Style-GAN AI-Generated aka Synthetic faces
        as part of Counter-Information Operations

    REQUIRED ML-TRAINING FILE:
        - shape_predictor_68_face_landmarks.dat

    HOW-TO-RUN (DEFAULT):
        ~/foo/bar/: python among_us.py

    HOW-TO-RUN (OPTIONS):
        ~/foo/bar/: python among_us.py      # TBD

    FILE TREE:
    ./BLADERUNNER
        /amongus
            - among_us.py
            /src_predictor
                - shape_predictor_68_face_landmarks.dat
            /INPUT
            /LOGGING
            /MARKED
            /OUTPUT


    OUTPUT VERDICT:
    - CONFIRMED:
    - POSSIBLE:
    - NEGATIVE:
    - INCONCLUSIVE:
    - UNKNOWN:

    ASSUMPTIONS:
    - Script and samples are in the same relative folder.
    - Samples are in sub-folders relative to parent project folder.
    - Correct packages are installed (pip / requirements, see IMPORTS)
    - LEFT/RIGHT based on MIRRORed image (ie: image is the viewer looking into a mirror.
    - Not all Unit Tests (UTs) are currently used for detection, saved for future analysis.

    KNOWN ISSUES:
    - OpenCV imwrite() function is compatible only w/ ASCII characters, not UTF-16
        - Stopgap remediation:
            - May need to down-convert to UTF-8 with "temp" then restore original input name.
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
    - Add other face-mapping types.


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

## IMPORT - Pipped
import cv2                  # PIPPED: OpenCV-python
import dlib                 # PIPPED: Eye Detection

## IMPORT - Specific Modules
from imutils import face_utils                  # PIPPED: Face Landmarks w/ dlib
from shapely.geometry import Point, Polygon     # PIPPED: Unit Tests

## IMPORT - Other Noted Dependencies
# cmake                                         # PIPPED: dlib
# $conda install -c conda-forge dlib            # CONDA INSTALL Command

### VARIABLES ###
# VARIABLES - Flags (Production)
sensitive_detection = False
exact_mode = False
production_mode = True
send_to_file = True

# VARIABLES - Dot-plotting
target_lockon_face = False
show_landmark_dotplots = False
show_landmark_counter = False

# VARIABLES - Debugging
debugging = True
write_test = False
test_image_save = False

# SRC Predictor
predictor_dir = os.path.join(os.getcwd(), 'src_predictor')
DATFILE = os.path.join(predictor_dir, 'shape_predictor_68_face_landmarks.dat')

# INPUT
input_dir = os.path.join(os.getcwd(), 'INPUT')

# LOGGING
logging_dir = os.path.join(os.getcwd(), 'LOGGING')
marked_dir = os.path.join(os.getcwd(), 'MARKED')
debug_dir = os.path.join(os.getcwd(), 'DEBUG')

# OUTPUT
output_dir = os.path.join(os.getcwd(), 'OUTPUT')
amongus_csv = os.path.join(output_dir, 'output.csv')

# TESTBED DATA
samp_down_dir = os.path.join(os.getcwd(), 'INPUT')

# SINGLE TEST
TEST_img_sample = r'C:\Users\WONG\PROJECTS\bladerunner\amongus\INPUT\001.jpg'

# VARIABLES - Data Storage
unit_test_results = {}
out_data_dict = {}

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


# VARIABLES - TESTING Eye Data from PapersPlease.py
eye_pts = {
    # Base-10
    (100, 100): [(37, 47), (62, 47)],
    (200, 200): [(74, 94), (124, 94)],
    (300, 300): [(112, 141), (188, 141)],
    (400, 400): [(149, 189), (250, 189)],
    (500, 500): [(187, 236), (314, 236)],
    (600, 600): [(224, 284), (376, 284)],
    (700, 700): [(262, 331), (438, 331)],
    (800, 800): [(300, 378), (500, 378)],
    (900, 900): [(337, 425), (562, 425)],
    (1000, 1000): [(374, 473), (626, 473)],

    # Base-2
    (128, 128): [(48, 60), (80, 60)],
    (256, 256): [(96, 121), (160, 121)],
    (512, 512): [(192, 242), (320, 242)],
    (1024, 1024): [(384, 484), (640, 484)]
}


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

        # Draw GREEN thin rectangle around detected perimeter of face.
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
            # Mark faces with RED circle on landmarks
            if show_landmark_dotplots:
                cv2.circle(img_obj, (x, y), 1, (0, 0, 255), -1)

            # Tag landmarks with BLUE numeral
            if show_landmark_counter:
                cv2.putText(img_obj, str(index_LM), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

            # Save Image OBJ to Output Image Path
            if test_image_save:
                cv2.imwrite(path, img_obj)

            # Increment to change LMs and move to the next in the for-loop
            index_LM += 1

    # TEST - CYAN
    if debugging:
        # Mark eyes-coordinates as CYAN, change (x,y) for manual testing
        cv2.circle(img_obj, (386, 484), 1, (255, 255, 0), -1)
        cv2.circle(img_obj, (643, 484), 1, (255, 255, 0), -1)

        # Mark eyes-coordinates as CYAN, change (x,y) for individual testing
        # cv2.circle(img_obj, (x, y), 1, (255, 255, 0), -1)

    # Save image object to output as TEST.jpg
    if write_test:
        reso = detect_resolution(img_path)

        for key in eye_pts:
            if reso == key:
                # Mark eyes-coordinates as CYAN, change (x,y) for individual testing
                # LEFT
                cv2.circle(img_obj, eye_pts[key][0], 1, (255, 255, 0), -1)
                # RIGHT
                cv2.circle(img_obj, eye_pts[key][1], 1, (255, 255, 0), -1)

        # Mark Left-Side
        for i in LM_eyes[0]:
            cv2.circle(img_obj, marker[i], 1, (0, 0, 255), -1)

        # Mark Right-Side
        for k in LM_eyes[1]:
            cv2.circle(img_obj, marker[k], 1, (0, 0, 255), -1)

        
        marked_filename = os.path.basename(img_path)
        output_path = os.path.join(marked_dir, marked_filename)
        cv2.imwrite(output_path, img_obj)

        #test_name = "TEST.jpg"
        #test_path = os.path.join(debug_dir, test_name)
        #cv2.imshow("TEST", img_obj)
        #cv2.imwrite(test_path, img_obj)
        #cv2.waitKey(0)


    # Return [LM{}, Boolean]
    return marker, face_detected

## Image Metrics ##
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

def get_PapersPlease_data(resolution):
    ''' Retrieve coordinates (x,y) from PapersPlease for specific resolution '''
    # Saving Parameters as local variables
    r = resolution                  # Resolution to access eye_pts dictionary
    # Extract suspected coordinates for pupils (from PapersPlease)
    suspected_pupil = eye_pts[r]

    # Return (x, y)
    return suspected_pupil

def get_measured_distance(coord_a, coord_b):
    ''' '''
    # Save parameters as local variables
    left_pt = coord_a
    right_pt = coord_b

    # Difference between coordinates (tuples)
    # i: left-eye coordinate
    # j: right-eye coordinate
    # deviation = right_coord - left_coord
    dev = tuple(map(lambda i, j: j - i, left_pt, right_pt))

    # Returns a tuple (dev_x, dev_y)
    return dev

def check_eye_overlap(dict_lm, polygon_lm, pp_coord):
    ''' Check if Mapped_Eye overlaps with PapersPlease coordinates '''
    # Saving Parameters as local variables
    landmarks = dict_lm         # All landmark coordinates
    eye_lm = polygon_lm         # List of indexes for specific eye
    pp_pupil = pp_coord         # Resolution to access eye_pts dictionary

    # Create point object with suspected coordinate location (from PapersPlease)
    pt_obj = Point(pp_pupil)

    # Create a polygon over the center 4x points of the detected eye on the input image (in AmongUs).
    eye_Polygon = Polygon([
        landmarks[eye_lm[0]],       # Coordinates L/LM:37 or R/LM:43 (T/R)
        landmarks[eye_lm[1]],       # Coordinates L/LM:38 or R/LM:44 (T/L)
        landmarks[eye_lm[2]],       # Coordinates L/LM:40 or R/LM:46 (B/L)
        landmarks[eye_lm[3]]        # Coordinates L/LM:41 or R/LM:47 (B/R)
    ])

    # METHOD 1: High-Sensitivity - Coordinate must be inside the Polygon (inside but cannot touch edges)
    if sensitive_detection:
        result = eye_Polygon.contains(pt_obj)       # Check Eye coordinate is inside Mapped_Eye Polygon borders

    # METHOD 2: Low-Sensitivity - Coordinate must be within Polygon (on or inside edges)
    else:
        result = eye_Polygon.contains(pt_obj)       # Check Eye coordinate touches Mapped_Eye Polygon

    # Return Boolean
    return result

def get_coordinate_offset(coord_a, coord_b):
    ''' Measure x/y-offset between two points '''
    # Saving Parameters as local variables
    left_coord = coord_a
    right_coord = coord_b

    # Get deviations between two points
    deviation = get_measured_distance(left_coord, right_coord)

    # Return tuple (x,y)
    return deviation


## Unit Tests ##

def unit_test_pupil_overlap(dict_lm, polygon_lm, pp_coord):
    ''' Unit Test to check if both Mapped_Eyes overlap with PapersPlease coordinates '''
    # Saving Parameters as local variables
    landmarks = dict_lm         # All landmark coordinates
    eye_lm = polygon_lm         # List of indexes for specific eye
    pp_pupil = pp_coord         # Resolution to access eye_pts dictionary

    # Test both eyes
    left_match = check_eye_overlap(landmarks, eye_lm[0], pp_pupil[0])
    right_match = check_eye_overlap(landmarks, eye_lm[1], pp_pupil[1])

    # Return Boolean-Pair
    return [left_match, right_match]

def unit_test_pupil_offset(est_coord_set, pp_coord_set):
    ''' Measure x/y-offset between estimated eyes and suspected TPDNE eyes. '''
    # Saving Parameters as local variables
    estimated_pupil = est_coord_set     # Coordinate-Pair Calculated by GetApproxCenter
    suspected_pupil = pp_coord_set      # Coordinate-Pair Derived from Papers Please

    # Coordinate deviations between estimated (sample-calculated) and actual (PapersPlease)
    left_dev = get_coordinate_offset(estimated_pupil[0], suspected_pupil[0])
    right_dev = get_coordinate_offset(estimated_pupil[1], suspected_pupil[1])

    # Return tuple [LEFT:(x,y), RIGHT:(x,y)]
    return [left_dev, right_dev]

def unit_test_pupil_separation(est_coord_set):
    ''' Measure x/y-offset between estimated eyes (calculated only) '''
    # Save parameter as local variables
    left, right = est_coord_set

    # Get distance between two points
    deviation = get_coordinate_offset(left, right)

    # Return tuple (x,y) for distance/deviation
    return deviation

def unit_test_smile_dimensions():
    ''' Run unit tests to assess mouth width (at horizontal axis) and height (at vertical central axis) '''
    ## FIX ##

    return None

def unit_test_smile2pupil():
    ''' Run unit tests to get vertical distance from mouth-corner to pupil '''
    ## FIX ##

    return None


## Image Analysis ##
def process_image(path):
    ''' Extract image landmarks '''
    # Save parameter as local variable
    img_path = path

    # Map a face and return landmarks & detection status
    marks, detect_status = detect_face(img_path)
    resolution = detect_resolution(img_path)

    return marks, detect_status, resolution

def run_unit_tests(landmarks, image_resolution):
    # Save parameters as local variables
    marks = landmarks
    resolution = image_resolution

    # Calculate eye-coordinates in sample image
    estimated_coord = eye_catcher(marks, LM_eyes)

    # Retrieve PapersPlease coordinates
    pp_locations = get_PapersPlease_data(resolution)

    ## Unit Test
    # UT 1: Eye Location Test
    eye_verdict = unit_test_pupil_overlap(marks, LM_eyes, pp_locations)
    ut1_res = [eye_verdict]

    # UT 2: Eye-Offset Test (between estimated and known coordinates)
    marking_offset = unit_test_pupil_offset(estimated_coord, pp_locations)
    ut2_res = [marking_offset]

    # UT 3: Eye Separation Test
    eye_separation = unit_test_pupil_separation(estimated_coord)
    ut3_res = [eye_separation]

    # UT 4:

    # UT 5:

    # Compound List of UT results (which UT results are individually cast as lists)
    result_set = ut1_res + ut2_res + ut3_res

    # Returns [[bool L/E, bool R/E], [LE-offset(x,y), R/E-offset(x,y)], [Distance: (x,y)]]
    return result_set

def run_data_analysis(results):
    ''' Analyze Unit Test results and return results '''
    # Save parameter as local variable
    eye_catcher, offsets, spacing = results

    # UT 1: Pupil-Overlap
    left_eye, right_eye = eye_catcher

    # UT 2: Reserved for future

    # UT 3: Reserved for future

    # Decision Logic
    if left_eye & right_eye:
        decision = "AI_Confirmed"
    elif left_eye ^ right_eye:
        decision = "AI_Probable"
    else:
        decision = "AI_Negative"

    # Return String
    return decision


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


### MAIN ###

if production_mode:

    out_headerline = ['filename', 'resolution', 'verdict', 'UT1: eye_overlap', 'UT2: eye_offset', 'UT3: eye_dist']

    with open(amongus_csv, 'w', encoding='utf-8', newline='\n') as output_csv:
        out_writer = csv.writer(output_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        out_writer.writerow(out_headerline)

        # target_dir = get_sample_directory(os.getcwd())
        imgfiles = get_sample_files(input_dir)

        # Process each image in a folder
        print("\nDetector Operations\n")
        for img in imgfiles:
            landmarks, status, img_res = process_image(img)
            filename = os.path.basename(img)

            try:
                # Run Unit Test for results
                ut_data = run_unit_tests(landmarks, img_res)

                # Process UT Data for a verdict on StyleGAN status
                verdict = run_data_analysis(ut_data)

                # Build Output Report
                output_data = [filename, img_res, verdict]

                # Write to CSV or Print
                if send_to_file:
                    out_writer.writerow(output_data + ut_data)
                else:
                    print(output_data + ut_data)

            except KeyError as e:
                # Print (a, b) indicates resolution mismatch - not coded in dictionary.
                print(e)

        # Close the CSV File
        output_csv.close()

else:
    # Process the image to detect face and extract metadata
    landmarks, status, img_res = process_image(TEST_img_sample)

    # Filename Extraction
    filename = os.path.basename(TEST_img_sample)

    # Attempt to run unit tests on image
    try:
        # Run Unit Test for results
        ut_data = run_unit_tests(landmarks, img_res)

        # Process UT Data for a verdict on StyleGAN status
        verdict = run_data_analysis(ut_data)

        # Build Output Report
        output_data = [filename, img_res, verdict]

        ## TEST
        print(output_data)

    except KeyError as e:
        # Print (a, b) indicates resolution mismatch - not coded in dictionary.
        print(e)
