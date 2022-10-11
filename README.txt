PROJECT:
    BLADERUNNER

AUTHOR:
    Adam D. Wong (@MalwareMorghulis/"Gh0st")
    
SUPPORTING ANALYSTS:
    Beau J. Guidry
    Raul Harnasch
    Steve Castellarin
    Joshua Nadeau
    
ABOUT:
    BladeRunner is an over-arching project consisting of 2x scripts to counter StyleGAN-based synethetic faces.
    - (Detector) Papers_Please.py takes given known hostile data sets, scales photos down, and produces CSVs of coordinates of faces to load back into AmongUs.
    - (Analyzer) Among_Us.py takes unknown photos and tests PapersPlease coordinates against predicted eye-location. Based on mappings, they image may or may not be StyleGAN.


INSTALLATION INSTRUCTIONS:
1) Install Conda
2) Conda Install Environment or Requirements File for BladeRunner
3) Set up folders based on Installation File Tree
4) Download Pre-trained ML Predictor file from DLIB (See Credits)
5) Place predictor file into correct folders
6) Set known-hostile dataset from the Internet
7) Rename known-hostile samples to follow file naming schema (See Below in PapersPlease Image Standardization)

RUNTIME INSTRUCTIONS:
8) Conda Activate BladeRunner Environment
9) Navigate to PapersPlease Directory
10) Run Papers Please (preferably with IDE like PyCharm or VB Community)
11) Extract data from PapersPlease
12) Change Directory to AmongUs Directory
13) Run AmongUs (preferably with IDE like PyCharm or VB Community)
14) Move calculated data from PapersPlease into AmongUs (hard-coded in Variables Section) IAW dictionary formatting.


INSTALLATION FILE TREE:
    ./BLADERUNNER
        /papersplease
          - papers_please.py
          /src_predictor
            - shape_predictor_68_face_landmarks.dat
          /DATA
            # All results combined across different folders or resolutions
            - ALL_papersplease_results.csv
            # All results from A
            - pp_A.csv
            # All results from A-type photos with (r, r) resolution.
            - pp_A_(r, r).csv
            # etc...
          /INPUT
            # Pre-labeled or pre-sorted images
            - A_###.jpg (Face looking forward)
            - B_###.jpg (Face lookign left)
            ...
            - H_###.jpg (Faces with sunglasses)
          /MARKED
            # Input images have been scaled and placed here
            - A_001_100.jpg
            - A_001_128.jpg
            ...
            - A_001_1024.jpg
            # etc...
          /OUTPUT
            # Lists average coordinates for every resolution.
            - avg_resolutions.csv
            # Lists average coordinates for every resolution and for every face-type (A... B... C...)             
            - avg_type_resolutions.csv
        /amongus
          - among_us.py
          /src_predictor
            - shape_predictor_68_face_landmarks.dat
          /INPUT
            # Photos aree given arbitrary names
            - 001.jpg
            - 002.jpg
            ...
            - n.jpg
            # etc...
          /LOGGING
            # Placeholder for future logging to text/CSV for debugging.
          /MARKED
            # If activated - photos are merely saved here with marked eyes
            - 001.jpg
            - 002.jpg
            ...
            - n.jpg
            # etc...
          /OUTPUT
            # CSV file containing list of photos, verdicts, or metadata
            - output.csv      
 
 Tested Base-10 Resolutions:
    (1000, 1000)
    (900, 900)
    (800, 800)
    (700, 700)
    (600, 600)
    (500, 500)
    (400, 400)
    (300, 300)
    (200, 200)
    (100, 100)
    
Tested Base-2 Resolutions   
    (1024, 1024)
    (512, 512)
    (256, 256)
    (128, 128)
 
REQUIRED ML-TRAINING FILE:
    # See Reference Below in Credits
    - shape_predictor_68_face_landmarks.dat

PapersPlease Image Labeling Standardization Key (LETTER_####.jpg):
    A: Forward Facing
    B: Left Facing
    C: Right Facing
    D: Upward Facing
    E: Downward Facing
    F: Warped Faces (Monstrocities)
    G: Generic Backgrounds (Control Group)
    H: Hidden Faces (Sunglasses)

REQUIRED ML-TRAINING FILE:
    - shape_predictor_68_face_landmarks.dat

HOW-TO-RUN (DEFAULT):
    ~/foo/bar/: python among_us.py
    ~/foo/bar/: python papers_please.py

HOW-TO-RUN (OPTIONS, TBD):
    # Tools with options are not yet ready for production
    ~/foo/bar/: python among_us.py 
    ~/foo/bar/: python papers_please.py

ASSUMPTIONS:
  - Script and samples are in the same relative folder.
  - Samples are in sub-folders relative to parent project folder.
  - Correct packages are installed (pip / requirements, see IMPORTS)
  - LEFT/RIGHT based on MIRRORed image (ie: image is the viewer looking into a mirror).
  - Not all Unit Tests (UTs) are currently used for detection, saved for future analysis.
  - Using large enough StyleGAN dataset
  - Data is properly marked for PapersPlease Ingestion (A - forward facing images, B - left/right facing images, etc through H - sunglasses)
  - Tested photos are squared (r, r) resolutions defined in the code.
    
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

FUTURE WORK:
  - (Both) Adding flexibility to code (no longer hardcoded coordinates, extracting from file)
  - (Both) Using alternative predictors than 68-W Landmarks by DLIB
  - (AmongUs) Adding additional Unit Tests


DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2022 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis.

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
