# pose-estimation-dlib
This repo provide a simple pose estimation code using dlib. Before using the program, you should download [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and unzip it first.

## Requisite
* python==3.8
* dlib==19.24.0
* opencv-python==4.5.3.56
* scipy==1.7.1

## Getting Started
* Clone this repo

      git clone https://github.com/Rayhchs/pose-estimation-dlib.git
      cd pose-estimation-dlib/src
      
* Download [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and put it into pose-estimation-dlib/

* Run
      python -m main

* Arguments
 | Keyboard arguments | Description |
 | ------------- | ------------- |
 | s | start program |
 | v | record frame as image |
 | q | quit |
 
 ## Acknowledgements
 Thank dlib provides amazing landmark detector.
