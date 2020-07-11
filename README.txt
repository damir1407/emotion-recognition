REQUIREMENTS:
	-Python 3.4, 3.5 or 3.6
	-pip install tensorflow
	-pip install keras
	-pip install pandas
	-pip install numpy
	-pip install opencv-python
	-pip install opencv-contrib-python
	-pip install pillow
	
USAGE FOR TRAINING:
	python train.py --batch X --epochs Y
	X - batch size as integer
	Y - number of training epochs as integer
	Dataset is not included, because the .csv file is too large. Training will not work without the file.

USAGE FOR MAIN:
	python main.py
	Press "q" to terminate the program
	Press "k" to predict emotion from captured image (make sure lighting is good)
