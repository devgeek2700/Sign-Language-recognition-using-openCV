
# Sign Language Recognition(ASL)-openCV

The project aims at building a machine learning model that will be able to classify the various
hand gestures used for fingerspelling in sign language. In this user independent model,
classification machine learning algorithms are trained using a set of image data and testing is
done on a completely different set of data.

For the image dataset, depth images are used, which
gave better results than some of the previous literatures, owing to the reduced pre-processing time.

Various machine learning algorithms are applied on the datasets, including Convolutional
Neural Network (CNN). An attempt is made to increase the accuracy of the CNN model by pretraining it on the dataset. However, a small dataset was used for pre-training, which gave an accuracy of 85% during training.

In this project we had predicated American Sign Language (ASL).
In this project we have predicated numbers[0-9], letters[A-Y] and 10 basic gestures ["Don't Want","Hello","Help","I Love You","Money","No","Ok", "Thank You","Want","Yes"].

## Proccess

We use openCV to get input through webcam. We detect landmark through cvzone library. After
detecting landmark, once we save the gesture in folder it gives the meaning to the gesture. 250 to
300 pictures are given to individual gesture for accuracy. We have taken each picture in teachable
machine or training machine and trained it and converted it into one model file respectively and
gave its path in the output part of code. Lastly through openCV we access webcam and then the
user shows the sign which is predicted and analyzed to give the final detected result.

This our Sem-3 mini project.
## Hardware

- OS: Windows 10 and 11

- Processor: i5 and 8th generation

- RAM 8GB


## Libraries used

- python 3.10

- Tensorflow framework- 2.9.1,keras API -2.9.0

-  Real-time Computer vision using openCV - 4.6

- Google teachable Machine for Training part

- Numpy as np

- cvzone - 1.5.6

- Hands detector using matplotlib - 3.5.3 




## Roadmap 

- Create Dataset (In this project I have made my own dataset)
- Train the dataset
- Predicate the sign


## Create Dataset

![App Screenshot](https://github.com/devgeek2700/Sign-Language-recognition-using-openCV/blob/master/Output/dataset_img.png?raw=true)

## Train the dataset

![App Screenshot](https://github.com/devgeek2700/Sign-Language-recognition-using-openCV/blob/master/Output/trainning_img.png?raw=true)

## Predicate the sign

![App Screenshot](https://github.com/devgeek2700/Sign-Language-recognition-using-openCV/blob/master/Output/output_letter.png?raw=true)

## 



## Demo video

![App Screenshot](https://github.com/devgeek2700/Sign-Language-recognition-using-openCV/blob/master/Output/srl_output_demo_gif.gif?raw=true)





## 🛠 Skills
Python, openCV....

