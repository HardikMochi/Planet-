<h1 align='center'>Planet: Understanding the Amazon from Space</h1>
Every minute, the world loses an area of forest the size of 48 football fields. And deforestation in the Amazon Basin accounts for the largest share, contributing to reduced biodiversity, habitat loss, climate change, and other devastating effects. But better data about the location of deforestation and human encroachment on forests can help governments and local stakeholders respond more quickly and effectively.

<br>In our Planet: Understanding the Amazon from Space competition, Planet challenged the Kaggle community to label satellite images from the Amazon basin, in order to better track and understand causes of deforestation.
The competition contained over 40,000 training images, each of which could contain multiple labels, generally divided into the following groups:
1. Atmospheric conditions: clear, partly cloudy, cloudy, and haze
2. Common land cover and land use types: rainforest, agriculture, rivers, towns/cities, roads, cultivation, and bare ground
3. Rare land cover and land use types: slash and burn, selective logging, blooming, conventional mining, artisanal mining, and blow down



## Table of Contents
<details open>
<summary>Show/Hide</summary>
<br>

1. [ File Descriptions ](#File_Description)
2. [ Technologies Used ](#Technologies_Used)    
3. [ Executive Summary ](#Executive_Summary)
   * [ 1. Webscraping, Early EDA, and Cleaning ](#Webscraping_Early_EDA_and_Cleaning)
       * [ Webscraping ](#Webscraping)
       * [ Early EDA and Cleaning](#Early_EDA_and_Cleaning)
   * [ 2. Further EDA and Preprocessing ](#Further_EDA_and_Preprocessing) 
   * [ 3. Modelling and Hyperparameter Tuning ](#Modelling)
   * [ 4. Evaluation ](#Evaluation)
       * [ Future Improvements ](#Future_Improvements)
   * [ 5. Neural Network Modelling ](#Neural_Network_Modelling)
   * [ 6. Revaluation and Deployment ](#Revaluation)
</details>

## File Descriptions
<details>
<a name="File_Description"></a>
<summary>Show/Hide</summary>
<br>

* <strong>[ Images ](https://github.com/awesomeahi95/Hotel_Review_NLP/tree/master/Images)</strong>: folder containing images used for testing and in readme file.
* <strong>[ Models ](https://github.com/awesomeahi95/Hotel_Review_NLP/tree/master/Models)</strong>: folder containing trained models saved with pickle
    * <strong>model and weight file of Base Model, VGG19, Resnet50, InceptionV3</strong>
    
* <strong>[ static ](https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/1.Webscraping_Early_EDA_and_Cleaning.ipynb)</strong>: folder containing css and javascript file
* <strong>[ templates ](https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/2.Further_EDA_and_Preprocessing.ipynb)</strong>: folder containing html files
* <strong>[ Planet.ipynb ](https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/3.Modelling_and_Hyperparameter_Tuning.ipynb)</strong>: Main File for project
* <strong>[ Procfile ](https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/4.Evaluation.ipynb)</strong>: file containing script to run app in heroku app
* <strong>[ app.py ](https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/5.Neural_Network_Modelling.ipynb)</strong>: this file contain script to create web application
* <strong>[ inference.ipynb ](https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/6.Revaluation_and_Deployment.ipynb)</strong>: this file containing script for prediction on image
* <strong>[ util.py ](https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Classification.py)</strong>: this file contain function used in web application
</details>

## Tecnologies Used:
<details>
<a name="Technologies_Used"></a>
<summary>Show/Hide</summary>
<br>
    
* <strong>Python</strong>
* <strong>Pandas</strong>
* <strong>Numpy</strong>
* <strong>Matplotlib</strong>
* <strong>Scikit-Learn</strong>
* <strong>Open-Cv</strong>
* <strong>Keras</strong>
* <strong>Tensorflow</strong>
* <strong>Flask</strong>
* <strong>Heroku</strong>
</details>

   
<a name="Executive_Summary"></a>
## Executive Summary
<details open>


<a name="Dataset_Description"></a>
### Dataset_Description:
<details open>
<summary>Show/Hide</summary>
<br>
The training set consists of 40479 images, while there are 61191 images in testing set.
<br>the dataset is downloaded from https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data
<br>The dimension of image is  (256, 256, 3).
<br> training the models, a validation split of 0.3 is used which splits the training set such that 70% of the training data is used for training and the remaining 30% for testing. The batch size used is 128
and the number of epochs used is 50.

<h5>Images</h5>
<table><tr><td><img src='https://github.com/HardikMochi/Planet-Understanding-the-Amazon-from-Space/blob/main/Images/3.jpg' ></td><td><img src='https://github.com/HardikMochi/Planet-Understanding-the-Amazon-from-Space/blob/main/Images/4.jpg' ></td></tr></table>
</detail>
<h4>Labels</h4>
Each image that used for training has several labels. Totally, 17 labels are involved as shown in next section
<br><h2></h2>
<table>
  <tr>
    <td>clear</td>
    <td>water</td>
    <td>agriculture</td>
    <td>habitation</td>
    <td>artisinal_mine</td>
    <td>conventional_mine</td>
  </tr>
  <tr>
    <td>haze</td>
    <td>cloudy</td>
    <td>blooming</td>
    <td>blow_down</td>
    <td>bare_ground</td>
    <td>selective_logging</td>
  </tr>
  <tr>
    <td>road</td>
    <td>primary</td>
    <td>cultivation</td>
    <td>slash_burn</td>
    <td>partly_cloudy</td>
    <td></td>
  </tr>
</table>

</detail>
<a name="Data_Preprocessing"></a>
### Data_Preprocessing:
<details open>
<summary>Show/Hide</summary>
<br>
The images are first pre-processed by reading image and  Each  image is then scaled and reshaped to the size: height- 128, width-128 and the number of channels as 3.
the apply Augmentation and normalize the images.
<br><b>One-Hot-Encoding</b> was used to encode the Labels

<h3>Augmentation<h3>

An example of image augmentation for one image.

Original | 90° | 180° | 270° | Flip H | Flip V
---------|-----|------|------|--------|-------
<img src="https://github.com/HardikMochi/Planet-Understanding-the-Amazon-from-Space/blob/main/Images/7.jpg" width="100"> | <img src="https://github.com/HardikMochi/Planet-Understanding-the-Amazon-from-Space/blob/main/Images/7_90.jpg" width="100"> | <img src="https://github.com/HardikMochi/Planet-Understanding-the-Amazon-from-Space/blob/main/Images/7_180.jpg" width="100"> | <img src="https://github.com/HardikMochi/Planet-Understanding-the-Amazon-from-Space/blob/main/Images/7_270.jpg" width="100"> | <img src="https://github.com/HardikMochi/Planet-Understanding-the-Amazon-from-Space/blob/main/Images/7_flipH.png" width="100"> | <img src="https://github.com/HardikMochi/Planet-Understanding-the-Amazon-from-Space/blob/main/Images/7_flipV.png" width="100">

</details>



<a name="Model_Development"></a>
### Model_Development:
<details open>
<summary>Show/Hide</summary>
<br>
In this Project I I experimented with different models to get the better.
<br>


1. Base Model :  I build simple base model using  input layer, convolutional layers, max pooling layers, dense layers, flatten layers  and dropout layers.
2. Transfet Learning :
   * VGG19 
   * ResNet50
   * InceptionV3
   
<br>In this case, we are working neither with a binary or multi-class classification task; instead, it is a multi-label classification task and the number of labels are not balanced, with some used more heavily than others.

As such, we chose the F-beta metric, specifically the F2 score. This is a metric that is related to the F1 score (also called F-measure).   

<br> The F2 score I got from this model is below:

<h5 align="center">Table Comparing Best Models</h5>
<p align="center">
  <img src="https://github.com/HardikMochi/Planet-Understanding-the-Amazon-from-Space/blob/main/Images/15.PNG" width=700,height=400>
</p>
</details>

<a name="Evaluation"></a>
### Evaluation
<details open>
<summary>Show/Hide</summary>
<br>

I focused on 3 factors of defining a good model:

1. Good Validation F2 score
2. Good Training  F2 score
3. Small Difference between Training and Validation F2 score

<br>I chose the VGG19 model as my best model. because it has the highest validation score is 0.9056 and  train score is 0.9197. and thers is not more diffrence in train validation score.

<h5 align="center">Test Results</h5>
<p align="center">
  <img src="https://github.com/HardikMochi/Planet-Understanding-the-Amazon-from-Space/blob/main/Images/16.PNG" width=600 height=200>
</p>

    
</details>
