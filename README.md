<div align="center">
  <center><h1>The Sorting Hat 🎩<br> EDA and Multinomial Logistic Regression 📈</h1></center>
  </div>

## Objectives:
This is a four parts project, the three first parts are about Exploratory Data analysis. The last part implements a class that performs classification via the Multinomial Logistic Regression algorithm (One vs All), with batch size as option to chose an optimization algorithm (such as stochastic gradient descent, batch or mini-batch gradient descent).
<br>
The goal of the project is to recreate the Sorting Hat from the Harry Potter series. We are provided with datasets that contain features such as the scores of students from different houses. Using these features, we need to make inferences on a new dataset to determine which student belongs to which house.


<div align="center">
  <center><h1>Exploratory Data Analysis</h1></center>
 </div>
 
## describe.py:

Custom implementation of the panda's library [describe()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html) function.
<br>
I added the mean absolute deviation (MAD) and the coefficient of variation (CV) to the output:

<p align="center">
<img src="https://user-images.githubusercontent.com/91064070/238584138-4a9e1a82-85a6-4435-b7d0-d7e8f8988a76.png"/>
</p>

## histogram.py:

This script tries to answer the question "Which Hogwarts course has a homogeneous score distribution between all four houses?".
<br>
We basically just look for the course that has the lowest standard deviation between houses:

<p align="center">
<img src="https://user-images.githubusercontent.com/91064070/238587039-2a6ade9c-c2cf-4863-b507-c67bc315eab3.png"/>
</p>

The script will also display those distributions, one histogram per course:

<p align="center">
<img src="https://user-images.githubusercontent.com/91064070/238592295-baafd09d-7f32-406b-861b-d229f59d5960.png"/>
</p>

## scatter_plot.py:

This script tries to answer the question "What are the two features that are similar ?".
<br>
We compare the distribution of each feature by pair and see that the two similar features are Astronomy and Defense Against the Dark Arts because of the pattern they follow:

<p align="center">
<img src=""/>
</p>

## pair_plot.py:

This script will display a pair plot, so basically all the histograms plus scatter plots that compare features by pair:

<p align="center">
<img src=""/>
</p>

<div align="center">
  <center><h1>Multinomial Logistic Regression (One vs All)</h1></center>
 </div>

## Usage:

<!--![image](https://user-images.githubusercontent.com/91064070/217234438-dbcb4473-bef4-44d6-8efb-eee9a3378c30.png)-->

1/ To generate the model, plot the results and save the model in the <theta.csv> file, use the [logreg_train.py] script:

```shell
$> python3 logreg_train.py <datasets/dataset_train.csv>
```

Loss function evolution with mini-batch gradient descent (batch size of 64):
<p align="center">
<img src="https://user-images.githubusercontent.com/91064070/238607684-ff579ad5-a44a-4ef5-a409-695a46a2ee68.png"/>
</p>

2/ Then use the model to predict classes. The script will use the model generated by the [logreg_train.py] script and saved in the <theta.csv> file.
<br>
It will make an inference on the [datasets/dataset_test.csv], and save the results in a [house.csv] file.

```shell
$> python3 logreg_predict.py
```
Results on the [dataset_test.csv] file:
<p align="center">
<img src="https://user-images.githubusercontent.com/91064070/238607155-f00cd52d-7f9a-4ebb-b56c-06385aa6d118.png"/>
</p>
