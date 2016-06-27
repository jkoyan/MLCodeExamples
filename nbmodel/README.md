#Naive Bayes Model

The directory contains 3 sub directories and 2 files, lets go through each of the sub directories and the individual files

1. data : The first sub directory is called data and it contains two files viz.

        * iris.data.txt - The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. 
          One class is linearly separable from the other 2; the latter are NOT linearly separable from each other. 
          The file was downloaded from the UCI Machine learning repository @ http://archive.ics.uci.edu/ml/datasets/Iris
          The file was modified to include a header row as the first line to help identifying the different features.
          
        * iris.names.txt - The file was downloaded from the same UCI Machine learning repository as stated above, it is part
         of the dataset, the contents describe the dataset in detail.

2. images : The directory contains 4 different images that were produced either using the nbclassifier.py or the plots.r 
   R code from the r-code directory
        * confusion-matrix.png - The file was produced using the nbclassifier.py file by calling the plot_confusion_matrix
          method. The image helps in visualizing the accuracy of the classifier.
          
        * normalized-confusion-matrix.png - This image was produced using the same method stated above. The predicted outcomes
          were normalized and then used to create the confusion matrix, this helps in understanding how
          well the classifier performed across all classes.
        
        * correlation-coefficients-matrix.png - The image was created using the R code within the plots.r file from the 
          r-code directory. Correlation coefficients are a measure of the linear relationship between two variables.
          The stronger the linear relationship the coefficient will be closer to 1, a -ve coefficient means the variable 
          on the y axis decreases when the variable on the x axis increases.
          
        * iris_data-scatter-plot-1.png - This image was created using the R code within the plots.r file from the r-code 
          directory, it presents a visual on the relationship between the different features. The names on the rows represent
          the y axis and the names in the column represents the x axis.
          
3. r-code : The directory contains single code file called plots.r.

        * plots.r - Contain the R code that was used to generate the scatter plot for features from the iris dataset 
          called iris_data-scatter-plot-1.png, it was also used to plot the correlation coefficients between the
          features.

4. nbclassifier.py : The file contains python code to load the iris datasets, create a naive bayes classifier, train
   the classifier and then have it predict the class names for the test datasets, the code employs the train_test_split
   method from the cross_validation package to split the dataset into training and test sets. At the end we measure
   the performance of the classifier by calculating the accuracy ratio of correctly predicted outcomes to that of the
   total test observations.
   The plot_confusion_matrix method is used to plot the confusion matrix of the predicted outcomes.