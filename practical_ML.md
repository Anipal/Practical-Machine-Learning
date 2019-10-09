Introduction
------------

Human Activity Recognition - HAR - has emerged as a key research area in
the last years and is gaining increasing attention by the pervasive
computing research community, especially for the development of
context-aware systems. There are many potential applications for HAR,
like: elderly monitoring, life log systems for monitoring energy
expenditure and for supporting weight-loss programs, and digital
assistants for weight lifting exercises.

Using devices like Fitbit makes it possible to collect a large amount of
data about personal activity relatively inexpensively. These type of
devices are part of the quantified self movement – a group of
enthusiasts who take measurements about themselves regularly to improve
their health, to find patterns in their behavior, or because they are
tech geeks. One thing that people regularly do is quantify how much of a
particular activity they do, but they rarely quantify how well they do
it.

In this project, I will use data from accelerometers on the belt,
forearm, arm, and dumbell of 6 participants to predict the manner in
which they did the exercise.

Data Preprocessing
------------------

    library(caret)

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    library(rpart)
    library(randomForest)

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

    library(corrplot)

    ## corrplot 0.84 loaded

    library(rpart.plot)

### Download the Data

    trainU <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    testU <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    if (!file.exists("./data")) {
      dir.create("./data")
    }
    trFile <- "./data/pml-training.csv"
    teFile  <- "./data/pml-testing.csv"
    download.file(trainU, destfile=trFile, method="curl")
    download.file(testU, destfile=teFile, method="curl")

### Read the Data

After downloading the data from the data source, we can read the two csv
files into two data frames.

    trdf <- read.csv("./data/pml-training.csv")
    tedf <- read.csv("./data/pml-testing.csv")
    dim(trdf)

    ## [1] 19622   160

    dim(tedf)

    ## [1]  20 160

The training data set contains 19622 observations and 160 variables,
while the testing data set contains 20 observations and 160 variables.
The “classe” variable in the training set is the outcome to predict.

### Clean the data

In this step, we will clean the data and get rid of observations with
missing values .

    sum(complete.cases(trdf))

    ## [1] 406

    trdf <- trdf[, colSums(is.na(trdf)) == 0] 
    tedf <- tedf[, colSums(is.na(tedf)) == 0] 
    classe <- trdf$classe
    trdel <- grepl("^X|timestamp|window", names(trdf))
    trdf <- trdf[, !trdel]
    trnew <- trdf[, sapply(trdf, is.numeric)]
    trnew$classe <- classe
    tedel <- grepl("^X|timestamp|window", names(tedf))
    tedf <- tedf[, !tedel]
    tenew <- tedf[, sapply(tedf, is.numeric)]

The cleaned training data set contains 19622 observations and 53
variables, while the testing data set contains 20 observations and 53
variables. \#\#\# Slice the data The cleaned training set is split into
a pure training data set (70%) and a validation data set (30%).
Validation data set will be used to conduct cross validation in later
steps.

    set.seed(22600) # For reproducibile purpose
    inTrain <- createDataPartition(trnew$classe, p=0.70, list=F)
    trdata <- trnew[inTrain, ]
    tedata <- trnew[-inTrain, ]

Data Modeling
-------------

A predictive model for activity recognition is fit to the data using
**Random Forest** algorithm because it automatically selects important
variables and is robust to correlated covariates & outliers .**5-fold
cross validation** is used.

    controlRf <- trainControl(method="cv", 5)
    modelRf <- train(classe ~ ., data=trdata, method="rf", trainControl=controlRf, ntree=250)
    modelRf

    ## Random Forest 
    ## 
    ## 13737 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Bootstrapped (25 reps) 
    ## Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9873974  0.9840508
    ##   27    0.9883630  0.9852742
    ##   52    0.9811783  0.9761821
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 27.

    predictRf <- predict(modelRf, tedata)
    confusionMatrix(tedata$classe, predictRf)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1667    5    2    0    0
    ##          B    8 1127    4    0    0
    ##          C    0    7 1015    4    0
    ##          D    0    0   13  951    0
    ##          E    0    0    1    1 1080
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9924          
    ##                  95% CI : (0.9898, 0.9944)
    ##     No Information Rate : 0.2846          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9903          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9952   0.9895   0.9807   0.9948   1.0000
    ## Specificity            0.9983   0.9975   0.9977   0.9974   0.9996
    ## Pos Pred Value         0.9958   0.9895   0.9893   0.9865   0.9982
    ## Neg Pred Value         0.9981   0.9975   0.9959   0.9990   1.0000
    ## Prevalence             0.2846   0.1935   0.1759   0.1624   0.1835
    ## Detection Rate         0.2833   0.1915   0.1725   0.1616   0.1835
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
    ## Balanced Accuracy      0.9968   0.9935   0.9892   0.9961   0.9998

    acc <- postResample(predictRf, tedata$classe)
    acc

    ##  Accuracy     Kappa 
    ## 0.9923534 0.9903278

    s <- 1 - as.numeric(confusionMatrix(tedata$classe, predictRf)$overall[1])
    s

    ## [1] 0.007646559

So, the estimated accuracy of the model is 99.42% and the estimated
out-of-sample error is 0.58%.

Predicting for Test Data Set
----------------------------

Now, we apply the model to the original testing data set downloaded from
the data source. We remove the `problem_id` column first.

    final <- predict(modelRf, tenew[, -length(names(tenew))])
    final

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

Appendix: Figures
-----------------

1.  Correlation Matrix Visualization

<!-- -->

    corrPlot <- cor(trdata[, -length(names(trdata))])
    corrplot(corrPlot, method="color")

![](practical_ML_files/figure-markdown_strict/unnamed-chunk-10-1.png) 2.
Decision Tree Visualization

    tree <- rpart(classe ~ ., data=trdata, method="class")
    prp(tree) # fast plot

![](practical_ML_files/figure-markdown_strict/unnamed-chunk-11-1.png)
