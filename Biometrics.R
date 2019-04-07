-## Dataset

#Column_3C_weka.csv will be imported as dataset and it has 7 colummns data. First six columns are six biomechanics and last column is used to classify patients. In file, 100 patients are as normal, 60 patients are as Disk Hernia and 150 patients are Spondilolistez. Total 310 patients.  

# Structure of data
#  https://www.kaggle.com/uciml/biomechanical-features-of-orthopedic-patients/downloads/biomechanical-features-of-orthopedic-patients.zip/1
#https://www.kaggle.com/uciml/biomechanical-features-of-orthopedic-patients

dl <- tempfile()
download.file("https://www.kaggle.com/uciml/biomechanical-features-of-orthopedic-patients/downloads/biomechanical-features-of-orthopedic-patients.zip/1", dl)  

Data_3class <- read.csv(file="column_3C_weka.csv", header=TRUE, sep=",")
str(Data_3class)

# Summary of data
summary(Data_3class)

# First 6 data lines
head(Data_3class)
##ANALYSES
# Data Visulization
## Distribution of Patients in 3 class items dataset
require(ggplot2)
require(gridExtra)
ggplot(Data_3class,aes(x=class,fill=class))+geom_bar(stat = 'count')+labs(x = 'Distribution of patients') +
  geom_label(stat='count',aes(label=..count..), size=4) +theme_dark(base_size = 12)
require(GGally)
# Pair graphs of data.
ggpairs(data=Data_3class, columns=c(1:7))
#It is seen that from diagonally looking there is an outlier problem for degree_spondylolisthesis. it should be corrected first by redefine it as mean of column, mean of degree_spondylolisthesis.
outlier_3class <- which.max(Data_3class$degree_spondylolisthesis)
Data_3class$degree_spondylolisthesis[outlier_3class] <- mean(Data_3class$degree_spondylolisthesis)
#Refreshing pair graph it is viewed that outlier problem is Solved.
ggpairs(data=Data_3class, columns=c(1:7))
# Correlation between six biometrics
suppressMessages(library(corrplot))
corr_mat <- cor(Data_3class[,1:6])
corrplot(corr_mat, method = "number")
#There aren't good correlations and relations between 6 biometrics that easily formalize.

### Machine learning techniques
# Prepare Train set and test set
require(dplyr)
require(caret)
y <- Data_3class$class
set.seed(1)
test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)
train_set <- Data_3class %>% slice(-test_index)
test_set <- Data_3class %>% slice(test_index)
# Classification (decision) trees.
require(rpart.plot)
class.tree <- rpart(Data_3class$class~.,data = Data_3class,control = rpart.control(cp = 0.01))
rpart.plot(class.tree, 
box.palette="GnBu",
branch.lty=10, shadow.col="gray", nn=TRUE)
#First branch of decision tree is degree_spondylolisthesis whether degree_spondylolisthesis is bigger than 16, If it is yes then patient class is spondylolisthesis.
#Numbers in the box.
#0.19 - 0.193548387 - Number of Hernia in the dataset - (60 / 310)
#0.32 - 0.322580645 - Number of Normal in the dataset - (100 / 310)
#0.48 - 0.483870968 - Number of spondylolisthesis in the dataset - (150 / 310)

#Right side of the first branche is spondylolisthesis patients
#48% of all data
#Numbers in the box.
#0.00 - 0 - Number of Hernia in the dataset - (0 / 150)
#0.02 - 0.2 - Number of Normal in the dataset - (3 / 150)
#0.98 - 0.98 - Number of spondylolisthesis in the dataset - (147 / 150)
#and so on.

#cp is selected  as 0.01 by controlling accuracy with respect to below approach.    
train_rpart <- train(class ~ .,  method = "rpart",
tuneGrid = data.frame(cp = seq(0, 0.05, len = 50)),
data = train_set)
ggplot(train_rpart)
#And Accuracy is 
confusionMatrix(predict(train_rpart, test_set),test_set$class)$overall["Accuracy"]
#random forest
require(randomForest)
fit <- randomForest(class~., data = Data_3class, ntree=500, proximity=T) 
plot(fit)
fit.legend <- colnames(Data_3class)
legend("top", cex =0.3, legend=fit.legend, lty=c(1,2,3,4,5,6,7), col=c(1,2,3,4,5,6,7), horiz=T)
#lowest err.rate. It is 100.
which.min(fit$err.rate[,1])
suppressMessages(library(randomForest))
rf.train_set <-randomForest(class ~., data=train_set)
rf.train_set
#Accuracy is better as expected.
train_rf <- randomForest(class ~ ., data=train_set)
confusionMatrix(predict(train_rf, test_set), test_set$class)$overall["Accuracy"]
