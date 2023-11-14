# Import libraries

# install.packages("ggDCA")
# install.packages("devtools")
# Attaching package: 'ggDCA'
# devtools::install_github('yikeshu0611/ggDCA')
library(ggplot2)
library(ROCR)
library(e1071)
library(pROC)
library(caret)
names(getModelInfo())
library(modeldata)
library(dplyr)
library(reportROC)

library(tidyverse)
library(mlr3)
library(mlr3learners)
library(randomForest)
library(varSelRF)
library(rms)

library(ggalt)
library(dcurves)
library(survival)

library(mlr3viz)
library(mlr3tuning) 
library(data.table)
library(magrittr)
library(reshape2)

library(rmda)

library(Hmisc)
library(grid)
library(lattice)
library(Formula) 
library(corrplot)
library(glmnet)

# Dataset
data<-read.csv("~/Desktop/U orginal feature_1se.csv")
manual<-read.csv("~/Desktop/manual.csv")

data_train<-read.csv("~/Desktop/U orginal train.csv")
data_test<-read.csv("~/Desktop/U orginal test.csv")

# Check dataset 
# 206 obs. of  5 variables
str(data)

# Change the variable types.
a <- sub("1","One",data$ID)
b <- sub("2","Two",a)
data$ID <- b

# Split the train and test datasets
# Testset: 44 obs. of 5 variables
# Trainset: 162 obs. of 5 variables
s <- data[1:162,]
trainset = data[1:162,]
testset = data[163:206,]

# Define the parameters used in the training process.
control = trainControl(method = "repeatedcv",
                       number = 10,
                       repeats =1,
                       classProbs = TRUE,
                       summaryFunction = twoClassSummary)
# Model training.

# KNN
knn.model = train(ID ~ .,
                  data= trainset,
                  method = "knn",
                  metric = "ROC",
                  trControl = control,tuneLength = 15)

# SVM
svm.model = train(ID ~ .,
                  data= trainset,
                  method = "svmRadial",
                  metric = "ROC",
                  trControl = control,tuneLength = 15)

# Check the models.
rf.model = train(ID ~ .,
                    data = trainset,
                    method = "rf",
                    metric = "ROC",
                    trControl = control,tuneLength = 15)
print(rf.model)

# Decision Tree model
dt.model = train(ID ~ .,
                 data = trainset,
                 method = "C5.0Tree",
                 metric = "ROC",
                 trControl = control,
                 verbose=F,,tuneLength = 15)

# Predict using the trained models and visualize.
knn.probs = predict(knn.model,testset,type = "prob")
svm.probs = predict(svm.model,testset,type = "prob")
rf.probs = predict(rf.model,testset,type = "prob")
dt.probs= predict(dt.model,testset,type = "prob")

# Graphing
knn.ROC = roc(response = testset$ID,
              predictor = knn.probs$One,ci=TRUE, boot.n=100, ci.alpha=0.9, stratified=FALSE)
plot(knn.ROC,xlim=c(1,0), ylim=c(0,1),type = "S",col = "red",print.auc=F,auc.polygon=F,print.thres = F,lty=1)

svm.ROC = roc(response = testset$ID,
              predictor = svm.probs[,1])
plot(svm.ROC,add = TRUE,col = "green",print.auc=F,auc.polygon=F,print.thres = F,lty=2)

rf.ROC = roc(response = testset$ID,
                predictor = rf.probs[,1])
plot(rf.ROC,add = TRUE,col = "blue",print.auc=F,auc.polygon=F,print.thres = F,lty=3)

dt.ROC = roc(response = testset$ID,
             predictor = dt.probs[,1])
plot(dt.ROC,add = TRUE,col = "yellow",print.auc=F,auc.polygon=F,print.thres = F,lty=4)

#plot(manual_roc, add = TRUE,col = "black",print.auc=F,auc.polygon=F,print.thres = F,lty=5)
grid()
grid(nx=8,ny=8,lwd=1,lty=2,col="gray")

legend("bottomright",cex=01,inset=0.1,title=NULL,c("KNN","SVM","RF","DT","Radiologist"),
       lty=c(1,2,3,4,5),col=c("red","green","blue","yellow","black"),lwd=2,text.font=6)

# Delong Test
knn_vs_rf <- roc.test(knn.ROC,rf.ROC)
knn_vs_rf
knn_vs_dt <- roc.test(knn.ROC,dt.ROC)
knn_vs_dt
rf_vs_dt <- roc.test(rf.ROC,dt.ROC)
rf_vs_dt

# Checing ROC manually
manual_roc <- roc(manual$answer,as.numeric(manual$guess))
plot(manual_roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE,main='manual')

reportROC(gold=testset$ID,predictor=knn.probs$One,import="se",plot=T)
#reportROC(gold=testset$ID,predictor=svm.probs$M,import="se",plot=T)
reportROC(gold=testset$ID,predictor=rf.probs$One,import="se",plot=T)
reportROC(gold=testset$ID,predictor=dt.probs$One,import="se",plot=T)

# Roc graphs of training dataset.

knn.train = predict(knn.model,trainset,type = "prob")
rf.train = predict(rf.model,trainset,type = "prob")
dt.train= predict(dt.model,trainset,type = "prob")

knntrain.ROC = roc(response = trainset$ID,
                   predictor = knn.train$One,ci=TRUE, boot.n=100, ci.alpha=0.9, stratified=FALSE)
plot(knntrain.ROC,xlim=c(1,0), ylim=c(0,1),type = "S",col = "red",print.auc=F,auc.polygon=F,print.thres = F,lty=1)

rftrain.ROC = roc(response = trainset$ID,
                  predictor = rf.train[,1])
plot(rftrain.ROC,add = TRUE,col = "blue",print.auc=F,auc.polygon=F,print.thres = F,lty=3)

dttrain.ROC = roc(response = trainset$ID,
                  predictor = dt.train[,1])
plot(dttrain.ROC,add = TRUE,col = "orange",print.auc=F,auc.polygon=F,print.thres = F,lty=4)

#plot(manual_roc, add = TRUE,col = "black",print.auc=F,auc.polygon=F,print.thres = F,lty=5)
grid(lwd=1,lty=2,col="gray")
#grid(nx=8,ny=8,lwd=1,lty=2,col="gray")

legend("bottomright",cex=1.6,inset=0.05,title=NULL,c("KNN","RF","DT"),
       lty=c(1,2,3),col=c("red","blue","orange"),lwd=2,text.font=6)

# Manual check
manual_roc <- roc(manual$answer,as.numeric(manual$guess))
plot(manual_roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE,main='manual')

reportROC(gold=trainset$ID,predictor=knn.train$One,import="se",plot=T)
reportROC(gold=trainset$ID,predictor=rf.train$One,import="se",plot=T)
reportROC(gold=trainset$ID,predictor=dt.train$One,import="se",plot=T)




# Calibration curves

# Test set
knn_pred<-predict(knn.model, newdata=testset,type = "prob")
p <- data.frame(knn_pred)

p_positive <- p$One
sor <- order(p_positive)
p_positive <- p_positive[sor]

y <- testset$ID[sor]
y <- ifelse(y == "One",1,0)

groep <- cut2(p_positive, g = 5)
meanpred <- round(tapply(p_positive, groep, mean), 3)
meanobs <- round(tapply(y, groep, mean), 3)
finall <- data.frame(meanpred = meanpred,
                     meanobs = meanobs)

rf_pred<-predict(rf.model, newdata=testset,type = "prob")
p1 <- data.frame(rf_pred)
p1_positive <- p1$One

sor1 <- order(p1_positive)
p1_positive <- p1_positive[sor1]
y1 <- testset$ID[sor1]
y1 <- ifelse(y1 == "One",1,0)
groep1 <- cut2(p1_positive, g = 5)
meanpred1 <- round(tapply(p1_positive, groep1, mean), 3)
meanobs1 <- round(tapply(y1, groep1, mean), 3)
finall1 <- data.frame(meanpred = meanpred1,
                      meanobs = meanobs1)

ggplot(finall1,aes(x = meanpred1,y = meanobs1))+geom_point()+
  geom_line(data=finall1, aes(x = meanpred1,y = meanobs1),lty=3,col = "blue")+ylim(0, 1)+xlim(0,1)+
  geom_abline(slope = 1,intercept = 0,lty="solid",color = "black")

ggplot(finall,aes(x = meanpred,y = meanobs))+geom_line(data=finall1,aes(x = meanpred,y = meanobs))


dt_pred<-predict(dt.model, newdata=testset,type = "prob")
p2 <- data.frame(dt_pred)
p2_positive <- p2$One

sor2 <- order(p2_positive)
p2_positive <- p2_positive[sor2]
y2 <- testset$ID[sor2]
y2 <- ifelse(y2 == "One",1,0)
groep2 <- cut2(p2_positive, g = 5)
meanpred2 <- round(tapply(p2_positive, groep2, mean), 3)
meanobs2 <- round(tapply(y2, groep2, mean), 3)
finall2 <- data.frame(meanpred = meanpred2,
                      meanobs = meanobs2)

ggplot(finall2,aes(x = meanpred2,y = meanobs2))+geom_point()+
  geom_line(data=finall2, aes(x = meanpred2,y = meanobs2),lty=5,col = "orange")+ylim(0, 1)+xlim(0,1)+
  geom_abline(slope = 1,intercept = 0,lty="solid",color = "black")

# Final curve
ggplot(finall,aes(x = meanpred,y = meanobs))+
  geom_point()+
  geom_line(lwd=1.5,lty=1,col = "red")+ylim(0, 1)+xlim(0,1)+
  geom_abline(slope = 1,intercept = 0,lty="solid",color = "black")+
  geom_line(lwd=1.5,lty=1,col = "blue",data=finall1,aes(x = meanpred1,y = meanobs1))+
  geom_line(lwd=1.5,lty=1,col = "orange",data=finall2,aes(x = meanpred2,y = meanobs2))+
  theme_classic()

# Smooth
ggplot(finall,aes(x = meanpred,y = meanobs))+
  geom_xspline(lwd=1.5,lty=1,col = "red")+ylim(0, 1.05)+xlim(0,1)+
  geom_abline(slope = 1,intercept = 0,lty=2,color = "black")+
  geom_xspline(lwd=1.5,lty=1,col = "blue",data=finall1,aes(x = meanpred1,y = meanobs1))+
  geom_xspline(lwd=1.5,lty=1,col = "orange",data=finall2,aes(x = meanpred2,y = meanobs2))+
  theme_classic()


# Train set
knn_train<-predict(knn.model, newdata=trainset,type = "prob")
p3 <- data.frame(knn_train)
p3_positive <- p3$One

sor3 <- order(p3_positive)
p3_positive <- p3_positive[sor3]
y3 <- trainset$ID[sor3]
y3 <- ifelse(y3 == "One",1,0)
groep3 <- cut2(p3_positive, g = 4)
meanpred3 <- round(tapply(p3_positive, groep3, mean), 3)
meanobs3 <- round(tapply(y3, groep3, mean), 3)
finall3 <- data.frame(meanpred = meanpred3,
                      meanobs = meanobs3)

rf_train<-predict(rf.model, newdata=trainset,type = "prob")
p4 <- data.frame(rf_train)
p4_positive <- p4$One
sor4 <- order(p4_positive)
p4_positive <- p4_positive[sor4]
y4 <- trainset$ID[sor4]
y4 <- ifelse(y4 == "One",1,0)
groep4 <- cut2(p4_positive, g = 10)
meanpred4 <- round(tapply(p4_positive, groep4, mean), 3)
meanobs4 <- round(tapply(y4, groep4, mean), 3)
finall4 <- data.frame(meanpred = meanpred4,
                      meanobs = meanobs4)

dt_train<-predict(dt.model, newdata=trainset,type = "prob")
p5 <- data.frame(dt_train)
p5_positive <- p5$One
sor5 <- order(p5_positive)
p5_positive <- p5_positive[sor5]
y5 <- trainset$ID[sor5]
y5 <- ifelse(y5 == "One",1,0)
groep5 <- cut2(p5_positive, g = 5)
meanpred5 <- round(tapply(p5_positive, groep5, mean), 3)
meanobs5 <- round(tapply(y5, groep5, mean), 3)
finall5 <- data.frame(meanpred = meanpred5,
                      meanobs = meanobs5)

ggplot(finall3,aes(x = meanpred3,y = meanobs3))+
  
  geom_xspline(lwd=1,lty=1,col = "red")+ylim(-0.05, 1.05)+xlim(0,1)+
  geom_abline(slope = 1,intercept = 0,lty=2,color = "black")+
  geom_xspline(lwd=1,lty=1,col = "blue",data=finall4,aes(x = meanpred4,y = meanobs4))+
  geom_xspline(lwd=1,lty=1,col = "orange",data=finall5,aes(x = meanpred5,y = meanobs5))+
  theme_classic()

# Decision curve
dcurves::dca(ID ~ knn_pred,
             data = testset) %>% 
  plot(smooth = T)

dcurves::dca(ID ~ rf_pred,
             data = data_test) %>% 
  plot(smooth = T)

dcurves::dca(ID ~ dt_pred,
             data = data_test) %>% 
  plot(smooth = T)

# DCA——test
knn<- decision_curve(ID~knn_pred,data = testset, family = binomial(link ='logit'),
                     thresholds= seq(0,1, by = 0.01),
                     confidence.intervals =0.95,
                     study.design = 'cohort')
rf<- decision_curve(ID~rf_pred,data = data_test, family = binomial(link ='logit'),
                    thresholds= seq(0,1, by = 0.01),
                    confidence.intervals =0.95,study.design ='cohort')
dt<- decision_curve(ID~dt_pred,data = data_test,family = binomial(link='logit'),
                    thresholds= seq(0,1, by = 0.01),
                    confidence.intervals =0.95,study.design ='cohort')

List<-list(knn,rf,dt)
List1<-list(knn)

plot_decision_curve(List,curve.names= c('knn','rf','dt'),
                    cost.benefit.axis =FALSE,col = c('red','blue','orange'),
                    confidence.intervals =FALSE,standardize = FALSE, xlim=c(0,1), ylim=c(-0.3, 0.8))


# DCA——train
knn_t<- decision_curve(ID~KNN_train,data = data_train, family = binomial(link ='logit'),
                       thresholds= seq(0,1, by = 0.01),
                       confidence.intervals =0.95,
                       study.design = 'cohort')
rf_t<- decision_curve(ID~RF_train,data = data_train, family = binomial(link ='logit'),
                      thresholds= seq(0,1, by = 0.01),
                      confidence.intervals =0.95,study.design ='cohort')
dt_t<- decision_curve(ID~DT_train,data = data_train,family = binomial(link='logit'),
                      thresholds= seq(0,1, by = 0.01),
                      confidence.intervals =0.95,study.design ='cohort')

List<-list(knn_t,rf_t,dt_t)
plot_decision_curve(List,curve.names= c('knn_train','rf_train','dt_train'),
                    cost.benefit.axis =FALSE,col = c('red','blue','orange'),
                    confidence.intervals =FALSE,standardize = FALSE, xlim=c(0,1), ylim=c(-0.3, 1))



# Lasso
data <-na.omit(data_train)

lambdas <-seq(0,09,length.out=1000)
set.seed(123) 

X <- as.matrix(data[,2:856])
Y <-as.matrix(data[,1])

alpha1_fit <-glmnet(X,Y,alpha=1) 

plot(alpha1_fit,xvar="lambda",label=TRUE)

cv.lasso <-cv.glmnet(X,Y,alpha=1,lambda=lambdas,nfolds=10,family="binomial")
plot(cv.lasso)

coef <-coef(cv.lasso$glmnet.fit,cv.lasso$lambda)
lasso_min <- cv.lasso$lambda.min
lasso_min
lasso.coef <-coef(cv.lasso$glmnet.fit, s=lasso_min, exact=T)
lasso.coef