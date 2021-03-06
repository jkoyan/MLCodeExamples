library(fpp)
library(dplyr)
library(corrplot)
iris <- tbl_df(read.csv('//Users//GraysTECH//BigQLabs//ml//MLCodeExamples//nbmodel//data//iris.data.txt',header=TRUE,sep=','))
pairs(iris[-5], main = "Fisher's Iris Data -- 3 species",pch = 21, bg = c("red", "green3", "blue")[unclass(iris$iris_class)],oma=c(4,4,6,12))
par(xpd=TRUE)
legend(0.85, 0.7, as.vector(unique(iris$iris_class)),
       fill=c("red", "green3", "blue"))
> i <- cor(iris[-5])
> corrplot(i,method='number')
