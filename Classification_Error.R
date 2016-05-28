# Plotting classification errors of different linear discriminant score threshold
# Just want to see whether 0.5 is where total error minimizes

require(ISLR)
require(data.table)
require(dplyr)
require(ggplot2)

default<-default
fit<-lda(default~.,default)
X<-data.frame(default$student,default$balance,default$income)
names(X)<-c('student','balance','income')
pred<-predict(fit,X)
# TP<-data.frame(default$default,pred$class)
# names(TP)<-c("Actual_Value","Prediction")
truth<-default$default=="Yes"
truthT<-length(which(truth==T))
truthF<-length(which(truth==F))
error.rates<-data.frame()

error.generator<-function(threshold=0.5){
    # prediction of +
    prediction<-pred$posterior[,1]<threshold
    
    Type.I<-length(which((truth==F)*(prediction==T)==1))
    Type.II<-length(which((truth==T)*(prediction==F)==1))
    
    bind_rows(error.rates,data.frame("threshold"=threshold,"Type.I"=Type.I,"Type.II"=Type.II))
}

sequence<-seq(0,1,0.01)
error.rates<-data.table(t(sapply(sequence,error.generator)))
error.rates<-mutate(error.rates,I.rate=as.numeric(Type.I)/truthF,
                    II.rate=as.numeric(Type.II)/truthT,
                    Total.rate=(as.numeric(Type.I)+as.numeric(Type.II))/10000)
g<-ggplot()
g<-g+geom_line(data=error.rates,aes(x=sequence,y=error.rates$I.rate,color='Type I'))
g<-g+geom_line(data=error.rates,aes(x=sequence,y=error.rates$II.rate,color='Type II'))
g<-g+geom_line(data=error.rates,aes(x=sequence,y=error.rates$Total.rate,color='Total Error'))
g<-g+labs(x="Threshold",y="Error Rates",color=NULL)
g

# Actually not. It's 0.64. But 0.5 is expected.
error.rates[threshold<0.7&threshold>0.45]
# The analysis assumes a normal distribution, and there is noise. So.