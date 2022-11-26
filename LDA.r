library(readxl)
df <- read_excel("C:/Users/alema/Desktop/perR2.xlsx")
head(df)

#install.packages("Rcpp")
library(topicmodels)
library(tm)
library(Rcpp)
library(writexl)

df<-as.data.frame(df)
df[df == 0] <- "Void"
df[is.na(df)] <- "Void"
#df<-subset(df, Lyrics!="Void")

docs1 <- Corpus(VectorSource(df$Lyrics) )
dtm <- DocumentTermMatrix(docs1)
lda <- LDA(dtm, k = 10, method = "Gibbs", 
            control = list(seed = 42, iter = 5000, verbose = 50, alpha = 0.0)) 

#cambiando alpha mi cambia il modello, co 5 veniva 0.103 il primo pvalue su ARE, co 2.5 0.004, co 1.0 è 0.002, co 0 è 0 :DDDD
attr(lda, "alpha")

tmResult <- posterior(lda)
attributes(tmResult)
terms(lda, 10)

theta <- tmResult$topics
theta <- as.data.frame(theta)
write_xlsx(theta,"C:/Users/alema/Desktop/perPydaR2.xlsx")
