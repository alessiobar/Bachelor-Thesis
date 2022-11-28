"""
This script runs LDA using Gibbs sampling at 5,000 iterations, as specified in the supplementary material of Berger et al.
"""
library(readxl)
library(topicmodels)
library(tm)
library(Rcpp)
library(writexl)

df <- read_excel("PyToR.xlsx")
head(df)
df <- as.data.frame(df)
df[df == 0] <- "Void"
df[is.na(df)] <- "Void"
df <- subset(df, Lyrics!="Void")

docs1 <- Corpus(VectorSource(df$Lyrics))
dtm <- DocumentTermMatrix(docs1, control = list(weighting=weightTf, stopwords = TRUE ))
lda <- LDA(dtm, k = 10, method = "Gibbs", 
            control = list(seed = 42, iter = 5000, verbose = 100, alpha = 5)) 

attr(lda, "alpha")
tmResult <- posterior(lda)
attributes(tmResult)
terms(lda, 10)

theta <- tmResult$topics
theta <- as.data.frame(theta)
write_xlsx(theta,"perPydaR.xlsx")
