#Import libraries and subset the data
library(jsonlite)
library(tidyverse)
library(caret)
library(tm)
library(e1071)
library(rpart)
library(wordcloud)
library(tidytext)
library(reshape2)
library(RWeka)

json <- stream_in(file("C:\\Users\\dan-cash\\Documents\\edx\\capstone CYO\\Kindle_Store_5.json"))
json_tbl <- tbl_df(json)
set.seed(1)
reviews <- json_tbl[sample(nrow(json_tbl), 1000), ]

#Classification Rubric, turns ratings into a binary field
divideSet <- function(data) {
  result <- ""
  if(data < 3)
    result <- "negative"
  else 
    result <- "positive"
  return(result)
}

#Construct a word cloud using the bing lexicon
tidy_samplet <- reviews %>% 
  unnest_tokens(word, reviewText)
tidy_samplet %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort = TRUE) %>%
  acast(word ~ sentiment, value.var = "n", fill = 0) %>%
  comparison.cloud(colors = c("gray20", "gray80"),
                   max.words = 100)

#Format the data to be just the reviews and the binary rating
#Table to show the totals of each rating
text <- unlist(reviews$reviewText)
sc <- unlist(reviews$overall)
sc <- map(sc, divideSet)
sc <- unlist(sc)
reviews <- cbind(text,sc)

table(reviews[,"sc"])

#Construct the corpus, then the document term matrix, and then split the data into training and testing sets
corpus <- VCorpus(VectorSource(text))
corpus.clean <- corpus %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace)

dtm <- DocumentTermMatrix(corpus.clean)

split_val <- floor(0.8 * nrow(reviews))
train_ind <- sample(seq_len(nrow(reviews)), size = split_val)

df.train <- reviews[train_ind, ]
df.test <- reviews[-train_ind, ]

dtm.train <- dtm[train_ind, ]
dtm.test <- dtm[-train_ind, ]

corpus.clean.train <- corpus.clean[train_ind]
corpus.clean.test <- corpus.clean[-train_ind]

#Create the three different models
fivefreq <- findFreqTerms(dtm.train, 5)

dtm.train.nb <- DocumentTermMatrix(corpus.clean.train, control=list(dictionary = fivefreq))
dtm.test.nb <- DocumentTermMatrix(corpus.clean.test, control=list(dictionary = fivefreq))

tfidf.dtm.train.nb <- DocumentTermMatrix(corpus.clean.train, control=list(weighting = weightTfIdf))
tfidf.dtm.test.nb <- DocumentTermMatrix(corpus.clean.test, control=list(weighting = weightTfIdf))

BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
bi.dtm.train.nb <- DocumentTermMatrix(corpus.clean.train, control = list(tokenize = BigramTokenizer))
bi.dtm.train.nb <- removeSparseTerms(bi.dtm.train.nb,0.99)
bi.dtm.test.nb <- DocumentTermMatrix(corpus.clean.test, control = list(tokenize = BigramTokenizer))
bi.dtm.test.nb <- removeSparseTerms(bi.dtm.test.nb,0.99)

#Training and testing
classifier <- naiveBayes(trainNB, as.factor(df.train[,"sc"]), laplace = 1)
tfidf.classifier <- naiveBayes(tfidf.trainNB, as.factor(df.train[,"sc"]), laplace = 1)
bi.classifier <- naiveBayes(bi.trainNB, as.factor(df.train[,"sc"]), laplace = 1)

pred <- predict(classifier, newdata=testNB)
tfidf.pred <- predict(tfidf.classifier, newdata=tfidf.testNB)
bi.pred <- predict(bi.classifier, newdata=bi.testNB)

#Truth tables and connstructing confusion matrices
print("Five")
table("Predictions"= pred,  "Actual" = as.factor(df.test[,"sc"]) )
print("Tf-Idf")
table("Predictions"= tfidf.pred,  "Actual" = as.factor(df.test[,"sc"]) )
print("Bigrams")
table("Predictions"= bi.pred,  "Actual" = as.factor(df.test[,"sc"]) )

conf.mat <- confusionMatrix(pred, as.factor(df.test[,"sc"]))
tfidf.conf.mat <- confusionMatrix(tfidf.pred, as.factor(df.test[,"sc"]))
bi.conf.mat <- confusionMatrix(bi.pred, as.factor(df.test[,"sc"]))

#Results
print("Five")
conf.mat$byClass
conf.mat$overall
print("Tf-Idf")
tfidf.conf.mat$byClass
tfidf.conf.mat$overall
print("Bigrams")
bi.conf.mat$byClass
bi.conf.mat$overall

#Using Cross Validation
#Repeat prior steps but inside loop to find optimal model and number of folds
folded <- cut(seq(1,nrow(reviews)),breaks=10,labels=FALSE)
class <- vector(mode = "list", length = 11)
overall <- vector(mode = "list", length = 7)
class.tfidf <- vector(mode = "list", length = 11)
overall.tfidf <- vector(mode = "list", length = 7)
class.bi <- vector(mode = "list", length = 11)
overall.bi <- vector(mode = "list", length = 7)

for(i in 1:10){
  testIndexes <- which(folded==i,arr.ind=TRUE)
  
  df.train <- reviews[-testIndexes, ]
  df.test <- reviews[testIndexes, ]
  
  dtm.train <- dtm[-testIndexes, ]
  dtm.test <- dtm[testIndexes, ]
  
  corpus.clean.train <- corpus.clean[-testIndexes]
  corpus.clean.test <- corpus.clean[testIndexes]
  
  fivefreq <- findFreqTerms(dtm.train, 5)
  
  dtm.train.nb <- DocumentTermMatrix(corpus.clean.train, control=list(dictionary = fivefreq))
  dtm.test.nb <- DocumentTermMatrix(corpus.clean.test, control=list(dictionary = fivefreq))
  tfidf.dtm.train.nb <- DocumentTermMatrix(corpus.clean.train, control=list(weighting = weightTfIdf))
  tfidf.dtm.test.nb <- DocumentTermMatrix(corpus.clean.test, control=list(weighting = weightTfIdf))
  bi.dtm.train.nb <- DocumentTermMatrix(corpus.clean.train, control = list(tokenize =  BigramTokenizer))
  bi.dtm.train.nb <- removeSparseTerms(bi.dtm.train.nb,0.99)
  bi.dtm.test.nb <- DocumentTermMatrix(corpus.clean.test, control = list(tokenize =  BigramTokenizer))
  bi.dtm.test.nb <- removeSparseTerms(bi.dtm.test.nb,0.99)
  
  trainNB <- apply(dtm.train.nb, 2, convert_count)
  testNB <- apply(dtm.test.nb, 2, convert_count)
  tfidf.trainNB <- apply(tfidf.dtm.train.nb, 2, convert_count)
  tfidf.testNB <- apply(tfidf.dtm.test.nb, 2, convert_count)
  bi.trainNB <- apply(bi.dtm.train.nb, 2, convert_count)
  bi.testNB <- apply(bi.dtm.test.nb, 2, convert_count)
  
  classifier <- naiveBayes(trainNB, as.factor(df.train[,"sc"]), laplace = 1)
  tfidf.classifier <- naiveBayes(tfidf.trainNB, as.factor(df.train[,"sc"]), laplace = 1)
  bi.classifier <- naiveBayes(bi.trainNB, as.factor(df.train[,"sc"]), laplace = 1)
  
  pred <- predict(classifier, newdata=testNB)
  tfidf.pred <- predict(tfidf.classifier, newdata=tfidf.testNB)
  bi.pred <- predict(bi.classifier, newdata=bi.testNB)
  
  conf.mat <- confusionMatrix(pred, as.factor(df.test[,"sc"]))
  tfidf.conf.mat <- confusionMatrix(tfidf.pred, as.factor(df.test[,"sc"]))
  bi.conf.mat <- confusionMatrix(bi.pred, as.factor(df.test[,"sc"]))
  
  class[[i]] <- conf.mat$byClass
  overall[[i]] <- conf.mat$overall
  class.tfidf[[i]] <- tfidf.conf.mat$byClass
  overall.tfidf[[i]] <- tfidf.conf.mat$overall
  class.bi[[i]] <- bi.conf.mat$byClass
  overall.bi[[i]] <- bi.conf.mat$overall
}

#Results
#number of folds, F1 score, and accuracy
df <- data.frame(do.call(rbind, class))[1:11]
dfo <- data.frame(do.call(rbind, overall))[1:7]
df.tfidf <- data.frame(do.call(rbind, class.tfidf))[1:11]
dfo.tfidf <- data.frame(do.call(rbind, overall.tfidf))[1:7]
df.bi <- data.frame(do.call(rbind, class.bi))[1:11]
dfo.bi <- data.frame(do.call(rbind, overall.bi))[1:7]
#Number of folds with highest F1 score
n_folds <- order(df['F1'],decreasing=T)[1]
message("Five - # of folds:",n_folds)
n_folds.tfidf <- order(df.tfidf['F1'],decreasing=T)[1]
message("Tf-Idf - # of folds:",n_folds.tfidf)
n_folds.bi <- order(df.bi['F1'],decreasing=T)[1]
message("Bigrams - # of folds:",n_folds.bi)
#F1 score
message("Five - F1:",df[n_folds, 'F1'])
message("Tf-Idf - F1:",df.tfidf[n_folds.tfidf, 'F1'])
message("Bigrams - F1:",df.bi[n_folds.bi, 'F1'])
#Accuracy
message("Five - Accuracy:",dfo[n_folds,"Accuracy"])
message("Tf-Idf - Accuracy:",dfo.tfidf[n_folds.tfidf,"Accuracy"])
message("Bigrams - Accuracy:",dfo.bi[n_folds.bi,"Accuracy"])
#Guess Accuracy
# 57      943 
message("Always guess positive:", (943)/(943+57))

