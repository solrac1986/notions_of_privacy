
library(dplyr)
library(plyr)
library(tidyr)
library("e1071")

library(ggplot2)
library(stats)
library(ggExtra)

library(reshape2)
library(psych)

library(plotly)
library(plyr)

library(ellipse)
library(RColorBrewer)

library(rgeolocate)
library(Publish)

library(MASS)
library(tidyverse)
library(caret)

# Lattice package
require(lattice)

library(Publish) 
library(lsr)
library(ggpubr)
library(ez)

#Levene test
library(rstatix)

# Lmm
library(lme4)
#AdaBoost classifier
library(adabag)

library("OneR")

folder_privacy_results <- "../results/survey_js/amt_definition_total"
filename <- 'merged_labelled_surve_custom.csv'
filename <- 'merged_labelled_survey_14.csv'
filename <- 'keywords_label_top_5.csv'

dataframe_survey_labelled <- read.csv(file = file.path(folder_privacy_results, filename), head =TRUE, sep=",", stringsAsFactors = TRUE)

# Rename columns

dataframe_survey_labelled <- dataframe_survey_labelled %>%
                  rename(
                    gender = What.is.your.gender.,
                    age = What.is.your.age.,
                    profession = What.is.your.profession.,
                    education = Education ,
                    country.residence = Country.of.residence ,
                    iot.have = Do.you.have.any.smart.device.at.home..e.g...smart.tv..smart.speaker.. ,
                    privacy.definition = Define.privacy.in.your.own.words,
                    Topic = Dominant_Topic
                  )

dataframe_survey_labelled <- dataframe_survey_labelled %>%
                rename(
                  gender = What.is.your.gender.,
                  age = What.is.your.age.,
                  profession = What.is.your.profession.,
                  education = Education ,
                  country.residence = Country.of.residence ,
                  iot.have = Do.you.have.any.smart.device.at.home..e.g...smart.tv..smart.speaker.. ,
                  privacy.definition = Define.privacy.in.your.own.words
  )


# Get only columns 
# dataframe_survey_labelled <- dataframe_survey_labelled[, c('gender', 'age', 'profession', 
#                                                            'education','country.residence',
#                                                            'iot.have', 'privacy.definition', 
#                                                            'client_os', 'familiarity_technology',
#                                                            'familiarity_smart_home', 'familiarity_computer_sec',
#                                                            'average_concern',
#                                                            'Westin', 'Solove',
#                                                            'Topic')]

##### GLMR - Models #####

dataframe_filtered <- dataframe_survey_labelled
# dataframe_filtered$gender <- factor(dataframe_filtered$gender)

# Westin
dataframe_filtered <- dataframe_filtered[complete.cases(dataframe_filtered$Westin),]
dataframe_filtered <- subset(dataframe_filtered, Westin != "")
dataframe_filtered$Westin <- as.numeric(factor(dataframe_filtered$Westin))

# Solove 
dataframe_filtered <- dataframe_filtered[complete.cases(dataframe_filtered$Solove),]
dataframe_filtered <- subset(dataframe_filtered, Westin != "")
dataframe_filtered$Solove <- as.numeric(factor(dataframe_filtered$Westin))

# Thematic topic
dataframe_filtered <- dataframe_filtered[complete.cases(dataframe_filtered$Topic),]


# Keywords_number
dataframe_filtered <- dataframe_filtered[complete.cases(dataframe_filtered$Num_keywords),]


# Remove countries outliers
dataframe_filtered <- dataframe_filtered[dataframe_filtered$country.residence %in% names(which(table(dataframe_filtered$country.residence) > 10)), ]
dataframe_filtered$KeywordClass <- 0
dataframe_filtered$KeywordClass[which(dataframe_filtered$Num_keywords > 0)] <- 1

## LDA

training.samples <- dataframe_filtered$Topic %>%
  createDataPartition(p = 0.8, list = FALSE)
train.data <- dataframe_filtered[training.samples, ]
test.data <- dataframe_filtered[-training.samples, ]

# Clean gender, education, profession, country that is not in train.data

test.data.clean <- subset(test.data, (age %in% train.data$age) )
test.data.clean <- subset(test.data.clean, (gender %in% train.data$gender) )
test.data.clean <- subset(test.data.clean, (education %in% train.data$education) )
test.data.clean <- subset(test.data.clean, (profession %in% train.data$profession) )
test.data.clean <- subset(test.data.clean, (country.residence %in% train.data$country.residence) )
test.data.clean <- subset(test.data.clean, (client_os %in% train.data$client_os) )
test.data.clean <- subset(test.data.clean, (country.residence %in% train.data$country.residence) )

test.data <- test.data.clean

preproc.param <- train.data %>% 
  preProcess(method = c("center", "scale"))
# Transform the data using the estimated parameters
train.transformed <- preproc.param %>% predict(train.data)
test.transformed <- preproc.param %>% predict(test.data)


# data.pca <- prcomp (dataBoxPlotFilteredModel)
# pairs(data.pca$x[,1:5])
# 
# train.transformed <- train.transformed[-c(13
#                                           ), ]

# model.lda <- lda(vignette_answer~., data = train.transformed)
model.lda <- lda(KeywordClass ~  gender
                 # + age
                 # + profession
                 # + education
                 + country.residence
                 # + iot.have
                 # + client_os
                 # + familiarity_technology
                 + familiarity_smart_home
                 + familiarity_computer_sec
                 # + average_concern,
                 , data = train.transformed)
# summary(model.lda)

# model.lda$xlevels$description <- union(model.lda$xlevels$description, levels(test.transformed$profession))

# test.transformed.subset <- test.transformed %>%

predictions <- model.lda %>% predict(test.transformed)
mean(predictions$class==test.transformed$KeywordClass)

lda.data <- cbind(train.transformed, predict(model.lda)$x)

ggplot(lda.data, aes(LD1, LD2)) +
  geom_point(aes(color = KeywordClass), alpha=0.5)

# Westin
f.1 <- ~  gender + age + profession + education + country.residence + iot.have + client_os +  familiarity_technology + familiarity_smart_home +  familiarity_computer_sec + average_concern
model.vignette.glmm.full <- glm(Westin ~ gender, data= train.data, family=poisson())
model.vignette.glmm.1 <- update(model.vignette.glmm.full, f.1)
m.backward <- step(model.vignette.glmm.1, scope = c(lower = ~ gender), direction="backward")
model.vignette.glmm.backward <- stepAIC(model.vignette.glmm.1, trace = TRUE, direction = "backward", data = train.data)
summary(model.vignette.glmm.backward)

model.vignette.glmm.backward.1 <- glm(Westin ~ client_os, data= train.data, family=poisson())
summary(model.vignette.glmm.backward.1)

# Solove
f.1 <- ~  gender + age + profession + education + country.residence + iot.have + client_os +  familiarity_technology + familiarity_smart_home +  familiarity_computer_sec + average_concern
model.vignette.glmm.full <- glm(Solove ~ gender, data= train.data, family=poisson())
model.vignette.glmm.1 <- update(model.vignette.glmm.full, f.1)
m.backward <- step(model.vignette.glmm.1, scope = c(lower = ~ gender), direction="backward")
model.vignette.glmm.backward <- stepAIC(model.vignette.glmm.1,  trace = TRUE, direction = "backward", data = train.data)
summary(model.vignette.glmm.backward)

model.vignette.glmm.backward.1 <- glm(Solove ~ familiarity_technology, data= train.data, family=poisson())
summary(model.vignette.glmm.backward.1)

# Topic
f.1 <- ~  gender + age + profession + education + country.residence + iot.have + client_os +  familiarity_technology + familiarity_smart_home +  familiarity_computer_sec + average_concern
model.vignette.glmm.full <- glm(Topic ~ gender, data= train.data, family=poisson())
model.vignette.glmm.1 <- update(model.vignette.glmm.full, f.1)
m.backward <- step(model.vignette.glmm.1, scope = c(lower = ~ gender), direction="backward")
model.vignette.glmm.backward <- stepAIC(model.vignette.glmm.1, trace = TRUE, direction = "backward", data = train.data)
summary(model.vignette.glmm.backward)

model.vignette.glmm.backward.1 <- glm(Topic ~ profession*iot.have, data= train.data, family=poisson())
summary(model.vignette.glmm.backward.1)


# Num Keywords
f.1 <- ~  age + gender + profession + education + country.residence + iot.have + client_os +  familiarity_technology + familiarity_smart_home +  familiarity_computer_sec + average_concern
model.vignette.glmm.full <- glm(KeywordClass ~ age, data= train.data, family=binomial())
# model.vignette.glmm.full <- lm(KeywordClass ~ gender, data= train.data)
model.vignette.glmm.1 <- update(model.vignette.glmm.full, f.1)
m.backward <- step(model.vignette.glmm.1,  direction="backward", k = log(nrow(train.data)))
model.vignette.glmm.backward <- stepAIC(model.vignette.glmm.1,  trace = TRUE, direction = "backward", data = train.data)
summary(model.vignette.glmm.backward)

model.vignette.glmm.backward.1 <- glm(KeywordClass ~ gender+country.residence, data= train.data, family=binomial())
summary(model.vignette.glmm.backward.1)

## classifiers #########

# Adaboost

train.data.factorized <- train.data
train.data.factorized$Westin <- factor(train.data.factorized$Westin)

# using naeni method neutral - > correct
model.abc.correct = boosting(Topic ~  gender
                             + age
                             + profession
                             + education
                             + country.residence
                             # + iot.have
                             + client_os
                             # + familiarity_technology
                             + familiarity_smart_home
                             # + familiarity_computer_sec
                             + average_concern
                             , data=train.data.factorized, boos=TRUE, mfinal=50)


model.abc <- model.abc.correct


# u_topics <- union(train.data$Topic, test.data$Topic)
predictions.abc <- model.abc %>% predict(test.data)
# t <- table(factor(predictions.abc, u_topics), factor(test.data, u_topics))

print(confusionMatrix(predictions.abc$confusion))
print(predictions.abc$error)

TP = predictions.abc$confusion[2,2]
FP = predictions.abc$confusion[1,2]
FN = predictions.abc$confusion[2,1]

precision <- TP/(TP+FP)
recall <- TP/(TP+FN)

sprintf("Precision: %.5f", precision)
sprintf("Recall: %.5f", recall)

mean(predictions.abc$class == test.data$KeywordClass)



# Logistic regression

train.data.factorized <- train.data
train.data.factorized$Topic <- factor(train.data.factorized$Topic)

model.logistic <- glm (Topic ~  gender
                          + age
                          + profession
                          + education
                          + country.residence
                          # + iot.have
                          + client_os
                          # + familiarity_technology
                          + familiarity_smart_home
                          # + familiarity_computer_sec
                          + average_concern
                          , data=train.data.factorized, family=binomial)


summary(model.logistic)

probabilities <- model.logistic %>% predict(test.data, type='response')
predicted.classes <- ifelse(probabilities > 0.5, 1, 0)
predicted.classes

mean(predicted.classes == test.data$KeywordClass)

confusion.matrix <- table(predicted.classes,test.data$KeywordClass)



TP = confusion.matrix[2,2]
FP = confusion.matrix[1,2]
FN = confusion.matrix[2,1]

precision <- TP/(TP+FP)
recall <- TP/(TP+FN)

sprintf("Precision: %.5f", precision)
sprintf("Recall: %.5f", recall)


model.multi <- nnet::multinom (Topic ~  gender
                       + age
                       + profession
                       + education
                       + country.residence
                       # + iot.have
                       + client_os
                       # + familiarity_technology
                       + familiarity_smart_home
                       # + familiarity_computer_sec
                       + average_concern
                       , data=train.data.factorized,MaxNWts=4000)

summary(model.multi)

predictions.multi <- model.multi %>% predict(test.data)
confusion.matrix <- table(predictions.multi, test.data$Topic)

accuracy <- mean(predictions.multi == test.data$Topic)
sprintf("Accuracy: %.5f", recall)

TP = confusion.matrix[2,2]
FP = confusion.matrix[1,2]
FN = confusion.matrix[2,1]

precision <- TP/(TP+FP)
recall <- TP/(TP+FN)

sprintf("Precision: %.5f", precision)
sprintf("Recall: %.5f", recall)




## Boxplots topics - frequency

dataBoxplot <- dataframe_filtered


agg_dd_reasons <- dataBoxplot %>%
  select(Topic) %>%
  group_by(Topic) %>%
  summarise(freq = n()) %>%
  # group_by(Topic) %>%
  mutate(Percentage=freq/sum(freq)*100) %>%
  # group_by_(.dots=c(filter_vignette)) %>%
  # group_by_(.dots=c(filter_vignette, filter_reason)) %>%
  # group_by(device_type,vignette_answer) %>%
  # mutate(Frequency=n()/nrow(.)) %>%
  ungroup

agg_dd_reasons$freq <- agg_dd_reasons$freq*2  
agg_dd_reasons$Topic <- as.factor(agg_dd_reasons$Topic)

p <- ggplot(na.omit(agg_dd_reasons), aes(x=(Topic), y=freq, fill=Topic))
# p <- p + geom_tile()
# p <- p + geom_bar(stat='identity')
p <- p + geom_bar(stat = "summary", fun.y = "mean", position=position_dodge())
# p <- p + stat_summary(fun.data = mean_se, geom = "errorbar", position=position_dodge(0.9), aes(width=0.5))
# p <- p + stat_boxplot(geom = "errorbar", width = 0.5)
# p <- p + geom_point(aes(x=device_type, y=necessity))
# p <- p + facet_wrap(~ get(filter_vignette_2), ncol=5)
# p <- p + ggtitle((filter_reason))
p <- p + theme(axis.text=element_text(size=32),
               axis.title=element_text(size=32, face='bold'))
p <- p + theme(legend.text=element_text(size=46),
               legend.title=element_text(size=38))
p <- p + guides(fill = FALSE)
p <- p + theme(legend.position = "none")
# p <- p + scale_fill_gradient(palette="Blues")
p <- p + labs(x="Topic",y="Frequency")
# p <- p + theme(axis.text.x = element_text(face="bold", color="black", 
                                          # size=14, angle=0))
print (p)


p <- ggplot(na.omit(agg_dd_reasons), aes(x=(Topic), y=Percentage, fill=Topic))
# p <- p + geom_tile()
# p <- p + geom_bar(stat='identity')
p <- p + geom_bar(stat = "summary", fun.y = "mean", position=position_dodge())
# p <- p + stat_summary(fun.data = mean_se, geom = "errorbar", position=position_dodge(0.9), aes(width=0.5))
# p <- p + stat_boxplot(geom = "errorbar", width = 0.5)
# p <- p + geom_point(aes(x=device_type, y=necessity))
# p <- p + facet_wrap(~ get(filter_vignette_2), ncol=5)
# p <- p + ggtitle((filter_reason))
p <- p + theme(axis.text=element_text(size=32),
               axis.title=element_text(size=32, face='bold'))
p <- p + theme(legend.text=element_text(size=46),
               legend.title=element_text(size=38))
p <- p + guides(fill = FALSE)
p <- p + theme(legend.position = "none")
# p <- p + scale_fill_gradient(palette="Blues")
p <- p + labs(x="Topic",y="Frequency (%)")
# p <- p + theme(axis.text.x = element_text(face="bold", color="black", 
# size=14, angle=0))
print (p)

## Length (number of words) of definitions

sapply(strsplit(dataBoxplot$privacy.definition, " "), length)

df_length <- dataBoxplot %>% 
            select(privacy.definition) %>%
            rowwise() %>%
            mutate(length=lengths(gregexpr("\\W+", privacy.definition)))

# lengths(gregexpr("\\W+", dataBoxplot$privacy.definition[2]))


agg_dd_length <- df_length %>% na.omit() %>%
  select(length) %>%
  group_by(length) %>%
  summarise(freq = n()) %>%
  # group_by_(.dots=c(filter_vignette)) %>%
  # group_by_(.dots=c(filter_vignette, filter_reason)) %>%
  # group_by(device_type,vignette_answer) %>%
  # mutate(Frequency=n()) %>%
  ungroup

p <- ggplot(na.omit(agg_dd_length), aes(x=(length), y=freq))
# p <- p + geom_tile()
# p <- p + geom_bar(stat='identity')
p <- p + stat_ecdf(geom = "line", position='identity')
# p <- p + geom_density(stat='identity', alpha=0.6 )
# p <- p + geom_bar(stat = "summary", fun.y = "mean", position=position_dodge())
# p <- p + stat_summary(fun.data = mean_se, geom = "errorbar", position=position_dodge(0.9), aes(width=0.5))
# p <- p + stat_boxplot(geom = "errorbar", width = 0.5)
# p <- p + geom_point(aes(x=device_type, y=necessity))
# p <- p + facet_wrap(~ get(filter_vignette_2), ncol=5)
# p <- p + ggtitle((filter_reason))
p <- p + theme(axis.text=element_text(size=36),
               axis.title=element_text(size=36,face="bold"))
p <- p + theme(legend.text=element_text(size=46),
               legend.title=element_text(size=48,face="bold"))
p <- p + guides(fill = FALSE)
p <- p + theme(legend.position = "none")
# p <- p + scale_fill_gradient(palette="Blues")
p <- p + labs(x="Number of words",y="CDF")
# p <- p + theme(axis.text.x = element_text(face="bold", color="black", 
# size=14, angle=0))
print (p)
k

## Kruskal test for topic

model.kruskal <- kruskal.test(Dominant_Topic ~ country.residence, data=dataframe_filtered)
print(model.kruskal)


## Kruskal test for keywords

model.kruskal <- kruskal.test(Num_keywords ~ country.origin, data=dataframe_filtered)
print(model.kruskal)
   
