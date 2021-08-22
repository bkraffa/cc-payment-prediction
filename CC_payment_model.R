setwd('C:/Users/bcpython/Documents/PBIparaDataScience/Cap15')
getwd()

install.packages('Amelia')
install.packages('caret')
install.packages('dplyr')
install.packages('reshape')
install.packages('randomForest')
install.packages('e1071')
install.packages('ggplot2')
install.packages('remotes')

library(Amelia)
library(caret)
library(dplyr)
library(reshape)
library(randomForest)
library(e1071)
library(ggplot2)

dados_clientes <- read.csv('dataset.csv')

View(dados_clientes)
dim(dados_clientes)
str(dados_clientes)
summary(dados_clientes)
 
#Removing col ID
dados_clientes$ID <- NULL

#Renaming class column
colnames(dados_clientes)[24] <- "Inadimplente"

#Checking for missing values and removing them from dataset
sapply(dados_clientes,function(x) sum(is.na(x)))
missmap(dados_clientes, main='Valores Missing Observados')
dados_clientes <- na.omit(dados_clientes)

#Converting the attributes gender, education, marriage and age to factors
colnames(dados_clientes)[2] <- 'Genero'
colnames(dados_clientes)[3] <- 'Escolaridade'
colnames(dados_clientes)[4] <- 'Estado_Civil'
colnames(dados_clientes)[5] <- 'Idade'

#Gender
dados_clientes$Genero <- cut(dados_clientes$Genero,
                             c(0,1,2),
                             labels = c('Masculino',
                                        'Feminino'))
View(dados_clientes$Genero)

#Education
summary(dados_clientes$Escolaridade)
dados_clientes$Escolaridade <- cut(dados_clientes$Escolaridade,
                                   c(0,1,2,3,4),
                                   labels = c('Pós Graduado',
                                              'Graduado',
                                              'Ensino Médio',
                                              'Outros'))
str(dados_clientes$Escolaridade)

#Marriage
dados_clientes$Estado_Civil <- cut(dados_clientes$Estado_Civil,
                                     c(-1,0,1,2,3),
                                     labels = c('Desconhecido',
                                                  'Casado',
                                                  'Solteiro',
                                                  'Outros'))
View(dados_clientes$Estado_Civil)

#Age
hist(dados_clientes$Idade)
dados_clientes$Idade <- cut(dados_clientes$Idade,
                            c(0,30,50,100),
                            labels = c('Jovem',
                                       'Adulto',
                                       'Idoso'))
View(dados_clientes$Idade)
str(dados_clientes$Idade)
summary(dados_clientes$Idade)

#Converting the payment variables to factor
dados_clientes$PAY_0 <- as.factor(dados_clientes$PAY_0)
dados_clientes$PAY_2 <- as.factor(dados_clientes$PAY_2)
dados_clientes$PAY_3 <- as.factor(dados_clientes$PAY_3)
dados_clientes$PAY_4 <- as.factor(dados_clientes$PAY_4)
dados_clientes$PAY_5 <- as.factor(dados_clientes$PAY_5)
dados_clientes$PAY_6 <- as.factor(dados_clientes$PAY_6)
str(dados_clientes)

#Checking for missing values
missmap(dados_clientes, main='Valores Missing Observados')
dados_clientes <- na.omit(dados_clientes)
sapply(dados_clientes,function(x) sum(is.na(x)))

#Converting the independent variable to factor
str(dados_clientes$Inadimplente)
colnames(dados_clientes)
dados_clientes$Inadimplente <- as.factor(dados_clientes$Inadimplente)
str(dados_clientes$Inadimplente)

#Comparing payments vs missing payments
table(dados_clientes$Inadimplente)

#Proportion among classes
prop.table(table(dados_clientes$Inadimplente))

#Distribution plot
qplot(Inadimplente,data=dados_clientes, geom = 'bar')+
  theme(axis.text.x = element_text(angle=90,hjust=1))

#Set seed -> random variable
set.seed(12345)

#Stratifying the sample
indice <- createDataPartition(dados_clientes$Inadimplente, p=0.75, list=FALSE)
dim(indice)

#Creating training data
dados_treino <- dados_clientes[indice,]
table(dados_treino$Inadimplente)
dim(dados_treino)

#Comparing the proportion between training and original classes
compara_dados <- cbind(prop.table(table(dados_treino$Inadimplente)),
                       prop.table(table(dados_clientes$Inadimplente)))
colnames(compara_dados) <- c("Treinamento",'Original')
compara_dados

#Melt data - convert columns in rows
melt_compara_dados <- melt(compara_dados)
melt_compara_dados

#Plot to check the distrbution of the training vs original dataset
ggplot(melt_compara_dados, aes (x=X1, y = value))+
  geom_bar(aes(fill = X2), stat = 'identity', position = 'dodge')+
  theme(axis.text.x = element_text (angle = 90, hjust=1))

#Test Dataset - Everything that is not on the training dataset
dados_teste <- dados_clientes[-indice,]
dim(dados_teste)

#Building first version of the model
modelo_v1 <- randomForest(Inadimplente~.,data = dados_treino)
plot(modelo_v1)

#Predictions with test data
previsoes_v1 <- predict(modelo_v1,dados_teste)

#Confusion matrix
cm_v1 <- caret::confusionMatrix(previsoes_v1,dados_teste$Inadimplente,positive='1')
cm_v1

#Calculating precision, recall, f1score -> Evaluation of the predictive model metrics
y <- dados_teste$Inadimplente
y_pred_v1 <- previsoes_v1

precision <- posPredValue(y_pred_v1,y)
precision

recall <- sensitivity(y_pred_v1,y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

#Balancing classes -> increasing the nr os samples '1' 
remotes::install_github("cran/DMwR")
library(DMwR)

#Applying SMOTE (Synthetic Minority Over-Sampling Technique)
table(dados_treino$Inadimplente)
prop.table(table(dados_treino$Inadimplente))
set.seed(9560)
dados_treino_balanceados <- SMOTE(Inadimplente ~., data = dados_treino)
table(dados_treino_balanceados$Inadimplente)
prop.table(table(dados_treino_balanceados$Inadimplente))

#2nd version of the model - with balanced data
modelo_v2 <- randomForest(Inadimplente ~.,data = dados_treino_balanceados)
modelo_v2

#Predictions with test data
previsoes_v2 <- predict(modelo_v2,dados_teste)

#Confusion matrix
cm_v2 <- caret::confusionMatrix(previsoes_v2,dados_teste$Inadimplente,positive='1')
cm_v2

#Calculating precision, recall, f1score -> Evaluation of the predictive model metrics
y <- dados_teste$Inadimplente
y_pred_v2 <- previsoes_v2

precision <- posPredValue(y_pred_v2,y)
precision

recall <- sensitivity(y_pred_v2,y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

#Checkin the importance of prediction variables
varImpPlot(modelo_v2)

#Storing the most important variables on imp_var object
imp_var <- importance(modelo_v2)
varImportancia <- data.frame(Variables = row.names(imp_var),
                             Importance = round(imp_var[,'MeanDecreaseGini'],2))

#Creating the rank of variables based on importance
rankImportancia <- varImportancia %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

#Plotting the relative importance of the variables
ggplot(rankImportancia,
       aes(x=reorder(Variables,Importance),
           y = Importance,
           fill = Importance)) +
  geom_bar(stat = 'identity') +
  geom_text(aes(x=Variables,y=.5, label = Rank),
            hjust = 0,
            vjust = .55,
            size = 4,
            colour = 'red') +
  labs (x = 'variables') +
  coord_flip()

#Creating the 3rd version of the model only with the most important variables
colnames(dados_treino_balanceados)
modelo_v3 <- randomForest(Inadimplente ~ PAY_0 + PAY_2 + PAY_3 + PAY_AMT1 + PAY_AMT2 + PAY_5 + BILL_AMT1,
                          data = dados_treino_balanceados)
modelo_v3


#Predictions with test data
previsoes_v3 <- predict(modelo_v3,dados_teste)

#Confusion matrix
cm_v3 <- caret::confusionMatrix(previsoes_v3,dados_teste$Inadimplente,positive='1')
cm_v3

#Calculating precision, recall, f1score -> Evaluation of the predictive model metrics
y <- dados_teste$Inadimplente
y_pred_v3 <- previsoes_v3

precision <- posPredValue(y_pred_v3,y)
precision

recall <- sensitivity(y_pred_v3,y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

#Saving the model
saveRDS(modelo_v3,file = 'modelo_v3.rds')

#Loading the model
modelo_final <- readRDS('modelo_v3.rds')

#Predictions with 3 new clients data

#Clients data
PAY_0 <- c(0,0,0)
PAY_2 <- c(0,0,0)
PAY_3 <- c(1,0,0)
PAY_AMT1 <- c(1100,1000,1200)
PAY_AMT2 <- c(1500,1300,1150)
PAY_5 <- c(0,0,0)
BILL_AMT1 <- c(350,420,280)

#Consolidate them in a new dataframe
novos_clientes <- data.frame(PAY_0,PAY_2,PAY_3,PAY_AMT1,PAY_AMT2,PAY_5,BILL_AMT1)
View(novos_clientes)

#Converting the data types
novos_clientes$PAY_0 <- factor(novos_clientes$PAY_0, levels = levels(dados_treino_balanceados$PAY_0))
novos_clientes$PAY_2 <- factor(novos_clientes$PAY_2, levels = levels(dados_treino_balanceados$PAY_2))
novos_clientes$PAY_3 <- factor(novos_clientes$PAY_3, levels = levels(dados_treino_balanceados$PAY_3))
novos_clientes$PAY_5 <- factor(novos_clientes$PAY_5, levels = levels(dados_treino_balanceados$PAY_5))
str(novos_clientes)

#Predictions
previsoes_novos_clientes <- predict(modelo_final, novos_clientes)
previsoes_novos_clientes