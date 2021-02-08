library(AppliedPredictiveModeling)
library(broom)
library(caret)
library(caTools)
library(class)
library(corrplot)
library(DataExplorer)
library(dplyr)
library(e1071) 
library(funModeling)
library(ggfortify) 
library(ggplot2)
library(gridExtra)
library(Hmisc)
library(ISLR)
library(kableExtra)
library(kknn)
library(knitr)
library(lattice)
library(mgcv)
library(MLeval)
library(multiROC)
library(nnet)
library(pander)
library(party)
library(pROC)
library(quantmod) 
library(readxl)
library(rpart)
library(rpart.plot)
library(scatterplot3d)
library( splines)
library(tidyverse)
library(visreg)
theme_set(theme_classic())


bitcoin_dataset <- read.csv(file = "/Users/nselvarajan/Desktop/R/Assignment5/bitcoin_dataset.csv",
                            header = T,stringsAsFactors = T)
bitcoin_dataset<-na.omit(bitcoin_dataset)

plot_missing(bitcoin_dataset)

cor <- cor(bitcoin_dataset[,2:12])
corrplot(cor)
#btc_market_cap, btc_hash_rate, btc_difficulty are most correlated with btc_market_price
plot(bitcoin_dataset$Date, bitcoin_dataset$btc_market_price)


bitcoin_dataset$Days <- 1:nrow(bitcoin_dataset) #Add column that adds a count of days for each row
bitcoin_dataset$Date <- as.Date(bitcoin_dataset$Date) #Format Date as Date
bitcoin_dataset <- subset(bitcoin_dataset, bitcoin_dataset$btc_median_confirmation_time>0) #Subset of data that is clean



ggplot(bitcoin_dataset, aes(bitcoin_dataset$Date, bitcoin_dataset$btc_market_price)) + 
  geom_point(color="firebrick") +
  ggtitle('BTC Value vs. Time') +
  theme(plot.title = element_text(size=20, face="bold", 
                                  margin = margin(10, 0, 10, 0)))+
  labs(x="Date", y="USD")+
  theme(axis.text.x=element_text(angle=50, vjust=0.5)) +
  theme(panel.background = element_rect(fill = 'grey75'))

ggplot(bitcoin_dataset, aes(bitcoin_dataset$btc_market_cap, bitcoin_dataset$btc_market_price)) + 
  geom_point(color="firebrick") +
  ggtitle('BTC Market Capitalization vs. Market Price') +
  theme(plot.title = element_text(size=19.5, face="bold", 
                                  margin = margin(10, 0, 10, 0)))+
  labs(x="Market Cap (USD)", y="Market Price (USD)")+
  theme(axis.text.x=element_text(angle=50, vjust=0.5)) +
  theme(panel.background = element_rect(fill = 'grey75'))+
  stat_smooth(method = "lm",  formula = y ~ x, col = "yellow")




str(bitcoin_dataset[,c(2:6,11,12,13,14,16,24)])

#cor <- cor(bitcoin_dataset[,c(2:6,11,12,13,14,16,24)]) #selecting variables to include in correlation analysis
#transparentTheme(trans = .4)

correlation_r <- rcorr(as.matrix(bitcoin_dataset[,c(2:6,11,12,13,14,16,24)]))
correlation_Matrix <- correlation_r$r
p_mat <- correlation_r$P


#col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
corr_graph<-corrplot(correlation_Matrix, type = "upper", order = "hclust", 
                     p.mat = p_mat, sig.level = 0.05)
par(mfrow = c(1, 1))
png(height=1200, width=1500, pointsize=15, file="overlapmnb.png")
corsig<-corrplot(correlation_Matrix,number.cex=0.75,
                 method = "color",
                 col = col(200),  
                 type = "upper", order = "hclust", 
                 addCoef.col = "black", # Add coefficient of correlation
                 tl.col = "darkblue", tl.srt = 45, #Text label color and rotation
                 # Combine with significance level
                 p.mat = p_mat, sig.level = 0.05, insig = "blank", 
                 # hide correlation coefficient on the principal diagonal
                 diag = FALSE,
                 title = "Correlation Between Significant Biomarkers",
                 mar=c(0,0,1,0)
)
knitr::include_graphics("overlapmnb.png")


colnames(cor) <- c("MarketPrice", "TotalBTC", "MarketCap", "BlocksSize","Ntransactions","HashRate","BTCDifficulty","MinersRevenue","TransactionFees","CostPerTransaction%","CostPerTransaction","EstTransactionVolUSD")
rownames(cor) <-c("MarketPrice", "TotalBTC", "MarketCap", "BlocksSize","Ntransactions","HashRate","BTCDifficulty","MinersRevenue","TransactionFees","CostPerTransaction%","CostPerTransaction","EstTransactionVolUSD")
corrplot(cor, method = "square",  tl.srt = 50, tl.col = "black", tl.cex = 0.6, title = "Correlation of Variables", mar=c(0,0,1,0))


w1 <- ggplot(bitcoin_dataset, aes(y=btc_market_price, x=btc_market_cap)) + geom_point(colour="blue")
w1<- w1 + stat_smooth(method="lm", formula = y~poly(x,2))+ ggtitle("Regression Plot \n Polynomial \n Market Cap")

w2 <- ggplot(bitcoin_dataset, aes(y=btc_market_price, x=btc_hash_rate)) + geom_point(colour="blue")
w2<- w2 + stat_smooth(method="lm", formula = y~poly(x,2))+ ggtitle("Regression Plot \n Polynomial \n Hash Rate")

w3 <- ggplot(bitcoin_dataset, aes(y=btc_market_price, x=btc_difficulty)) + geom_point(colour="blue")
w3<- w3 + stat_smooth(method="lm", formula = y~poly(x,2))+ ggtitle("Regression Plot \n Polynomial \n Difficulty")

w4 <- ggplot(bitcoin_dataset, aes(y=btc_market_price, x=btc_miners_revenue)) + geom_point(colour="blue")
w4<- w4 + stat_smooth(method="lm", formula = y~poly(x,2))+ ggtitle("Regression Plot \n Polynomial \n Miners Revenue")

w5 <- ggplot(bitcoin_dataset, aes(y=btc_market_price, x=btc_estimated_transaction_volume_usd)) + geom_point(colour="blue")
w5<- w5 + stat_smooth(method="lm", formula = y~poly(x,2))+ ggtitle("Regression Plot \n Polynomial \n Transaction Volume USD")

grid.arrange(w1,w2,w3,w4,w5, ncol=2)








####Linear Regression

bitcoin_datasetlm <- bitcoin_dataset

trainIndex1 <- createDataPartition(bitcoin_datasetlm$btc_market_price, p = 0.8, list=FALSE, times=3)
subTrain1 <- bitcoin_datasetlm[trainIndex1,]
subTest1 <- bitcoin_datasetlm[-trainIndex1,]

# setup cross validation and control parameters
control <- trainControl(method="repeatedcv", number=3, repeats = 3, verbose = TRUE, search = "grid")
metric <- "RMSE"
tuneLength <- 10

# Training process 
# Fit / train a Linear Regression model to  dataset
linearModelReg <- caret::train(btc_market_price~
                                 btc_market_cap+btc_hash_rate+
                                 btc_difficulty+btc_miners_revenue+
                                 btc_estimated_transaction_volume_usd
                       ,data=subTrain1, method="lm", metric=metric, 
                       preProc=c("center", "scale"), trControl=control, tuneLength = tuneLength)

linearplotmodel<-lm( btc_market_price~
                       btc_market_cap+btc_hash_rate+
                       btc_difficulty+btc_miners_revenue+
                       btc_estimated_transaction_volume_usd, data = subTrain1)


summary(linearModelReg)

predictions<-predict(linearModelReg,newdata = subTest1)

rmse<-RMSE( predictions, subTest1$btc_market_price)
rmse

error.rate.linear=rmse/mean(subTest1$btc_market_price)
error.rate.linear

linearr2= R2( predictions, subTest1$btc_market_price) 
linearr2

lineardf <- data.frame( RMSE = rmse, R2 = linearr2 , Error =error.rate.linear) 

####Polynominal Regression

poly_reg<-lm( btc_market_price~
                poly( btc_market_cap,2)+ poly( btc_hash_rate,2)+
             poly( btc_difficulty,2)+ poly( btc_miners_revenue,2)+
             poly( btc_estimated_transaction_volume_usd,2), data = subTrain1)

predictionpoly1<-predict(poly_reg,newdata = subTest1)


rmsepoly<-RMSE( predictions, subTest1$btc_market_price)
rmsepoly

error.rate.poly=rmse/mean(subTest1$btc_market_price)
error.rate.poly

polyrsquare = R2( predictionpoly1, subTest1$btc_market_price) 
polyrsquare

polydf <- data.frame( RMSE = rmsepoly, R2 = polyrsquare , Error =error.rate.poly) 


###Spline

knots <- quantile( subTrain1$btc_market_price, p = c( 0.25, 0.5, 0.75))

splinemodel<-lm( btc_market_price~
                bs( btc_market_cap, knots = knots)+ bs( btc_hash_rate, knots = knots)+
                bs( btc_difficulty, knots = knots)+ bs( btc_miners_revenue, knots = knots)+
                bs( btc_estimated_transaction_volume_usd, knots = knots), data = subTrain1)


predictionspline<-predict(splinemodel,newdata = subTest1)

rmsespline<-RMSE( predictionspline, subTest1$btc_market_price)
rmsespline

error.rate.spline=rmsespline/mean(subTest1$btc_market_price)
error.rate.spline

splinersquare = R2( predictionspline, subTest1$btc_market_price) 
splinersquare
# Model performance 

splinedf <- data.frame( RMSE = rmsespline, R2 = splinersquare , Error =error.rate.spline) 




### Generalized Linear Model



mod_lm2 <- gam(btc_market_price ~ btc_difficulty+btc_miners_revenue+btc_estimated_transaction_volume_usd, data=bitcoin_dataset)

lmfit6 <- gam(btc_market_price ~ btc_estimated_transaction_volume_usd + btc_miners_revenue, data=bitcoin_dataset)

summary(lmfit6)

predictiongam<-predict(lmfit6,newdata = subTest1)

rmsegam<-RMSE( predictiongam, subTest1$btc_market_price)
rmsegam

error.rate.gam=rmsegam/mean(subTest1$btc_market_price)
error.rate.gam

rsquaregam = R2( predictiongam, subTest1$btc_market_price) 
rsquaregam

gamdf <- data.frame( RMSE = rmsegam, R2 = rsquaregam , Error =error.rate.gam) 


s3d <- scatterplot3d(bitcoin_dataset$btc_estimated_transaction_volume_usd,
                     bitcoin_dataset$btc_miners_revenue,
                     bitcoin_dataset$btc_market_price, 
                     pch=16, highlight.3d = TRUE, type = "h", 
                     main = "Multi-Variable Regression 
                     \nMarket Price ~ Transaction Volume + Miners Revenue", 
                     xlab="Transaction Volume", 
                     ylab="Miners Revenue", 
                     zlab="Value (USD)", 
                     angle=35)
s3d$plane3d(lmfit6)

#Plotting the Model
par(mfrow=c(1,1)) #to partition the Plotting Window
plot(mod_lm2, all.terms = TRUE) 
#se stands for standard error Bands
gam.check(mod_lm2, k.rep=1000)


#Specify A Smoothing Spline Fit In A GAM Formula.Cubic regression splines

mod_lm4 <- gam(btc_market_price ~ s(btc_total_bitcoins, bs="cr")+s(btc_avg_block_size, bs="cr")+
                 s(btc_transaction_fees, bs="cr"),
               data=bitcoin_dataset)
summary(mod_lm4)

gam.check(mod_lm4, k.rep=1000)

vis.gam(mod_lm4, type='response', plot.type='persp',
         phi=30, theta=30, n.grid=500, border=NA)
visreg2d(mod_lm4, xvar='btc_avg_block_size', yvar='btc_market_price', scale='response')


predictiongamsmooth<-predict(mod_lm4,newdata = subTest1)

rmsegamsmooth<-RMSE( predictiongamsmooth, subTest1$btc_market_price)
rmsegamsmooth

error.rate.gam.smooth=rmsegamsmooth/mean(subTest1$btc_market_price)
error.rate.gam.smooth

rsquaregamsmooth = R2( predictiongamsmooth, subTest1$btc_market_price) 
rsquaregamsmooth

gamsmoothdf <- data.frame( RMSE = rmsegamsmooth, R2 = rsquaregamsmooth , Error =error.rate.gam.smooth) 



### Stochastic Gradient Boosting 

bitcoin_dataset4 <- bitcoin_dataset[,c(2:6,11,12,13,14,16,24)]

trainIndex <- createDataPartition(bitcoin_dataset4$btc_market_price, p = 0.8, list=FALSE, times=3)
subTrain <- bitcoin_dataset4[trainIndex,]
subTest <- bitcoin_dataset4[-trainIndex,]

model_gbm <- train(btc_market_price ~ .,
                          data = subTrain,
                          method = "gbm",
                          preProcess = c("scale", "center"),
                           trControl = trainControl(method = "repeatedcv", 
                                                    number = 5, 
                                                    repeats = 3, 
                                                    verboseIter = FALSE),
                          verbose = 0)

model_gbm
model_gbm$results   
plot(model_gbm)

pred_y = predict(model_gbm, subTest)

mse = mean((subTest[, 1] - pred_y)^2)
mae = caret::MAE(subTest[, 1], pred_y)
rmse = caret::RMSE(subTest[, 1], pred_y)

cat("MSE: ", mse, "MAE: ", mae, " RMSE: ", rmse)

x = 1:length(subTest[, 1])
plot(x, subTest[, 1], col = "red", type = "l",lty=3, lwd=3, xlab='x', ylab='y')
lines(x, pred_y, col = "blue", type = "l")
legend(x = 1, y = 5000,  legend = c("original test_y", "predicted test_y"), 
       col = c("red", "blue"), box.lty = 1, cex = 0.8, lty = c(1, 1))



rmseStochastic<-RMSE( pred_y, subTest$btc_market_price)
rmseStochastic

error.rate.gam.stochastic=rmseStochastic/mean(subTest$btc_market_price)
error.rate.gam.stochastic

rsquarestochastic = R2( predictiongamsmooth, subTest$btc_market_price) 
rsquarestochastic

gamstochasticdf <- data.frame( RMSE = rmseStochastic, R2 = rsquarestochastic , Error =error.rate.gam.stochastic) 



#### Extreme Gradient Boosting
xgbGrid <- expand.grid(nrounds = c(140,160),  # this is n_estimators in the python code above
                       max_depth = c(10, 15, 20, 25),
                       colsample_bytree = seq(0.5, 0.9, length.out = 5),
                       ## The values below are default values in the sklearn-api. 
                       eta = 0.3,
                       gamma=0,
                       min_child_weight = 1,
                       subsample = 1
)

model_xgb <- train(btc_market_price ~ .,
                   data = subTrain,
                   method = "xgbTree",
                   preProcess = c("scale", "center"),
                   trControl = trainControl(method = "repeatedcv", 
                                            number = 5, 
                                            repeats = 3, 
                                            verboseIter = FALSE),
                   tuneGrid = xgbGrid,
                   verbose = 0)

model_xgb$results   
plot(model_xgb)

pred_xgb = predict(model_xgb, subTest)

mse = mean((subTest[, 1] - pred_xgb)^2)
mae = caret::MAE(subTest[, 1], pred_xgb)
rmse = caret::RMSE(subTest[, 1], pred_xgb)

cat("MSE: ", mse, "MAE: ", mae, " RMSE: ", rmse)

x = 1:length(subTest[, 1])
plot(x, subTest[, 1], col = "red", type = "l",lty=3, lwd=3, xlab='x', ylab='y')
lines(x, pred_xgb, col = "blue", type = "l")
legend(x = 1, y = 5000,  legend = c("original test_y", "predicted test_y"), 
       col = c("red", "blue"), box.lty = 1, cex = 0.8, lty = c(1, 1))


rmseXgb<-RMSE( pred_xgb, subTest$btc_market_price)
rmseXgb

error.rate.Xgb=rmseXgb/mean(subTest$btc_market_price)
error.rate.Xgb

rsquarexgb = R2( pred_xgb, subTest$btc_market_price) 
rsquarexgb

gamstochasticdf <- data.frame( RMSE = rmseXgb, R2 = rsquarexgb , Error =error.rate.Xgb) 






head(linearModelReg.diagnostics)

par(mfrow = c(2, 2))
plot(splinemodel)
plot(poly_reg)
plot(linearplotmodel)


plot(poly_reg, 4)
plot(splinemodel, 4)
plot(linearplotmodel, 4)


# Residuals vs Leverage
plot(poly_reg, 5)
plot(splinemodel, 5)
plot(linearplotmodel, 5)




plot(linearplotmodel$fitted, linearplotmodel$residuals, xlab = "Fitted Values", ylab = "Residuals")
abline(0,0)

plot(splinemodel$fitted, splinemodel$residuals, xlab = "Fitted Values", ylab = "Residuals")
abline(0,0)

plot(poly_reg$fitted, poly_reg$residuals, xlab = "Fitted Values", ylab = "Residuals")
abline(0,0)

bitcoin_datasetplot <- bitcoin_datasetlm
bitcoin_datasetplot$btc_market_price <- as.factor(bitcoin_datasetplot$btc_market_price)
