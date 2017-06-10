library(MASS)

confusion.glm <- function(data, model) {
  prediction <- ifelse(predict(model, data, type='response') > 0.5, TRUE, FALSE)
  confusion  <- table(prediction, as.logical(model$y))
  confusion  <- cbind(confusion, c(1 - confusion[1,1]/(confusion[1,1]+confusion[2,1]), 1 - confusion[2,2]/(confusion[2,2]+confusion[1,2])))
  confusion  <- as.data.frame(confusion)
  names(confusion) <- c('FALSE', 'TRUE', 'class.error')
  confusion
}

combi <- read.csv('input/train_201701_201704_mod100.csv')
trainIndex <- sample(1:nrow(combi), size=round(0.6*nrow(combi)), replace=FALSE)
train <- combi[trainIndex, ]
valid <- combi[-trainIndex, ]

formula <- label ~ read_tag_num
formula <- label ~ user_sex + user_age + user_city + read_tag_num

lr.glm <- glm(
    formula=formula,
    data=train,
    family=binomial("logit"))

# lr.stepAIC = stepAIC(lr.glm, direction="backward") #变量筛选方法-逐步回归对方程修正 向后回归法

summary(lr.glm)
# summary(lr.stepAIC)

confusion.glm(train, lr.glm)