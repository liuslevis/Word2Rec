library(MASS)

confusion.glm <- function(data, model) {
  prediction <- ifelse(predict(model, data, type='response') > 0.5, TRUE, FALSE)
  # confusion  <- table(prediction, as.logical(model$y))
  confusion  <- table(prediction, as.logical(data$label))
  confusion  <- cbind(confusion, c(1 - confusion[1,1]/(confusion[1,1]+confusion[2,1]), 1 - confusion[2,2]/(confusion[2,2]+confusion[1,2])))
  confusion  <- as.data.frame(confusion)
  names(confusion) <- c('FALSE', 'TRUE', 'class.error')
  confusion
}

score <- function(confusion) {
    tp <- c[2,2]
    tn <- c[1,1]
    fp <- c[2,1]
    fn <- c[1,2]
    precision <- tp / (tp + fp)
    recall <- tp / (tp + fn)
    df <- as.data.frame(c(precision, recall))
    df
}

combi <- read.csv('input/train_201701_201704_mod100.csv')
trainIndex <- sample(1:nrow(combi), size=round(0.6*nrow(combi)), replace=FALSE)
train <- combi[trainIndex, ]
valid <- combi[-trainIndex, ]

# formula <- label ~ past_tag_num + 
# user_sex_0 +
# user_sex_1 +
# user_sex_2 +
# user_sex_3 +
# user_age_0 +
# user_age_1 +
# user_age_2 +
# user_age_3 +
# user_age_4 +
# user_city_0 +
# user_city_1 +
# user_city_2 +
# user_city_3 +
# user_city_4 +
#   item_sex_0_ctr + 
#   item_sex_1_ctr + 
#   item_sex_2_ctr + 
#   item_sex_3_ctr + 
#   item_age_0_ctr + 
#   item_age_1_ctr + 
#   item_age_2_ctr + 
#   item_age_3_ctr + 
#   item_age_4_ctr + 
#   item_city_0_ctr + 
#   item_city_1_ctr + 
#   item_city_2_ctr + 
#   item_city_3_ctr + 
#   item_city_4_ctr

# Coefficients:
#                   Estimate Std. Error z value Pr(>|z|)
# (Intercept)     -1.744e+00  2.019e-02 -86.393  < 2e-16 ***
# past_tag_num     1.524e-01  1.723e-03  88.432  < 2e-16 ***
# user_sex_0       4.641e-02  2.581e-02   1.798 0.072234 .
# user_sex_1       4.877e-02  1.345e-02   3.627 0.000287 ***
# user_age_1      -2.831e-02  1.695e-02  -1.670 0.094853 .
# user_age_2      -4.489e-02  1.960e-02  -2.290 0.022008 *
# user_age_3      -1.341e-01  2.683e-02  -4.999 5.75e-07 ***
# user_city_0     -7.355e-02  1.483e-02  -4.958 7.11e-07 ***
# user_city_1     -1.011e-01  2.591e-02  -3.903 9.51e-05 ***
# user_city_3     -3.812e-02  2.150e-02  -1.773 0.076188 .
# item_sex_0_ctr  -9.259e+02  9.567e+01  -9.678  < 2e-16 ***
# item_sex_1_ctr  -1.264e+03  1.188e+02 -10.637  < 2e-16 ***
# item_age_0_ctr   5.243e+02  5.606e+01   9.353  < 2e-16 ***
# item_age_1_ctr  -1.965e+02  1.036e+02  -1.896 0.057948 .
# item_age_2_ctr  -8.925e+01  6.272e+01  -1.423 0.154726
# item_age_3_ctr   2.229e+02  2.651e+01   8.408  < 2e-16 ***
# item_city_0_ctr -4.994e+02  9.359e+01  -5.336 9.51e-08 ***
# item_city_1_ctr -5.695e+02  7.808e+01  -7.294 3.02e-13 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

formula <- label ~ past_tag_num +
  user_sex_0 +
  user_sex_1 +
  user_age_1 +
  user_age_2 +
  user_age_3 +
  user_city_0 +
  user_city_1 +
  user_city_3 +
  item_sex_0_ctr +
  item_sex_1_ctr +
  item_age_0_ctr +
  item_age_1_ctr +
  item_age_3_ctr +
  item_city_0_ctr +
  item_city_1_ctr

lr.glm <- glm(
    formula=formula,
    data=train,
    family=binomial("logit"))

lr.stepAIC <- stepAIC(lr.glm, direction="backward") #变量筛选方法-逐步回归对方程修正 向后回归法

summary(lr.glm)
summary(lr.stepAIC)

confusion.glm(train, lr.glm)
c <- confusion.glm(valid, lr.glm)
score(c)

