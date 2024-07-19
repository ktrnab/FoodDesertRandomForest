if (!require("smotefamily")) install.packages("smotefamily", dependencies = TRUE)
if (!require("PRROC")) install.packages("PRROC", dependencies = TRUE)
if (!require("sf")) install.packages("sf", dependencies = TRUE)
if (!require("gridExtra")) install.packages("gridExtra", dependencies = TRUE)
if (!require("pROC")) install.packages("pROC", dependencies = TRUE)
if (!require("doParallel")) install.packages("doParallel", dependencies = TRUE)

library(randomForest)
library(caret)
library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(pROC)
library(PRROC)
library(sf)
library(gridExtra)
library(smotefamily)
library(doParallel)

file_path <- "~/Downloads/FoodAccessResearchAtlasData2019.csv"  # Update this path to your actual file location
data <- read_delim(file_path, delim = ";")
problems(data)

data <- data %>%
  mutate(across(c(Pop2010, OHU2010, MedianFamilyIncome, PovertyRate, lapophalf, laseniorshalf, lawhitehalf,
                  lablackhalf, laasianhalf, lahisphalf, lasnaphalf, lahunvhalf, TractKids, TractSeniors,
                  TractWhite, TractBlack, TractAsian, TractHispanic, TractSNAP), 
                ~ as.numeric(gsub(",", ".", gsub("\\.", "", .)))))

# for missing values in all columns
data <- data %>%
  mutate(across(everything(), ~ifelse(is.na(.), median(., na.rm = TRUE), .)))

# population under 28, over 65, and snap 
data <- data %>%
  mutate(
    proportion_under_18 = TractKids / Pop2010,
    proportion_over_65 = TractSeniors / Pop2010,
    proportion_receiving_snap = TractSNAP / OHU2010
  )

outcome_var <- 'LILATracts_1And10'  
predictor_vars <- c(
  'Pop2010',  # total population
  'proportion_under_18',  # proportion under 18
  'proportion_over_65',  # proportion over 65
  'TractWhite',  # racial composition
  'TractBlack',  # racial composition
  'TractAsian',  # racial composition
  'TractHispanic',  # racial composition
  'MedianFamilyIncome',  # median household income
  'PovertyRate',  # poverty rate
  'proportion_receiving_snap',  # proportion households receiving SNAP benefits
  'lapophalf',  # population beyond 1/2 mile from supermarket
  'lahunvhalf'  # housing units without vehicle beyond 1/2 mile from supermarket
)

data <- data %>% drop_na(all_of(c(outcome_var, predictor_vars)))

data[[outcome_var]] <- as.factor(data[[outcome_var]])

# split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(data[[outcome_var]], p = .8, list = FALSE, times = 1)
data_train <- data[ trainIndex,]
data_test  <- data[-trainIndex,]

# smote
set.seed(123)
smote_result <- SMOTE(X = data_train[predictor_vars], target = data_train[[outcome_var]], K = 5, dup_size = 0)
data_train_smote <- smote_result$data
data_train_smote[[outcome_var]] <- as.factor(data_train_smote$class)
data_train_smote$class <- NULL

# remove NA/NaN/Inf values in the training data
data_train_smote <- data_train_smote %>%
  filter_all(all_vars(!is.na(.))) %>%
  filter_all(all_vars(!is.infinite(.)))

# check for errors
print(any(is.na(data_train_smote)))
print(sapply(data_train_smote, function(x) sum(is.na(x) | is.nan(x) | is.infinite(x))))

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# hyperparameter tuning 
control <- trainControl(method="cv", number=5, search="grid")
tunegrid <- expand.grid(.mtry=c(2:5))  # Reduced grid for faster tuning
set.seed(123)
rf_gridsearch <- train(as.formula(paste(outcome_var, "~", paste(predictor_vars, collapse = "+"))), 
                       data=data_train_smote, method="rf", metric="Accuracy", tuneGrid=tunegrid, trControl=control)

stopCluster(cl)
registerDoSEQ()

# best hyperparameters
print(rf_gridsearch$bestTune)

# 
mtry_best <- rf_gridsearch$bestTune$mtry
if (is.na(mtry_best) || mtry_best < 1) {
  mtry_best <- 2  
}

# training the Random Forest model with best hyperparameters
set.seed(123)
rf_model <- randomForest(as.formula(paste(outcome_var, "~", paste(predictor_vars, collapse = "+"))),
                         data = data_train_smote,
                         mtry = mtry_best,
                         importance = TRUE,
                         ntree = 500)

# evaluate
predictions <- predict(rf_model, data_test)
conf_matrix <- confusionMatrix(predictions, data_test[[outcome_var]])
# print
print(conf_matrix)
#save
write.csv(as.table(conf_matrix), 'results/confusion_matrix.csv', row.names = FALSE)

#Plot: feature importance
importance <- importance(rf_model)
var_importance <- data.frame(Variables = row.names(importance),
                             Importance = round(importance[ , 'MeanDecreaseGini'], 2))

plot1 <- ggplot(var_importance, aes(x = reorder(Variables, Importance), y = Importance)) +
  geom_bar(stat = 'identity') +
  coord_flip() +
  theme_minimal() +
  labs(title = 'Feature Importance', x = 'Features', y = 'Importance')

# Plot: ROC curve
rf_probs <- predict(rf_model, data_test, type = "prob")[,2]
roc_obj <- roc(data_test[[outcome_var]], rf_probs)
plot2 <- ggroc(roc_obj) +
  ggtitle("ROC Curve for Random Forest Model")

# plot: Precision-Recall curve
pr_curve <- pr.curve(scores.class0 = rf_probs, weights.class0 = ifelse(data_test[[outcome_var]] == 1, 1, 0), curve = TRUE)
pr_data <- data.frame(Recall = pr_curve$curve[,1], Precision = pr_curve$curve[,2])
plot3 <- ggplot(pr_data, aes(x = Recall, y = Precision)) +
  geom_line() +
  theme_minimal() +
  labs(title = "Precision-Recall Curve", x = "Recall", y = "Precision")

#boxplot for poverty Rate
plot4 <- ggplot(data, aes(x = as.factor(LILATracts_1And10), y = PovertyRate, fill = as.factor(LILATracts_1And10))) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Poverty Rate in Food Deserts vs Non-Food Deserts", x = "Food Desert", y = "Poverty Rate")

grid.arrange(plot1, plot2, plot3, plot4, ncol = 2)
