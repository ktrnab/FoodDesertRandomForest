# Load necessary libraries
library(randomForest)
library(caret)
library(dplyr)
library(tidyr)
library(readr)

# Define the local path of the dataset
file_path <- "~/Downloads/FoodAccessResearchAtlasData2019.csv"  

# Load the dataset from the CSV file with the correct delimiter
data <- read_delim(file_path, delim = ";")

# Convert necessary columns to numeric and handle commas in numeric columns
data <- data %>%
  mutate(across(c(Pop2010, OHU2010, MedianFamilyIncome, PovertyRate, lapophalf, laseniorshalf, lawhitehalf,
                  lablackhalf, laasianhalf, lahisphalf, lasnaphalf, lahunvhalf, TractKids, TractSeniors,
                  TractWhite, TractBlack, TractAsian, TractHispanic, TractSNAP), 
                ~ as.numeric(gsub(",", ".", gsub("\\.", "", .)))))

# Display the first few rows and column names to understand the structure of the dataset
print(head(data))
print(colnames(data))

# Check for NA values in the dataset
print(colSums(is.na(data)))

# handle missing values in all columns
data <- data %>%
  mutate(across(everything(), ~ifelse(is.na(.), median(., na.rm = TRUE), .)))

# verify that there are no NA values in the dataset
print(colSums(is.na(data)))

# new features
data <- data %>%
  mutate(
    proportion_under_18 = TractKids / Pop2010,
    proportion_over_65 = TractSeniors / Pop2010,
    proportion_receiving_snap = TractSNAP / OHU2010
  )

# outcome variable and predictor variables
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

# remove rows with NA values in the outcome and predictor variables
data <- data %>% drop_na(all_of(c(outcome_var, predictor_vars)))

# no NA values in predictor variables
print(colSums(is.na(data[predictor_vars])))

# check for any remaining NA, NaN, or Inf values in the predictor variables
data <- data %>%
  filter_all(all_vars(!is.na(.))) %>%
  filter_all(all_vars(!is.infinite(.)))

# check if no NA, NaN, or Inf values in the dataset
print(sapply(data[predictor_vars], function(x) sum(is.na(x) | is.nan(x) | is.infinite(x))))

# outcome variable to factor for classification
data[[outcome_var]] <- as.factor(data[[outcome_var]])

# Split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(data[[outcome_var]], p = .8, 
                                  list = FALSE, 
                                  times = 1)
data_train <- data[ trainIndex,]
data_test  <- data[-trainIndex,]

# Random Forest model
set.seed(123)
rf_model <- randomForest(as.formula(paste(outcome_var, "~", paste(predictor_vars, collapse = "+"))), 
                         data = data_train, 
                         importance = TRUE,
                         ntree = 500)

# Evaluate the model on the test set
predictions <- predict(rf_model, data_test)
conf_matrix <- confusionMatrix(predictions, data_test[[outcome_var]])

# Print model evaluation metrics
print(conf_matrix)

# Feature importance analysis
importance <- importance(rf_model)
var_importance <- data.frame(Variables = row.names(importance), 
                             Importance = round(importance[ , 'MeanDecreaseGini'], 2))

# Plot feature importance
library(ggplot2)
ggplot(var_importance, aes(x = reorder(Variables, Importance), y = Importance)) +
  geom_bar(stat = 'identity') + 
  coord_flip() + 
  theme_minimal() +
  labs(title = 'Feature Importance', x = 'Features', y = 'Importance')

# save the model
save(rf_model, file = 'random_forest_model.RData')

# save the variable importance
write.csv(var_importance, 'variable_importance.csv', row.names = FALSE)
