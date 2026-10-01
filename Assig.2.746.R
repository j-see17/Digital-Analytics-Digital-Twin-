##########################
# Individual Assignment #2
# Julia See
# BUMK746
# Jacobs
##########################

############################################
# Getting Working Libraries and Variables
###########################################
options(stringAsFactors = FALSE)

library(class)
library(klaR)
library(e1071)
library(rpart)
library(randomForest)
library(leaps)

reviewer_path <- "/Users/juliasee/Downloads/reviewer_withavg.csv"
bank_path <- "/Users/juliasee/Downloads/bank-full-shuffled.csv"

f1_score <- function(actual, predicted, positive = "yes"){
  tp <- sum(actual == positive & predicted == positive)
  fp <- sum(actual != positive & predicted == positive)
  fn <- sum(actual == positive & predicted != positive)
  
  precision <- if ((tp + fp) == 0) 0 else tp / (tp + fp)
  recall <- if ((tp + fn) == 0) 0 else tp / (tp + fn)
  
  if ((precision + recall) == 0) {
    return(0)
  }
  
  2 * precision * recall / (precision + recall)
}

reviewer <- read.csv(reviewer_path)
reviewer$Popular <- factor(
  ifelse(reviewer$Top10 == 1 | reviewer$Top50 == 1 | reviewer$Top100 == 1,1,0)
         )

reviewer_features <- c(
    "avg_centrality",
    "avg_content",
    "avg_viewership",
    "avg_enhcontent"
  )

reviewer_test <- reviewer[1:50,]
reviewer_train <- reviewer[51:nrow(reviewer),]

train_x <- scale(reviewer_train[, reviewer_features])

test_x <- scale(
  reviewer_test[, reviewer_features],
  center = attr(train_x, "scaled:center"),
  scale = attr(train_x, "scaled:scale")
)

train_y <- reviewer_train$Popular
test_y <- reviewer_test$Popular


##############################
# Part 1 : Classification
##############################
knn_summary <- data.frame(
  k = 1:10,
  train_error = integer(10),
  test_error = integer(10)
)

for (k in 1:10) {
  knn_train_pred <- knn(train = train_x, test = train_x, cl = train_y, k = k)
  knn_test_pred <- knn(train = train_x, test = test_x, cl = train_y, k = k)
  
  knn_summary$train_error[k] <- sum(knn_train_pred != train_y)
  knn_summary$test_error[k] <- sum(knn_test_pred != test_y)
}

print(knn_summary)

# 1. k-NN: best K for training data
best_k_train <- knn_summary$k[which.min(knn_summary$train_error)]
best_train_error <- min(knn_summary$train_error)

cat("1. For k-NN, the k value that minimizes classification error in the TRAINING dataset is:",
    best_k_train, "\n")
cat(" Error count:", best_train_error, "\n\n")

# 2. k-NN: best K for testing data
best_k_test <- knn_summary$k[which.min(knn_summary$test_error)]
best_test_error <- min(knn_summary$test_error)

cat("2. For k-NN, the value that minimizes classification error in the TESTING dataset is:",
    best_k_test, "\n")
cat(" Error count:", best_test_error, "\n\n")

# 3. Naive Bayes
naive_bayes_fit <- NaiveBayes(x = reviewer_train[, reviewer_features], grouping = train_y)
naive_bayes_test_pred <- predict(naive_bayes_fit, reviewer_test[, reviewer_features]) $class
nb_test_error <- sum(naive_bayes_test_pred != test_y)

cat("3. For Naive Bayes, the error count for the TESTING dataset is:",
    nb_test_error, "\n\n")

# 4. SVM linear and polynomial: training errors
svm_formula <- Popular ~ avg_centrality + avg_content + avg_viewership + avg_enhcontent

svm_linear <- svm(svm_formula, data = reviewer_train, kernel = "linear")
svm_poly <- svm(svm_formula, data = reviewer_train, kernel = "polynomial")

svm_linear_train_pred <- predict(svm_linear, reviewer_train)
svm_linear_test_pred <- predict(svm_linear, reviewer_test)
svm_poly_train_pred <- predict(svm_poly, reviewer_train)
svm_poly_test_pred <- predict(svm_poly, reviewer_test)

svm_linear_train_error <- sum(svm_linear_train_pred != train_y)
svm_poly_train_error <- sum(svm_poly_train_pred != train_y)

cat("4. For SVM, the TRAINING error counts are:\n")
cat("   Linear kernel:", svm_linear_train_error, "\n")
cat("   Polynomial kernel:", svm_poly_train_error, "\n\n")

# 5. SVM linear and polynomial: testing errors
svm_linear_test_error <- sum(svm_linear_test_pred != test_y)
svm_poly_test_error <- sum(svm_poly_test_pred != test_y)

cat("5. For SVM, the TESTING error counts are:\n")
cat(" Linear kernel:", svm_linear_test_error, "\n")
cat(" Polynomial kernel:", svm_poly_test_error, "\n\n")

# 6. Best algorithm by testing error count
method_names <- c(
  paste0("k-NN (k=", best_k_test,")"),
  "Naive Bayes",
  "SVM Linear",
  "SVM Polynomial"
)

method_test_errors <- c(
  best_test_error,
  nb_test_error,
  svm_linear_test_error,
  svm_poly_test_error
)

best_method <- method_names[which.min(method_test_errors)]
best_method_error <- min(method_test_errors)

cat("6. Judging by TESTING error count, the best-performing algorithm is:",
    best_method, "\n")
cat(" Testing error count:", best_method_error, "\n")

#################################
# Part 2 : Tree and Random Forest
#################################
bank <- read.csv(bank_path, sep = ";")

split_row <- floor(0.8 * nrow(bank))
bank_train <- bank[1:split_row,]
bank_test <- bank[(split_row +1):nrow(bank),]

bank_train$y <- factor(bank_train$y)
bank_test$y <- factor(bank_test$y, levels = levels(bank_train$y))

bank_formula <- y ~ duration + month + poutcome + job + education + margin.table

# 1: Original Classification Tree
bank <- read.csv(bank_path, sep = ";")

split_row <- floor(0.8 * nrow(bank))
bank_train <- bank[1:split_row, ]
bank_test <- bank[(split_row + 1):nrow(bank),]

bank_train$y <- factor(bank_train$y)
bank_test$y <- factor(bank_test$y, levels = levels(bank_train$y))

bank_formula <- y ~ duration + month + poutcome + job + education + marital

set.seed(1)
tree_fit <- rpart(
  bank_formula, 
  data = bank_train,
  method = "class",
  control = rpart.control(cp = 0.001)
)

tree_train_pred <- predict(tree_fit, bank_train, type = "class")
tree_test_pred <- predict(tree_fit, bank_test, type = "class")

tree_train_cm <- table(actual = bank_train$y, predicted = tree_train_pred)
tree_test_cm <- table(actual = bank_test$y, predicted = tree_test_pred)

tree_train_f1 <- f1_score(bank_train$y, tree_train_pred)
tree_test_f1 <- f1_score(bank_test$y, tree_test_pred)

cat("1. Original classification tree\n")
cat(" Training confusion matrix:\n")
print(tree_train_cm)
cat(" Testing F1:", round(tree_test_f1, 4),"\n\n")

# 2: Pruned Tree
cp_table <- tree_fit$cptable
prune_cp <- cp_table[which.min(cp_table[, "xerror"]), "CP"]

pruned_tree <- prune(tree_fit, cp = prune_cp)

pruned_train_pred <- predict(pruned_tree, bank_train, type = "class")
pruned_test_pred <- predict(pruned_tree, bank_test, type = "class")

pruned_train_cm <- table(actual = bank_train$y, predicted = pruned_train_pred)
pruned_test_cm <- table(actual = bank_test$y, predicted = pruned_test_pred)

pruned_train_f1 <- f1_score(bank_train$y, pruned_train_pred)
pruned_test_f1 <- f1_score(bank_test$y, pruned_test_pred)

cat("2. Pruned Tree\n")
cat(" Chosen complexity parameter (cp):", prune_cp, "\n")
cat(" Reason: this cp gives the minimum cross-validated xerror in the CP table.\n\n")

cat(" Training confusion matrix:\n")
print(pruned_train_cm)
cat(" Training F1:", round(pruned_train_f1, 4), "\n\n")

cat(" Testing confusion matrix:\n")
print(pruned_test_cm)
cat(" Testing F1:", round(pruned_test_f1, 4), "\n\n")

cat(" Comparison with original tree:\n")
cat(" Original tree testing F1:",round(tree_test_f1, 4),"\n")
cat( " Pruned tree testing F1:", round(pruned_test_f1,4),"\n\n")

# 3: Random Forest
set.seed(1)

rf_fit <- randomForest(
  bank_formula,
  data = bank_train,
  ntree = 50,
  importance = TRUE
)

rf_train_pred <- predict(rf_fit, bank_train, type = "class")
rf_test_pred <- predict(rf_fit, bank_test, type = "class")

rf_train_cm <- table(actual = bank_train$y, predicted = rf_train_pred)
rf_test_cm <- table(actual = bank_test$y, predicted = rf_test_pred)

rf_train_f1 <- f1_score(bank_train$y, rf_train_pred)
rf_test_f1 <- f1_score(bank_test$y, rf_test_pred)

rf_importance <- importance(rf_fit)
top_vars <- rownames(rf_importance)[order(rf_importance[, "MeanDecreaseAccuracy"], decreasing = TRUE)][1:2]

cat("3. Random forest (50 trees)\n")
cat("   Training confusion matrix:\n")
print(rf_train_cm)
cat("   Training F1:", round(rf_train_f1, 4), "\n\n")

cat("   Testing confusion matrix:\n")
print(rf_test_cm)
cat("   Testing F1:", round(rf_test_f1, 4), "\n\n")

cat("   Comparison of testing F1 scores:\n")
cat("   Original tree:", round(tree_test_f1, 4), "\n")
cat("   Pruned tree:", round(pruned_test_f1, 4), "\n")
cat("   Random forest:", round(rf_test_f1, 4), "\n\n")

cat("   Two most important variables (MeanDecreaseAccuracy):\n")
print(top_vars)
cat("\n")

########################
# Part 3 : Regression
########################
regression_formula <- avg_content ~ avg_centrality + avg_viewership + avg_enhcontent + Top10 + Top50 + Top100 + Advisor + Lead

full_search <- regsubsets(
  regression_formula,
  data = reviewer,
  nvmax = 8,
  method = "exhaustive"
)
full_summary <- summary(full_search)

forward_search <- regsubsets(
  regression_formula,
  data = reviewer, 
  nvmax = 8,
  method = "forward"
)
forward_summary <- summary(forward_search)

full_best_size <- which.min(full_summary$cp)
forward_best_size <- which.min(forward_summary$cp)

full_best_model <- names(coef(full_search, full_best_size))[-1]
forward_best_model <- names(coef(forward_search, forward_best_size))[-1]

full_3var <- names(coef(full_search, 3))[-1]
forward_3var <- names(coef(full_search, 3))[-1]

full_5var <- names(coef(full_search, 5))[-1]
forward_5var <- names(coef(full_search, 5))[-1]

cat("1. Best model according to full model search:\n")
print(full_best_model)
cat("\n")

cat("2. Best model according to forward-stepwise regression:\n ")
print(forward_best_model)
cat("\n")

cat("3. Best model using only 3 independant variables:\n")
cat(" Full model search:\n")
print(full_3var)
cat(" Forward-stepwise regression:\n")
print(forward_3var)
cat(" Same set?:", identical(full_3var, forward_3var), "\n\n")

cat("4. Best model using 5 independant variables:\n")
cat(" Full model search:\n")
print(full_5var)
cat(" Forward-stepwise regression:\n")
print(forward_5var)
cat(" Same set?:", identical(full_5var, forward_5var), "\n\n")
