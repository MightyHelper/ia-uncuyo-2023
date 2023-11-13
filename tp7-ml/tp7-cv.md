
# 7
```{r}
create_folds <- function(dataframe, num_folds, seed=123) {
  set.seed(seed)
  folds <- createFolds(y = seq_len(nrow(dataframe)), k = num_folds)
  folds_list <- vector("list", length = num_folds)
  for (i in seq_along(folds)) folds_list[[i]] <- folds[[i]]
  names(folds_list) <- paste0("Fold", seq_along(folds))
  return(folds_list)
}
df <- data.frame(A = 1:100, B = rnorm(100))
folds <- create_folds(df, 10, seed=111)
print(folds)
```

```{r}
cross_validation <- function(dataframe, num_folds, seed=123) {
    set.seed(seed)
    folds <- createFolds(y = seq_len(nrow(dataframe)), k = num_folds)
    accuracies <- numeric(num_folds)
    precisions <- numeric(num_folds)
    sensitivities <- numeric(num_folds)
    specificities <- numeric(num_folds)
    for (fold_index in seq_along(folds)) {
        train_indices <- unlist(folds[-fold_index])
        test_indices <- folds[[fold_index]]
        train_data <- dataframe[train_indices, ]
        test_data <- dataframe[test_indices, ]
        model <- rpart(formula = inclinacion_peligrosa ~ ., data = train_data, method = "class")
        predictions <- predict(model, newdata = test_data, type = "class")
        predictions <- factor(predictions)
        test_data$inclinacion_peligrosa <- factor(test_data$inclinacion_peligrosa)
        confusion_matrix <- confusionMatrix(predictions, test_data$inclinacion_peligrosa)
        accuracies[fold_index] <- confusion_matrix$overall["Accuracy"]
        precisions[fold_index] <- confusion_matrix$byClass["Precision"]
        sensitivities[fold_index] <- confusion_matrix$byClass["Sensitivity"]
        specificities[fold_index] <- confusion_matrix$byClass["Specificity"]
    }
    mean_accuracy <- mean(accuracies)
    sd_accuracy <- sd(accuracies)
    mean_precision <- mean(precisions)
    sd_precision <- sd(precisions)
    mean_sensitivity <- mean(sensitivities)
    sd_sensitivity <- sd(sensitivities)
    mean_specificity <- mean(specificities)
    sd_specificity <- sd(specificities)
    result <- list(
        mean_accuracy = mean_accuracy,
        sd_accuracy = sd_accuracy,
        mean_precision = mean_precision,
        sd_precision = sd_precision,
        mean_sensitivity = mean_sensitivity,
        sd_sensitivity = sd_sensitivity,
        mean_specificity = mean_specificity,
        sd_specificity = sd_specificity
    )

    return(result)
}

set.seed(42)
results <- cross_validation(arbolado.mza.dataset[, !colnames(arbolado.mza.dataset) %in% c("especie", "ultima_modificacion") ], 50)
print(results)
```
