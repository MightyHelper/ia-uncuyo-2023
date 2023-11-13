# Kaggle
```{r}
encode_danger <- function(data, column, out) data %>%
        dplyr::group_by(!!sym(column)) %>%
        dplyr::summarize(count = sum(inclinacion_peligrosa == 1), total = n()) %>%
        dplyr::mutate({{out}} := count / total) %>%
        dplyr::select(-count, -total)
```

```{r}
test_model_output <- function(model, filename){
  root_path <- "~/workspace/arbolado-mza/"
  test.data.set <- read.csv(paste0(root_path, "arbolado-mza-dataset-test.csv/arbolado-mza-dataset-test.csv"))
  
  result.test <- model(testing_data)
  print(result.test)
  confusion_matrix <- get_confusion(result.test$inclinacion_peligrosa, result.test$y_hat_tree)
  print(confusion_matrix)
  print(get_stats(confusion_matrix))
  
  result.validation <- model(test.data.set)
  columns <- data.frame(id = as.numeric(result.validation$id), inclinacion_peligrosa = as.numeric(result.validation$y_hat_tree) - 1)
  write.csv(columns, file = paste0(root_path, "predictions/", filename), row.names=F, append=FALSE, quote=FALSE)
}
```
Con undersampling
```{r}
t <- random_classifier(training_data, ratio)(training_data) %>% filter(inclinacion_peligrosa == 1 | y_hat_5050 == 0)
t %>% group_by(inclinacion_peligrosa) %>% summarize(count=n())
model <- tree_classifier(t)
test_model_output(model, "simple_tree_123.csv")
```

```{r}

weights <- ifelse(training_data$inclinacion_peligrosa == 1, (1-ratio), ratio)
model <- tree_classifier(training_data, weights)
test_model_output(model, "weighted_tree_123.csv")

```

```{r}
rf_classifier <- function(train_data, weights=NULL){
  train_data$inclinacion_peligrosa <- factor(train_data$inclinacion_peligrosa)
  rf_model<-randomForest(inclinacion_peligrosa~altura+circ_tronco_cm+lat+long+seccion, data=train_data, weights = weights, minsplit=1, minbucket=1, ntree=500)
  print(rf_model)
  summary(rf_model)
  return(function (test_data){
    temp <- data.frame(test_data)
    temp$y_hat_tree <- predict(rf_model, test_data, type="c")
    return(temp)
  })
}

weights <- ifelse(training_data$inclinacion_peligrosa == 1, 1*(1-ratio), ratio)
model <- rf_classifier(training_data, weights)
test_model_output(model, "weighted_rf_123.csv")
```

```{r}
# F Engineered RF
rf_fe_classifier <- function(train_data, weights=NULL){
  train_data <- data.frame(train_data)
  especie_a <- encode_danger(train_data, "especie", "especie_a")
  altura_a <- encode_danger(train_data, "altura", "altura_a")
  diametro_tronco_a <- encode_danger(train_data, "diametro_tronco", "diametro_tronco_a")
  nombre_seccion_a <- encode_danger(train_data, "nombre_seccion", "nombre_seccion_a")
  
  train_data <- train_data %>%
    left_join(especie_a, by = "especie") %>%
    left_join(nombre_seccion_a, by = "nombre_seccion") %>%
    left_join(diametro_tronco_a, by = "diametro_tronco") %>%
    left_join(altura_a, by = "altura")
  print(train_data)
  train_data$inclinacion_peligrosa <- factor(train_data$inclinacion_peligrosa)
  rf_model<-randomForest(inclinacion_peligrosa~altura+circ_tronco_cm+lat+long+seccion+especie_a+nombre_seccion_a+diametro_tronco_a+altura_a, data=train_data, weights = weights, minsplit=1, minbucket=1, ntree=1000)
  print(rf_model)
  summary(rf_model)
  return(function (test_data){
    temp <- data.frame(test_data) %>%
      left_join(especie_a, by = "especie") %>%
      left_join(nombre_seccion_a, by = "nombre_seccion") %>%
      left_join(diametro_tronco_a, by = "diametro_tronco") %>%
      left_join(altura_a, by = "altura")
    temp$y_hat_tree <- predict(rf_model, temp, type="c")
    return(temp)
  })
}

t <- random_classifier(training_data, ratio)(training_data) %>% filter(inclinacion_peligrosa == 1 | y_hat_5050 == 0)
t %>% group_by(inclinacion_peligrosa) %>% summarize(count=n())

weights <- ifelse(training_data$inclinacion_peligrosa == 1, 1*(1-ratio), ratio)
weights_t <- ifelse(t$inclinacion_peligrosa == 1, 1*(1-ratio), ratio)

model <- rf_fe_classifier(training_data, weights)
test_model_output(model, "weighted_rf_fe_123.csv")
model <- rf_fe_classifier(t, weights_t)
test_model_output(model, "weighted_rf_fe_123_2.csv")

```

```{r}
rf_cv_classifier <- function(train_data, weights=NULL, folds=10){
  target_column_index <- which(colnames(train_data) == "inclinacion_peligrosa")

  preProc <- preProcess(train_data[, -target_column_index], method = c("center", "scale"))
  training_data_norm <- predict(preProc, train_data[, -target_column_index])
  training_data_norm <- cbind(training_data_norm, inclinacion_peligrosa=train_data$inclinacion_peligrosa)
  print(training_data_norm)
  
  ctrl <- trainControl(method = "cv", number = folds, verboseIter = TRUE)
  set.seed(123)  # for reproducibility
  model <- train(inclinacion_peligrosa~altura+circ_tronco_cm+lat+long+seccion, data = training_data_norm, method = "rf", ntree=50, trControl = ctrl,weights=weights)

  results <- resamples(list(model,model))
  print(model)
  summary(results)
  


  
  # train_data$inclinacion_peligrosa <- factor(train_data$inclinacion_peligrosa)
  # rf_model<-randomForest(inclinacion_peligrosa~altura+circ_tronco_cm+lat+long+seccion, data=train_data, weights = weights, minsplit=1, minbucket=1, ntree=5)
  
  return(function (test_data){
    temp <- data.frame(test_data)
    new_data_norm <- predict(preProc, test_data)
    print(new_data_norm)
    temp$y_hat_tree <- ifelse(predict(model, new_data_norm) > 0.5, 2, 1)
    return(temp)
  })
}



weights <- ifelse(training_data$inclinacion_peligrosa == 1, 1*(1-ratio), ratio)

model <- rf_cv_classifier(training_data, weights, folds=10)
#model <- rf_cv_classifier(t)
test_model_output(model, "weighted_rf_cv_123.csv")
```


```{r}
ggplot(arbolado.mza.dataset, aes(x = lat, y = long, color = especie)) +
  geom_point(aes(alpha=altura), size = 0.3) +
  labs(
    title = "Scatter Plot of Lat vs. Long",
    x = "Latitude",
    y = "Longitude",
    color = "Inclinacion Peligrosa"
  ) +
  #scale_color_gradient(low = "red", high = "green")
  scale_color_manual(values = c(
  "#FF5733", "#337DFF", "#33FF49", "#A833FF", "#FF9B33", "#FF33B6", "#754A23", "#33F4FF",
  "#E133FF", "#FFFA33", "#000000", "#808080", "#AA0000", "#0000AA", "#00AA00", "#AA00AA",
  "#FF5500", "#FF0088", "#5C3317", "#0088FF", "#FF33A1", "#D9B066", "#3E4E50", "#D35400",
  "#E74C3C", "#3498DB", "#229954", "#7D3C98", "#F4D03F", "#34495E", "#F22613", "#2E86C1"
)
)
```


```{r}
# install.packages("plotly")
library(plotly)

# Assuming df is your dataframe with columns lat, long, altura, and especie
plot_ly(data = arbolado.mza.dataset, x = ~lat, y = ~long, z = ~altura, color = ~especie, colors = "Set1") %>%
  add_markers(marker = list(size = 2)) %>%
  layout(
    scene = list(
      xaxis = list(title = "Latitude"),
      yaxis = list(title = "Longitude"),
      zaxis = list(title = "Altura")
    ),
    legend = list(title = "Especie"),
    title = "3D Scatter Plot"
  )
```
