print("Grid Search...")
def make_model(lstm_layer_sizes):
    model = Sequential()

    # model.add(LSTM(layers[1], input_shape=(None, layers[0]), return_sequences=True))
    # model.add(Dropout(0.2))

    for layer1_size in lstm_layer_sizes:
        model.add(LSTM(layer1_size, input_shape=(None, X_train.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))
    
    model.add(LSTM(15))
    model.add(Dropout(0.2))

    # for layer2_size in lstm_layer_sizes:
    #     model.add(LSTM(layer2_size))
    #     model.add(Dropout(0.2))

    model.add(Dense(y_train.shape[1]))
    model.add(Activation('linear'))
    
    model.compile(loss="mse", optimizer='rmsprop', metrics=['accuracy'])
    
    # print(model.summary())
          
    return model

lstm_size_candidates = [[50, 45, 40, 35, 30, 25, 20, 15]]
my_regressor = KerasRegressor(make_model, batch_size=1024)
validator = GridSearchCV(my_regressor, 
                    param_grid={'lstm_layer_sizes': lstm_size_candidates,
                                'epochs': [2, 3]},
                    scoring='neg_log_loss',
                    n_jobs=1)

validator.fit(X_train, y_train)
print('the best param are: ')
print(validator.best_params_)

best_model = validator.best_estimator_.model
metric_names = best_model.metric_names
metric_values = best_model.evaluate(X_test, y_test)
for metric, value in zip(metric_names, metric_values):
    print(metric, ': ', value)