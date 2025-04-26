def preprocessing(data):
 y = df.Potability
 xtrain,xtest,y_train,y_test = train_test_split(data.drop("Potability",axis = 1), y ,stratify=y, test_size = 0.2, random_state = 42)
 num_features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
 num_transformer = Pipeline([
     ("imputer", SimpleImputer(strategy = 'mean')),
     ("scaler", MinMaxScaler())
])

 preprocessor = ColumnTransformer([
     ("num", num_transformer,num_features)
])


       # Encode categorical variables and split features/target as needed
 X_train = num_transformer.fit_transform(xtrain)
 X_test = num_transformer.transform(xtest)
 new_columns = num_features

       # Convert in Pandas DataFrame
 X_train = pd.DataFrame(X_train, columns=new_columns)
 X_test = pd.DataFrame(X_test, columns=new_columns)

return X_train,X_test,y_train,y_test

X_train,X_test,y_train,y_test = preprocessing(df)
