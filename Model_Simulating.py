from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import random
def class_funct(value):
    if value > 0:
        return 1
    if value < 0:
        return 0
    else:
        return 0
if __name__ == "__main__":


    """Basic data loading and removing unnessicary features for training data"""
    df = pd.read_csv("Model_CSV.csv")
    df = df.iloc[:,1:]
    df.dropna(inplace = True)
    training_df = df.drop(["Date","High","Low","Close","Adj Close", "Open", "Volume"], axis = 1)

    training_df["Class_Target"] = training_df["Target"].apply(class_funct)

    """Setting up training/test split (Test data is last month of data and is untouched during training to
    simulate unknown future market conditions"""
    X_train = training_df.drop(["Target","Class_Target"],axis =1)[0:1405]
    y_train = training_df["Class_Target"][0:1405]
    X_test = training_df.drop(["Target","Class_Target"],axis =1)[1405:]
    y_test = training_df["Class_Target"][1405:]

    """Defining and training Model, hyperparamter tuning was done in Jupyter, random grid search was used 
    valiadation data created from training data"""
    
    model_class = RandomForestClassifier(n_estimators = 500, max_depth = 13, min_samples_split = 8, random_state = 8)
    model_class.fit(X_train,y_train)

    """Test df is created for market simulation, Open and target columns are brought back in"""
    prediction = model_class.predict(X_test)
    test = X_test
    test["Prediction"] = prediction
    test["Target"] = df["Target"][1405:]
    test.reset_index(inplace = True, drop = True)
    """Simulating model deployment, with $30 investments every day based on prediction and gain/loss
    determined by actual Target."""
    total_gain_loss = []
    for index in test.index:
        """Code Block for purchase positions"""
        if test.iloc[index]["Prediction"] == 1:
            """Conditional based off false positives rates from training data"""
            if random.randint(0,(804+38)) < 38:
                value = (-30*test.iloc[index]["Target"])+30
                total_gain_loss.append(value)
            else:
                value = (30*test.iloc[index]["Target"])+30
                total_gain_loss.append(value)
            """Code block for shorting positions""" 
        else:
            if test.iloc[index]["Target"] <= 0:
                value = (-30*test.iloc[index]["Target"])+30
                total_gain_loss.append(value)
            else:
                value = (30*test.iloc[index]["Target"])+30
                total_gain_loss.append(value)
    print(sum(total_gain_loss))

