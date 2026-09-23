"""
1 nmess7o comuln unnamed
2 devision de x (features) et y (target) 
3 n9essmo data l 80% training et 20% testing 
4 n7owelo string l encoding 
5 scaling 3La 7ssab model 

"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
class Processing:
    def __init__(self , data):
        self.data = data.drop(columns=["Unnamed: 0"])
        self.categorical_features = [
            "airline",
            "flight",
            "source_city",
            "departure_time",
            "stops",
            "arrival_time",
            "destination_city",
            "class"
        ]
        self.numerical_features = [
            "duration",
            "days_left"
        ]
        self.encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore"
        )

    def devision(self):
        x = self.data.drop(columns=["price"])
        y = self.data["price"]
        return x , y

    def training_testing(self):
        x , y = self.devision()
        X_train , X_test , y_train , y_test = train_test_split(
            x , y , test_size=0.2 , random_state=42
        )
        X_train_encoded = self.encoder.fit_transform(
            X_train[self.categorical_features]
        )

        X_test_encoded = self.encoder.transform(
            X_test[self.categorical_features]
        )

        return X_train_encoded , X_test_encoded

        





data = pd.read_csv("c:/Users/Youcode/Desktop/model_billets_avion/data/Clean_Dataset.csv")
proce = Processing(data)
print(proce.training_testing())