import pandas as pd
import pickle


with open("../knowledgeBase/currentOlsSolution.pkl", "rb") as file:
    model_OLS = pickle.load(file)

activation_data = pd.read_csv(f'../activationBase/activation_data.csv')
label = activation_data['Label']

activation_data_input_ols = activation_data[['Input1', 'Input2', 'Input3', 'Input4', 'Input5']]
prediction_OLS = model_OLS.predict(activation_data_input_ols)


print("OLS model prediction:")
print("Input:")
print(activation_data_input_ols)
print("------------------------------------------")
print("Prediction: ", prediction_OLS[0])
print("actual label: ", label[0])


