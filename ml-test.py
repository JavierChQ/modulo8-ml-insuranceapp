
import joblib
import numpy as np
import sklearn

model = joblib.load('./model/insurance.pkl')
sc_x = joblib.load('./model/scaler_x.pkl')
sc_y = joblib.load('./model/scaler_y.pkl')

# print(f'sc_x: {sc_x.mean_}')
# print(f'sc_y: {sc_y.mean_}')

edad = int(input('Ingrese la edad: '))
edad_sc = sc_x.transform(np.array([edad]).reshape(-1,1))
prediction = model.predict(edad_sc)
prediction_sc = sc_y.inverse_transform(prediction)
print(f"LOS GASTOS MEDICOS PARA UNA PERSONA CON {edad} años son {prediction_sc}")
#print(f"LOS GASTOS MEDICOS PARA UNA PERSONA CON {edad} años son {sc_y.inverse_transform(model.predict(edad_sc))}")



# edad_sc = sc_x.transform(np.array([[edad]]))
# #print(f'edad_sc: {edad_sc}')

# prediction = model.predict(edad_sc)

# # #print(f'prediction: {prediction}')

# prediction_sc = sc_y.inverse_transform(prediction) * 1000
# print(f'Los gastos medicos para una persona de {edad} años es: $ {prediction_sc[0][0]:.2f}')
