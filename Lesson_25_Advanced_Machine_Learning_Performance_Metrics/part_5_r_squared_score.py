from sklearn.metrics import r2_score

r2 = r2_score(y_actual, y_predicted)
print(f'R-squared Score: {r2:.2f}')
