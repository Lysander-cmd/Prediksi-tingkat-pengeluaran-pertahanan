from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def evaluate_models(y_test, lr_preds, rf_preds):
    print("=== Linear Regression ===")
    print("MSE:", mean_squared_error(y_test, lr_preds))
    print("R² :", r2_score(y_test, lr_preds))

    print("\n=== Random Forest ===")
    print("MSE:", mean_squared_error(y_test, rf_preds))
    print("R² :", r2_score(y_test, rf_preds))

    # Visualisasi
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual', marker='o')
    plt.plot(lr_preds, label='Linear Regression', linestyle='--')
    plt.plot(rf_preds, label='Random Forest', linestyle='-.')
    plt.title('Defense Expenditure Prediction')
    plt.xlabel('Index')
    plt.ylabel('Expenditure')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
