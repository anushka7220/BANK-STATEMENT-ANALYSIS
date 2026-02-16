def predict_loan_eligibility(df):
    total_credits = df[df['Amount'] > 0]['Amount'].sum()
    total_debits = abs(df[df['Amount'] < 0]['Amount'].sum())
    num_transactions = len(df)
    avg_transaction_amount = df['Amount'].mean()
    transaction_variability = df['Amount'].std()
    balance_trend = df['Balance'].iloc[-1] - df['Balance'].iloc[0]

    # Prepare feature array
    new_features = np.array([[total_credits, total_debits, num_transactions, avg_transaction_amount, transaction_variability, balance_trend]])
    new_features_scaled = scaler.transform(new_features)

    # Make prediction
    prediction = xgb_model.predict(new_features_scaled)[0]
    return "Eligible for Loan" if prediction == 1 else "Not Eligible for Loan"

# Example usage with extracted transaction data
eligibility_result = predict_loan_eligibility(df)
print(f"Loan Eligibility Result: {eligibility_result}")
