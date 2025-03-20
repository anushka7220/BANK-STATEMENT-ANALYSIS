import streamlit as st
import pdfplumber
import re
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title='Bank Statement Analysis', page_icon=':moneybag:')

st.markdown("""
    <h1 style="text-align: center;">Upload Your Bank Statement</h1>
    <h3 style='text-align: center; color: green;'>Congratulations! You've successfully passed the initial eligibility check. Please submit your bank statement so we can complete the final eligibility review.</h3>
    """, unsafe_allow_html=True)

st.write('<p style="text-align: center;">Please upload your bank statement</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drag and drop file here", type="pdf", label_visibility="collapsed")

def parse_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ''
        for page_num, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
            else:
                st.write(f"Warning: No text extracted from page {page_num + 1}")
    return text

def categorize_expense(description):
    if 'salary' in description.lower():
        return 'Income'
    elif 'grocery' in description.lower():
        return 'Groceries'
    elif 'restaurant' in description.lower() or 'dining' in description.lower():
        return 'Dining'
    elif 'utility' in description.lower() or 'electricity' in description.lower() or 'water' in description.lower():
        return 'Utilities'
    elif 'rent' in description.lower() or 'mortgage' in description.lower():
        return 'Housing'
    elif 'transfer' in description.lower() or 'atm' in description.lower() or 'withdrawal' in description.lower():
        return 'Transfers'
    elif 'investment' in description.lower() or 'mutual fund' in description.lower():
        return 'Investments'
    elif 'credit' in description.lower() or 'refund' in description.lower():
        return 'Credits'
    else:
        return 'Other'

def process_text_to_df(text):
    transactions = []
    transaction_pattern = re.compile(r'(\d{2}-\d{2}-\d{4})\s+(.+?)\s+([+-]?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s+([+-]?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)')
    
    lines = text.split('\n')
    combined_lines = []
    current_line = ""
    
    for line in lines:
        if re.match(r'\d{2}-\d{2}-\d{4}', line):
            if current_line:
                combined_lines.append(current_line)
            current_line = line
        else:
            current_line += " " + line.strip()
    if current_line:
        combined_lines.append(current_line)
    
    for line in combined_lines:
        match = transaction_pattern.search(line)
        if match:
            date_str, description, amount_str, balance_str = match.groups()
            amount = float(amount_str.replace(',', '').replace(' ', ''))
            balance = float(balance_str.replace(',', '').replace(' ', ''))
            transactions.append([date_str, description.strip(), amount, balance])
    
    df = pd.DataFrame(transactions, columns=['Date', 'Description', 'Amount', 'Balance'])
    return df

def prepare_features(df):
    total_credits = df[df['Amount'] > 0]['Amount'].sum()
    total_debits = df[df['Amount'] < 0]['Amount'].sum()
    num_transactions = len(df)
    return total_credits, total_debits, num_transactions

def build_random_forest_model():
    absa_csv_path = '/Users/anushkasharma/Downloads/BANK-STATEMENT-ANALYSIS/ABSA Bank/absa.csv'
    df = pd.read_csv(absa_csv_path)
    X = df.drop('Eligibility (y)', axis=1)
    y = df['Eligibility (y)']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model

def compute_metrics(df):
    avg_daily_expense = df[df['Amount'] < 0]['Amount'].mean()
    total_expense = df[df['Amount'] < 0]['Amount'].sum()
    max_expense = df[df['Amount'] < 0]['Amount'].min()
    min_expense = df[df['Amount'] < 0]['Amount'].max()
    num_transactions = len(df)
    return avg_daily_expense, total_expense, max_expense, min_expense, num_transactions

def plot_expense_distribution(df):
    expense_distribution = df[df['Amount'] < 0].groupby('Category')['Amount'].sum().reset_index()
    if not expense_distribution.empty:
        fig_pie_category = px.pie(expense_distribution, values='Amount', names='Category', title='Expense Distribution by Category')
        st.plotly_chart(fig_pie_category)
    else:
        st.write("No expense data available for plotting.")

def analyze_spending(df):
    # Calculate spending by category
    spending_by_category = df[df['Amount'] < 0].groupby('Category')['Amount'].sum().reset_index()
    spending_by_category['Percentage'] = (spending_by_category['Amount'] / spending_by_category['Amount'].sum()) * 100
    return spending_by_category

def categorize_essential_spending(category):
    essential_categories = ['Groceries', 'Utilities', 'Housing', 'Transport']
    return 'Essential' if category in essential_categories else 'Non-Essential'

def provide_financial_advice(df, financial_goal, target_amount=0, timeframe=0, debt_amount=0):
    spending_by_category = analyze_spending(df)
    total_expenses = df[df['Amount'] < 0]['Amount'].sum()
    total_income = df[df['Amount'] > 0]['Amount'].sum()
    
    # Calculate essential vs. non-essential spending
    df['Spending_Type'] = df['Category'].apply(categorize_essential_spending)
    essential_spending = df[df['Spending_Type'] == 'Essential']['Amount'].sum()
    non_essential_spending = df[df['Spending_Type'] == 'Non-Essential']['Amount'].sum()
    
    advice = []
    
    # General Spending Analysis
    advice.append("### Spending Analysis")
    advice.append(f"Total Income: R{total_income:.2f}")
    advice.append(f"Total Expenses: R{total_expenses:.2f}")
    advice.append(f"Essential Spending: R{essential_spending:.2f} ({abs(essential_spending / total_expenses * 100):.2f}%)")
    advice.append(f"Non-Essential Spending: R{non_essential_spending:.2f} ({abs(non_essential_spending / total_expenses * 100):.2f}%)")
    
    # Identify Overspending Categories
    for _, row in spending_by_category.iterrows():
        if row['Percentage'] > 20:  # Example threshold for high spending
            advice.append(f"⚠️ You're spending a significant portion ({row['Percentage']:.2f}%) on {row['Category']}. Consider reducing this expense.")
    
    # Goal-Based Recommendations
    if financial_goal == 'Save for a house':
        monthly_savings_required = (target_amount - total_income) / timeframe
        advice.append(f"### Goal: Save for a House")
        advice.append(f"To save R{target_amount:.2f} in {timeframe} months, you need to save R{monthly_savings_required:.2f} per month.")
        if monthly_savings_required > (total_income + total_expenses):
            advice.append("⚠️ This goal may not be achievable with your current income and expenses. Consider increasing your income or extending the timeframe.")
        else:
            advice.append("💡 Automate savings of 10% of your income to reach your goal faster.")
    
    elif financial_goal == 'Reduce debt':
        advice.append(f"### Goal: Reduce Debt")
        advice.append(f"Total Debt: R{debt_amount:.2f}")
        
        if debt_amount > 0 and timeframe > 0:
            monthly_installment = debt_amount / timeframe
            advice.append(f"To pay off R{debt_amount:.2f} in {timeframe} months, you need to pay R{monthly_installment:.2f} per month.")
            if monthly_installment > (total_income + total_expenses):
                advice.append("⚠️ This goal may not be achievable with your current income and expenses. Consider increasing your income or extending the timeframe.")
            else:
                advice.append("💡 Prioritize paying off high-interest debt first. Consider consolidating your debt for lower interest rates.")
    
    elif financial_goal == 'Build an emergency fund':
        emergency_fund_target = total_income * 6  # Recommended: 6 months of income
        advice.append(f"### Goal: Build an Emergency Fund")
        advice.append(f"Target Emergency Fund: R{emergency_fund_target:.2f} (6 months of income)")
        advice.append("💡 Automate savings of 5% of your income to build your emergency fund.")
    
    elif financial_goal == 'Invest more':
        investment_opportunities = df[df['Category'] == 'Investments']['Amount'].sum()
        advice.append(f"### Goal: Invest More")
        advice.append(f"Current Investments: R{investment_opportunities:.2f}")
        advice.append("💡 Consider diversifying your investments into mutual funds, stocks, or real estate.")
    
    # Budgeting Tips
    advice.append("### Budgeting Tips")
    advice.append(f"Your current savings rate: {((total_income + total_expenses) / total_income * 100):.2f}%")
    advice.append("💡 Aim to save at least 20% of your income. Here's a suggested budget:")
    advice.append("- 50% on Essentials (Groceries, Utilities, Housing)")
    advice.append("- 30% on Non-Essentials (Dining, Entertainment)")
    advice.append("- 20% on Savings and Investments")
    
    return advice

if uploaded_file is not None:
    try:
        text = parse_pdf(uploaded_file)
        df = process_text_to_df(text)
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

        if not df.empty:
            df['Category'] = df['Description'].apply(categorize_expense)
            total_credits, total_debits, num_transactions = prepare_features(df)
            model = build_random_forest_model()

            features = pd.DataFrame({'total_credits': [total_credits],
                                     'total_debits': [total_debits],
                                     'num_transactions': [num_transactions],
                                     'avg_transaction_amount': [df['Amount'].mean()],
                                     'transaction_variability': [df['Amount'].std()],
                                     'balance_trend': [df['Balance'].iloc[-1] - df['Balance'].iloc[0]]})
            
            rf_prediction = model.predict(features)[0]
            eligible = total_credits > abs(total_debits) and total_credits > 1.25 * abs(total_debits) and rf_prediction == 1

            if eligible:
                result = 'Eligible for Loan'
                color = 'green'
            else:
                result = 'Not Eligible for Loan'
                color = 'red'

            st.markdown(f'<p style="color:{color};font-size:24px;">{result}</p>', unsafe_allow_html=True)

            st.subheader('Extracted Transactions')
            st.dataframe(df, use_container_width=True)

            avg_daily_expense, total_expense, max_expense, min_expense, num_transactions = compute_metrics(df)

            st.subheader('Key Metrics')
            st.write(f'Average Daily Expense: R{avg_daily_expense:.2f}')
            st.write(f'Total Expense: R{total_expense:.2f}')
            st.write(f'Maximum Expense: R{max_expense:.2f}')
            st.write(f'Minimum Expense: R{min_expense:.2f}')
            st.write(f'Number of Transactions: {num_transactions}')

            st.subheader('Expense Overview')
            fig_bar = px.bar(df, x='Date', y='Amount', color='Category', title='Total Expenses per Date')
            st.plotly_chart(fig_bar)

            st.subheader('Expense Distribution by Category')
            plot_expense_distribution(df)

            fig_pie_description = px.pie(df, values='Amount', names='Description', title='Expense Distribution by Description')
            st.plotly_chart(fig_pie_description)

            fig_line = px.line(df, x='Date', y='Amount', title='Daily Expense Trend')
            st.plotly_chart(fig_line)

            # Streamlit UI for financial goals
            st.subheader("Personalized Financial Advice")
            financial_goal = st.selectbox("What is your financial goal?", ["Save for a house", "Reduce debt", "Build an emergency fund", "Invest more"])
            
            if financial_goal == 'Save for a house':
                target_amount = st.number_input("Enter your target amount for the house (R):", min_value=0.0, value=500000.0)
                timeframe = st.number_input("Enter your target timeframe (in months):", min_value=1, value=60)
                debt_amount = 0
            elif financial_goal == 'Reduce debt':
                debt_amount = st.number_input("Enter the amount of debt you need to pay off (R):", min_value=0.0, value=100000.0)
                timeframe = st.number_input("Enter your target timeframe to pay off debt (in months):", min_value=1, value=24)
                target_amount = 0
            else:
                target_amount = 0
                timeframe = 0
                debt_amount = 0
            
            if st.button("Get Financial Advice"):
                advice = provide_financial_advice(df, financial_goal, target_amount, timeframe, debt_amount)
                for tip in advice:
                    if tip.startswith("###"):
                        st.markdown(tip)
                    elif tip.startswith("💡"):
                        st.success(tip)
                    elif tip.startswith("⚠️"):
                        st.warning(tip)
                    else:
                        st.write(tip)
        else:
            st.write("No transactions found in the uploaded statement.")
    except Exception as e:
        st.error(f"Error: {e}")

hide_streamlit_style = """
    <style>
    # MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    footer:after {
    content:'Made with ❤️ by Akshat'; 
    visibility: visible;
    display: block;
    position: relative;
    padding: 15px;
    top: 2px;
    }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
