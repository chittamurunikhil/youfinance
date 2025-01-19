import streamlit as st
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_display_BS(ticker1, ticker2):
    try:
        # Fetch P&L data
        ticker1_data = yf.Ticker(ticker1).balance_sheet
        ticker2_data = yf.Ticker(ticker2).balance_sheet

        # Transpose for better visualization
        ticker1_df = ticker1_data.transpose()
        ticker2_df = ticker2_data.transpose()

        # Basic data cleaning (adjust as needed)
        ticker1_df = ticker1_df.fillna(0)  # Fill missing values with 0
        ticker2_df = ticker2_df.fillna(0)

        
        #MARGINS & Ratios
        #ticker one
        ticker1_df['Debt-to-Equity Ratio'] = (ticker1_df['Total Debt'] /ticker1_df['Common Stock Equity'])
        ticker1_df['Debt-to-Assets Ratio'] = (ticker1_df['Total Debt'] /ticker1_df['Total Assets'])
        ticker1_df['Current Ratio'] = (ticker1_df['Current Assets'] / ticker1_df['Current Liabilities'])
        #ticker1_df['Quick Ratio'] = (ticker1_df['Current Assets'] - ticker1_df['Inventory'] /ticker1_df['Current Liabilities'])
        ticker1_df['Working Capital Ratio'] = ticker1_df['Working Capital'] / ticker1_df['Total Assets']
        ticker1_df['Cash Ratio'] = ticker1_df['Cash Cash Equivalents And Short Term Investments'] / ticker1_df['Current Liabilities']

        ticker2_df['Debt-to-Equity Ratio'] = (ticker2_df['Total Debt'] /ticker2_df['Common Stock Equity'])
        ticker2_df['Debt-to-Assets Ratio'] = (ticker2_df['Total Debt'] /ticker2_df['Total Assets'])
        ticker2_df['Current Ratio'] = (ticker2_df['Current Assets'] / ticker2_df['Current Liabilities'])
        #ticker2_df['Quick Ratio'] = (ticker2_df['Current Assets'] -ticker2_df['Inventory'] /ticker2_df['Current Liabilities'])
        ticker2_df['Working Capital Ratio'] = ticker2_df['Working Capital'] / ticker2_df['Total Assets']
        ticker2_df['Cash Ratio'] = ticker2_df['Cash Cash Equivalents And Short Term Investments'] / ticker2_df['Current Liabilities']


        Solvency_Ratios_1 = ticker1_df[['Debt-to-Equity Ratio', 'Debt-to-Assets Ratio', 'Current Ratio' ]]
        Solvency_Ratios_2 = ticker2_df[['Debt-to-Equity Ratio', 'Debt-to-Assets Ratio', 'Current Ratio' ]]

        cash_and_working_captial_1 = ticker1_df[['Working Capital Ratio', 'Cash Ratio']]
        cash_and_working_captial_2 = ticker2_df[['Working Capital Ratio', 'Cash Ratio']]


        assets_1 = ticker1_df[['Total Assets', 'Current Assets', 'Cash Cash Equivalents And Short Term Investments',  'Inventory','Gross PPE', 'Properties', 'Land And Improvements', 'Accumulated Depreciation', 'Goodwill And Other Intangible Assets',  'Invested Capital', 'Net Tangible Assets']]
        Debt_1 = ticker1_df[['Total Liabilities Net Minority Interest', 'Total Debt', 'Current Liabilities', 'Current Provisions', 'Other Current Liabilities', 'Total Non Current Liabilities Net Minority Interest', 'Long Term Debt', 'Long Term Debt And Capital Lease Obligation',  'Long Term Provisions', 'Stockholders Equity']]


        assets_2 = ticker2_df[['Total Assets', 'Current Assets', 'Cash Cash Equivalents And Short Term Investments', 'Inventory', 'Gross PPE','Properties', 'Land And Improvements', 'Accumulated Depreciation', 'Goodwill And Other Intangible Assets', 'Invested Capital', 'Net Tangible Assets']]
        Debt_2 = ticker2_df[[ 'Total Liabilities Net Minority Interest', 'Total Debt', 'Current Liabilities', 'Current Provisions', 'Other Current Liabilities', 'Total Non Current Liabilities Net Minority Interest', 'Long Term Debt', 'Long Term Debt And Capital Lease Obligation', 'Long Term Provisions' ,'Stockholders Equity']]

        #calculate the vertical analysis 
        assets_1_vertical = assets_1.T.div(assets_1['Total Assets'], axis=1).round(2) * 100
        Debt_1_vertical = Debt_1.T.div(Debt_1['Total Liabilities Net Minority Interest'], axis=1).round(2) * 100

        #calculate the vertical analysis 
        assets_2_vertical = assets_2.T.div(assets_2['Total Assets'], axis=1).round(2) * 100
        Debt_2_vertical = Debt_2.T.div(Debt_2['Total Liabilities Net Minority Interest'], axis=1).round(2) * 100

        # Calculate and visualize horizontal analysis
        ticker1_asset_1_pct_change = assets_1.pct_change(axis=1) * 100
        ticker2_asset_2_pct_change = assets_2.pct_change(axis=1) * 100





        # Display side-by-side
        
        col1, col2 = st.columns(2)
        with col1:

            
            st.title(f"{ticker1}:Ratios")
            st.line_chart(Solvency_Ratios_1)
            

            plt.title(f"{ticker1}:Ratios")
            st.dataframe(cash_and_working_captial_1)
            # sns.heatmap(cash_and_working_captial_1, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            st.subheader(f"Vertical Analysis of Assets in % : {ticker1}")
            st.dataframe(assets_1_vertical)
            st.subheader(f"Vertical Analysis of Debts in % : {ticker1}")
            
            fig, ax =plt.subplots()
            plt.title(f"Vertical Analysis of Debts in %: {ticker1}")
            sns.heatmap(Debt_1_vertical, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            st.pyplot(fig)
            st.dataframe(ticker1_asset_1_pct_change)

            
            



        with col2:
            
            st.title(f"{ticker2}:Ratios")
            st.line_chart(Solvency_Ratios_2)
            

            plt.title(f"Cash and Equaivalents & working captial Ratios : {ticker2}")
            st.dataframe(cash_and_working_captial_2)
            # sns.heatmap(cash_and_working_captial_2, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            st.subheader(f"Vertical Analysis of Assets in % : {ticker2}")
            st.dataframe(assets_2_vertical)
            st.subheader(f"Vertical Analysis of Debts in % : {ticker2}")
            
            fig, ax =plt.subplots()
            plt.title(f"Vertical Analysis of Debts in % : {ticker2}")
            sns.heatmap(Debt_2_vertical, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            st.pyplot(fig)
            st.dataframe(ticker2_asset_2_pct_change)

            

    except Exception as e:
        st.error(f"An error occurred: {e}")
    return load_and_display_BS

