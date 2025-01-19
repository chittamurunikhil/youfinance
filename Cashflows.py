import streamlit as st
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_display_cashflows(ticker1, ticker2):
    try:
        # Fetch P&L data
        ticker1_data = yf.Ticker(ticker1).cashflow
        ticker2_data = yf.Ticker(ticker2).cashflow

        # Transpose for better visualization
        ticker1_df = ticker1_data.transpose()
        ticker2_df = ticker2_data.transpose()

        # Basic data cleaning (adjust as needed)
        ticker1_df = ticker1_df.fillna(0)  # Fill missing values with 0
        ticker2_df = ticker2_df.fillna(0)

        
        #MARGINS & Ratios
        #ticker one
        ticker1_df['CFO_to_Revenue'] = (ticker1_df['Operating Cash Flow'] /ticker1_df['Net Income From Continuing Operations'])
        ticker1_df['FCF_to_Revenue'] = (ticker1_df['Free Cash Flow'] /ticker1_df['Net Income From Continuing Operations'])

        ticker2_df['CFO_to_Revenue'] = (ticker2_df['Operating Cash Flow'] /ticker2_df['Net Income From Continuing Operations'])
        ticker2_df['FCF_to_Revenue'] = (ticker2_df['Free Cash Flow'] /ticker2_df['Net Income From Continuing Operations'])


        cash_flow_ratios_1 = ticker1_df[['CFO_to_Revenue', 'FCF_to_Revenue']]
        cash_flow_ratios_2 = ticker2_df[['CFO_to_Revenue', 'FCF_to_Revenue']]





        # Display side-by-side
        
        col1, col2 = st.columns(2)
        with col1:

            
            st.title(f"{ticker1}:Ratios")
            st.line_chart(cash_flow_ratios_1)
            

            plt.title(f"{ticker1}:Ratios")
            st.dataframe(cash_flow_ratios_1)
            # sns.heatmap(cash_and_working_captial_1, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            
            



        with col2:
            
            st.title(f"{ticker2}:Ratios")
            st.line_chart(cash_flow_ratios_2)
            
            plt.title(f"Cash and Equaivalents & working captial Ratios : {ticker2}")
            st.dataframe(cash_flow_ratios_2)
            # sns.heatmap(cash_and_working_captial_2, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            

    except Exception as e:
        st.error(f"An error occurred: {e}")
    return load_and_display_cashflows

