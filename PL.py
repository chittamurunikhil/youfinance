import streamlit as st
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import openai


def load_and_display_pl(ticker1, ticker2):
    try:
        # Fetch P&L data
        ticker1_data = yf.Ticker(ticker1).financials
        ticker2_data = yf.Ticker(ticker2).financials

        # Transpose for better visualization
        ticker1_df = ticker1_data.transpose()
        ticker2_df = ticker2_data.transpose()

        # Basic data cleaning (adjust as needed)
        ticker1_df = ticker1_df.fillna(0)  # Fill missing values with 0
        ticker2_df = ticker2_df.fillna(0)

        ticker1_df = ticker1_df[['Total Revenue', 'Cost Of Revenue', 'Gross Profit', 'Operating Income', 'Operating Expense', 'EBIT', 'EBITDA', 'Pretax Income','Tax Provision','Net Interest Income','Net Income', 'Basic EPS']]
        ticker2_df = ticker2_df[['Total Revenue', 'Cost Of Revenue', 'Gross Profit', 'Operating Income', 'Operating Expense', 'EBIT', 'EBITDA', 'Pretax Income','Tax Provision','Net Interest Income','Net Income', 'Basic EPS']]

        # Calculate and visualize horizontal analysis
        ticker1_pct_change = ticker1_df.pct_change(axis=1) * 100
        ticker2_pct_change = ticker2_df.pct_change(axis=1) * 100

        ticker1_pct_change_1 = ticker1_pct_change[['Gross Profit',  'Pretax Income', 'Net Income']]
        ticker2_pct_change_2 = ticker2_pct_change[['Gross Profit',  'Pretax Income', 'Net Income']]

        #calculate the vertical analysis 
        ticker_1_vertical = ticker1_df.T.div(ticker1_df['Total Revenue'], axis=1).round(2) * 100
        ticker_2_vertical = ticker2_df.T.div(ticker2_df['Total Revenue'], axis=1).round(2) * 100

        #MARGINS & Ratios
        #ticker one
        ticker1_df['Operating_profit_Margin'] = (ticker1_df['Operating Income'] /ticker1_df['Total Revenue']) *100
        ticker1_df['Operating_Expense_Ratio'] = (ticker1_df['Operating Expense'] /ticker1_df['Total Revenue']) *100
        ticker1_df['Interest_Coverage_Ratio'] = ticker1_df['EBIT'] / abs(ticker1_df['Net Interest Income']) 
        ticker1_df['Effective_Tax_Rate'] = (ticker1_df['Tax Provision'] / ticker1_df['Pretax Income']) * 100
        ticker1_df['Net_profit_Margin'] = (ticker1_df['Net Income'] /ticker1_df['Total Revenue']) *100

        ticker2_df['Operating_profit_Margin'] = (ticker2_df['Operating Income'] /ticker2_df['Total Revenue']) *100
        ticker2_df['Operating_Expense_Ratio'] = (ticker2_df['Operating Expense'] /ticker2_df['Total Revenue']) *100
        ticker2_df['Interest_Coverage_Ratio'] = ticker2_df['EBIT'] / abs(ticker2_df['Net Interest Income']) 
        ticker2_df['Effective_Tax_Rate'] = (ticker2_df['Tax Provision'] / ticker2_df['Pretax Income']) * 100
        ticker2_df['Net_profit_Margin'] = (ticker2_df['Net Income'] /ticker2_df['Total Revenue']) *100


        Profit_Margins_1 = ticker1_df[['Operating_profit_Margin', 'Operating_Expense_Ratio', 'Interest_Coverage_Ratio', 'Effective_Tax_Rate', 'Net_profit_Margin']]
        Profit_Margins_2 = ticker2_df[['Operating_profit_Margin', 'Operating_Expense_Ratio', 'Interest_Coverage_Ratio', 'Effective_Tax_Rate', 'Net_profit_Margin']]



        # Display side-by-side
        
        col1, col2 = st.columns(2)
        with col1:
            st.header(f"Vertical Analysis : {ticker1}")
            st.dataframe(ticker_1_vertical)
            st.header(f"Horizantal Analysis : {ticker1}")
            st.dataframe(ticker1_pct_change)
            st.header(f"Trend Analysis : {ticker1}")
            st.line_chart(ticker1_pct_change_1)
            # st.dataframe(Profit_Margins)
            fig, ax =plt.subplots()
            plt.title(f"Margins & Tax coverage Ratio of : {ticker1}")
            sns.heatmap(Profit_Margins_1, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            st.pyplot(fig)
            



        with col2:
            st.header(f"Vertical Analysis of : {ticker2}")
            st.dataframe(ticker_2_vertical)
            st.header(f"Horizantal Analysis : {ticker2}")
            st.dataframe(ticker2_pct_change)
            st.header(f"Trend Analysis of :{ticker2}")
            st.line_chart(ticker2_pct_change_2)
            fig, ax =plt.subplots()
            plt.title(f"Margins & Tax coverage Ratio of : {ticker2}")
            sns.heatmap(Profit_Margins_2, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            st.pyplot(fig)



        # GPT-3 Integration
        openai.api_key = "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" 

        # Prepare data for GPT-3
        ticker1_summary = f"**{ticker1}**\n\n" \
                         f"**Vertical Analysis:**\n{ticker_1_vertical.to_markdown()}\n\n" \
                         f"**Horizontal Analysis:**\n{ticker1_pct_change.to_markdown()}\n\n" \
                         f"**Key Ratios:**\n{Profit_Margins_1.to_markdown()}\n\n"
        ticker2_summary = f"**{ticker2}**\n\n" \
                         f"**Vertical Analysis:**\n{ticker_2_vertical.to_markdown()}\n\n" \
                         f"**Horizontal Analysis:**\n{ticker2_pct_change.to_markdown()}\n\n" \
                         f"**Key Ratios:**\n{Profit_Margins_2.to_markdown()}\n\n"

        prompt = f"Compare and contrast the financial performance of the two companies based on their P&L data:\n\n" \
                 f"{ticker1_summary}\n\n" \
                 f"{ticker2_summary}\n\n" \
                 f"Provide insights into their profitability, growth trends, and operational efficiency."

        # Generate response from GPT-3
        response = openai.Completion.create(
            engine="text-ada-001",  # Choose an appropriate engine
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.7
        )

        # Display GPT-3's analysis
        st.header("GPT-3 Analysis")
        st.write(response.choices[0].text)

    except Exception as e:
        st.error(f"An error occurred: {e}")
    return load_and_display_pl

    

