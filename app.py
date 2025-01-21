import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from PL import load_and_display_pl
from Balance_sheet import load_and_display_BS
from Cashflows import load_and_display_cashflows
# from news import classify_sentiment
from daily_prices import load_and_display_prices
from macd_prices import load_and_display_macd


def main():
    # st.title("Compare Companies fundamental Statements")

    # Using Markdown to make the title bold
    st.markdown("# **ANALYSE YOUR BETS**")

    # User input for ticker symbols
    ticker1 = st.text_input("Enter Ticker Symbol 1")
    ticker2 = st.text_input("Enter Ticker Symbol 2")

    # Tabbed interface
    tab1, tab2, tab3, tab5, tab6  = st.tabs([  "P/L Statement","Balance Sheet", "Cashflow", 'Compare Prices', "MACD Indicatords"])

    with tab1:
    # Button to trigger data loading and display
      if st.button("Load and Compare Profit and loss statement"):
        load_and_display_pl(ticker1, ticker2)

    with tab2:
    # Button to trigger data loading and display the Balance sheets Ratios
      if st.button("Load and Compare Balance statement"):
        load_and_display_BS(ticker1, ticker2)

    with tab3:
    # Button to trigger data loading and display the cashflow statement Ratios
      if st.button("Load and Compare cashflow statement"):
        load_and_display_cashflows(ticker1, ticker2)

    # with tab4:
    # # Button to trigger data loading and display the news updates
    #   if st.button("Load and Compare News updates"):
            
        

    with tab5:
      
      if st.button("Load and Compare Prices"): 
        load_and_display_prices(ticker1, ticker2)
    
    with tab6:
      # User input for MACD parameters
       price_column = st.selectbox("Price Column", ["High", "Low", "Close", "Open"], index=2)  # Default to "Close"
       ema_fast = st.number_input("Fast EMA Span", min_value=1, step=1, value=12)
       ema_slow = st.number_input("Slow EMA Span", min_value=1, step=1, value=26)
       signal_span = st.number_input("Signal Span", min_value=1, step=1, value=9)
       user_inputs = {
      "price_column": price_column,
      "ema_fast": ema_fast,
      "ema_slow": ema_slow,
      "signal_span": signal_span
          }
      
       if st.button("Load and Compare MACD Indicators"): 
        load_and_display_macd(ticker1, ticker2, user_inputs)

            

if __name__ == "__main__":
    main()
