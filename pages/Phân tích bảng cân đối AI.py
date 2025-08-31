from vnstock import Vnstock
import streamlit as st
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai


# Input for stock symbol
st.markdown(
    "<h1 style='color: darkblue;'>📊 Phân tích bảng cân đối và dòng tiền trên sàn chứng khoán với AI</h1>",
    unsafe_allow_html=True
)

symbol = st.text_input("Nhập mã chứng khoán vào đây (ví dụ, VCB):", value="VCB")

if symbol:  # Ensure symbol is not empty
    stock = Vnstock().stock(symbol=symbol, source='VCI')

    # Exporting results as DataFrames
    balance_sheet_year = stock.finance.balance_sheet(period='year', lang='vi', dropna=True)
    balance_sheet_quarter = stock.finance.balance_sheet(period='quarter', lang='vi', dropna=True)
    income_statement = stock.finance.income_statement(period='year', lang='vi', dropna=True)
    cash_flow = stock.finance.cash_flow(period='year', lang='vi', dropna=True)
    financial_ratios = stock.finance.ratio(period='year', lang='vi', dropna=True)

    # Remove the first level of the multi-index in "Chỉ số tài chính"
    if isinstance(financial_ratios.columns, pd.MultiIndex):
        financial_ratios.columns = financial_ratios.columns.get_level_values(1)

    # Create a dictionary to map indicators to DataFrames
    dataframes = {
        "Bảng cân đối kế toán theo Năm": balance_sheet_year,
        "Bảng cân đối kế toán theo Quý": balance_sheet_quarter,
        "Kết quả hoạt động kinh doanh": income_statement,
        "Lưu chuyển tiền tệ": cash_flow,
        "Chỉ số tài chính": financial_ratios,
    }

        # Selectbox for choosing an indicator
    selected_indicator = st.selectbox(
        "Chọn chỉ số tài chính để hiển thị:",
        list(dataframes.keys())
    )

    # Display the selected DataFrame
    st.write(f"### {selected_indicator}")
    st.write(dataframes[selected_indicator])
    
    # Set the Google Generative AI API key (authentication)
    API_KEY = "AIzaSyDgpMnXlUyC-Ebi7z4xRkdmxBzvfWkcw2Q"

    # Authenticate with Google Generative AI
    genai.configure(api_key=API_KEY)

    # Define the generative model
    model = genai.GenerativeModel('gemini-2.0-flash')

    # Add custom CSS for button styling
    st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: darkblue;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
        cursor: pointer;
    }
    div.stButton > button:hover {
        background-color: #00008b;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

     # Button to fetch results using Gemini AI API
    if st.button("Phân tích dữ liệu với AI"):
        # Convert the selected DataFrame to a string for input to the Gemini AI API
        input_data = dataframes[selected_indicator].to_string()

        # Define the prompt for Gemini AI
        prompt = f"""
        Phân tích dữ liệu tài chính sau đây và cung cấp các nhận xét chi tiết về tình hình tài chính, xu hướng, và các điểm nổi bật. Đưa ra các khuyến nghị nếu có:

        {input_data}

        Hãy tập trung vào các yếu tố quan trọng như:
        1. Tăng trưởng doanh thu hoặc lợi nhuận.
        2. Tình hình tài sản và nợ phải trả.
        3. Dòng tiền hoạt động, đầu tư, và tài chính.
        4. Các chỉ số tài chính quan trọng (nếu có).
        5. Bất kỳ rủi ro hoặc cơ hội nào được thể hiện qua dữ liệu.

        Trả lời bằng tiếng Việt và sử dụng ngôn ngữ dễ hiểu.
        """

        # Use Gemini AI API to analyze the data
        try:
            model_response = model.generate_content(prompt)          
            st.write("### Kết quả phân tích từ Gemini AI:")
            st.write(model_response.text)
        except Exception as e:
             st.error(f"Đã xảy ra lỗi khi gọi API Gemini AI: {e}")

    st.write("")
 # Selectbox for choosing a data field (column) to display in the graph
# Exclude specific fields from the selectbox
    st.markdown(
    "<h3 style='color: darkblue;'> Hiển thị biểu đồ theo chỉ tiêu</h3>",
    unsafe_allow_html=True
)
    excluded_fields = ['CP', 'Năm', 'Kỳ']
    available_fields = [col for col in dataframes[selected_indicator].columns if col not in excluded_fields]

    # Selectbox for choosing a data field (column) to display in the graph
    field_name = st.selectbox(
        "Chọn trường dữ liệu để hiển thị biểu đồ:",
        available_fields
    )

    # Selectbox for choosing the chart style
    chart_style = st.selectbox(
        "Chọn kiểu biểu đồ:",
        ["Biểu đồ cột", "Biểu đồ đường", "Biểu đồ miền"]
    )

    # Button to display the bar chart
    if st.button("Hiển thị biểu đồ"):     
        # Extract the data for the selected field
        chart_data = dataframes[selected_indicator][['year', field_name]].dropna()

        # Plot the bar chart
        fig, ax = plt.subplots()
        if chart_style == "Biểu đồ cột":
           ax.bar(chart_data['year'], chart_data[field_name], color='darkblue')
           ax.set_title(f"Biểu đồ {field_name} theo Năm (Dạng cột)")
        elif chart_style == "Biểu đồ đường":
           ax.plot(chart_data['year'], chart_data[field_name], marker='o', color='darkblue')
           ax.set_title(f"Biểu đồ {field_name} theo Năm (Dạng đường)")
        elif chart_style == "Biểu đồ miền":
           ax.fill_between(chart_data['year'], chart_data[field_name], color='darkblue', alpha=0.5)
           ax.set_title(f"Biểu đồ {field_name} theo Năm (Dạng miền)")

        ax.set_xlabel("Năm")
        ax.set_ylabel(field_name)
        st.pyplot(fig)

else:
    st.write("Vui lòng nhập mã chứng khoán để xem kết quả.")
