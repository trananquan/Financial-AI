import streamlit as st
import google.generativeai as genai
import pandas as pd
import matplotlib.pyplot as plt

# Set the Google Generative AI API key (authentication)
API_KEY = "AIzaSyAD5-tRTbhtr17baOAVq307Fguv5oa49hY"

# Authenticate with Google Generative AI
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel('gemini-2.0-flash')

# Add custom CSS for button styling
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: darkblue;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
    }
    div.stButton > button:first-child:hover {
        background-color: blue;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def get_financial_analysis(text_input):
    """Use Google's gemini-2.0-flash model to analyze financial indicators and generate a report."""
    try:
        # Generate financial analysis report using gemini-2.0-flash model
        response = model.generate_content( contents=f"Phân tích các chỉ tiêu tài chính sau và xuất ra báo cáo tài chính: {text_input}",)
       
        return response.text  # Return the generated text
    except Exception as e:
        st.error(f"Lỗi trong quá trình phân tích: {e}")
        return None

def display_financial_report(report):
    """Display the financial report in Streamlit."""
    if not report:
        st.warning("Không có báo cáo hiển thị.")
        return

    st.subheader("Báo cáo phân tích tài chính dựa trên AI")
    st.write(report)

def parse_indicators(input_text):
    """Parse the input text into a dictionary of financial indicators."""
    if not input_text.strip():  # Check if the input is empty or contains only whitespace
        return {}  # Return an empty dictionary without showing warnings
    
    try:
        # Split the input by commas
        pairs = input_text.split(",")
        data = {}
        for pair in pairs:
            if ':' in pair:
                try:
                    key, value = pair.split(':', 1)
                    data[key.strip()] = float(value.strip())
                except ValueError:
                    st.warning(f"Các giá trị không hợp lệ: '{pair}'. Skipping...")
            else:
                st.warning(f"Các giá trị không hợp lệ: '{pair}'. Skipping...")
        return data
    except Exception as e:
        st.error(f"Bị lỗi khi xử lý các chỉ tiêu tài chính: {e}")
        return None

# Display the input data as a table

def generate_bar_chart(data):
    """Generate a bar chart based on the financial indicators."""
    if not data:
        st.warning("Không có dữ liệu để xuất ra biểu đồ.")
        return

    df = pd.DataFrame(list(data.items()), columns=["Indicator", "Value"])

    st.subheader("Biểu đồ cột của các chỉ tiêu tài chính")
    fig, ax = plt.subplots(figsize=(20, 10))
    df.plot(kind="bar", x="Indicator", y="Value", ax=ax, legend=False)
    ax.set_ylabel("Giá trị")
    st.pyplot(fig)

def generate_pie_chart(data):
    """Xuất biểu đồ quạt dựa trên các chỉ tiêu tài chính ."""
    if not data:
        st.warning("Không có dữ liệu cho biểu đồ.")
        return

    df = pd.DataFrame(list(data.items()), columns=["Indicator", "Value"])

    st.subheader("Biểu đồ quạt của các chỉ tiêu tài chính")
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.pie(df["Value"], labels=df["Indicator"], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    st.pyplot(fig)

# Streamlit app UI
st.set_page_config(page_icon="📊",layout="wide")
st.title("📊 Ứng dụng phân tích chỉ tiêu tài chính bằng AI")
st.write("Nhập vào các chỉ tiêu tài chính quan trọng, App sẽ xuất ra báo cáo phân tích sử dụng mô hình AI Gemini. Sau đó, có thể so sánh với các doanh nghiệp khác trong ngành")

# Text area for financial indicators input
user_input = st.text_area(
    "Nhập vào các chỉ tiêu tài chính quan trọng, ngăn cách bởi dấu hai chấm và dấu phẩy (Ví dụ: Doanh thu: 120000, Chi phí: 80000, Lợi nhuận: 40000...). Gõ lệnh gợi ý để so sánh với các doanh nghiệp khác trong ngành",
    placeholder="Gõ các chỉ tiêu tài chính tại đây (Ngăn cách bởi dấu hai chấm và dấu phẩy)...",
    height=200,
)

if st.button("Phân tích & xuất báo cáo"):
    if user_input.strip():
        with st.spinner("Analyzing financial data with Google Generative AI (gemini-2.0-flash)..."):
            report = get_financial_analysis(user_input)
            display_financial_report(report)
    else:
        st.warning("Xin hãy nhập chỉ tiêu tài chính để hệ thống xử lý.")

# Parse indicators and generate charts
indicators = parse_indicators(user_input)

# Create two columns for the buttons
col1, spacer, col2 = st.columns([3, 0.8, 3]) 

with col1:
    if st.button("Xuất ra biểu đồ cột"):
        generate_bar_chart(indicators)
with col2:
    if st.button("Xuất ra biểu đồ quạt"):
        generate_pie_chart(indicators)
