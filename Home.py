import streamlit as st

st.set_page_config(page_title="Multi-App",page_icon="🏠", layout="centered")
st.sidebar.title("Thanh công cụ")
st.sidebar.info("Sử dụng thanh công cụ để chuyển giữa các App.")

st.markdown("<h1 style='color: darkblue;'>🏠 Dự báo và phân tích tài chính AI</h1>", unsafe_allow_html=True)
st.subheader("Chào mừng đến với Trang chủ Dự báo và phân tích tài chính với AI!")
st.image("images/app1.jpg", use_container_width=True, caption="AI-generated Mindmap")

