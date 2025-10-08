import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # ******************************* PHẦN SỬA LỖI BẮT ĐẦU *******************************
    # Lỗi xảy ra khi dùng .replace() trên giá trị đơn lẻ (numpy.int64).
    # Sử dụng điều kiện ternary để xử lý giá trị 0 thủ công cho mẫu số.
    
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    # ******************************* PHẦN SỬA LỖI KẾT THÚC *******************************
    
    return df

# --- Hàm gọi API Gemini cho Phân tích tự động (Chức năng 5) ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets trên Streamlit Cloud."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"
        
# ------------------------------------------------------------------
## Khung Chatbot Hỏi Đáp với Gemini (Chức năng mới)
# ------------------------------------------------------------------

def run_chatbot(df_processed):
    """Xây dựng giao diện và logic cho khung chat hỏi đáp."""
    
    st.subheader("Hỏi đáp Chuyên sâu với AI về Dữ liệu đã Tải lên 💬")
    
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets để sử dụng tính năng này.")
        return

    # Khởi tạo client và model
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
    except Exception as e:
        st.error(f"Lỗi khởi tạo Gemini Client: {e}")
        return

    # Thiết lập lịch sử chat trong session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Chuẩn bị context dữ liệu để gửi cùng prompt
    data_context = df_processed.to_markdown(index=False)
    
    # Prompt hệ thống, cung cấp bối cảnh cho Gemini
    system_prompt = f"""
    Bạn là một trợ lý phân tích tài chính chuyên nghiệp, thân thiện. 
    Nhiệm vụ của bạn là trả lời các câu hỏi của người dùng dựa trên dữ liệu Báo cáo Tài chính đã được phân tích sau.
    Hãy sử dụng dữ liệu sau để trả lời (nếu cần):
    
    {data_context}
    
    Hãy giữ câu trả lời ngắn gọn, súc tích và chỉ sử dụng thông tin từ dữ liệu được cung cấp. 
    Nếu câu hỏi không liên quan đến dữ liệu tài chính, hãy lịch sự từ chối trả lời.
    """
    
    # Hiển thị lịch sử chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Xử lý input từ người dùng
    if prompt := st.chat_input("Hỏi Gemini về báo cáo tài chính này..."):
        # Thêm tin nhắn người dùng vào lịch sử
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Tạo nội dung (contents) gửi đến API (bao gồm system prompt và lịch sử chat)
        contents = [
            {"role": "system", "parts": [{"text": system_prompt}]}
        ]
        
        # Thêm các tin nhắn cũ vào contents
        for message in st.session_state.messages:
            contents.append({"role": message["role"], "parts": [{"text": message["content"]}]})
        
        # Gọi Gemini API
        with st.chat_message("assistant"):
            with st.spinner("Gemini đang phân tích..."):
                try:
                    # Gọi API với toàn bộ lịch sử và system prompt
                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents
                    )
                    
                    ai_response = response.text
                    st.markdown(ai_response)
                
                except APIError as e:
                    ai_response = f"Lỗi gọi API: {e}. Vui lòng kiểm tra Khóa API."
                    st.error(ai_response)
                
                except Exception as e:
                    ai_response = f"Lỗi không xác định: {e}"
                    st.error(ai_response)
            
            # Thêm phản hồi của AI vào lịch sử chat
            st.session_state.messages.append({"role": "assistant", "content": ai_response})

# ------------------------------------------------------------------
# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

# Sử dụng st.tabs để tổ chức nội dung
tab_analysis, tab_chat = st.tabs(["Phân Tích & Nhận Xét Tự Động", "Hỏi Đáp Chuyên Sâu với AI"])

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            
            # --- Tab Phân Tích & Nhận Xét Tự Động ---
            with tab_analysis:
                
                # --- Chức năng 2 & 3: Hiển thị Kết quả ---
                st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
                st.dataframe(df_processed.style.format({
                    'Năm trước': '{:,.0f}',
                    'Năm sau': '{:,.0f}',
                    'Tốc độ tăng trưởng (%)': '{:.2f}%',
                    'Tỷ trọng Năm trước (%)': '{:.2f}%',
                    'Tỷ trọng Năm sau (%)': '{:.2f}%'
                }), use_container_width=True)
                
                # --- Chức năng 4: Tính Chỉ số Tài chính ---
                st.subheader("4. Các Chỉ số Tài chính Cơ bản")
                
                # Khởi tạo giá trị mặc định để tránh lỗi trước khi tính
                thanh_toan_hien_hanh_N = "N/A"
                thanh_toan_hien_hanh_N_1 = "N/A"
                
                try:
                    # Lấy Tài sản ngắn hạn
                    tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                    tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                    # Lấy Nợ ngắn hạn
                    # Xử lý ZeroDivisionError bằng cách kiểm tra trước khi chia
                    no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                    no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                    # Tính toán
                    thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N if no_ngan_han_N != 0 else float('inf')
                    thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1 if no_ngan_han_N_1 != 0 else float('inf')
                    
                    # Định dạng lại giá trị cho st.metric
                    val_N = f"{thanh_toan_hien_hanh_N:.2f} lần" if thanh_toan_hien_hanh_N != float('inf') else "∞"
                    val_N_1 = f"{thanh_toan_hien_hanh_N_1:.2f} lần" if thanh_toan_hien_hanh_N_1 != float('inf') else "∞"
                    
                    # Tính delta
                    delta_val = None
                    if thanh_toan_hien_hanh_N != float('inf') and thanh_toan_hien_hanh_N_1 != float('inf'):
                        delta_val = f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}"
                        
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                            value=val_N_1
                        )
                    with col2:
                        st.metric(
                            label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                            value=val_N,
                            delta=delta_val
                        )
                        
                except IndexError:
                    st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
                except Exception: # Bắt lỗi chung cho việc tính toán và gán giá trị N/A
                    thanh_toan_hien_hanh_N = "N/A"
                    thanh_toan_hien_hanh_N_1 = "N/A"
                    
                # --- Chức năng 5: Nhận xét AI ---
                st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
                
                # Chuẩn bị dữ liệu để gửi cho AI
                # Xử lý giá trị infinity cho data_for_ai
                tt_N = f"{thanh_toan_hien_hanh_N:.2f}" if isinstance(thanh_toan_hien_hanh_N, (int, float)) and thanh_toan_hien_hanh_N != float('inf') else str(thanh_toan_hien_hanh_N)
                tt_N_1 = f"{thanh_toan_hien_hanh_N_1:.2f}" if isinstance(thanh_toan_hien_hanh_N_1, (int, float)) and thanh_toan_hien_hanh_N_1 != float('inf') else str(thanh_toan_hien_hanh_N_1)
                
                data_for_ai = pd.DataFrame({
                    'Chỉ tiêu': [
                        'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                        'Tăng trưởng Tài sản ngắn hạn (%)', 
                        'Thanh toán hiện hành (N-1)', 
                        'Thanh toán hiện hành (N)'
                    ],
                    'Giá trị': [
                        df_processed.to_markdown(index=False),
                        f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%" if not df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)].empty else "N/A",
                        tt_N_1,
                        tt_N
                    ]
                }).to_markdown(index=False) 

                if st.button("Yêu cầu AI Phân tích"):
                    api_key = st.secrets.get("GEMINI_API_KEY") 
                    
                    if api_key:
                        with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                            ai_result = get_ai_analysis(data_for_ai, api_key)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                    else:
                        st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

            # --- Tab Hỏi Đáp Chuyên Sâu ---
            with tab_chat:
                run_chatbot(df_processed)

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file và cột.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích và hỏi đáp.")
