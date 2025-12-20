# TỐI ƯU HÓA CHIẾC LƯỢC TIẾP THỊ THẺ TÍN DỤNG THÔNG QUA PHÂN KHÚC KHÁCH HÀNG DỰA TRÊN HÀNH VI

**Môn:** Khai phá dữ liệu  
**Lớp:** DDU1231

**Giảng viên hướng dẫn:** TS.Đỗ Như Tài

## Phát biểu bài toán

Ngân hàng thu thập rất nhiều dữ liệu giao dịch thẻ tín dụng, nhưng việc phân khúc khách hàng vẫn thường dựa trên kinh nghiệm hoặc một vài tiêu chí đơn giản (tuổi, thu nhập, khu vực…). Cách làm này khó phản ánh đầy đủ hành vi chi tiêu, thói quen thanh toán và mức độ sử dụng hạn mức của từng khách hàng, dẫn đến:

- Chiến dịch marketing dàn trải, hiệu quả thấp  
- Khách hàng giá trị cao chưa được chăm sóc đúng mức  
- Khách hàng tiềm ẩn rủi ro tín dụng không được phát hiện sớm  

Dự án hướng tới xây dựng một cách tiếp cận **dựa trên dữ liệu hành vi**, dùng mô hình phân cụm để tự động nhóm khách hàng thành các phân khúc có ý nghĩa và dễ diễn giải.

---

## Mục tiêu

- Xây dựng pipeline phân khúc khách hàng thẻ tín dụng bằng **thuật toán K-Means** (unsupervised learning).  
- Làm sạch, tiền xử lý và chuẩn hóa bộ dữ liệu **CC_GENERAL.csv**.  
- Lựa chọn số cụm phù hợp và huấn luyện mô hình với **K = 4**.  
- Diễn giải đặc trưng của từng cụm và đề xuất một số gợi ý chiến lược cho Marketing, CSKH và Quản trị rủi ro.  

---

# sơ đồ quy trình của dự án
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/0959c324-3b26-4e6b-952b-67c1cabb0acf" />
