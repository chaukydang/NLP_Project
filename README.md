# NLP_Project

![image](https://github.com/chaukydang/NLP_Project/assets/94186949/768df910-aaf8-4fe2-98a7-0500fbb68786)

### Nhận biết cảm xúc dựa vào bộ dữ liệu Emotion Dataset for Emotion Recognition Tasks



Bộ dữ liệu tập hợp các tin nhắn Twitter bằng tiếng Anh với sáu cảm xúc cơ bản: anger, disgust, fear, joy, sadness and surprise. Để biết thêm thông tin chi tiết xin vui lòng tham khảo bài viết dưới đây.

Các tác giả đã xây dựng một bộ hashtag để thu thập một tập dữ liệu riêng gồm các tweet tiếng Anh từ API Twitter thuộc tám cảm xúc cơ bản, bao gồm anger, anticipation, disgust, fear, joy, sadness, surprise and trust. Dữ liệu đã được xử lý trước dựa trên phương pháp được mô tả trong bài báo của họ.

Tập tin "NLP_Cuối_Kì.ipynb" là source code của các mô hình bao gồm những phần như sau:
  - Cài đặt môi trường lập trình
  - Đọc dữ liệu & tiền xử lí dữ liệu
  - Vector hóa dữ liệu
  - Áp dụng dữ liệu vào mô hình
    * Mô hình máy học Decision Tree kết hợp với kĩ thuật Ensemble learning (Adaboost)
    * Mô hình Maximum Entropy
    * Mô hình Deep Learning (LSTM)
    * So sánh ba mô hình và chọn mô hình tốt nhất
    * Ứng dụng mô hình (không giao diện)
  
Tập tin "model.py" là source chứa code của mô hình deep learning (LSTM) có bao gồm giao diện 

Và tập tin "main.py" dùng để chạy mô hình đó

- Giao diện demo:
  ![image](https://github.com/chaukydang/NLP_Project/assets/94186949/6f11f59f-fa7d-4519-a44e-026d57df30c7)

