# Lab: Reinforcement Learning - Blackjack và FrozenLake

**Họ tên:** Nguyễn Lâm Anh
**MSSV:** SE194272
**Môn học:** REL301m
**Kỳ:** Summer 2026

## Mô tả

Bài lab huấn luyện agent chơi 2 game Blackjack và FrozenLake bằng thuật toán Tabular Q-Learning, sử dụng thư viện Gymnasium. Công thức cập nhật Q-value:

```
Q(s, a) = Q(s, a) + alpha * [R + gamma * max Q(s', a') - Q(s, a)]
```


## Bài 1: Blackjack

### Tham số huấn luyện

| Tham số         | Giá trị                                          |
| --------------- | ------------------------------------------------ |
| Learning rate   | 0.01                                             |
| Discount factor | 0.95                                             |
| Số episodes     | 100,000                                          |
| Epsilon         | 1.0 giảm dần xuống 0.1 trong 50,000 episodes đầu |

### Nhận xét kết quả

Agent đạt **win rate khoảng 43-47%** (không tính hòa) sau 100,000 episodes huấn luyện. Kết quả này hợp lý vì bản chất Blackjack là game bất lợi cho người chơi - nhà cái luôn có lợi thế thống kê (house edge). Win rate tầm 47% đã gần sát với basic strategy tối ưu.

Nhìn vào biểu đồ **Training Progress**, reward trung bình tăng dần từ -0.5 lên khoảng -0.1 trong suốt quá trình huấn luyện, cho thấy agent cải thiện đều đặn. TD Error cũng hội tụ dần về 0, nghĩa là Q-values đã ổn định.

Về **policy đã học được**:

- Trường hợp **không có usable ace**: agent học được Stick khi tổng >= 17-18 (tùy bài dealer), Hit khi tổng thấp. Khi dealer show bài nhỏ (2-6), agent dừng sớm hơn vì dealer có xác suất bust cao. Khi dealer show bài lớn (7-10, A), agent phải hit nhiều hơn.
- Trường hợp **có usable ace**: agent hit mạnh tay hơn (đến khoảng 17-18) vì có ace làm "lưới an toàn" - nếu bust thì ace tự chuyển từ 11 xuống 1.

Cả 2 policy đều khớp với lý thuyết basic strategy trong sách Sutton & Barto, chứng tỏ Q-Learning đã hội tụ đúng.


## Bài 2: FrozenLake

### Tham số huấn luyện

| Tham số         | Giá trị                 |
| --------------- | ----------------------- |
| Learning rate   | 0.8                     |
| Discount factor | 0.95                    |
| Epsilon         | 0.1 (cố định)           |
| Số episodes/map | 2,000                   |
| Số runs/map     | 20 (lấy trung bình)     |
| Map sizes       | 4x4, 7x7, 9x9, 11x11    |
| Slippery        | Tắt (is_slippery=False) |

### Nhận xét kết quả

Em chạy trên 4 kích thước map khác nhau để so sánh khả năng học của agent khi state space tăng dần.

**Map 4x4 (16 states):** Agent hội tụ rất nhanh, chỉ sau vài trăm episodes đã tìm được đường tối ưu đến Goal. Policy heatmap cho thấy các mũi tên hướng rõ ràng về đích, tránh hố. Win rate cao. Biểu đồ cumulated rewards tăng gần như tuyến tính, chứng tỏ agent thắng đều đặn mỗi episode.

**Map 7x7 (49 states):** Vẫn học tốt nhưng chậm hơn 4x4. Nhìn policy heatmap thấy agent biết đi vòng tránh các Hole, chọn đường an toàn dù không phải đường ngắn nhất. Cumulated rewards vẫn tăng đều.

**Map 9x9 (81 states):** Bắt đầu thấy rõ sự khó khăn. Agent cần nhiều episodes hơn để explore hết các state. Cumulated rewards tăng chậm hơn, average steps giảm chậm hơn so với 2 map nhỏ.

**Map 11x11 (121 states):** Win rate = 0%. Nguyên nhân chính là 2,000 episodes không đủ cho state space 121 ô. Với epsilon cố định 0.1, agent không explore đủ các trạng thái xa, dẫn đến Q-values ở nhiều ô vẫn bằng 0. Để cải thiện có thể tăng episodes lên 10,000-20,000 hoặc áp dụng epsilon decay giống bài Blackjack.

Nhìn tổng thể biểu đồ **Training Progress** so sánh 4 map sizes, thấy rõ quy luật: map nhỏ hội tụ nhanh (average steps giảm sớm, cumulated rewards dốc hơn), map lớn cần nhiều thời gian hơn theo cấp số nhân. Điều này phù hợp với lý thuyết - khi state space tăng, Q-Learning cần exponentially nhiều samples hơn để hội tụ.


## So sánh 2 bài

| Điểm so sánh | Blackjack                      | FrozenLake                             |
| ------------ | ------------------------------ | -------------------------------------- |
| Q-table      | Dictionary (vì state là tuple) | Numpy array 2D (vì state là số nguyên) |
| Epsilon      | Decay 1.0 xuống 0.1            | Cố định 0.1                            |
| Số episodes  | 100,000                        | 2,000 x 20 runs                        |
| Đánh giá     | 1,000 episodes greedy          | 100 episodes/map                       |

Em cố ý dùng 2 cách lưu Q-table khác nhau (dict vs array) và 2 chiến lược epsilon (decay vs cố định) để so sánh. Kết luận rút ra là epsilon decay cho kết quả tốt hơn vì agent explore nhiều lúc đầu rồi exploit dần, còn epsilon cố định thì bị giới hạn ở map lớn.
