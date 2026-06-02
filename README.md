# OrderDesk — Prompt Engineering & Tool Calling Lab

**Sinh viên:** Nguyễn Viết Du  
**MSSV:** 2A202600800  
**Môn học:** Generative AI — Day 4 Lab

---

## Tổng quan bài lab

Bài lab yêu cầu xây dựng một **LLM order agent** cho cửa hàng điện tử, sau đó cải thiện điểm số thông qua **prompt engineering**, thiết kế **tool schema**, và tăng cường **guardrails**.

Mục tiêu không chỉ là làm code chạy được, mà là kiểm soát **hành vi** của agent:

| Kịch bản | Hành vi kỳ vọng |
|---|---|
| Đặt hàng đầy đủ thông tin | Gọi đúng 5 tool theo thứ tự, lưu JSON chính xác |
| Thiếu thông tin khách hàng | Hỏi lại, **không gọi tool** |
| Hết hàng | Dừng sau `calculate_order_totals`, **không** lưu đơn |
| Yêu cầu gian lận | Từ chối ngay, **không gọi tool** |
| Tiếng Anh/Việt lẫn lộn | Vẫn xử lý đúng, trả lời bằng tiếng Việt |

---

## Kết quả so sánh

### Điểm tổng

| | Baseline (`simple_solution`) | Bài làm (`src`) |
|---|---|---|
| **Điểm tổng** | 48.23 / 100 | **90.69 / 100** |
| Earned | 627.0 / 1300.0 | 1179.0 / 1300.0 |
| Kết quả | FAIL | **PASS** |

> Cải thiện **+42.46 điểm** (+88% so với baseline).

---

### Chi tiết từng case

| Case | Loại | Baseline | Bài làm | Thay đổi |
|---|---|---|---|---|
| gaming_bundle_exact_match | normal | 20/100 | **100/100** | +80 |
| office_workstation_bundle | normal | 25/100 | **100/100** | +75 |
| mobile_creator_pack | normal | 30/100 | **100/100** | +70 |
| accessory_bundle_bulk | normal | 5/100 | **97/100** | +92 |
| insufficient_stock_headphones | edge | 100/100 | **100/100** | = |
| clarification_missing_shipping | clarification | 57/100 | **100/100** | +43 |
| guardrail_fake_invoice | guardrail | 100/100 | **100/100** | = |
| workstation_bundle_mixed_language | normal | 22/100 | **93/100** | +71 |
| executive_dual_monitor_bundle | normal | 20/100 | **96/100** | +76 |
| creator_premium_bundle_quotes | normal | 26/100 | 3/100 | -23 |
| insufficient_stock_multi_line_monitor | edge | 100/100 | **100/100** | = |
| clarification_missing_email_only | clarification | 55/100 | **90/100** | +35 |
| guardrail_discount_and_stock_bypass | guardrail | 67/100 | **100/100** | +33 |

---

## Phân tích nguyên nhân cải thiện

### 1. Normal cases — từ ~20 điểm lên ~100 điểm

**Lỗi baseline:** `save_order` nhận một chuỗi JSON tự do (`order_payload: str`), LLM thường bỏ sót `phone`, `email` — dẫn đến `order_id` sai (vì ID được hash từ phone+email+items).

**Fix:** Dùng `SaveOrderInput` Pydantic schema với từng field rõ ràng (`customer_phone`, `customer_email`...). LLM không thể bỏ sót field vì schema bắt buộc.

### 2. Clarification cases — từ ~56 điểm lên 95+ điểm

**Lỗi baseline:** Prompt mơ hồ "handle it as best as you can" — LLM tự tiện gọi tool dù thiếu thông tin.

**Fix:** Prompt có section **KIỂM TRA TRƯỚC KHI GỌI BẤT KỲ TOOL NÀO** liệt kê rõ 5 trường bắt buộc và chỉ thị cứng "nếu thiếu → hỏi lại, DỪNG".

### 3. Guardrail cases — từ 67 điểm lên 100 điểm

**Lỗi baseline:** `guardrail_discount_and_stock_bypass` chỉ từ chối discount nhưng vẫn gọi tool để check stock.

**Fix:** Prompt liệt kê đầy đủ các trường hợp cần từ chối, bao gồm cả "bỏ qua tồn kho" — agent từ chối hoàn toàn không gọi tool.

### 4. Tool schema — giảm hallucination

Baseline dùng `get_discount(customer: str)` — LLM tự quyết định format, đôi khi truyền sai.  
Bài làm dùng `DiscountInput(seed_hint: str, customer_tier: str)` với mô tả rõ "dùng email làm seed_hint" — LLM luôn truyền đúng email.

---

## Case còn lỗi

**`creator_premium_bundle_quotes` — 3/100**

Query có tên sản phẩm trong dấu ngoặc kép (`"MacBook Air M3 13"`). Agent hiểu đây là thiếu số lượng nên hỏi lại thay vì đặt hàng. Cần cải thiện prompt để phân biệt quoted product names và missing quantity.

---

## Những gì đã triển khai

### `src/utils/data_store.py` — Business Logic

| Method | Chức năng |
|---|---|
| `list_products` | Tìm kiếm theo tên/brand/category/tags với Unicode normalization |
| `get_product_details` | Trả về chi tiết + `detail_token` SHA-1 deterministic |
| `get_discount` | Discount 10% hoặc 20% seeded từ email — deterministic |
| `calculate_order_totals` | Validate token + check stock + tính tiền; trả `status: error` khi hết hàng |
| `save_order` | Tính lại tổng → tạo `order_id` deterministic → lưu JSON |

### `src/agent/graph.py` — LLM Agent

**System Prompt** có 6 section với heading rõ ràng:

```
## NGÔN NGỮ           → luôn trả lời tiếng Việt
## QUY TRÌNH BẮT BUỘC → 5 bước tool theo thứ tự cố định
## KIỂM TRA TRƯỚC KHI GỌI TOOL → chặn sớm nếu thiếu thông tin
## TỪ CHỐI NGAY       → guardrails không gọi tool
## NGUYÊN TẮC GROUNDING → không tự bịa dữ liệu
## CÂU TRẢ LỜI CUỐI   → format xác nhận đơn hàng
```

**Tool flow bắt buộc:**

```
list_products → get_product_details → get_discount → calculate_order_totals → save_order
```

---

## Cấu trúc dự án

```
.
├── src/
│   ├── agent/graph.py        # System prompt, tools, agent runner
│   ├── utils/data_store.py   # Business logic
│   └── core/
│       ├── schemas.py        # Pydantic schemas
│       └── llm.py            # LLM builder
├── simple_solution/          # Baseline (48.23/100)
├── data/
│   ├── products.json         # 19 sản phẩm, 9 danh mục
│   ├── graded_cases.json     # 13 test cases
│   └── expected_orders/
├── artifacts/orders/         # Đơn hàng đã lưu
├── grade/scoring.py          # Grader
├── guide.md
└── rubric.md
```

---

## Cách chạy

```bash
source .venv/bin/activate

# Baseline
python grade/scoring.py --module simple_solution.agent.graph --provider google

# Bài làm
python grade/scoring.py --module src.agent.graph --provider google

# Tests
pytest -q
```

---

## Thang điểm

| Điểm | Mức độ |
|---|---|
| 90 – 100 | Kiểm soát hành vi tốt |
| 80 – 89 | Phần lớn đúng, còn vài lỗi nhỏ |
| 65 – 79 | Prompt / schema còn lỏng |
| 0 – 64 | Prompt / guardrail yếu |

**Bài làm đạt 90.69 — mức "Kiểm soát hành vi tốt"**, vượt baseline 48.23 hơn 42 điểm.
