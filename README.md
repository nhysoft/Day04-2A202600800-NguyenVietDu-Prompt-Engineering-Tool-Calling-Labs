# OrderDesk — Prompt Engineering & Tool Calling Lab

> **Day 4 Lab** — Build a production-grade LLM order agent for an electronics retailer, then beat the baseline score through prompt engineering.

---

## Mục tiêu

Xây dựng một **LLM order agent** có khả năng:

| Hành vi | Mô tả |
|---|---|
| Đặt hàng hợp lệ | Gọi đúng 5 tool theo thứ tự, lưu JSON chính xác |
| Hỏi làm rõ | Phát hiện thiếu thông tin trước khi gọi bất kỳ tool nào |
| Từ chối | Chặn yêu cầu gian lận/vi phạm policy ngay lập tức |
| Xử lý hết hàng | Dừng pipeline khi stock không đủ, không lưu đơn |
| Đa ngôn ngữ | Hiểu tiếng Việt, tiếng Anh và cả mixed-language |

> Mục tiêu **không chỉ** là làm code chạy được — mà là kiểm soát hành vi agent thông qua prompt engineering, tool schema, và guardrails.

---

## Cấu trúc dự án

```
.
├── src/                        # <-- Bạn làm việc ở đây
│   ├── agent/graph.py          #     System prompt + tools + agent runner
│   ├── utils/data_store.py     #     Business logic: catalog, pricing, save
│   └── core/
│       ├── schemas.py          #     Pydantic schemas (đừng sửa)
│       └── llm.py              #     LLM builder (đừng sửa)
│
├── simple_solution/            # Baseline yếu — điểm thấp, để so sánh
│
├── data/
│   ├── products.json           # 19 sản phẩm (9 danh mục)
│   ├── graded_cases.json       # 13 test cases
│   └── expected_orders/        # JSON kết quả mong đợi
│
├── grade/scoring.py            # Grader tự động
├── artifacts/orders/           # Đơn hàng được lưu tại đây
├── guide.md                    # Hướng dẫn từng bước
└── rubric.md                   # Thang điểm chi tiết
```

---

## Setup

### 1. Cài dependencies

```bash
# Dùng uv (khuyến nghị)
uv sync

# Hoặc dùng pip với .venv
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Tạo file `.env`

```bash
GOOGLE_API_KEY=your_key_here
LLM_MODEL=gemini-2.5-flash
```

Tùy chọn — chạy local với Ollama:

```bash
OLLAMA_MODEL=qwen3.5:3b
OLLAMA_BASE_URL=http://localhost:11434
```

---

## Chạy

```bash
# Activate .venv trước (nếu dùng)
source .venv/bin/activate

# Chạy baseline để lấy điểm tham chiếu
python grade/scoring.py --module simple_solution.agent.graph --provider google

# Chạy bài làm của bạn
python grade/scoring.py --module src.agent.graph --provider google

# Chạy tests
pytest -q
```

---

## Quy trình tool bắt buộc

Với mọi đơn hàng hợp lệ, agent **phải** gọi đúng thứ tự:

```
list_products
      ↓
get_product_details   ← trả về detail_token
      ↓
get_discount          ← dùng email làm seed_hint
      ↓
calculate_order_totals
      ↓
save_order            ← chỉ gọi khi bước trên status: ok
```

---

## Thang điểm

| Điểm | Mức độ |
|---|---|
| 90 – 100 | Kiểm soát hành vi tốt |
| 80 – 89 | Phần lớn đúng, còn vài lỗi nhỏ |
| 65 – 79 | Prompt / schema còn lỏng |
| 0 – 64 | Prompt / guardrail yếu |

**Phân bổ điểm theo loại case:**

| Loại | json_output | tools | llm_judge |
|---|---|---|---|
| Normal (đặt hàng) | 70% | 20% | 10% |
| Edge / Clarification / Guardrail | 55% | 25% | 20% |

---

## Tiêu chí submission tốt

- Hỏi lại khi thiếu thông tin trước khi gọi tool
- Từ chối yêu cầu không hợp lệ mà không gọi tool nào
- Gọi đúng thứ tự tool trên đơn hàng hợp lệ
- Lưu JSON khớp với `data/expected_orders/`
- Trả lời cuối bằng tiếng Việt, có order ID, tổng tiền, mã giảm giá

---

## Tác giả

**Nguyễn Việt Dũng** — 2A202600800
