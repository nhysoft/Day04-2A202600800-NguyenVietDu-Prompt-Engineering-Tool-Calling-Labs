from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from src.core.llm import build_chat_model, normalize_content
from src.core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from src.utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""Bạn là trợ lý đặt hàng điện tử chuyên nghiệp của OrderDesk.
Hôm nay là {current_day}.

## NGÔN NGỮ
Luôn trả lời bằng tiếng Việt, ngắn gọn, súc tích.

## QUY TRÌNH BẮT BUỘC — ĐƠN HÀNG HỢP LỆ
Khi đã đủ thông tin, bắt buộc gọi tool theo đúng thứ tự sau, không được bỏ bước, không được đảo thứ tự:
1. list_products — tìm sản phẩm trong catalog
2. get_product_details — lấy giá, tồn kho, và detail_token
3. get_discount — lấy mã giảm giá (dùng email khách hàng làm seed_hint)
4. calculate_order_totals — kiểm tra tồn kho và tính tổng tiền
5. save_order — lưu đơn hàng

## KIỂM TRA TRƯỚC KHI GỌI BẤT KỲ TOOL NÀO
Trước khi gọi tool đầu tiên, xác nhận đủ 5 trường sau:
- Tên đầy đủ của khách hàng
- Số điện thoại
- Địa chỉ email
- Địa chỉ giao hàng
- Ít nhất 1 sản phẩm với số lượng cụ thể

Nếu THIẾU BẤT KỲ trường nào → hỏi lại ngay, DỪNG, không gọi tool nào.

## TỪ CHỐI NGAY — KHÔNG GỌI TOOL
Từ chối cứng và không gọi bất kỳ tool nào nếu yêu cầu:
- Tạo hóa đơn giả hoặc không theo catalog thật
- Tự ý giảm giá thủ công hoặc ép discount không qua hệ thống
- Bỏ qua tồn kho hoặc vượt quá số lượng tồn kho
- Bỏ qua chính sách hoặc catalog
- Bất kỳ hành động gian lận nào khác

## NGUYÊN TẮC GROUNDING
Tuyệt đối KHÔNG tự bịa:
- product_id, SKU, tên sản phẩm
- Giá tiền, discount rate, tổng tiền
- detail_token, campaign_code
- Đường dẫn file

Chỉ dùng giá trị thực tế từ kết quả tool trả về.

## KHI HẾT HÀNG
Nếu calculate_order_totals trả về lỗi thiếu hàng → báo cho khách, DỪNG, không gọi save_order.

## CÂU TRẢ LỜI CUỐI
Sau khi save_order thành công, xác nhận bằng tiếng Việt với:
- Order ID
- Tổng tiền sau giảm giá
- Mã campaign
- Đường dẫn file đã lưu
""".strip()


def build_tools(store: OrderDataStore):
    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the product catalog by name, brand, category, or tags. Always call this first to discover product IDs."""
        result = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags,
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(result, ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Return exact price, stock, SKU, and a detail_token for the given product IDs. Call this second, after list_products."""
        result = store.get_product_details(product_ids)
        return json.dumps(result, ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Return the campaign discount rate and campaign_code. Use the customer email as seed_hint. Call this third, after get_product_details."""
        result = store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier)
        return json.dumps(result, ensure_ascii=False)

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items: list, detail_token: str, discount_rate: float) -> str:
        """Validate stock and compute subtotal, discount_amount, and final_total. Call this fourth. If status is error, do NOT call save_order."""
        from src.core.schemas import OrderLineInput
        parsed_items = [
            item if isinstance(item, OrderLineInput) else OrderLineInput(**item)
            for item in items
        ]
        result = store.calculate_order_totals(
            items=parsed_items,
            detail_token=detail_token,
            discount_rate=discount_rate,
        )
        return json.dumps(result, ensure_ascii=False)

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items: list,
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist the final order to a JSON file. Call this last, only when calculate_order_totals returned status ok."""
        from src.core.schemas import OrderLineInput
        parsed_items = [
            item if isinstance(item, OrderLineInput) else OrderLineInput(**item)
            for item in items
        ]
        result = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=parsed_items,
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )
        return json.dumps(result, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )


def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = response["messages"] if isinstance(response, dict) else response
    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)
    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def extract_final_answer(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            return None, None
        return payload.get("saved_order"), payload.get("path")
    return None, None
