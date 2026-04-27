import asyncio
from dataclasses import dataclass
from enum import Enum
from src.broker.groww_client import GrowwClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_M = "SL-M"


class ProductType(str, Enum):
    INTRADAY = "INTRADAY"
    DELIVERY = "DELIVERY"


class TransactionType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class OrderRequest:
    symbol: str
    exchange: str
    transaction_type: str
    quantity: int
    order_type: OrderType
    product_type: ProductType
    price: float = 0.0
    trigger_price: float = 0.0
    tag: str = "algo-trader"


class OrderManager:
    """
    Handles all order operations against Groww API.
    Supports PAPER_TRADE mode where orders are logged but not sent.
    """

    def __init__(self, client: GrowwClient, paper_trade: bool = True):
        self.client = client
        self.paper_trade = paper_trade

    async def place_order(self, req: OrderRequest) -> dict:
        """Places a single order. Returns order_id and status."""
        payload = {
            "symbol": req.symbol,
            "exchange": req.exchange,
            "transactionType": req.transaction_type,
            "quantity": req.quantity,
            "orderType": req.order_type.value,
            "productType": req.product_type.value,
            "price": req.price,
            "triggerPrice": req.trigger_price,
            "tag": req.tag,
        }

        if self.paper_trade:
            logger.info(
                "PAPER_TRADE_order",
                symbol=req.symbol,
                action=req.transaction_type,
                qty=req.quantity,
                price=req.price,
                order_type=req.order_type.value,
            )
            return {
                "order_id": f"PAPER-{req.symbol}-{req.transaction_type}",
                "status": "COMPLETE",
                "paper_trade": True,
            }

        logger.info("placing_order", symbol=req.symbol, action=req.transaction_type, qty=req.quantity)
        return await self.client.post("/orders/place", payload)

    async def place_bracket_order(
        self,
        entry: OrderRequest,
        stop_loss_price: float,
        target_price: float,
        poll_interval: float = 2.0,
        max_wait: float = 30.0,
    ) -> dict:
        """
        Places entry order then GTT OCO after fill.
        Polls order status until COMPLETE before placing GTT.
        Returns {"entry_order_id": ..., "gtt_id": ...}
        """
        entry_result = await self.place_order(entry)
        order_id = entry_result.get("order_id", "")

        if self.paper_trade:
            logger.info("PAPER_TRADE_gtt", symbol=entry.symbol, sl=stop_loss_price, target=target_price)
            return {
                "entry_order_id": order_id,
                "gtt_id": f"PAPER-GTT-{entry.symbol}",
                "paper_trade": True,
            }

        # Poll until entry fills
        elapsed = 0.0
        while elapsed < max_wait:
            status_data = await self.get_order_status(order_id)
            if status_data.get("status") == "COMPLETE":
                break
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        else:
            logger.warning("entry_order_not_filled", order_id=order_id, symbol=entry.symbol)
            return {"entry_order_id": order_id, "gtt_id": None, "warning": "entry not filled in time"}

        gtt_payload = {
            "symbol": entry.symbol,
            "exchange": entry.exchange,
            "quantity": entry.quantity,
            "stopLossPrice": stop_loss_price,
            "targetPrice": target_price,
            "triggerType": "OCO",
            "productType": entry.product_type.value,
        }
        gtt_result = await self.client.post("/orders/gtt", gtt_payload)
        logger.info("gtt_placed", symbol=entry.symbol, gtt_id=gtt_result.get("gtt_id"))
        return {"entry_order_id": order_id, "gtt_id": gtt_result.get("gtt_id")}

    async def get_order_status(self, order_id: str) -> dict:
        return await self.client.get(f"/orders/{order_id}")

    async def cancel_order(self, order_id: str) -> dict:
        logger.info("cancelling_order", order_id=order_id)
        if self.paper_trade:
            return {"order_id": order_id, "status": "CANCELLED", "paper_trade": True}
        return await self.client.delete(f"/orders/{order_id}")

    async def get_open_positions(self) -> list[dict]:
        """Returns open positions from broker (live)."""
        try:
            data = await self.client.get("/user/positions")
            return data.get("positions", data.get("data", []))
        except Exception as e:
            logger.error("get_positions_failed", error=str(e))
            return []

    async def square_off_all_intraday(self, positions: list[dict]) -> list[dict]:
        """Market-exits all open INTRADAY positions. Called at SQUARE_OFF_TIME."""
        results = []
        for pos in positions:
            if pos.get("product_type", pos.get("productType", "")) in ("INTRADAY", "MIS"):
                qty = abs(int(pos.get("quantity", pos.get("qty", 0))))
                if qty == 0:
                    continue
                direction = pos.get("direction", pos.get("transactionType", "BUY"))
                exit_side = "SELL" if direction == "BUY" else "BUY"
                req = OrderRequest(
                    symbol=pos["symbol"],
                    exchange=pos.get("exchange", "NSE"),
                    transaction_type=exit_side,
                    quantity=qty,
                    order_type=OrderType.MARKET,
                    product_type=ProductType.INTRADAY,
                )
                result = await self.place_order(req)
                logger.info("square_off", symbol=pos["symbol"], qty=qty, side=exit_side)
                results.append(result)
        return results
