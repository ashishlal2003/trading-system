"""
Unit tests for src/risk/manager.py — RiskManager.

All async methods on trade_store are mocked with AsyncMock so no database
or broker connections are required.
"""

from __future__ import annotations

import math
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.risk.manager import RiskManager, PositionSize


# ---------------------------------------------------------------------------
# Shared factory
# ---------------------------------------------------------------------------


def make_risk_manager(
    total_capital: float = 100_000.0,
    max_risk_per_trade_pct: float = 1.0,
    max_daily_loss_pct: float = 3.0,
    max_open_positions: int = 5,
    daily_pnl: float = 0.0,
    open_positions: int = 0,
) -> RiskManager:
    """
    Build a RiskManager with a mocked trade_store.

    Parameters
    ----------
    daily_pnl       : Value returned by trade_store.get_daily_pnl().
    open_positions  : Value returned by trade_store.count_open_positions().
    """
    trade_store = MagicMock()
    trade_store.get_daily_pnl = AsyncMock(return_value=daily_pnl)
    trade_store.count_open_positions = AsyncMock(return_value=open_positions)

    return RiskManager(
        total_capital=total_capital,
        max_risk_per_trade_pct=max_risk_per_trade_pct,
        max_daily_loss_pct=max_daily_loss_pct,
        max_open_positions=max_open_positions,
        trade_store=trade_store,
    )


# ---------------------------------------------------------------------------
# compute_position_size
# ---------------------------------------------------------------------------


class TestComputePositionSize:
    """Tests for RiskManager.compute_position_size()."""

    def test_position_size_basic(self) -> None:
        """
        Capital=100 000, risk=1% → risk_amount=1 000.
        entry=500, sl=490 → risk_per_share=10.
        qty = floor(1000 / 10) = 100.
        """
        rm = make_risk_manager(total_capital=100_000.0, max_risk_per_trade_pct=1.0)
        ps = rm.compute_position_size(entry_price=500.0, stop_loss=490.0)

        assert isinstance(ps, PositionSize)
        assert ps.quantity == 100
        assert ps.entry_price == 500.0
        assert ps.stop_loss == 490.0
        assert math.isclose(ps.capital_at_risk, 1000.0, rel_tol=1e-9)

    def test_position_size_short_side(self) -> None:
        """
        Short trade: entry < stop_loss → risk_per_share = |entry - sl| still positive.
        entry=490, sl=500 → risk_per_share=10, qty=100.
        """
        rm = make_risk_manager(total_capital=100_000.0, max_risk_per_trade_pct=1.0)
        ps = rm.compute_position_size(entry_price=490.0, stop_loss=500.0)

        assert ps.quantity == 100
        assert math.isclose(ps.capital_at_risk, 1000.0, rel_tol=1e-9)

    def test_position_size_minimum_one(self) -> None:
        """
        Very small capital or wide stop-loss must yield at least 1 share.
        Capital=500, risk=0.1% → risk_amount=0.50.
        entry=500, sl=100 → risk_per_share=400 → raw_qty=0.00125 → floor=0 → clipped to 1.
        """
        rm = make_risk_manager(total_capital=500.0, max_risk_per_trade_pct=0.1)
        ps = rm.compute_position_size(entry_price=500.0, stop_loss=100.0)

        assert ps.quantity >= 1, (
            f"Expected quantity >= 1 for near-zero raw qty, got {ps.quantity}"
        )

    def test_position_size_zero_sl_raises(self) -> None:
        """entry == stop_loss must raise ValueError (zero risk per share)."""
        rm = make_risk_manager()

        with pytest.raises(ValueError, match="zero risk per share"):
            rm.compute_position_size(entry_price=500.0, stop_loss=500.0)

    def test_position_size_returns_position_size_dataclass(self) -> None:
        """Return value must be a PositionSize with the expected fields populated."""
        rm = make_risk_manager(total_capital=200_000.0, max_risk_per_trade_pct=2.0)
        ps = rm.compute_position_size(entry_price=1000.0, stop_loss=980.0)

        # risk_amount = 200_000 * 0.02 = 4_000
        # risk_per_share = 20
        # qty = floor(4000 / 20) = 200
        assert ps.quantity == 200
        assert math.isclose(ps.max_loss, ps.capital_at_risk, rel_tol=1e-9)
        assert 0 < ps.risk_pct <= ps.capital_at_risk / rm.total_capital * 100 + 0.001

    def test_position_size_fractional_qty_floors(self) -> None:
        """Non-integer raw quantity should be floored (not rounded)."""
        rm = make_risk_manager(total_capital=100_000.0, max_risk_per_trade_pct=1.0)
        # risk_amount=1000, risk_per_share=7 → raw=142.857 → floor=142
        ps = rm.compute_position_size(entry_price=507.0, stop_loss=500.0)
        assert ps.quantity == 142


# ---------------------------------------------------------------------------
# can_trade
# ---------------------------------------------------------------------------


class TestCanTrade:
    """Tests for the async RiskManager.can_trade() gate."""

    @pytest.mark.asyncio
    async def test_can_trade_allowed(self) -> None:
        """
        Safe daily P&L and fewer open positions than the cap
        must return (True, "").
        """
        rm = make_risk_manager(
            total_capital=100_000.0,
            max_daily_loss_pct=3.0,
            max_open_positions=5,
            daily_pnl=0.0,       # no loss
            open_positions=2,    # well under the cap of 5
        )
        allowed, reason = await rm.can_trade()

        assert allowed is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_can_trade_blocked_by_daily_loss(self) -> None:
        """
        When daily P&L <= -3 % of capital (= -3 000 for 100 000 capital),
        can_trade must return (False, <reason>).
        """
        rm = make_risk_manager(
            total_capital=100_000.0,
            max_daily_loss_pct=3.0,
            max_open_positions=5,
            daily_pnl=-3001.0,   # exceeds the -3 000 limit
            open_positions=0,
        )
        allowed, reason = await rm.can_trade()

        assert allowed is False
        assert "Daily loss limit" in reason
        rm.trade_store.get_daily_pnl.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_can_trade_blocked_at_exact_loss_limit(self) -> None:
        """
        Daily P&L exactly equal to the limit (-3 000) must also block trading.
        (The check is <=, so equality triggers the halt.)
        """
        rm = make_risk_manager(
            total_capital=100_000.0,
            max_daily_loss_pct=3.0,
            max_open_positions=5,
            daily_pnl=-3000.0,   # exactly at the limit
            open_positions=0,
        )
        allowed, reason = await rm.can_trade()

        assert allowed is False
        assert "Daily loss limit" in reason

    @pytest.mark.asyncio
    async def test_can_trade_blocked_by_max_positions(self) -> None:
        """
        When open positions reach the configured cap (5),
        can_trade must return (False, <reason>).
        """
        rm = make_risk_manager(
            total_capital=100_000.0,
            max_daily_loss_pct=3.0,
            max_open_positions=5,
            daily_pnl=200.0,     # profitable day — daily loss check passes
            open_positions=5,    # at the cap
        )
        allowed, reason = await rm.can_trade()

        assert allowed is False
        assert "Max open positions" in reason
        rm.trade_store.count_open_positions.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_can_trade_blocked_by_max_positions_over_cap(self) -> None:
        """
        open_positions > cap must also block trading.
        """
        rm = make_risk_manager(
            total_capital=100_000.0,
            max_open_positions=5,
            daily_pnl=0.0,
            open_positions=6,
        )
        allowed, reason = await rm.can_trade()

        assert allowed is False
        assert "Max open positions" in reason

    @pytest.mark.asyncio
    async def test_can_trade_daily_loss_check_runs_before_position_check(
        self,
    ) -> None:
        """
        Daily loss is checked first; if it fails the position check is irrelevant.
        Verify that get_daily_pnl is awaited and count_open_positions may not be.
        """
        rm = make_risk_manager(
            total_capital=100_000.0,
            max_daily_loss_pct=3.0,
            max_open_positions=5,
            daily_pnl=-5000.0,   # triggers daily loss halt
            open_positions=0,    # positions would be fine
        )
        allowed, reason = await rm.can_trade()

        assert allowed is False
        rm.trade_store.get_daily_pnl.assert_awaited_once()
        # count_open_positions must NOT have been called — early exit
        rm.trade_store.count_open_positions.assert_not_awaited()


# ---------------------------------------------------------------------------
# validate_signal_rr
# ---------------------------------------------------------------------------


class TestValidateSignalRR:
    """Tests for RiskManager.validate_signal_rr()."""

    def test_validate_rr_passes(self) -> None:
        """
        entry=100, sl=95, target=115.
        reward=15, risk=5, rr=3.0 >= 1.5 → True.
        """
        rm = make_risk_manager()
        assert rm.validate_signal_rr(entry=100.0, stop_loss=95.0, target=115.0) is True

    def test_validate_rr_fails(self) -> None:
        """
        entry=100, sl=95, target=102.
        reward=2, risk=5, rr=0.4 < 1.5 → False.
        """
        rm = make_risk_manager()
        assert rm.validate_signal_rr(entry=100.0, stop_loss=95.0, target=102.0) is False

    def test_validate_rr_exactly_at_minimum(self) -> None:
        """
        rr == 1.5 exactly should pass (>= check).
        entry=100, sl=90, target=115 → reward=15, risk=10, rr=1.5.
        """
        rm = make_risk_manager()
        assert rm.validate_signal_rr(entry=100.0, stop_loss=90.0, target=115.0) is True

    def test_validate_rr_just_below_minimum(self) -> None:
        """
        rr = 1.49 should fail.
        entry=100, sl=90, target=114.9 → reward=14.9, risk=10, rr=1.49.
        """
        rm = make_risk_manager()
        assert rm.validate_signal_rr(entry=100.0, stop_loss=90.0, target=114.9) is False

    def test_validate_rr_zero_risk_returns_false(self) -> None:
        """entry == stop_loss → risk = 0 → should return False, not raise."""
        rm = make_risk_manager()
        result = rm.validate_signal_rr(entry=100.0, stop_loss=100.0, target=120.0)
        assert result is False

    def test_validate_rr_custom_min_rr(self) -> None:
        """Custom min_rr=2.0: rr=1.8 should fail, rr=2.1 should pass."""
        rm = make_risk_manager()
        # rr = (110 - 100) / (100 - 95) = 10 / 5 = 2.0 → passes with min_rr=2.0
        assert rm.validate_signal_rr(entry=100.0, stop_loss=95.0, target=110.0, min_rr=2.0) is True
        # rr = (108 - 100) / (100 - 95) = 8 / 5 = 1.6 → fails with min_rr=2.0
        assert rm.validate_signal_rr(entry=100.0, stop_loss=95.0, target=108.0, min_rr=2.0) is False

    def test_validate_rr_short_trade(self) -> None:
        """
        Short trade: entry < stop_loss, target < entry.
        entry=200, sl=210, target=185 → reward=15, risk=10, rr=1.5 → True.
        """
        rm = make_risk_manager()
        assert rm.validate_signal_rr(entry=200.0, stop_loss=210.0, target=185.0) is True


# ---------------------------------------------------------------------------
# validate_entry_proximity
# ---------------------------------------------------------------------------


class TestValidateEntryProximity:
    """Tests for RiskManager.validate_entry_proximity() — intraday use-case."""

    def test_validate_entry_proximity_intraday_within_limit(self) -> None:
        """
        current price within 0.3% of entry_price must return True.
        entry=1000, current=1002 → deviation = 0.2% < 0.3% → True.
        """
        rm = make_risk_manager()
        assert rm.validate_entry_proximity(entry_price=1000.0, current_price=1002.0) is True

    def test_validate_entry_proximity_intraday_exceeds_limit(self) -> None:
        """
        current price 1% away from entry_price must return False.
        entry=1000, current=1010 → deviation = 1.0% > 0.3% → False.
        """
        rm = make_risk_manager()
        assert rm.validate_entry_proximity(entry_price=1000.0, current_price=1010.0) is False

    def test_validate_entry_proximity_exactly_at_limit(self) -> None:
        """
        deviation exactly equal to max_pct (0.3%) must pass (<=).
        entry=1000, current=1003 → deviation = 0.3% → True.
        """
        rm = make_risk_manager()
        assert rm.validate_entry_proximity(entry_price=1000.0, current_price=1003.0) is True

    def test_validate_entry_proximity_just_over_limit(self) -> None:
        """
        deviation just above 0.3% must fail.
        entry=1000, current=1003.1 → deviation ≈ 0.31%.
        """
        rm = make_risk_manager()
        assert rm.validate_entry_proximity(entry_price=1000.0, current_price=1003.1) is False

    def test_validate_entry_proximity_zero_current_price(self) -> None:
        """current_price == 0 must return False (avoid ZeroDivisionError)."""
        rm = make_risk_manager()
        result = rm.validate_entry_proximity(entry_price=100.0, current_price=0.0)
        assert result is False

    def test_validate_entry_proximity_custom_max_pct(self) -> None:
        """
        Custom max_pct=1.0: 0.9% deviation should pass, 1.1% should fail.
        """
        rm = make_risk_manager()
        # 0.9% deviation → passes with max_pct=1.0
        assert rm.validate_entry_proximity(
            entry_price=1000.0, current_price=1009.0, max_pct=1.0
        ) is True
        # 1.1% deviation → fails with max_pct=1.0
        assert rm.validate_entry_proximity(
            entry_price=1000.0, current_price=1011.0, max_pct=1.0
        ) is False

    def test_validate_entry_proximity_below_current_price(self) -> None:
        """
        Entry below current price — uses absolute deviation, so direction is symmetric.
        entry=990, current=1000 → deviation = 1.0% > 0.3% → False.
        """
        rm = make_risk_manager()
        assert rm.validate_entry_proximity(entry_price=990.0, current_price=1000.0) is False

        # entry=999, current=1000 → deviation = 0.1% < 0.3% → True
        assert rm.validate_entry_proximity(entry_price=999.0, current_price=1000.0) is True


# ---------------------------------------------------------------------------
# max_daily_loss_limit property
# ---------------------------------------------------------------------------


class TestMaxDailyLossLimit:
    """Verify that the circuit-breaker threshold is computed correctly."""

    def test_max_daily_loss_limit_computed_correctly(self) -> None:
        """
        3% of 100 000 = -3 000.
        The limit is stored as a negative number.
        """
        rm = make_risk_manager(total_capital=100_000.0, max_daily_loss_pct=3.0)
        assert math.isclose(rm.max_daily_loss_limit, -3000.0, rel_tol=1e-9)

    def test_max_daily_loss_limit_custom_capital(self) -> None:
        """2% of 500 000 = -10 000."""
        rm = make_risk_manager(total_capital=500_000.0, max_daily_loss_pct=2.0)
        assert math.isclose(rm.max_daily_loss_limit, -10_000.0, rel_tol=1e-9)
