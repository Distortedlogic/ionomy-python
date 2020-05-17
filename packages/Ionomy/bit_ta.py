from typing import Optional

import pandas as pd
import pandas_ta as ta
from pandas.core.frame import DataFrame, Series
from pandas.core.series import Series

from .bit_panda import BitPanda
from .utils.dataframes import _size_mask


class BitTA(BitPanda):
    """Technical Analysis Wrapper for BitPanda

    Arguments:
        api_key {str} -- BitTrex API Key
        secret_key {str} -- BitTrex API Secret
    """
    def __init__(self, api_key: str, secret_key: str) -> None:
        BitPanda.__init__(self, api_key, secret_key)

    def update(self, currency: str, base: str, time: str):
        """Set the current ohlcv dataframe to the latest data with the given args

        Arguments:
            currency {str}
            base {str}
            time {str}
        """
        self.df = self.ohlcv(currency, base, time)

    """
    ========
    Momentum
    ========
    """

    def ao(self, fast: int = None, slow: int = 0, offset: int = None, **kwargs) -> Series:
        """Indicator: Awesome Oscillator (AO)"""
        return self.df.ta.ao(fast=fast, slow=slow, offest=offset, **kwargs)

    def apo(self, fast=None, slow=None, offset=None, **kwargs) -> Series:
        """Indicator: Absolute Price Oscillator (APO)"""
        return self.df.ta.apo(fast=fast, slow=slow, offset=offset, **kwargs)

    def bop(self, offset=None, **kwargs) -> Series:
        """Indicator: Balance of Power (BOP)"""
        return self.df.ta.bop(offset=offset, **kwargs)

    def cci(self, length=None, c=None, offset=None, **kwargs) -> Series:
        """Indicator: Commodity Channel Index (CCI)"""
        return self.df.ta.cci(length=length, c=c, offset=offset, **kwargs)

    def cg(self, length=None, offset=None, **kwargs) -> Series:
        """Indicator: Center of Gravity (CG)"""
        return self.df.ta.cg(length=length, offset=offset, **kwargs)

    def cmo(self, length=None, drift=None, offset=None, **kwargs) -> Series:
        """Indicator: Chande Momentum Oscillator (CMO)"""
        return self.df.ta.cmo(length=length, drift=drift, offset=offset, **kwargs)

    def coppock(self, length=None, fast=None, slow=None, offset=None, **kwargs) -> Series:
        """Indicator: Coppock Curve (COPC)"""
        return self.df.ta.coppock(length=length, fast=fast, slow=slow, offset=offset, **kwargs)

    def fisher(self, length=None, offset=None, **kwargs) -> Series:
        """Indicator: Fisher Transform (FISHT)"""
        return self.df.ta.fisher(length=length, offset=offset, **kwargs)

    def kst(
        self,
        roc1=None,
        roc2=None,
        roc3=None,
        roc4=None,
        sma1=None,
        sma2=None,
        sma3=None,
        sma4=None,
        signal=None,
        drift=None,
        offset=None,
        **kwargs) -> Series:
        """Indicator: 'Know Sure Thing'"""
        return self.df.ta.kst(
            roc1=roc1,
            roc2=roc2,
            roc3=roc3,
            roc4=roc4,
            sma1=sma1,
            sma2=sma2,
            sma3=sma3,
            sma4=sma4,
            signal=signal,
            drift=drift,
            offset=offset,
            **kwargs
        )

    def macd(self, fast=None, slow=None, signal=None, offset=None, **kwargs) -> Series:
        return self.df.ta.macd(fast=fast, slow=slow, signal=signal, offset=offset, **kwargs)

    def momentum(self, length=None, offset=None, **kwargs) -> Series:
        return self.df.ta.mom(length=length, offset=offset, **kwargs)

    def ppo(self, fast=None, slow=None, signal=None, offset=None, **kwargs):
        """Indicator: Percentage Price Oscillator (PPO)"""
        return self.df.ta.ppo(fast=fast, slow=slow, signal=signal, offset=offset, **kwargs)

    def roc(self, length=None, offset=None, **kwargs) -> Series:
        return self.df.ta.roc(length=length, offset=offset, **kwargs)

    def rsi(self, length=None, drift=None, offset=None, **kwargs) -> Series:
        """Indicator: Relative Strength Index (RSI)"""
        return self.df.ta.rsi(length=length, drift=drift, offset=offset, **kwargs)

    def rvi(self, length=None, swma_length=None, offset=None, **kwargs):
        """Indicator: RVI"""
        return self.df.ta.rvi(length=length, swma_length=swma_length, offset=offset, **kwargs)

    def slope(self, length=None, as_angle=None, to_degrees=None, offset=None, **kwargs):
        """Indicator: Slope"""
        return self.df.ta.slope(
            length=length,
            as_angle=as_angle,
            to_degrees=to_degrees,
            offset=offset,
            **kwargs
        )

    def stoch(self, fast_k=None, slow_k=None, slow_d=None, offset=None, **kwargs):
        """Indicator: Stochastic Oscillator (STOCH)"""
        return self.df.ta.stoch(fast_k=fast_k, slow_k=slow_k, slow_d=slow_d, offset=offset, **kwargs)

    def trix(self, length=None, drift=None, offset=None, **kwargs):
        """Indicator: Trix (TRIX)"""
        return self.df.ta.trix(length=length, drift=drift, offset=offset, **kwargs)

    def tsi(self, fast=None, slow=None, drift=None, offset=None, **kwargs):
        """Indicator: True Strength Index (TSI)"""
        return self.df.ta.tsi(fast=fast, slow=slow, drift=drift, offset=offset, **kwargs)

    def uo(
        self,
        fast=None,
        medium=None,
        slow=None,
        fast_w=None,
        medium_w=None,
        slow_w=None,
        drift=None,
        offset=None,
        **kwargs
    ):
        """Indicator: Ultimate Oscillator (UO)"""
        return self.df.ta.uo(
            fast=fast,
            medium=medium,
            slow=slow,
            fast_w=fast_w,
            medium_w=medium_w,
            slow_w=slow_w,
            drift=drift,
            offset=offset,
            **kwargs
        )

    def willr(self, length=None, offset=None, **kwargs):
        """Indicator: William's Percent R (WILLR)"""
        return self.df.ta.willr(length=length, offset=offset, **kwargs)

    """
    =======
    Overlap
    =======
    """

    def dema(self, length=None, offset=None, **kwargs):
        """Indicator: Double Exponential Moving Average (DEMA)"""
        return self.df.ta.dema(length=length, offset=offset, **kwargs)

    def ema(self, length=None, offset=None, **kwargs) -> Series:
        """Indicator: Exponential Moving Average (EMA)"""
        return self.df.ta.ema(length=length, offset=offset, **kwargs)

    def fwma(self, length=None, asc=None, offset=None, **kwargs):
        """Indicator: Fibonacci's Weighted Moving Average (FWMA)"""
        return self.df.ta.fwma(length=length, asc=asc, offset=offset, **kwargs)

    def hl2(self, offset=None, **kwargs):
        """Indicator: HL2 """
        return self.df.ta.hl2(offset=offset, **kwargs)

    def hlc3(self, offset=None, **kwargs):
        """Indicator: HLC3"""
        return self.df.ta.hlc3(offset=offset, **kwargs)

    def hma(self, length=None, offset=None, **kwargs):
        """Indicator: Hull Moving Average (HMA)"""
        return self.df.ta.hma(length=length, offset=offset, **kwargs)

    def ichimoku(self, tenkan=None, kijun=None, senkou=None, offset=None, **kwargs):
        """Indicator: Ichimoku Kinkō Hyō (Ichimoku)"""
        return self.df.ta.ichimoku(tenkan=tenkan, kijun=kijun, senkou=senkou, offset=offset, **kwargs)

    def kama(self, length=None, fast=None, slow=None, drift=None, offset=None, **kwargs):
        """Indicator: Kaufman's Adaptive Moving Average (HMA)"""
        return self.df.kama(length=length, fast=fast, slow=slow, drift=drift, offset=offset, **kwargs)

    def linreg(self, length=None, offset=None, **kwargs):
        """Indicator: Linear Regression"""
        return self.df.ta.linreg(length=length, offset=offset, **kwargs)

    def midpoint(self, length=None, offset=None, **kwargs):
        """Indicator: Midpoint"""
        return self.df.ta.midpoint(length=length, offset=offset, **kwargs)

    def midprice(self, length=None, offset=None, **kwargs):
        """Indicator: Midprice"""
        return self.df.ta.midprice(length=length, offset=offset, **kwargs)

    def ohlc4(self, offset=None, **kwargs):
        """Indicator: OHLC4"""
        return self.df.ta.ohlc4(offset=offset, **kwargs)

    def pwma(self, length=None, asc=None, offset=None, **kwargs):
        """Indicator: Pascals Weighted Moving Average (PWMA)"""
        return self.df.ta.pwma(length=length, asc=asc, offset=offset, **kwargs)

    def rma(self, length=None, offset=None, **kwargs):
        """Indicator: wildeR's Moving Average (RMA)"""
        return self.df.ta.rma(length=length, offset=offset, **kwargs)

    def sinwma(self, length=None, asc=None, offset=None, **kwargs):
        """Indicator: Sine Weighted Moving Average (SINWMA) by Everget of TradingView"""
        return self.df.ta.sinwma(length=length, asc=asc, offset=offset, **kwargs)

    def sma(self, length=None, offset=None, **kwargs) -> Series:
        """Indicator: Simple Moving Average (SMA)"""
        return self.df.ta.ema(length=length, offset=offset, **kwargs)

    def swma(self, length=None, asc=None, offset=None, **kwargs):
        """Indicator: Symmetric Weighted Moving Average (SWMA)"""
        return self.df.ta.swma(length=length, asc=asc, offset=offset, **kwargs)

    def t3(self, length=None, a=None, offset=None, **kwargs):
        """Indicator: T3"""
        return self.df.ta.t3(length=length, a=a, offset=offset, **kwargs)

    def tema(self, length=None, offset=None, **kwargs):
        """Indicator: Triple Exponential Moving Average (TEMA)"""
        return self.df.ta.tema(length=length, offset=offset, **kwargs)

    def trima(self, length=None, offset=None, **kwargs):
        """Indicator: Triangular Moving Average (TRIMA)"""
        return self.df.ta.trima(length=length, offset=offset, **kwargs)

    def vwap(self, offset=None, **kwargs):
        """Indicator: Volume Weighted Average Price (VWAP)"""
        return self.df.ta.vwap(offset=offset, **kwargs)

    def vwma(self, length=None, offset=None, **kwargs) -> Series:
        """Indicator: Volume Weighted Moving Average (VWMA)"""
        return self.df.ta.vwma(length=length, offset=offset, **kwargs)

    def wma(self, length=None, asc=None, offset=None, **kwargs):
        """Indicator: Weighted Moving Average (WMA)"""
        return self.df.ta.wma(length=length, asc=asc, offset=offset, **kwargs)

    def zlma(self, length=None, offset=None, mamode=None, **kwargs):
        """Indicator: Zero Lag Moving Average (ZLMA)"""
        return self.df.ta.zlma(length=length, offset=offset, mamode=mamode, **kwargs)

    """
    ===========
    Performance
    ===========
    """

    def log_return(
        self,
        length=None,
        cumulative=False,
        percent=False,
        offset=None,
        **kwargs
    ) -> Series:
        """Indicator: Log Return"""
        return self.df.ta.log_return(
            length=length,
            cumulative=cumulative,
            percent=percent,
            offset=offset,
            **kwargs
        )

    def percent_return(
        self,
        length=None,
        cumulative=False,
        percent=False,
        offset=None,
        **kwargs
    ) -> Series:
        """Indicator: Percent Return"""
        return self.df.ta.percent_return(
            length=length,
            cumulative=cumulative,
            percent=percent,
            offset=offset,
            **kwargs
        )

    def trend_return(
        self,
        trend: Series,
        log: bool=True,
        cumulative: bool=None,
        offset: int=None,
        trend_reset: int=0,
        **kwargs
    ):
        """Indicator: Trend Return"""
        return self.df.ta.trend_return(
            trend, log=log, cumulative=cumulative, offset=offset, trend_reset=trend_reset, **kwargs
        )

    """
    ==========
    Statistics
    ==========
    """

    def kurtosis(self, length=None, offset=None, **kwargs):
        """Indicator: Kurtosis"""
        return self.df.ta.kurtosis(length=length, offset=offset, **kwargs)

    def mad(self, length=None, offset=None, **kwargs):
        """Indicator: Mean Absolute Deviation"""
        return self.df.ta.mad(length=length, offset=offset, **kwargs)

    def median(self, length=None, offset=None, **kwargs):
        """Indicator: Median"""
        return self.df.ta.median(length=length, offset=offset, **kwargs)

    def quantile(self, length=None, q=None, offset=None, **kwargs):
        """Indicator: Quantile"""
        return self.df.ta.quantile(length=length, q=q, offset=offset, **kwargs)

    def skew(self, length=None, offset=None, **kwargs):
        """Indicator: Skew"""
        return self.df.ta.skew(length=length, offset=offset, **kwargs)

    def stdev(self, length=None, offset=None, **kwargs):
        """Indicator: Standard Deviation"""
        return self.df.ta.stdev(length=length, offset=offset, **kwargs)

    def variance(self, length=None, offset=None, **kwargs):
        """Indicator: Variance"""
        return self.df.ta.variance(length=length, offset=offset, **kwargs)

    def zscore(self, length=None, std=None, offset=None, **kwargs):
        """Indicator: Z Score"""
        return self.df.ta.zscore(length=length, std=std, offset=offset, **kwargs)

    """
    =====
    Trend
    =====
    """

    def adx(self, length=None, drift=None, offset=None, **kwargs):
        """Indicator: ADX"""
        return self.df.ta.adx(length=length, drift=drift, offset=offset, **kwargs)

    def amat(self, fast=None, slow=None, mamode=None, lookback=None, offset=None, **kwargs):
        """Indicator: Archer Moving Averages Trends (AMAT)"""
        return self.df.ta.amat(
            fast=fast,
            slow=slow,
            mamode=mamode,
            lookback=lookback,
            offset=offset,
            **kwargs
        )

    def aroon(self, length=None, offset=None, **kwargs):
        """Indicator: Aroon Oscillator"""
        return self.df.ta.aroon(length=length, offset=offset, **kwargs)

    def decreasing(self, length=None, asint=True, offset=None, **kwargs):
        """Indicator: Decreasing"""
        return self.df.ta.decreasing(length=length, asint=asint, offset=offset, **kwargs)

    def dpo(self, length=None, centered=True, offset=None, **kwargs):
        """Indicator: Detrend Price Oscillator (DPO)"""
        return self.df.ta.dpo(length=length, centered=centered, offset=offset, **kwargs)

    def increasing(self, length=None, asint=True, offset=None, **kwargs):
        """Indicator: Increasing"""
        return self.df.ta.increasing(length=length, asint=asint, offset=offset, **kwargs)

    def linear_decay(self, length=None, offset=None, **kwargs):
        """Indicator: Linear Decay"""
        return self.df.ta.linear_decay(length=length, offset=offset, **kwargs)

    def long_run(self, fast: Series, slow: Series, length=None, offset=None, **kwargs):
        """Indicator: Long Run"""
        return self.df.ta.long_run(fast, slow, length=length, offset=offset, **kwargs)

    def qstick(self, length=None, offset=None, **kwargs):
        """Indicator: Q Stick"""
        return self.df.ta.qstick(length=length, offset=offset, **kwargs)

    def short_run(self, fast: Series, slow: Series, length=None, offset=None, **kwargs):
        """Indicator: Short Run"""
        return self.df.ta.short_run(fast, slow, length=length, offset=offset, **kwargs)

    def vortex(self, length=None, drift=None, offset=None, **kwargs):
        """Indicator: Vortex"""
        return self.df.ta.vortex(length=length, drift=drift, offset=offset, **kwargs)

    """
    ==========
    Volatility
    ==========
    """

    def accbands(
        self,
        length=None,
        c=None,
        drift=None,
        mamode=None,
        offset=None,
        **kwargs
    ):
        """Indicator: Acceleration Bands (ACCBANDS)"""
        return self.df.ta.accbands(
            length=length,
            c=c,
            drift=drift,
            mamode=mamode,
            offset=offset,
            **kwargs
        )

    def atr(self, length=None, mamode=None, offset=None, **kwargs) -> Series:
        """Indicator: Average True Range (ATR)"""
        return self.df.ta.atr(length=length, mamode=mamode, offset=offset, **kwargs)

    def bbands(self, length=None, std=None, mamode=None, offset=None, **kwargs):
        """Indicator: Bollinger Bands (BBANDS)"""
        return self.df.ta.bbands(length=length, std=std, mamode=mamode, offset=offset, **kwargs)

    def donchian(self, lower_length=None, upper_length=None, offset=None, **kwargs):
        """Indicator: Donchian Channels (DC)"""
        return self.df.ta.donchian(
            lower_length=lower_length,
            upper_length=upper_length,
            offset=offset,
            **kwargs
        )

    def kc(self, length=None, scalar=None, mamode=None, offset=None, **kwargs):
        """Indicator: Keltner Channels (KC)"""
        return self.df.ta.kc(length=length, scalar=scalar, mamode=mamode, offset=offset, **kwargs)

    def massi(self, fast=None, slow=None, offset=None, **kwargs):
        """Indicator: Mass Index (MASSI)"""
        return self.df.ta.massi(fast=fast, slow=slow, offset=offset, **kwargs)

    def natr(self, length=None, mamode=None, drift=None, offset=None, **kwargs):
        """Indicator: Normalized Average True Range (NATR)"""
        return self.df.ta.natr(length=length, mamode=mamode, drift=drift, offset=offset, **kwargs)

    def true_range(self, drift=None, offset=None, **kwargs):
        """Indicator: True Range"""
        return self.df.ta.true_range(drift=drift, offset=offset, **kwargs)

    """
    ======
    Volume
    ======
    """

    def ad(self, offset=None, **kwargs):
        """Indicator: Accumulation/Distribution (AD)"""
        return self.df.ta.ad(offset=offset, **kwargs)

    def adosc(self, fast=None, slow=None, offset=None, **kwargs):
        """Indicator: Accumulation/Distribution Oscillator"""
        return self.df.ta.adosc(fast=fast, slow=slow, offset=offset, **kwargs)

    def aobv(
        self,
        fast=None,
        slow=None,
        mamode=None,
        max_lookback=None,
        min_lookback=None,
        offset=None,
        **kwargs
    ):
        """Indicator: Archer On Balance Volume (AOBV)"""
        return self.df.ta.aobv(
            fast=fast,
            slow=slow,
            mamode=mamode,
            max_lookback=max_lookback,
            min_lookback=min_lookback,
            offset=offset,
            **kwargs
        )

    def cmf(self, length=None, offset=None, **kwargs):
        """Indicator: Chaikin Money Flow (CMF)"""
        return self.df.ta.cmf(length=length, offset=offset, **kwargs)

    def efi(self, length=None, drift=None, mamode=None, offset=None, **kwargs):
        """Indicator: Elder's Force Index (EFI)"""
        return self.df.ta.efi(length=length, drift=drift, mamode=mamode, offset=offset, **kwargs)

    def eom(self, length=None, divisor=None, drift=None, offset=None, **kwargs):
        """Indicator: Ease of Movement (EOM)"""
        return self.df.ta.eom(length=length, divisor=divisor, drift=drift, offset=offset, **kwargs)

    def mfi(self, length=None, drift=None, offset=None, **kwargs):
        """Indicator: Money Flow Index (MFI)"""
        return self.df.ta.mfi(length=length, drift=drift, offset=offset, **kwargs)

    def nvi(self, length=None, initial=None, offset=None, **kwargs):
        """Indicator: Negative Volume Index (NVI)"""
        return self.df.ta.nvi(length=length, initial=initial, offset=offset, **kwargs)

    def obv(self, offset=None, **kwargs):
        """Indicator: On Balance Volume (OBV)"""
        return self.df.ta.obv(offset=offset, **kwargs)

    def pvi(self, length=None, initial=None, offset=None, **kwargs):
        """Indicator: Positive Volume Index (PVI)"""
        return self.df.ta.pvi(length=length, initial=initial, offset=offset, **kwargs)

    def pvol(self, offset=None, **kwargs):
        """Indicator: Price-Volume (PVOL)"""
        return self.df.ta.pvol(offset=offset, **kwargs)

    def pvt(self, drift=None, offset=None, **kwargs):
        """Indicator: Price-Volume Trend (PVT)"""
        return self.df.ta.pvt(drift=drift, offset=offset, **kwargs)

    def vp(self, width=None, **kwargs):
        """Indicator: Volume Profile (VP)"""
        return self.df.ta.vp(width=width, **kwargs)


    # def _filter_orderbook(
    #     self,
    #     market: str,
    #     order_type: str,
    #     size_min: Optional[float] = None,
    #     size_max: Optional[float] = None
    # ):
    #     order_book_pd = self.order_book(market)
    #     order_book_pd = order_book_pd[order_book_pd['type']==order_type]
    #     mask = _size_mask(order_book_pd, size_min, size_max)
        
    #     return order_book_pd[mask]

    # def max_bid(
    #     self,
    #     market: str,
    #     size_min: Optional[float] = None,
    #     size_max: Optional[float] = None
    # ) -> float:
    #     return self._filter_orderbook(market, 'bid', size_min, size_max)['price'].max()
    
    # def min_ask(
    #     self,
    #     market: str,
    #     size_min: Optional[float] = None,
    #     size_max: Optional[float] = None
    # ) -> float:
    #     return self._filter_orderbook(market, 'ask', size_min, size_max)['price'].min()

    # def spread(self, market: str = 'HIVE'):
    #     return self.min_ask(market) - self.max_bid(market)
    