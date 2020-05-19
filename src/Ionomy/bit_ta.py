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

    def ao(self, fast: int = None, slow: int = None, offset: int = None, **kwargs) -> Series:
        """Awesome Oscillator (AO)

        The Awesome Oscillator is an indicator used to measure a security's momentum.
        AO is generally used to affirm trends or to anticipate possible reversals.

        Sources:
            https://www.tradingview.com/wiki/Awesome_Oscillator_(AO)
            https://www.ifcm.co.uk/ntx-indicators/awesome-oscillator

        Calculation:
            Default Inputs:
                fast=5, slow=34
            
            SMA = Simple Moving Average

            median = (high + low) / 2

            AO = SMA(median, fast) - SMA(median, slow)

        Args:
            high (pd.Series): Series of 'high's
            low (pd.Series): Series of 'low's
            fast (int): The short period.  Default: 5
            slow (int): The long period.   Default: 34
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.

        """
        return self.df.ta.ao(fast=fast, slow=slow, offest=offset, **kwargs)

    def apo(self, fast=None, slow=None, offset=None, **kwargs) -> Series:
        """Absolute Price Oscillator (APO)

        The Absolute Price Oscillator is an indicator used to measure a security's
        momentum.  It is simply the difference of two Exponential Moving Averages
        (EMA) of two different periods.  Note: APO and MACD lines are equivalent.

        Sources:
            https://www.investopedia.com/terms/p/ppo.asp

        Calculation:
            Default Inputs:
                fast=12, slow=26

            EMA = Exponential Moving Average

            APO = EMA(close, fast) - EMA(close, slow)

        Args:
            close (pd.Series): Series of 'close's
            fast (int): The short period.  Default: 12
            slow (int): The long period.   Default: 26
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.

        """
        return self.df.ta.apo(fast=fast, slow=slow, offset=offset, **kwargs)

    def bop(self, offset=None, **kwargs) -> Series:
        """Balance of Power (BOP)

        Balance of Power measure the market strength of buyers against sellers.

        Sources:
            http://www.worden.com/TeleChartHelp/Content/Indicators/Balance_of_Power.htm

        Calculation:
            BOP = (close - open) / (high - low)

        Args:
            open (pd.Series): Series of 'open's
            high (pd.Series): Series of 'high's
            low (pd.Series): Series of 'low's
            close (pd.Series): Series of 'close's
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)
            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.bop(offset=offset, **kwargs)

    def cci(self, length=None, c=None, offset=None, **kwargs) -> Series:
        """Commodity Channel Index (CCI)

        Commodity Channel Index is a momentum oscillator used to primarily identify
        overbought and oversold levels relative to a mean.

        Sources:
            https://www.tradingview.com/wiki/Commodity_Channel_Index_(CCI)

        Calculation:
            Default Inputs:
                length=20, c=0.015

            SMA = Simple Moving Average

            MAD = Mean Absolute Deviation

            tp = typical_price = hlc3 = (high + low + close) / 3

            mean_tp = SMA(tp, length)

            mad_tp = MAD(tp, length)

            CCI = (tp - mean_tp) / (c * mad_tp)

        Args:
            high (pd.Series): Series of 'high's
            low (pd.Series): Series of 'low's
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 20
            c (float):  Scaling Constant.  Default: 0.015
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.cci(length=length, c=c, offset=offset, **kwargs)

    def cg(self, length=None, offset=None, **kwargs) -> Series:
        """Center of Gravity (CG)

        The Center of Gravity Indicator by John Ehlers attempts to identify turning
        points while exhibiting zero lag and smoothing.

        Sources:
            http://www.mesasoftware.com/papers/TheCGOscillator.pdf

        Calculation:
            Default Inputs:
                length=10

        Args:
            close (pd.Series): Series of 'close's
            length (int): The length of the period.  Default: 10
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.cg(length=length, offset=offset, **kwargs)

    def cmo(self, length=None, drift=None, offset=None, **kwargs) -> Series:
        """Chande Momentum Oscillator (CMO)

        Attempts to capture the momentum of an asset with overbought at 50 and
        oversold at -50.

        Sources:
            https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/chande-momentum-oscillator-cmo/

        Calculation:
            Default Inputs:
                drift=1

            if close.diff(drift) > 0:
                PSUM = SUM(close - prev_close)

            else:
                NSUM = ABS(SUM(close - prev_close))

            CMO = 100 * (PSUM - NSUM) / (PSUM + NSUM)

        Args:
            close (pd.Series): Series of 'close's
            length (int): The length of the period.  Default: 14
            drift (int): The short period.  Default: 1
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.cmo(length=length, drift=drift, offset=offset, **kwargs)

    def coppock(self, length=None, fast=None, slow=None, offset=None, **kwargs) -> Series:
        """Coppock Curve (COPC)

        Coppock Curve (originally called the "Trendex Model") is a momentum indicator
        is designed for use on a monthly time scale.  Although designed for monthly
        use, a daily calculation over the same period can be made, converting the
        periods to 294-day and 231-day rate of changes, and a 210-day weighted
        moving average.

        Sources:
            https://en.wikipedia.org/wiki/Coppock_curve

        Calculation:
            Default Inputs:
                length=10, fast=11, slow=14

            SMA = Simple Moving Average

            MAD = Mean Absolute Deviation

            tp = typical_price = hlc3 = (high + low + close) / 3

            mean_tp = SMA(tp, length)

            mad_tp = MAD(tp, length)

            CCI = (tp - mean_tp) / (c * mad_tp)

        Args:
            close (pd.Series): Series of 'close's
            length (int): WMA period.  Default: 10
            fast (int): Fast ROC period.  Default: 11
            slow (int): Slow ROC period.  Default: 14
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.coppock(length=length, fast=fast, slow=slow, offset=offset, **kwargs)

    def fisher(self, length=None, offset=None, **kwargs) -> Series:
        """Fisher Transform (FISHT)

        Attempts to identify trend reversals.

        Sources:
            https://tulipindicators.org/fisher

        Calculation:
            Default Inputs:
                length=5

        Args:
            close (pd.Series): Series of 'close's
            length (int): WMA period.  Default: 5
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
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
        """'Know Sure Thing' (KST)

        The 'Know Sure Thing' is a momentum based oscillator and based on ROC.

        Sources:
            https://www.tradingview.com/wiki/Know_Sure_Thing_(KST)
            https://www.incrediblecharts.com/indicators/kst.php

        Calculation:
            Default Inputs:
                roc1=10, roc2=15, roc3=20, roc4=30,
                sma1=10, sma2=10, sma3=10, sma4=15, signal=9, drift=1

            ROC = Rate of Change

            SMA = Simple Moving Average

            rocsma1 = SMA(ROC(close, roc1), sma1)

            rocsma2 = SMA(ROC(close, roc2), sma2)

            rocsma3 = SMA(ROC(close, roc3), sma3)

            rocsma4 = SMA(ROC(close, roc4), sma4)

            KST = 100 * (rocsma1 + 2 * rocsma2 + 3 * rocsma3 + 4 * rocsma4)

            KST_Signal = SMA(KST, signal)

        Args:
            close (pd.Series): Series of 'close's
            roc1 (int): ROC 1 period.  Default: 10
            roc2 (int): ROC 2 period.  Default: 15
            roc3 (int): ROC 3 period.  Default: 20
            roc4 (int): ROC 4 period.  Default: 30
            sma1 (int): SMA 1 period.  Default: 10
            sma2 (int): SMA 2 period.  Default: 10
            sma3 (int): SMA 3 period.  Default: 10
            sma4 (int): SMA 4 period.  Default: 15
            signal (int): It's period.  Default: 9
            drift (int): The difference period.   Default: 1
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.DataFrame: kst and kst_signal columns
        """
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
        """Moving Average Convergence Divergence (MACD)

        The MACD is a popular indicator to that is used to identify a security's trend.
        While APO and MACD are the same calculation, MACD also returns two more series
        called Signal and Histogram.  The Signal is an EMA of MACD and the Histogram is
        the difference of MACD and Signal.

        Sources:
            https://www.tradingview.com/wiki/MACD_(Moving_Average_Convergence/Divergence)

        Calculation:
            Default Inputs:
                fast=12, slow=26, signal=9

            EMA = Exponential Moving Average

            MACD = EMA(close, fast) - EMA(close, slow)

            Signal = EMA(MACD, signal)

            Histogram = MACD - Signal

        Args:
            close (pd.Series): Series of 'close's
            fast (int): The short period.  Default: 12
            slow (int): The long period.   Default: 26
            signal (int): The signal period.   Default: 9
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.DataFrame: macd, histogram, signal columns.
        """
        return self.df.ta.macd(fast=fast, slow=slow, signal=signal, offset=offset, **kwargs)

    def mom(self, length=None, offset=None, **kwargs) -> Series:
        """Momentum (MOM)

        Momentum is an indicator used to measure a security's speed (or strength) of
        movement.  Or simply the change in price.

        Sources:
            http://www.onlinetradingconcepts.com/TechnicalAnalysis/Momentum.html

        Calculation:
            Default Inputs:
                length=1

            MOM = close.diff(length)

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 1
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.mom(length=length, offset=offset, **kwargs)

    def ppo(self, fast=None, slow=None, signal=None, offset=None, **kwargs):
        """Percentage Price Oscillator (PPO)

        The Percentage Price Oscillator is similar to MACD in measuring momentum.

        Sources:
            https://www.tradingview.com/wiki/MACD_(Moving_Average_Convergence/Divergence)

        Calculation:
            Default Inputs:
                fast=12, slow=26

            SMA = Simple Moving Average

            EMA = Exponential Moving Average

            fast_sma = SMA(close, fast)

            slow_sma = SMA(close, slow)

            PPO = 100 * (fast_sma - slow_sma) / slow_sma

            Signal = EMA(PPO, signal)

            Histogram = PPO - Signal

        Args:
            close(pandas.Series): Series of 'close's
            fast(int): The short period.  Default: 12
            slow(int): The long period.   Default: 26
            signal(int): The signal period.   Default: 9
            offset(int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.DataFrame: ppo, histogram, signal columns
        """
        return self.df.ta.ppo(fast=fast, slow=slow, signal=signal, offset=offset, **kwargs)

    def roc(self, length=None, offset=None, **kwargs) -> Series:
        """Rate of Change (ROC)

        Rate of Change is an indicator is also referred to as Momentum (yeah, confusingly).
        It is a pure momentum oscillator that measures the percent change in price with the
        previous price 'n' (or length) periods ago.

        Sources:
            https://www.tradingview.com/wiki/Rate_of_Change_(ROC)

        Calculation:
            Default Inputs:
                length=1

            MOM = Momentum

            ROC = 100 * MOM(close, length) / close.shift(length)

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 1
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.roc(length=length, offset=offset, **kwargs)

    def rsi(self, length=None, drift=None, offset=None, **kwargs) -> Series:
        """Relative Strength Index (RSI)

        The Relative Strength Index is popular momentum oscillator used to measure the
        velocity as well as the magnitude of directional price movements.

        Sources:
            https://www.tradingview.com/wiki/Relative_Strength_Index_(RSI)

        Calculation:
            Default Inputs:
                length=14, drift=1

            ABS = Absolute Value

            EMA = Exponential Moving Average

            positive = close if close.diff(drift) > 0 else 0

            negative = close if close.diff(drift) < 0 else 0

            pos_avg = EMA(positive, length)

            neg_avg = ABS(EMA(negative, length))

            RSI = 100 * pos_avg / (pos_avg + neg_avg)

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 1
            drift (int): The difference period.   Default: 1
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.rsi(length=length, drift=drift, offset=offset, **kwargs)

    def rvi(self, length=None, swma_length=None, offset=None, **kwargs):
        """Relative Vigor Index (RVI)

        The Relative Vigor Index attempts to measure the strength of a trend relative to
        its closing price to its trading range.  It is based on the belief that it tends 
        to close higher than they open in uptrends or close lower than they open in
        downtrends.

        Sources:
            https://www.investopedia.com/terms/r/relative_vigor_index.asp

        Calculation:
            Default Inputs:
                length=14, swma_length=4

            SWMA = Symmetrically Weighted Moving Average

            numerator = SUM(SWMA(close - open, swma_length), length)

            denominator = SUM(SWMA(high - low, swma_length), length)

            RVI = numerator / denominator

        Args:
            open_ (pd.Series): Series of 'open's
            high (pd.Series): Series of 'high's
            low (pd.Series): Series of 'low's
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 14
            swma_length (int): It's period.  Default: 4
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.rvi(length=length, swma_length=swma_length, offset=offset, **kwargs)

    def slope(self, length=None, as_angle=None, to_degrees=None, offset=None, **kwargs):
        """Slope

        Returns the slope of a series of length n.  Can convert the slope to angle, default as 
        radians or degrees.

        Sources: Algebra I

        Calculation:
            Default Inputs:
                length=1

            slope = close.diff(length) / length

            if as_angle:
                slope = slope.apply(atan)
                if to_degrees:
                    slope *= 180 / PI

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 1
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            as_angle (value, optional): Converts slope to an angle.  Default: False

            to_degrees (value, optional): Converts slope angle to degrees.  Default: False

            fillna (value, optional): pd.DataFrame.fillna(value)
            
            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.slope(
            length=length,
            as_angle=as_angle,
            to_degrees=to_degrees,
            offset=offset,
            **kwargs
        )

    def stoch(self, fast_k=None, slow_k=None, slow_d=None, offset=None, **kwargs):
        """Stochastic (STOCH)

        Stochastic Oscillator is a range bound momentum indicator.  It displays the location
        of the close relative to the high-low range over a period.

        Sources:
            https://www.tradingview.com/wiki/Stochastic_(STOCH)

        Calculation:
            Default Inputs:
                fast_k=14, slow_k=5, slow_d=3

            SMA = Simple Moving Average

            lowest_low   = low for last fast_k periods

            highest_high = high for last fast_k periods

            FASTK = 100 * (close - lowest_low) / (highest_high - lowest_low)

            FASTD = SMA(FASTK, slow_d)

            SLOWK = SMA(FASTK, slow_k)

            SLOWD = SMA(SLOWK, slow_d)

        Args:
            high (pd.Series): Series of 'high's
            low (pd.Series): Series of 'low's
            close (pd.Series): Series of 'close's
            fast_k (int): The Fast %K period.  Default: 14
            slow_k (int): The Slow %K period.  Default: 5
            slow_d (int): The Slow %D period.  Default: 3
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.DataFrame: fastk, fastd, slowk, slowd columns.
        """
        return self.df.ta.stoch(fast_k=fast_k, slow_k=slow_k, slow_d=slow_d, offset=offset, **kwargs)

    def trix(self, length=None, drift=None, offset=None, **kwargs):
        """Trix (TRIX)

        TRIX is a momentum oscillator to identify divergences.

        Sources:
            https://www.tradingview.com/wiki/TRIX

        Calculation:
            Default Inputs:
                length=18, drift=1

            EMA = Exponential Moving Average

            ROC = Rate of Change

            ema1 = EMA(close, length)

            ema2 = EMA(ema1, length)

            ema3 = EMA(ema2, length)

            TRIX = 100 * ROC(ema3, drift)

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 18
            drift (int): The difference period.   Default: 1
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.trix(length=length, drift=drift, offset=offset, **kwargs)

    def tsi(self, fast=None, slow=None, drift=None, offset=None, **kwargs):
        """True Strength Index (TSI)

        The True Strength Index is a momentum indicator used to identify short-term
        swings while in the direction of the trend as well as determining overbought
        and oversold conditions.

        Sources:
            https://www.investopedia.com/terms/t/tsi.asp

        Calculation:
            Default Inputs:
                fast=13, slow=25, drift=1

            EMA = Exponential Moving Average

            diff = close.diff(drift)

            slow_ema = EMA(diff, slow)

            fast_slow_ema = EMA(slow_ema, slow)

            abs_diff_slow_ema = absolute_diff_ema = EMA(ABS(diff), slow)

            abema = abs_diff_fast_slow_ema = EMA(abs_diff_slow_ema, fast)

            TSI = 100 * fast_slow_ema / abema

        Args:
            close (pd.Series): Series of 'close's
            fast (int): The short period.  Default: 13
            slow (int): The long period.   Default: 25
            drift (int): The difference period.   Default: 1
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
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
        """Ultimate Oscillator (UO)

        The Ultimate Oscillator is a momentum indicator over three different
        periods.  It attempts to correct false divergence trading signals.

        Sources:
            https://www.tradingview.com/wiki/Ultimate_Oscillator_(UO)

        Calculation:
            Default Inputs:
                fast=7, medium=14, slow=28,
                fast_w=4.0, medium_w=2.0, slow_w=1.0, drift=1

            min_low_or_pc  = close.shift(drift).combine(low, min)

            max_high_or_pc = close.shift(drift).combine(high, max)

            bp = buying pressure = close - min_low_or_pc

            tr = true range = max_high_or_pc - min_low_or_pc

            fast_avg = SUM(bp, fast) / SUM(tr, fast)

            medium_avg = SUM(bp, medium) / SUM(tr, medium)

            slow_avg = SUM(bp, slow) / SUM(tr, slow)

            total_weight = fast_w + medium_w + slow_w

            weights = (fast_w * fast_avg) + (medium_w * medium_avg) + (slow_w * slow_avg)

            UO = 100 * weights / total_weight

        Args:
            high (pd.Series): Series of 'high's
            low (pd.Series): Series of 'low's
            close (pd.Series): Series of 'close's
            fast (int): The Fast %K period.  Default: 7
            medium (int): The Slow %K period.  Default: 14
            slow (int): The Slow %D period.  Default: 28
            fast_w (float): The Fast %K period.  Default: 4.0
            medium_w (float): The Slow %K period.  Default: 2.0
            slow_w (float): The Slow %D period.  Default: 1.0
            drift (int): The difference period.   Default: 1
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
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
        """William's Percent R (WILLR)

        William's Percent R is a momentum oscillator similar to the RSI that
        attempts to identify overbought and oversold conditions.

        Sources:
            https://www.tradingview.com/wiki/Williams_%25R_(%25R)

        Calculation:
            Default Inputs:
                length=20

            lowest_low   = low.rolling(length).min()

            highest_high = high.rolling(length).max()

            WILLR = 100 * ((close - lowest_low) / (highest_high - lowest_low) - 1)

        Args:
            high (pd.Series): Series of 'high's
            low (pd.Series): Series of 'low's
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 14
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.willr(length=length, offset=offset, **kwargs)

    """
    =======
    Overlap
    =======
    """

    def dema(self, length=None, offset=None, **kwargs):
        """Double Exponential Moving Average (DEMA)

        The Double Exponential Moving Average attempts to a smoother average with less
        lag than the normal Exponential Moving Average (EMA).

        Sources:
            https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/double-exponential-moving-average-dema/

        Calculation:
            Default Inputs:
                length=10

            EMA = Exponential Moving Average

            ema1 = EMA(close, length)

            ema2 = EMA(ema1, length)

            DEMA = 2 * ema1 - ema2

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 10
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.dema(length=length, offset=offset, **kwargs)

    def ema(self, length=None, offset=None, **kwargs) -> Series:
        """Exponential Moving Average (EMA)

        The Exponential Moving Average is more responsive moving average compared to the
        Simple Moving Average (SMA).  The weights are determined by alpha which is
        proportional to it's length.  There are several different methods of calculating
        EMA.  One method uses just the standard definition of EMA and another uses the
        SMA to generate the initial value for the rest of the calculation.

        Sources:
            https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_averages
            https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp

        Calculation:
            Default Inputs:
                length=10
            SMA = Simple Moving Average
            if kwargs['presma']:
                initial = SMA(close, length)
                rest = close[length:]
                close = initial + rest

            EMA = close.ewm(span=length, adjust=adjust).mean()

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 10
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            adjust (bool, optional): Default: True
            sma (bool, optional): If True, uses SMA for initial value.
            fillna (value, optional): pd.DataFrame.fillna(value)
            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.ema(length=length, offset=offset, **kwargs)

    def fwma(self, length=None, asc=None, offset=None, **kwargs):
        """Fibonacci's Weighted Moving Average (FWMA)

        Fibonacci's Weighted Moving Average is similar to a Weighted Moving Average
        (WMA) where the weights are based on the Fibonacci Sequence.

        Source: Kevin Johnson

        Calculation:
            Default Inputs:
                length=10, 

            def weights(w):
                def _compute(x):
                    return np.dot(w * x)
                return _compute

            fibs = utils.fibonacci(length - 1)

            FWMA = close.rolling(length)_.apply(weights(fibs), raw=True)

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 10
            asc (bool): Recent values weigh more.  Default: True
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.fwma(length=length, asc=asc, offset=offset, **kwargs)

    def hl2(self, offset=None, **kwargs):
        """Indicator: HL2 """
        return self.df.ta.hl2(offset=offset, **kwargs)

    def hlc3(self, offset=None, **kwargs):
        """Indicator: HLC3"""
        return self.df.ta.hlc3(offset=offset, **kwargs)

    def hma(self, length=None, offset=None, **kwargs):
        """Hull Moving Average (HMA)

        The Hull Exponential Moving Average attempts to reduce or remove lag in moving
        averages.

        Sources:
            https://alanhull.com/hull-moving-average

        Calculation:
            Default Inputs:
                length=10

            WMA = Weighted Moving Average

            half_length = int(0.5 * length)

            sqrt_length = int(math.sqrt(length))

            wmaf = WMA(close, half_length)

            wmas = WMA(close, length)

            HMA = WMA(2 * wmaf - wmas, sqrt_length)

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 10
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.hma(length=length, offset=offset, **kwargs)

    def ichimoku(self, tenkan=None, kijun=None, senkou=None, offset=None, **kwargs):
        """Ichimoku Kinkō Hyō (ichimoku)

        Developed Pre WWII as a forecasting model for financial markets.

        Sources:
            https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/ichimoku-ich/

        Calculation:
            Default Inputs:
                tenkan=9, kijun=26, senkou=52

            MIDPRICE = Midprice

            TENKAN_SEN = MIDPRICE(high, low, close, length=tenkan)

            KIJUN_SEN = MIDPRICE(high, low, close, length=kijun)

            CHIKOU_SPAN = close.shift(-kijun)

            SPAN_A = 0.5 * (TENKAN_SEN + KIJUN_SEN)

            SPAN_A = SPAN_A.shift(kijun)

            SPAN_B = MIDPRICE(high, low, close, length=senkou)

            SPAN_B = SPAN_B.shift(kijun)

        Args:
            high (pd.Series): Series of 'high's
            low (pd.Series): Series of 'low's
            close (pd.Series): Series of 'close's
            tenkan (int): Tenkan period.  Default: 9
            kijun (int): Kijun period.  Default: 26
            senkou (int): Senkou period.  Default: 52
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.DataFrame: Two DataFrames.
                For the visible period: spanA, spanB, tenkan_sen, kijun_sen,
                    and chikou_span columns
                For the forward looking period: spanA and spanB columns
        """
        return self.df.ta.ichimoku(tenkan=tenkan, kijun=kijun, senkou=senkou, offset=offset, **kwargs)

    def kama(self, length=None, fast=None, slow=None, drift=None, offset=None, **kwargs):
        """Kaufman's Adaptive Moving Average (KAMA)

        Developed by Perry Kaufman, Kaufman's Adaptive Moving Average (KAMA) is a moving average
        designed to account for market noise or volatility. KAMA will closely follow prices when
        the price swings are relatively small and the noise is low. KAMA will adjust when the
        price swings widen and follow prices from a greater distance. This trend-following indicator
        can be used to identify the overall trend, time turning points and filter price movements.

        Sources:
            https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:kaufman_s_adaptive_moving_average

        Calculation:
            Default Inputs:
                length=10

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 10
            fast(int): The short period.  Default: 2
            slow(int): The long period.   Default: 30
            drift (int): The difference period.   Default: 1
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.kama(length=length, fast=fast, slow=slow, drift=drift, offset=offset, **kwargs)

    def linreg(self, length=None, offset=None, **kwargs):
        """Linear Regression Moving Average (linreg)

        Linear Regression Moving Average

        Source: TA Lib

        Calculation:
            Default Inputs:
                length=14

            x = [1, 2, ..., n]

            x_sum = 0.5 * length * (length + 1)

            x2_sum = length * (length + 1) * (2 * length + 1) / 6

            divisor = length * x2_sum - x_sum * x_sum

            lr(series):
                y_sum = series.sum()

                y2_sum = (series* series).sum()

                xy_sum = (x * series).sum()

                m = (length * xy_sum - x_sum * y_sum) / divisor

                b = (y_sum * x2_sum - x_sum * xy_sum) / divisor

                return m * (length - 1) + b

            linreg = close.rolling(length).apply(lr)

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 10
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

            angle (bool, optional): Default: False.  If True, returns the angle of the slope in radians
            
            degrees (bool, optional): Default: False.  If True, returns the angle of the slope in degrees
            
            intercept (bool, optional): Default: False.  If True, returns the angle of the slope in radians
            
            r (bool, optional): Default: False.  If True, returns it's correlation 'r'
            
            slope (bool, optional): Default: False.  If True, returns the slope
            
            tsf (bool, optional): Default: False.  If True, returns the Time Series Forecast value.

        Returns:
            pd.Series: New feature generated.
        """
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
        """Pascal's Weighted Moving Average (PWMA)

        Pascal's Weighted Moving Average is similar to a symmetric triangular
        window except PWMA's weights are based on Pascal's Triangle.

        Source: Kevin Johnson

        Calculation:
            Default Inputs:
                length=10

            def weights(w):
                def _compute(x):
                    return np.dot(w * x)
                return _compute

            triangle = utils.pascals_triangle(length + 1)

            PWMA = close.rolling(length)_.apply(weights(triangle), raw=True)

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 10
            asc (bool): Recent values weigh more.  Default: True
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.pwma(length=length, asc=asc, offset=offset, **kwargs)

    def rma(self, length=None, offset=None, **kwargs):
        """wildeR's Moving Average (RMA)

        The WildeR's Moving Average is simply an Exponential Moving Average (EMA)
        with a modified alpha = 1 / length.

        Sources:
            https://alanhull.com/hull-moving-average

        Calculation:
            Default Inputs:
                length=10

            EMA = Exponential Moving Average

            alpha = 1 / length

            RMA = EMA(close, alpha=alpha)

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 10
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.rma(length=length, offset=offset, **kwargs)

    def sinwma(self, length=None, offset=None, **kwargs):
        """Sine Weighted Moving Average (SWMA)

        A weighted average using sine cycles.  The middle term(s) of the average have the highest
        weight(s).

        Source:
            https://www.tradingview.com/script/6MWFvnPO-Sine-Weighted-Moving-Average/
            Author: Everget (https://www.tradingview.com/u/everget/)

        Calculation:
            Default Inputs:
                length=10

            def weights(w):
                def _compute(x):
                    return np.dot(w * x)
                return _compute

            sines = Series([sin((i + 1) * pi / (length + 1)) for i in range(0, length)])

            w = sines / sines.sum()

            SINWMA = close.rolling(length, min_periods=length).apply(weights(w), raw=True)

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 10
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.sinwma(length=length, offset=offset, **kwargs)

    def sma(self, length=None, offset=None, **kwargs) -> Series:
        """Simple Moving Average (SMA)

        The Simple Moving Average is the classic moving average that is the equally
        weighted average over n periods.

        Sources:
            https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/simple-moving-average-sma/

        Calculation:
            Default Inputs:
                length=10

            SMA = SUM(close, length) / length

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 10
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            adjust (bool): Default: True

            presma (bool, optional): If True, uses SMA for initial value.

            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.ema(length=length, offset=offset, **kwargs)

    def swma(self, length=None, asc=None, offset=None, **kwargs):
        """Symmetric Weighted Moving Average (SWMA)

        Symmetric Weighted Moving Average where weights are based on a symmetric
        triangle.  For example: n=3 -> [1, 2, 1], n=4 -> [1, 2, 2, 1], etc...  This moving
        average has variable length in contrast to TradingView's fixed length of 4.

        Source: 
            https://www.tradingview.com/study-script-reference/#fun_swma

        Calculation:
            Default Inputs:
                length=10

            def weights(w):
                def _compute(x):
                    return np.dot(w * x)
                return _compute

            triangle = utils.symmetric_triangle(length - 1)
            SWMA = close.rolling(length)_.apply(weights(triangle), raw=True)

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 10
            asc (bool): Recent values weigh more.  Default: True
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)
            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.swma(length=length, asc=asc, offset=offset, **kwargs)

    def t3(self, length=None, a=None, offset=None, **kwargs):
        """Tim Tillson's T3 Moving Average (T3)

        Tim Tillson's T3 Moving Average is considered a smoother and more responsive
        moving average relative to other moving averages.

        Sources:
            http://www.binarytribune.com/forex-trading-indicators/t3-moving-average-indicator/

        Calculation:
            Default Inputs:
                length=10, a=0.7
            c1 = -a^3
            c2 = 3a^2 + 3a^3 = 3a^2 * (1 + a)
            c3 = -6a^2 - 3a - 3a^3
            c4 = a^3 + 3a^2 + 3a + 1

            ema1 = EMA(close, length)
            ema2 = EMA(ema1, length)
            ema3 = EMA(ema2, length)
            ema4 = EMA(ema3, length)
            ema5 = EMA(ema4, length)
            ema6 = EMA(ema5, length)
            T3 = c1 * ema6 + c2 * ema5 + c3 * ema4 + c4 * ema3

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 10
            a (float): 0 < a < 1.  Default: 0.7
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            adjust (bool): Default: True
            presma (bool, optional): If True, uses SMA for initial value.
            fillna (value, optional): pd.DataFrame.fillna(value)
            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.t3(length=length, a=a, offset=offset, **kwargs)

    def tema(self, length=None, offset=None, **kwargs):
        """Triple Exponential Moving Average (TEMA)

        A less laggy Exponential Moving Average.

        Sources:
            https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/triple-exponential-moving-average-tema/

        Calculation:
            Default Inputs:
                length=10

            EMA = Exponential Moving Average

            ema1 = EMA(close, length)

            ema2 = EMA(ema1, length)

            ema3 = EMA(ema2, length)

            TEMA = 3 * (ema1 - ema2) + ema3

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 10
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            adjust (bool): Default: True

            presma (bool, optional): If True, uses SMA for initial value.

            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.tema(length=length, offset=offset, **kwargs)

    def trima(self, length=None, offset=None, **kwargs):
        """Triangular Moving Average (TRIMA)

        A weighted moving average where the shape of the weights are triangular and the
        greatest weight is in the middle of the period.

        Sources:
            https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/triangular-moving-average-trima/
            tma = sma(sma(src, ceil(length / 2)), floor(length / 2) + 1)  # Tradingview
            trima = sma(sma(x, n), n)  # Tradingview

        Calculation:
            Default Inputs:
                length=10

            SMA = Simple Moving Average

            half_length = math.round(0.5 * (length + 1))

            SMA1 = SMA(close, half_length)

            TRIMA = SMA(SMA1, half_length)

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 10
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            adjust (bool): Default: True

            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.trima(length=length, offset=offset, **kwargs)

    def vwap(self, offset=None, **kwargs):
        """Volume Weighted Average Price (VWAP)

        The Volume Weighted Average Price that measures the average typical price
        by volume.  It is typically used with intraday charts to identify general
        direction.

        Sources:
            https://www.tradingview.com/wiki/Volume_Weighted_Average_Price_(VWAP)
            https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/volume-weighted-average-price-vwap/

        Calculation:
            tp = typical_price = hlc3(high, low, close)

            tpv = tp * volume

            VWAP = tpv.cumsum() / volume.cumsum()

        Args:
            high (pd.Series): Series of 'high's
            low (pd.Series): Series of 'low's
            close (pd.Series): Series of 'close's
            volume (pd.Series): Series of 'volume's
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.vwap(offset=offset, **kwargs)

    def vwma(self, length=None, offset=None, **kwargs) -> Series:
        """Volume Weighted Moving Average (VWMA)

        Volume Weighted Moving Average.

        Sources:
            https://www.motivewave.com/studies/volume_weighted_moving_average.htm

        Calculation:
            Default Inputs:
                length=10

            SMA = Simple Moving Average

            pv = close * volume

            VWMA = SMA(pv, length) / SMA(volume, length)

        Args:
            close (pd.Series): Series of 'close's
            volume (pd.Series): Series of 'volume's
            length (int): It's period.  Default: 10
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.vwma(length=length, offset=offset, **kwargs)

    def wma(self, length=None, asc=None, offset=None, **kwargs):
        """Weighted Moving Average (WMA)

        The Weighted Moving Average where the weights are linearly increasing and
        the most recent data has the heaviest weight.

        Sources:
            https://en.wikipedia.org/wiki/Moving_average#Weighted_moving_average

        Calculation:
            Default Inputs:
                length=10, asc=True

            total_weight = 0.5 * length * (length + 1)

            weights_ = [1, 2, ..., length + 1]  # Ascending

            weights = weights if asc else weights[::-1]

            def linear_weights(w):
                def _compute(x):
                    return (w * x).sum() / total_weight
                return _compute

            WMA = close.rolling(length)_.apply(linear_weights(weights), raw=True)

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 10
            asc (bool): Recent values weigh more.  Default: True
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.wma(length=length, asc=asc, offset=offset, **kwargs)

    def zlma(self, length=None, offset=None, mamode=None, **kwargs):
        """Zero Lag Moving Average (ZLMA)

        The Zero Lag Moving Average attempts to eliminate the lag associated
        with moving averages.  This is an adaption created by John Ehler and Ric Way.

        Sources:
            https://en.wikipedia.org/wiki/Zero_lag_exponential_moving_average

        Calculation:
            Default Inputs:
                length=10, mamode=EMA

            EMA = Exponential Moving Average

            lag = int(0.5 * (length - 1))

            source = 2 * close - close.shift(lag)

            ZLMA = EMA(source, length)

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 10
            mamode (str): Two options: None or 'ema'.  Default: 'ema'
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
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
        """Log Return

        Calculates the logarithmic return of a Series.
        See also: help(df.ta.log_return) for additional **kwargs a valid 'df'.

        Sources:
            https://stackoverflow.com/questions/31287552/logarithmic-returns-in-pandas-dataframe

        Calculation:
            Default Inputs:
                length=1, cumulative=False

            LOGRET = log( close.diff(periods=length) )

            CUMLOGRET = LOGRET.cumsum() if cumulative

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 20
            cumulative (bool): If True, returns the cumulative returns.  Default: False
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
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
        """Percent Return

        Calculates the percent return of a Series.
        See also: help(df.ta.percent_return) for additional **kwargs a valid 'df'.

        Sources:
            https://stackoverflow.com/questions/31287552/logarithmic-returns-in-pandas-dataframe

        Calculation:
            Default Inputs:
                length=1, cumulative=False

            PCTRET = close.pct_change(length)

            CUMPCTRET = PCTRET.cumsum() if cumulative

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 20
            cumulative (bool): If True, returns the cumulative returns.  Default: False
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
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
        """Trend Return

        Calculates the (Cumulative) Returns of a Trend as defined by some conditional.
        By default it calculates log returns but can also use percent change.

        Sources: Kevin Johnson

        Calculation:
            Default Inputs:
                trend_reset=0, log=True, cumulative=False

            sum = 0

            returns = log_return if log else percent_return # These are not cumulative

            returns = (trend * returns).apply(zero)

            for i, in range(0, trend.size):
                if item == trend_reset:
                    sum = 0
                else:
                    return_ = returns.iloc[i]
                    if cumulative:
                        sum += return_
                    else:
                        sum = return_
                trend_return.append(sum)

            if cumulative and variable:
                trend_return += returns

        Args:
            close (pd.Series): Series of 'close's
            trend (pd.Series): Series of 'trend's.  Preferably 0's and 1's.
            trend_reset (value): Value used to identify if a trend has ended.  Default: 0
            log (bool): Calculate logarithmic returns.  Default: True
            cumulative (bool): If True, returns the cumulative returns.  Default: False
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

            variable (bool, optional): Whether to include if return fluxuations in the cumulative returns.

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.trend_return(
            trend, log=log, cumulative=cumulative, offset=offset, trend_reset=trend_reset, **kwargs
        )

    """
    ==========
    Statistics
    ==========
    """

    def kurtosis(self, length=None, offset=None, **kwargs):
        """Rolling Kurtosis

        Sources:

        Calculation:
            Default Inputs:
                length=30

            KURTOSIS = close.rolling(length).kurt()

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 30
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.kurtosis(length=length, offset=offset, **kwargs)

    def mad(self, length=None, offset=None, **kwargs):
        """Rolling Mean Absolute Deviation

        Sources:

        Calculation:
            Default Inputs:
                length=30

            mad = close.rolling(length).mad()

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 30
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.mad(length=length, offset=offset, **kwargs)

    def median(self, length=None, offset=None, **kwargs):
        """Rolling Median

        Rolling Median of over 'n' periods.  Sibling of a Simple Moving Average.

        Sources:
            https://www.incrediblecharts.com/indicators/median_price.php

        Calculation:
            Default Inputs:
                length=30

            MEDIAN = close.rolling(length).median()

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 30
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.median(length=length, offset=offset, **kwargs)

    def quantile(self, length=None, q=None, offset=None, **kwargs):
        """Rolling Quantile

        Sources:

        Calculation:
            Default Inputs:
                length=30, q=0.5

            QUANTILE = close.rolling(length).quantile(q)

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 30
            q (float): The quantile.  Default: 0.5
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.quantile(length=length, q=q, offset=offset, **kwargs)

    def skew(self, length=None, offset=None, **kwargs):
        """Rolling Skew

        Sources:

        Calculation:
            Default Inputs:
                length=30

            SKEW = close.rolling(length).skew()

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 30
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """

        return self.df.ta.skew(length=length, offset=offset, **kwargs)

    def stdev(self, length=None, offset=None, **kwargs):
        """Rolling Standard Deviation

        Sources:

        Calculation:
            Default Inputs:
                length=30

            VAR = Variance

            STDEV = variance(close, length).apply(np.sqrt)

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 30
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)
            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.stdev(length=length, offset=offset, **kwargs)

    def variance(self, length=None, offset=None, **kwargs):
        """Rolling Variance

        Sources:

        Calculation:
            Default Inputs:
                length=30

            VARIANCE = close.rolling(length).var()

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 30
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)
            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.variance(length=length, offset=offset, **kwargs)

    def zscore(self, length=None, std=None, offset=None, **kwargs):
        """Rolling Z Score

        Sources:

        Calculation:
            Default Inputs:
                length=30, std=1

            SMA = Simple Moving Average

            STDEV = Standard Deviation

            std = std * STDEV(close, length)

            mean = SMA(close, length)

            ZSCORE = (close - mean) / std

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 30
            std (float): It's period.  Default: 1
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.zscore(length=length, std=std, offset=offset, **kwargs)

    """
    =====
    Trend
    =====
    """

    def adx(self, length=None, drift=None, offset=None, **kwargs):
        """Average Directional Movement (ADX)

        Average Directional Movement is meant to quantify trend strength by measuring
        the amount of movement in a single direction.

        Sources:
            https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/average-directional-movement-adx/

        Calculation:
            DMI ADX TREND 2.0 by @TraderR0BERT, NETWORTHIE.COM
                //Created by @TraderR0BERT, NETWORTHIE.COM, last updated 01/26/2016
                //DMI Indicator
                //Resolution input option for higher/lower time frames
                study(title="DMI ADX TREND 2.0", shorttitle="ADX TREND 2.0")

                adxlen = input(14, title="ADX Smoothing")

                dilen = input(14, title="DI Length")

                thold = input(20, title="Threshold")

                threshold = thold

                //Script for Indicator
                dirmov(len) =>

                    up = change(high)

                    down = -change(low)

                    truerange = rma(tr, len)

                    plus = fixnan(100 * rma(up > down and up > 0 ? up : 0, len) / truerange)

                    minus = fixnan(100 * rma(down > up and down > 0 ? down : 0, len) / truerange)

                    [plus, minus]

                adx(dilen, adxlen) =>

                    [plus, minus] = dirmov(dilen)

                    sum = plus + minus

                    adx = 100 * rma(abs(plus - minus) / (sum == 0 ? 1 : sum), adxlen)

                    [adx, plus, minus]

                [sig, up, down] = adx(dilen, adxlen)
                
                osob=input(40,title="Exhaustion Level for ADX, default = 40")

                col = sig >= sig[1] ? green : sig <= sig[1] ? red : gray

                //Plot Definitions Current Timeframe
                p1 = plot(sig, color=col, linewidth = 3, title="ADX")

                p2 = plot(sig, color=col, style=circles, linewidth=3, title="ADX")

                p3 = plot(up, color=blue, linewidth = 3, title="+DI")

                p4 = plot(up, color=blue, style=circles, linewidth=3, title="+DI")

                p5 = plot(down, color=fuchsia, linewidth = 3, title="-DI")

                p6 = plot(down, color=fuchsia, style=circles, linewidth=3, title="-DI")

                h1 = plot(threshold, color=black, linewidth =3, title="Threshold")

                trender = (sig >= up or sig >= down) ? 1 : 0

                bgcolor(trender>0?black:gray, transp=85)

                //Alert Function for ADX crossing Threshold

                Up_Cross = crossover(up, threshold)

                alertcondition(Up_Cross, title="DMI+ cross", message="DMI+ Crossing Threshold")

                Down_Cross = crossover(down, threshold)

                alertcondition(Down_Cross, title="DMI- cross", message="DMI- Crossing Threshold")

        Args:
            high (pd.Series): Series of 'high's
            low (pd.Series): Series of 'low's
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 14
            drift (int): The difference period.   Default: 1
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.DataFrame: adx, dmp, dmn columns.
        """
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
        """Aroon (AROON)

        Aroon attempts to identify if a security is trending and how strong.

        Sources:
            https://www.tradingview.com/wiki/Aroon
            https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/aroon-ar/

        Calculation:
            Default Inputs:
                length=1

            def maxidx(x):
                return 100 * (int(np.argmax(x)) + 1) / length

            def minidx(x):
                return 100 * (int(np.argmin(x)) + 1) / length

            _close = close.rolling(length, min_periods=min_periods)

            aroon_up = _close.apply(maxidx, raw=True)

            aroon_down = _close.apply(minidx, raw=True)

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 1
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)
            
            fill_method (value, optional): Type of fill method

        Returns:
            pd.DataFrame: aroon_up, aroon_down columns.
        """
        return self.df.ta.aroon(length=length, offset=offset, **kwargs)

    def decreasing(self, length=None, asint=True, offset=None, **kwargs):
        """Decreasing

        Returns True or False if the series is decreasing over a periods.  By default,
        it returns True and False as 1 and 0 respectively with kwarg 'asint'.

        Sources:

        Calculation:
            decreasing = close.diff(length) < 0

            if asint:
                decreasing = decreasing.astype(int)

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 1
            asint (bool): Returns as binary.  Default: True
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.decreasing(length=length, asint=asint, offset=offset, **kwargs)

    def dpo(self, length=None, centered=True, offset=None, **kwargs):
        """Detrend Price Oscillator (DPO)

        Is an indicator designed to remove trend from price and make it easier to
        identify cycles.

        Sources:
            http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:detrended_price_osci

        Calculation:
            Default Inputs:
                length=1, centered=True

            SMA = Simple Moving Average

            drift = int(0.5 * length) + 1
            
            DPO = close.shift(drift) - SMA(close, length)

            if centered:
                DPO = DPO.shift(-drift)

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 1
            centered (bool): Shift the dpo back by int(0.5 * length) + 1.  Default: True
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.dpo(length=length, centered=centered, offset=offset, **kwargs)

    def increasing(self, length=None, asint=True, offset=None, **kwargs):
        """Increasing

        Returns True or False if the series is increasing over a periods.  By default,
        it returns True and False as 1 and 0 respectively with kwarg 'asint'.

        Sources:

        Calculation:
            increasing = close.diff(length) > 0
            if asint:
                increasing = increasing.astype(int)

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 1
            asint (bool): Returns as binary.  Default: True
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)
            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.increasing(length=length, asint=asint, offset=offset, **kwargs)

    def linear_decay(self, length=None, offset=None, **kwargs):
        """Linear Decay

        Adds a linear decay moving forward from prior signals like crosses.

        Sources:
            https://tulipindicators.org/decay

        Calculation:
            Default Inputs:
                length=5
            max(close, close[-1] - (1 / length), 0)

        Args:
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 1
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.linear_decay(length=length, offset=offset, **kwargs)

    def long_run(self, fast: Series, slow: Series, length=None, offset=None, **kwargs):
        """Indicator: Long Run"""
        return self.df.ta.long_run(fast, slow, length=length, offset=offset, **kwargs)

    def qstick(self, length=None, offset=None, **kwargs):
        """Q Stick

        The Q Stick indicator, developed by Tushar Chande, attempts to quantify and identify
        trends in candlestick charts.

        Sources:
            https://library.tradingtechnologies.com/trade/chrt-ti-qstick.html

        Calculation:
            Default Inputs:
                length=10

            xMA is one of: sma (default), dema, ema, hma, rma

            qstick = xMA(close - open, length)

        Args:
            open (pd.Series): Series of 'open's
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 1
            ma (str): The type of moving average to use.  Default: None, which is 'sma'
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.qstick(length=length, offset=offset, **kwargs)

    def short_run(self, fast: Series, slow: Series, length=None, offset=None, **kwargs):
        """Indicator: Short Run"""
        return self.df.ta.short_run(fast, slow, length=length, offset=offset, **kwargs)

    def vortex(self, length=None, drift=None, offset=None, **kwargs):
        """Vortex

        Two oscillators that capture positive and negative trend movement.

        Sources:
            https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:vortex_indicator

        Calculation:
            Default Inputs:
                length=14, drift=1

            TR = True Range

            SMA = Simple Moving Average

            tr = TR(high, low, close)

            tr_sum = tr.rolling(length).sum()

            vmp = (high - low.shift(drift)).abs()

            vmn = (low - high.shift(drift)).abs()

            VIP = vmp.rolling(length).sum() / tr_sum

            VIM = vmn.rolling(length).sum() / tr_sum

        Args:
            high (pd.Series): Series of 'high's
            low (pd.Series): Series of 'low's
            close (pd.Series): Series of 'close's
            length (int): ROC 1 period.  Default: 14
            drift (int): The difference period.   Default: 1
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.DataFrame: vip and vim columns
        """
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
        """Acceleration Bands (ACCBANDS)

        Acceleration Bands created by Price Headley plots upper and lower envelope
        bands around a simple moving average.

        Sources:
            https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/acceleration-bands-abands/

        Calculation:
            Default Inputs:
                length=10, c=4

            EMA = Exponential Moving Average

            SMA = Simple Moving Average

            HL_RATIO = c * (high - low) / (high + low)

            LOW = low * (1 - HL_RATIO)

            HIGH = high * (1 + HL_RATIO)

            if 'ema':
                LOWER = EMA(LOW, length)

                MID = EMA(close, length)

                UPPER = EMA(HIGH, length)
            else:
                LOWER = SMA(LOW, length)

                MID = SMA(close, length)

                UPPER = SMA(HIGH, length)

        Args:
            high (pd.Series): Series of 'high's
            low (pd.Series): Series of 'low's
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 10
            c (int): Multiplier.  Default: 4
            mamode (str): Two options: None or 'ema'.  Default: 'ema'
            drift (int): The difference period.   Default: 1
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.DataFrame: lower, mid, upper columns.
        """
        return self.df.ta.accbands(
            length=length,
            c=c,
            drift=drift,
            mamode=mamode,
            offset=offset,
            **kwargs
        )

    def atr(self, length=None, mamode=None, offset=None, **kwargs) -> Series:
        """Average True Range (ATR)

        Averge True Range is used to measure volatility, especially
        volatility caused by gaps or limit moves.

        Sources:
            https://www.tradingview.com/wiki/Average_True_Range_(ATR)

        Calculation:
            Default Inputs:
                length=14, drift=1

            SMA = Simple Moving Average

            EMA = Exponential Moving Average

            TR = True Range

            tr = TR(high, low, close, drift)

            if 'ema':
                ATR = EMA(tr, length)
            else:
                ATR = SMA(tr, length)

        Args:
            high (pd.Series): Series of 'high's
            low (pd.Series): Series of 'low's
            close (pd.Series): Series of 'close's
            length (int): It's period.  Default: 14
            mamode (str): Two options: None or 'ema'.  Default: 'ema'
            drift (int): The difference period.   Default: 1
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.atr(length=length, mamode=mamode, offset=offset, **kwargs)

    def bbands(self, length=None, std=None, mamode=None, offset=None, **kwargs):
        """Bollinger Bands (BBANDS)

        A popular volatility indicator.

        Sources:
            https://www.tradingview.com/wiki/Bollinger_Bands_(BB)

        Calculation:
            Default Inputs:
                length=20, std=2

            EMA = Exponential Moving Average

            SMA = Simple Moving Average

            STDEV = Standard Deviation

            stdev = STDEV(close, length)

            if 'ema':
                MID = EMA(close, length)
            else:
                MID = SMA(close, length)
            
            LOWER = MID - std * stdev

            UPPER = MID + std * stdev

        Args:
            close (pd.Series): Series of 'close's
            length (int): The short period.  Default: 20
            std (int): The long period.   Default: 2
            mamode (str): Two options: None or 'ema'.  Default: 'ema'
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.DataFrame: lower, mid, upper columns.
        """
        return self.df.ta.bbands(length=length, std=std, mamode=mamode, offset=offset, **kwargs)

    def donchian(self, lower_length=None, upper_length=None, offset=None, **kwargs):
        """Donchian Channels (DC)

        Donchian Channels are used to measure volatility, similar to 
        Bollinger Bands and Keltner Channels.

        Sources:
            https://www.tradingview.com/wiki/Donchian_Channels_(DC)

        Calculation:
            Default Inputs:
                length=20

            LOWER = close.rolling(length).min()

            UPPER = close.rolling(length).max()

            MID = 0.5 * (LOWER + UPPER)

        Args:
            close (pd.Series): Series of 'close's
            lower_length (int): The short period.  Default: 10
            upper_length (int): The long period.  Default: 20
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.DataFrame: lower, mid, upper columns.
        """
        return self.df.ta.donchian(
            lower_length=lower_length,
            upper_length=upper_length,
            offset=offset,
            **kwargs
        )

    def kc(self, length=None, scalar=None, mamode=None, offset=None, **kwargs):
        """Keltner Channels (KC)

        A popular volatility indicator similar to Bollinger Bands and
        Donchian Channels.

        Sources:
            https://www.tradingview.com/wiki/Keltner_Channels_(KC)

        Calculation:
            Default Inputs:
                length=20, scalar=2

            ATR = Average True Range

            EMA = Exponential Moving Average

            SMA = Simple Moving Average

            if 'ema':
                BASIS = EMA(close, length)

                BAND = ATR(high, low, close)
            else:
                hl_range = high - low

                tp = typical_price = hlc3(high, low, close)

                BASIS = SMA(tp, length)

                BAND = SMA(hl_range, length)
            
            LOWER = BASIS - scalar * BAND

            UPPER = BASIS + scalar * BAND

        Args:
            high (pd.Series): Series of 'high's
            low (pd.Series): Series of 'low's
            close (pd.Series): Series of 'close's
            length (int): The short period.  Default: 20
            scalar (float): A positive float to scale the bands.   Default: 2
            mamode (str): Two options: None or 'ema'.  Default: 'ema'
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.DataFrame: lower, basis, upper columns.
        """
        return self.df.ta.kc(length=length, scalar=scalar, mamode=mamode, offset=offset, **kwargs)

    def massi(self, fast=None, slow=None, offset=None, **kwargs):
        """Mass Index (MASSI)

        The Mass Index is a non-directional volatility indicator that utilitizes the
        High-Low Range to identify trend reversals based on range expansions.

        Sources:
            https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:mass_index
            mi = sum(ema(high - low, 9) / ema(ema(high - low, 9), 9), length)

        Calculation:
            Default Inputs:
                fast: 9, slow: 25

            EMA = Exponential Moving Average

            hl = high - low

            hl_ema1 = EMA(hl, fast)

            hl_ema2 = EMA(hl_ema1, fast)

            hl_ratio = hl_ema1 / hl_ema2

            MASSI = SUM(hl_ratio, slow)

        Args:
            high (pd.Series): Series of 'high's
            low (pd.Series): Series of 'low's
            fast (int): The short period.  Default: 9
            slow (int): The long period.   Default: 25
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.massi(fast=fast, slow=slow, offset=offset, **kwargs)

    def natr(self, length=None, mamode=None, drift=None, offset=None, **kwargs):
        """Normalized Average True Range (NATR)

        Normalized Average True Range attempt to normalize the average
        true range.

        Sources:
            https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/normalized-average-true-range-natr/

        Calculation:
            Default Inputs:
                length=20

            ATR = Average True Range

            NATR = (100 / close) * ATR(high, low, close)

        Args:
            high (pd.Series): Series of 'high's
            low (pd.Series): Series of 'low's
            close (pd.Series): Series of 'close's
            length (int): The short period.  Default: 20
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature
        """
        return self.df.ta.natr(length=length, mamode=mamode, drift=drift, offset=offset, **kwargs)

    def true_range(self, drift=None, offset=None, **kwargs):
        """True Range

        An method to expand a classical range (high minus low) to include
        possible gap scenarios.

        Sources:
            https://www.macroption.com/true-range/

        Calculation:
            Default Inputs:
                drift=1

            ABS = Absolute Value

            prev_close = close.shift(drift)

            TRUE_RANGE = ABS([high - low, high - prev_close, low - prev_close]) 

        Args:
            high (pd.Series): Series of 'high's
            low (pd.Series): Series of 'low's
            close (pd.Series): Series of 'close's
            drift (int): The shift period.   Default: 1
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature
        """
        return self.df.ta.true_range(drift=drift, offset=offset, **kwargs)

    """
    ======
    Volume
    ======
    """

    def ad(self, offset=None, **kwargs):
        """Accumulation/Distribution (AD)

        Accumulation/Distribution indicator utilizes the relative position
        of the close to it's High-Low range with volume.  Then it is cumulated.

        Sources:
            https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/accumulationdistribution-ad/

        Calculation:
            CUM = Cumulative Sum
            if 'open':
                AD = close - open
            else:
                AD = 2 * close - high - low

            hl_range = high - low
            AD = AD * volume / hl_range
            AD = CUM(AD)

        Args:
            high (pd.Series): Series of 'high's
            low (pd.Series): Series of 'low's
            close (pd.Series): Series of 'close's
            volume (pd.Series): Series of 'volume's
            open (pd.Series): Series of 'open's
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)
            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.ad(offset=offset, **kwargs)

    def adosc(self, fast=None, slow=None, offset=None, **kwargs):
        """Accumulation/Distribution Oscillator or Chaikin Oscillator

        Accumulation/Distribution Oscillator indicator utilizes 
        Accumulation/Distribution and treats it similarily to MACD
        or APO.

        Sources:
            https://www.investopedia.com/articles/active-trading/031914/understanding-chaikin-oscillator.asp

        Calculation:
            Default Inputs:
                fast=12, slow=26

            AD = Accum/Dist

            ad = AD(high, low, close, open)

            fast_ad = EMA(ad, fast)

            slow_ad = EMA(ad, slow)

            ADOSC = fast_ad - slow_ad

        Args:
            high (pd.Series): Series of 'high's
            low (pd.Series): Series of 'low's
            close (pd.Series): Series of 'close's
            open (pd.Series): Series of 'open's
            volume (pd.Series): Series of 'volume's
            fast (int): The short period.  Default: 12
            slow (int): The long period.   Default: 26
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
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
        """Chaikin Money Flow (CMF)

        Chailin Money Flow measures the amount of money flow volume over a specific
        period in conjunction with Accumulation/Distribution.

        Sources:
            https://www.tradingview.com/wiki/Chaikin_Money_Flow_(CMF)
            https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf

        Calculation:
            Default Inputs:
                length=20
            if 'open':
                ad = close - open
            else:
                ad = 2 * close - high - low
            
            hl_range = high - low

            ad = ad * volume / hl_range

            CMF = SUM(ad, length) / SUM(volume, length)

        Args:
            high (pd.Series): Series of 'high's
            low (pd.Series): Series of 'low's
            close (pd.Series): Series of 'close's
            open (pd.Series): Series of 'open's
            volume (pd.Series): Series of 'volume's
            length (int): The short period.  Default: 20
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.cmf(length=length, offset=offset, **kwargs)

    def efi(self, length=None, drift=None, mamode=None, offset=None, **kwargs):
        """Elder's Force Index (EFI)

        Elder's Force Index measures the power behind a price movement using price
        and volume as well as potential reversals and price corrections.

        Sources:
            https://www.tradingview.com/wiki/Elder%27s_Force_Index_(EFI)
            https://www.motivewave.com/studies/elders_force_index.htm

        Calculation:
            Default Inputs:
                length=20, drift=1, mamode=None

            EMA = Exponential Moving Average

            SMA = Simple Moving Average

            pv_diff = close.diff(drift) * volume
            if mamode == 'sma':
                EFI = SMA(pv_diff, length)
            else:
                EFI = EMA(pv_diff, length)

        Args:
            close (pd.Series): Series of 'close's
            volume (pd.Series): Series of 'volume's
            length (int): The short period.  Default: 13
            drift (int): The diff period.   Default: 1
            mamode (str): Two options: None or 'sma'.  Default: None
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.efi(length=length, drift=drift, mamode=mamode, offset=offset, **kwargs)

    def eom(self, length=None, divisor=None, drift=None, offset=None, **kwargs):
        """Ease of Movement (EOM)

        Ease of Movement is a volume based oscillator that is designed to measure the
        relationship between price and volume flucuating across a zero line.

        Sources:
            https://www.tradingview.com/wiki/Ease_of_Movement_(EOM)
            https://www.motivewave.com/studies/ease_of_movement.htm
            https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ease_of_movement_emv

        Calculation:
            Default Inputs:
                length=14, divisor=100000000, drift=1

            SMA = Simple Moving Average    

            hl_range = high - low

            distance = 0.5 * (high - high.shift(drift) + low - low.shift(drift))

            box_ratio = (volume / divisor) / hl_range

            eom = distance / box_ratio

            EOM = SMA(eom, length)

        Args:
            high (pd.Series): Series of 'high's
            low (pd.Series): Series of 'low's
            close (pd.Series): Series of 'close's
            volume (pd.Series): Series of 'volume's
            length (int): The short period.  Default: 14
            drift (int): The diff period.   Default: 1
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.eom(length=length, divisor=divisor, drift=drift, offset=offset, **kwargs)

    def mfi(self, length=None, drift=None, offset=None, **kwargs):
        """Money Flow Index (MFI)

        Money Flow Index is an oscillator indicator that is used to measure buying and
        selling pressure by utilizing both price and volume.

        Sources:
            https://www.tradingview.com/wiki/Money_Flow_(MFI)

        Calculation:
            Default Inputs:
                length=14, drift=1

            tp = typical_price = hlc3 = (high + low + close) / 3

            rmf = raw_money_flow = tp * volume

            pmf = pos_money_flow = SUM(rmf, length) if tp.diff(drift) > 0 else 0

            nmf = neg_money_flow = SUM(rmf, length) if tp.diff(drift) < 0 else 0

            MFR = money_flow_ratio = pmf / nmf

            MFI = money_flow_index = 100 * pmf / (pmf + nmf)

        Args:
            high (pd.Series): Series of 'high's
            low (pd.Series): Series of 'low's
            close (pd.Series): Series of 'close's
            volume (pd.Series): Series of 'volume's
            length (int): The sum period.  Default: 14
            drift (int): The difference period.   Default: 1
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.mfi(length=length, drift=drift, offset=offset, **kwargs)

    def nvi(self, length=None, initial=None, offset=None, **kwargs):
        """Negative Volume Index (NVI)

        The Negative Volume Index is a cumulative indicator that uses volume change in
        an attempt to identify where smart money is active.

        Sources:
            https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:negative_volume_inde
            https://www.motivewave.com/studies/negative_volume_index.htm

        Calculation:
            Default Inputs:
                length=1, initial=1000

            ROC = Rate of Change

            roc = ROC(close, length)

            signed_volume = signed_series(volume, initial=1)

            nvi = signed_volume[signed_volume < 0].abs() * roc_

            nvi.fillna(0, inplace=True)

            nvi.iloc[0]= initial

            nvi = nvi.cumsum()

        Args:
            close (pd.Series): Series of 'close's
            volume (pd.Series): Series of 'volume's
            length (int): The short period.  Default: 13
            initial (int): The short period.  Default: 1000
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.nvi(length=length, initial=initial, offset=offset, **kwargs)

    def obv(self, offset=None, **kwargs):
        """On Balance Volume (OBV)

        On Balance Volume is a cumulative indicator to measure buying and selling
        pressure.

        Sources:
            https://www.tradingview.com/wiki/On_Balance_Volume_(OBV)
            https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/on-balance-volume-obv/
            https://www.motivewave.com/studies/on_balance_volume.htm

        Calculation:
            signed_volume = signed_series(close, initial=1) * volume

            obv = signed_volume.cumsum()

        Args:
            close (pd.Series): Series of 'close's
            volume (pd.Series): Series of 'volume's
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.obv(offset=offset, **kwargs)

    def pvi(self, length=None, initial=None, offset=None, **kwargs):
        """Positive Volume Index (PVI)

        The Positive Volume Index is a cumulative indicator that uses volume change in
        an attempt to identify where smart money is active.  Used in conjunction with NVI.

        Sources:
            https://www.investopedia.com/terms/p/pvi.asp

        Calculation:
            Default Inputs:
                length=1, initial=1000

            ROC = Rate of Change

            roc = ROC(close, length)

            signed_volume = signed_series(volume, initial=1)

            pvi = signed_volume[signed_volume > 0].abs() * roc_

            pvi.fillna(0, inplace=True)

            pvi.iloc[0]= initial

            pvi = pvi.cumsum()

        Args:
            close (pd.Series): Series of 'close's
            volume (pd.Series): Series of 'volume's
            length (int): The short period.  Default: 13
            initial (int): The short period.  Default: 1000
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.pvi(length=length, initial=initial, offset=offset, **kwargs)

    def pvol(self, offset=None, **kwargs):
        """Price-Volume (PVOL)

        Returns a series of the product of price and volume.

        Calculation:
            if signed:
                pvol = signed_series(close, 1) * close * volume
            else:
                pvol = close * volume

        Args:
            close (pd.Series): Series of 'close's
            volume (pd.Series): Series of 'volume's
            signed (bool): Keeps the sign of the difference in 'close's.  Default: True
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.pvol(offset=offset, **kwargs)

    def pvt(self, drift=None, offset=None, **kwargs):
        """Price-Volume Trend (PVT)

        The Price-Volume Trend utilizes the Rate of Change with volume to
        and it's cumulative values to determine money flow.

        Sources:
            https://www.tradingview.com/wiki/Price_Volume_Trend_(PVT)

        Calculation:
            Default Inputs:
                drift=1

            ROC = Rate of Change

            pv = ROC(close, drift) * volume

            PVT = pv.cumsum()

        Args:
            close (pd.Series): Series of 'close's
            volume (pd.Series): Series of 'volume's
            drift (int): The diff period.   Default: 1
            offset (int): How many periods to offset the result.  Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method

        Returns:
            pd.Series: New feature generated.
        """
        return self.df.ta.pvt(drift=drift, offset=offset, **kwargs)

    def vp(self, width=None, **kwargs):
        """Volume Profile (VP)

        Calculates the Volume Profile by slicing price into ranges.  Note: Value Area is not calculated.

        Sources:
            https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:volume_by_price
            https://www.tradingview.com/wiki/Volume_Profile
            http://www.ranchodinero.com/volume-tpo-essentials/
            https://www.tradingtechnologies.com/blog/2013/05/15/volume-at-price/

        Calculation:
            Default Inputs:
                width=10
            
            vp = pd.concat([close, pos_volume, neg_volume], axis=1)

            vp_ranges = np.array_split(vp, width)

            result = ({high_close, low_close, mean_close, neg_volume, pos_volume} foreach range in vp_ranges)
            
            vpdf = pd.DataFrame(result)

            vpdf['total_volume'] = vpdf['pos_volume'] + vpdf['neg_volume']

        Args:
            close (pd.Series): Series of 'close's
            volume (pd.Series): Series of 'volume's
            width (int): How many ranges to distrubute price into.  Default: 10

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)

            fill_method (value, optional): Type of fill method
            
            sort_close (value, optional): Whether to sort by close before splitting into ranges.  Default: False

        Returns:
            pd.DataFrame: New feature generated.
        """
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
    