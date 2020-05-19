from typing import Optional

from pandas.core.frame import DataFrame
from pandas.core.series import Series


def _size_mask(
        df: DataFrame,
        size_min: Optional[float] = None,
        size_max: Optional[float] = None
    ) -> Series:
        if size_min and size_max:
            return (df['size']>=size_min) & (df['size']<=size_max)
        elif size_min:
            return (df['size']>=size_min)
        elif size_max:
            return (df['size']<=size_max)
        else:
            return [True] * len(df)
