"""Ken French block parser: right block, percent -> decimal, missing codes -> NaN (never 0). Offline."""
import numpy as np
import pandas as pd
import pytest

from quantlab.alpha.kf import parse_block

TEXT = """This file was created using the CRSP database.
Missing data are indicated by -99.99 or -999.

  Average Value Weight Returns -- Monthly
,Lo 30,Med 40,Hi 30
196307,   -0.87,    0.38,    0.03
196308,    5.00,  -99.99,    1.50

  Average Equal Weighted Returns -- Monthly
,Lo 30,Med 40,Hi 30
196307,   9.00,    9.00,    9.00
"""


def test_value_weighted_block_is_parsed_as_decimals():
    df = parse_block(TEXT, "Value Weight", "Monthly")
    assert list(df.columns) == ["Lo 30", "Med 40", "Hi 30"]
    assert list(df.index) == [pd.Timestamp("1963-07-31"), pd.Timestamp("1963-08-31")]
    assert df.loc["1963-07-31", "Lo 30"] == pytest.approx(-0.0087)
    assert df.loc["1963-08-31", "Hi 30"] == pytest.approx(0.015)


def test_missing_code_is_unknown_not_zero():
    df = parse_block(TEXT, "Value Weight", "Monthly")
    assert np.isnan(df.loc["1963-08-31", "Med 40"])


def test_equal_weight_block_is_separate_and_missing_block_raises():
    eq = parse_block(TEXT, "Equal Weight", "Monthly")
    assert eq.shape == (1, 3) and eq.iloc[0, 0] == pytest.approx(0.09)
    with pytest.raises(ValueError):
        parse_block(TEXT, "Annual")


def test_first_block_without_title():
    df = parse_block(",Mkt-RF,RF\n192607,   2.89,   0.22\n\n Annual Factors\n", )
    assert df.loc["1926-07-31", "RF"] == pytest.approx(0.0022)
