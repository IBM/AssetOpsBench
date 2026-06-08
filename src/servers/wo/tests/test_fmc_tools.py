"""Unit tests for the failure-mode classification (wo_fmc) tools."""

from unittest.mock import patch

import pandas as pd
import pytest

from servers.wo import fmc_tools
from servers.wo.models import (
    ErrorResult,
    FmcBatchWriteResult,
    FmcCodeAssignment,
    FmcCodeDistributionResult,
    FmcWorkOrder,
    FmcWorkOrdersResult,
)


def _make_fmc_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "wo_id": [
                "TRN-WO00001",
                "TRN-WO00002",
                "TRN-WO00003",
                "TRN-WO00004",
                "TST-WO00001",
                "TST-WO00002",
            ],
            "description": [
                "falure",
                "unserviceable",
                "bogged",
                "leaking seal",
                "ejected",
                "hot joint",
            ],
            "failure_code": [
                "Breakdown",
                "Breakdown",
                "Plugged / choked",
                "Leaking",
                None,
                None,
            ],
        }
    )


@pytest.fixture
def mock_load():
    with patch("servers.wo.fmc_tools.load", side_effect=lambda key: _make_fmc_df() if key == "wo_fmc" else None):
        yield


# --- get_work_order_failure_code -------------------------------------------


def test_get_labeled_record(mock_load):
    res = fmc_tools.get_work_order_failure_code("TRN-WO00001")
    assert isinstance(res, FmcWorkOrder)
    assert res.wo_id == "TRN-WO00001"
    assert res.description == "falure"
    assert res.failure_code == "Breakdown"


def test_get_blank_record_has_null_code(mock_load):
    res = fmc_tools.get_work_order_failure_code("TST-WO00001")
    assert isinstance(res, FmcWorkOrder)
    assert res.failure_code is None


def test_get_missing_record(mock_load):
    res = fmc_tools.get_work_order_failure_code("TST-WO99999")
    assert isinstance(res, ErrorResult)


def test_get_no_data():
    with patch("servers.wo.fmc_tools.load", return_value=None):
        res = fmc_tools.get_work_order_failure_code("TRN-WO00001")
        assert isinstance(res, ErrorResult)


# --- list_work_order_failure_codes -----------------------------------------


def test_list_labeled_only(mock_load):
    res = fmc_tools.list_work_order_failure_codes(labeled=True)
    assert isinstance(res, FmcWorkOrdersResult)
    assert res.total == 4
    assert res.labeled == 4
    assert res.unlabeled == 0
    assert all(wo.failure_code is not None for wo in res.work_orders)


def test_list_unlabeled_only(mock_load):
    res = fmc_tools.list_work_order_failure_codes(labeled=False)
    assert isinstance(res, FmcWorkOrdersResult)
    assert res.total == 2
    assert res.labeled == 0
    assert res.unlabeled == 2
    assert all(wo.failure_code is None for wo in res.work_orders)


def test_list_all_default(mock_load):
    res = fmc_tools.list_work_order_failure_codes()
    assert isinstance(res, FmcWorkOrdersResult)
    assert res.total == 6
    assert res.labeled == 4
    assert res.unlabeled == 2


# --- set_work_order_failure_codes ------------------------------------------


def _asg(wo_id, code):
    return FmcCodeAssignment(wo_id=wo_id, failure_code=code)


def test_set_single():
    with patch("servers.wo.fmc_tools.write_failure_codes", return_value={"TST-WO00001": True}) as mock_write:
        res = fmc_tools.set_work_order_failure_codes([_asg("TST-WO00001", "Overheating")])
        assert isinstance(res, FmcBatchWriteResult)
        assert res.total == 1
        assert res.updated == 1
        assert res.results[0].failure_code == "Overheating"
        mock_write.assert_called_once_with({"TST-WO00001": "Overheating"})


def test_set_batch_partial_missing():
    status = {"TST-WO00001": True, "TST-WO99999": False}
    with patch("servers.wo.fmc_tools.write_failure_codes", return_value=status):
        res = fmc_tools.set_work_order_failure_codes(
            [_asg("TST-WO00001", "Electrical"), _asg("TST-WO99999", "Breakdown")]
        )
        assert isinstance(res, FmcBatchWriteResult)
        assert res.total == 2
        assert res.updated == 1
        missing = [r.wo_id for r in res.results if not r.updated]
        assert missing == ["TST-WO99999"]


def test_set_no_db():
    with patch("servers.wo.fmc_tools.write_failure_codes", return_value=None):
        res = fmc_tools.set_work_order_failure_codes([_asg("TST-WO00001", "Overheating")])
        assert isinstance(res, ErrorResult)


def test_set_empty_list_rejected():
    res = fmc_tools.set_work_order_failure_codes([])
    assert isinstance(res, ErrorResult)


def test_set_empty_code_rejected():
    res = fmc_tools.set_work_order_failure_codes([_asg("TST-WO00001", "   ")])
    assert isinstance(res, ErrorResult)


def test_set_duplicate_wo_id_rejected():
    res = fmc_tools.set_work_order_failure_codes(
        [_asg("TST-WO00001", "Electrical"), _asg("TST-WO00001", "Breakdown")]
    )
    assert isinstance(res, ErrorResult)


# --- get_failure_code_distribution -----------------------------------------


def test_distribution_ranked(mock_load):
    res = fmc_tools.get_failure_code_distribution()
    assert isinstance(res, FmcCodeDistributionResult)
    assert res.total_records == 6
    assert res.labeled_records == 4
    # Breakdown (2) ranks first; remaining tied at 1
    assert res.distribution[0].failure_code == "Breakdown"
    assert res.distribution[0].count == 2


def test_distribution_top_n(mock_load):
    res = fmc_tools.get_failure_code_distribution(top_n=1)
    assert isinstance(res, FmcCodeDistributionResult)
    assert len(res.distribution) == 1
    assert res.distribution[0].failure_code == "Breakdown"


def test_distribution_empty_when_no_codes():
    blank = pd.DataFrame({"wo_id": ["TST-WO00001"], "description": ["x"], "failure_code": [None]})
    with patch("servers.wo.fmc_tools.load", return_value=blank):
        res = fmc_tools.get_failure_code_distribution()
        assert isinstance(res, ErrorResult)
