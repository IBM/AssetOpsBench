import pytest

from mcphub import workflows


def test_iot_sensor_workflows_are_not_registered():
    assert workflows.REGISTERED == []


def test_chiller_triage_is_disabled():
    with pytest.raises(RuntimeError, match="asset_ids\\(\\) and assets\\(\\)"):
        workflows.chiller_triage(object(), asset_id="Pump-1")


def test_sensor_inventory_gap_is_disabled():
    with pytest.raises(RuntimeError, match="asset_ids\\(\\) and assets\\(\\)"):
        workflows.sensor_inventory_gap(object(), asset_id="Pump-1")
