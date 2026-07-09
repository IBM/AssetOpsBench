from mcphub.workflows import chiller_triage


class FakeToolUniverse:
    def __init__(self):
        self.calls = []

    def run(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "iot.sensors":
            return {"sensors": ["Pressure sensor", "Vibration sensor"]}
        if name == "fmsr.get_failure_modes":
            return {
                "asset_class": "pump",
                "failure_modes": ["seal leakage", "impeller wear"],
            }
        if name == "fmsr.generate_failure_mode_sensor_mapping":
            return {"metadata": arguments}
        raise AssertionError(f"unexpected tool call: {name}")


def test_chiller_triage_passes_lists_to_mapping_tool():
    tu = FakeToolUniverse()

    result = chiller_triage(tu, asset_id="Pump-1", raise_work_order=False)

    mapping_call = [
        arguments
        for name, arguments in tu.calls
        if name == "fmsr.generate_failure_mode_sensor_mapping"
    ][0]
    assert mapping_call == {
        "asset_class": "pump",
        "failure_modes": ["seal leakage", "impeller wear"],
        "sensors": ["Pressure sensor", "Vibration sensor"],
    }
    assert result["failure_modes_result"]["asset_class"] == "pump"
    assert result["failure_modes"] == ["seal leakage", "impeller wear"]
