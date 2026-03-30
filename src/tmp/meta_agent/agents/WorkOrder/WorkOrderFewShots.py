WORKORDERAGENTFEWSHOTS = """Question: Create a work order for inspecting CHL-A100 due to suspected compressor overheating.
Thought 1: I need to generate a work order for CHL-A100 inspection focused on compressor overheating.
Action 1: WorkOrderAgent
Action Input 1: request=Create a work order for asset CHL-A100 — task: inspect compressor for signs of overheating.
Observation 1: Work order created successfully for compressor inspection on CHL-A100.

Question: Generate a maintenance work order to replace the vibration sensor on WT-22.
Thought 1: I need to initiate a work order for replacing the vibration sensor on WT-22.
Action 1: WorkOrderAgent
Action Input 1: request=Create a work order for asset WT-22 — task: replace faulty vibration sensor.
Observation 1: Work order created for replacing vibration sensor on WT-22.

Question: Schedule a corrective work order for CHL-X0 due to detected low refrigerant charge.
Thought 1: Low refrigerant charge detected — a corrective work order is needed.
Action 1: WorkOrderAgent
Action Input 1: request=Schedule a corrective work order for asset CHL-X0 — task: address low refrigerant charge.
Observation 1: Corrective work order scheduled for CHL-X0 refrigerant charge issue.

Question: Raise an emergency work order for CHL-A100 after sudden shutdown.
Thought 1: CHL-A100 has experienced a sudden shutdown — immediate work order needed.
Action 1: WorkOrderAgent
Action Input 1: request=Create an emergency work order for CHL-A100 — task: investigate sudden shutdown.
Observation 1: Emergency work order created for sudden shutdown investigation on CHL-A100.

Question: Close the work order WO-12345 after confirming successful repair of suction pressure sensor.
Thought 1: The repair for the suction pressure sensor is completed — the work order can be closed.
Action 1: WorkOrderAgent
Action Input 1: request=Close work order WO-12345 — repair of suction pressure sensor completed successfully.
Observation 1: Work order WO-12345 closed successfully.
"""

def get_workorder_agent_examples():
    return WORKORDERAGENTFEWSHOTS.strip()
