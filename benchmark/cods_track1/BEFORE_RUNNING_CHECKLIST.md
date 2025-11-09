# ✅ Pre-Run Checklist for new_by_dev Workflow

Before running Docker with the new `new_by_dev.py` workflow, verify these critical configurations:

---

## 1. 📁 docker-compose.yml - Volume Mappings

**File Location:** `benchmark/cods_track1/docker-compose.yml`

### ✓ Check Volume Mount (Line 6)
```yaml
volumes:
  - ./track1_result_iter_by_dev:/home/track1_result_iter_by_dev
```

**What it means:**
- Local directory: `./track1_result_iter_by_dev/`
- Container directory: `/home/track1_result_iter_by_dev/`
- **Must match** the `RESULT_DIR` in `run_track_1.py`

### ✓ Check Workflow File Mount (Line 9)
```yaml
- ../../src/agent_hive/workflows/new_by_dev.py:/opt/conda/envs/assetopsbench/lib/python3.12/site-packages/agent_hive/workflows/new_by_dev.py
```

**What it means:**
- Mounts the new workflow file into the container environment
- **Must be** `new_by_dev.py` (not `track1_planning.py`)

---

## 2. 🐍 run_track_1.py - Result Directory

**File Location:** `benchmark/cods_track1/run_track_1.py`

### ✓ Check RESULT_DIR (Line 56)
```python
RESULT_DIR = "/home/track1_result_iter_by_dev/"
PLAN_DIR = RESULT_DIR + "plan/"
TRAJECTORY_DIR = RESULT_DIR + "trajectory/"
```

**What it means:**
- **Must match** the container path in docker-compose.yml
- Should be `/home/track1_result_iter_by_dev/`
- ❌ NOT `/home/new_be_dev/`
- ❌ NOT `/home/track1_result_iter2/`

### ✓ Check Workflow Import (Line 47)
```python
from agent_hive.workflows.new_by_dev import NewPlanningWorkflow
```

**What it means:**
- **Must import** from `new_by_dev` module
- ❌ NOT from `track1_planning`

---

## 3. 🔧 entrypoint.sh - Execution Commands

**File Location:** `benchmark/cods_track1/entrypoint.sh`

### ✓ Check Plan Generation (Line 15)
```bash
python /home/run_track_1.py --utterance_ids 1,106 --generate_steps_only True
```

**What it means:**
- First generates plans and saves them to `plan/` directory
- Helps you see the LLM's decomposition strategy

### ✓ Check Full Execution (Line 18)
```bash
python /home/run_track_1.py --utterance_ids 1,106
```

**What it means:**
- Then runs full workflow
- Saves trajectories to `trajectory/` directory

---

## 4. 📝 new_by_dev.py - Workflow File

**File Location:** `src/agent_hive/workflows/new_by_dev.py`

### ✓ Check Class Name (Line 23)
```python
class NewPlanningWorkflow(Workflow):
```

**What it means:**
- Must be named `NewPlanningWorkflow`
- run_track_1.py imports this exact name

### ✓ Check Editable Sections
- **Edit Section 1:** Agent description formatting (lines ~65-95)
- **Edit Section 2:** Prompt template (lines ~165-210)

**Status:**
- ✅ Section 1 has metadata extraction and emojis
- ✅ Section 2 has enhanced prompt with query guidance

---

## 🚨 Quick Verification Script

Run this in your terminal before Docker:

```bash
# Check RESULT_DIR matches
grep "RESULT_DIR = " benchmark/cods_track1/run_track_1.py

# Check import is correct
grep "from agent_hive.workflows" benchmark/cods_track1/run_track_1.py

# Check docker-compose volume
grep "track1_result_iter_by_dev" benchmark/cods_track1/docker-compose.yml

# Verify new_by_dev.py exists
ls -la src/agent_hive/workflows/new_by_dev.py
```

**Expected Output:**
```
RESULT_DIR = "/home/track1_result_iter_by_dev/"
from agent_hive.workflows.new_by_dev import NewPlanningWorkflow
- ./track1_result_iter_by_dev:/home/track1_result_iter_by_dev
-rw-r--r--  ... new_by_dev.py
```

---

## ✅ Final Checklist

Before running `docker-compose up`:

- [ ] docker-compose.yml volume: `./track1_result_iter_by_dev:/home/track1_result_iter_by_dev`
- [ ] docker-compose.yml mounts: `new_by_dev.py`
- [ ] run_track_1.py RESULT_DIR: `/home/track1_result_iter_by_dev/`
- [ ] run_track_1.py imports: `from agent_hive.workflows.new_by_dev import NewPlanningWorkflow`
- [ ] entrypoint.sh: Runs both `--generate_steps_only True` and full execution
- [ ] new_by_dev.py: Has both edit sections with enhancements
- [ ] Local directory exists: `./track1_result_iter_by_dev/` (will be created by Docker)

---

## 🚀 Ready to Run

If all checks pass:

```bash
cd benchmark/cods_track1/
docker-compose up
```

Expected outputs in `./track1_result_iter_by_dev/`:
- `plan/Model_16_Q_1_plan.txt` - LLM planning output
- `plan/Model_16_Q_106_plan.txt` - LLM planning output
- `trajectory/Q_1_trajectory.json` - Execution trajectory
- `trajectory/Q_106_trajectory.json` - Execution trajectory

---

**Last Updated:** November 9, 2025  
**For:** new_by_dev.py Workflow Testing
