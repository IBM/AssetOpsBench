# FACT_SHEET.json - Complete Guide

## 📋 What is fact_sheet.json?

The **fact_sheet.json** is a **metadata file** that accompanies your competition submission. It's a simple JSON file that describes which track/task your submission is for.

---

## 📁 Current Fact Sheet Files

### Track 1: Task Planning
**File:** `src/agent_hive/workflows/track1_fact_sheet.json`

```json
{
    "Track": "Task Planning"
}
```

### Track 2: Task Execution
**File:** `src/agent_hive/workflows/track2_fact_sheet.json`

```json
{
    "Track": "Task Execution"
}
```

---

## 🎯 Purpose of fact_sheet.json

1. **Track Identification** - Tells the evaluation system which track this submission is for
2. **Submission Validation** - CodaBench uses this to route submissions to correct evaluators
3. **Organization** - Helps track organizers manage thousands of submissions
4. **Metadata** - Provides context about the submission approach

---

## 📦 Submission Structure

### Track 1 Submission (Task Planning)
```
submission_track1.zip
├── track1_planning.py          ← Your modified workflow (EDITABLE)
└── track1_fact_sheet.json      ← Metadata file (DO NOT MODIFY)
```

### Track 2 Submission (Task Execution)
```
submission_track2.zip
├── track2_execution.py         ← Your modified workflow (EDITABLE)
└── track2_fact_sheet.json      ← Metadata file (DO NOT MODIFY)
```

---

## ⚠️ Important Rules

### ❌ DO NOT:
- Modify the contents of fact_sheet.json
- Rename the fact_sheet.json file
- Change the "Track" value
- Add new fields to fact_sheet.json
- Delete the fact_sheet.json file

### ✅ DO:
- Include EXACTLY this file in your ZIP
- Keep the filename as `track1_fact_sheet.json` or `track2_fact_sheet.json`
- Submit both `*_planning.py`/`*_execution.py` AND `*_fact_sheet.json` together
- Verify both files are present before uploading

---

## 🚀 What COULD Be Extended (Future Competitions)

While the current fact_sheet is minimal, here are fields that could be added in future iterations:

### Possible Extensions:

```json
{
    "Track": "Task Planning",
    "Team": "Your Team Name",
    "Version": "1.0",
    "Date": "2025-11-04",
    "Description": "Improved prompting strategy with enhanced agent descriptions",
    "Methodology": [
        "Better LLM prompting",
        "Structured agent formatting",
        "Enhanced examples"
    ],
    "Confidence": 0.85,
    "Notes": "Focused on improving task decomposition quality",
    "Hyperparameters": {
        "max_planning_steps": 5,
        "reflection_iterations": 3,
        "temperature": 0.7
    },
    "Dependencies": [
        "track1_planning.py"
    ],
    "SubmissionMetadata": {
        "local_test_score": 0.92,
        "iteration": 3,
        "estimated_improvement": "+15%"
    }
}
```

### Why These Could Help:

| Field | Value | Purpose |
|-------|-------|---------|
| Team | Your Team Name | Attribution & leaderboard |
| Version | 1.0, 2.0, etc. | Track iteration history |
| Date | Submission timestamp | Timeline tracking |
| Description | Strategy summary | Help organizers understand approach |
| Methodology | List of techniques | Reproducibility |
| Confidence | 0.0-1.0 | Self-assessment score |
| Notes | Free text | Context for reviewers |
| Hyperparameters | JSON object | Configuration snapshot |
| Dependencies | File list | Dependency tracking |
| SubmissionMetadata | Local testing results | Pre-submission validation |

---

## 📊 Relationship to Your Code

```
fact_sheet.json (Metadata)
        ↓
    Tells CodaBench
        ↓
    Which evaluator to use
        ↓
    Which scoring function
        ↓
    Which benchmark dataset
        ↓
    Generates leaderboard entry
```

The fact_sheet doesn't change execution—it just **tells the competition system** what you're submitting.

---

## 🔄 Submission Workflow

```
1. Edit track1_planning.py      (YOUR WORK HERE)
              ↓
2. Create fact_sheet.json       (File already exists)
              ↓
3. Zip both files together      (ZIP format required)
              ↓
4. Upload to CodaBench          (Automatic evaluation)
              ↓
5. Fact_sheet tells system      (Route to correct track)
              ↓
6. Evaluation happens           (Using fact_sheet track info)
              ↓
7. Results published            (On your track's leaderboard)
```

---

## ✅ Checklist Before Submission

- [ ] `track1_planning.py` or `track2_execution.py` is edited and tested
- [ ] `track1_fact_sheet.json` or `track2_fact_sheet.json` exists unchanged
- [ ] Both files are in `src/agent_hive/workflows/` directory
- [ ] ZIP file contains exactly 2 files
- [ ] ZIP file is named `submission_track*.zip`
- [ ] No new files added to ZIP
- [ ] No files renamed
- [ ] Fact sheet JSON is valid (no syntax errors)
- [ ] Ready to upload to CodaBench

---

## 🎓 Key Takeaways

| Aspect | Detail |
|--------|--------|
| **What it is** | Metadata JSON file |
| **Current content** | Just the track name |
| **Can you modify it?** | NO - DO NOT MODIFY |
| **Must it be included?** | YES - Required for submission |
| **What happens if missing?** | Submission rejected |
| **What happens if modified?** | May cause evaluation errors |
| **Could be extended?** | YES - in future competitions |

---

## 📝 Example: Complete Submission

### File Structure
```
/tmp/submission_track1/
├── track1_planning.py
└── track1_fact_sheet.json
```

### Commands to Create ZIP
```bash
cd src/agent_hive/workflows

# Verify files exist
ls -la track1_planning.py track1_fact_sheet.json

# Create ZIP
zip submission_track1.zip track1_planning.py track1_fact_sheet.json

# Verify ZIP contents
unzip -l submission_track1.zip

# Output should show:
# -rw-r--r--   track1_planning.py
# -rw-r--r--   track1_fact_sheet.json
```

### What Gets Evaluated
- **Your Submission:** `track1_planning.py` (your edits in TODO sections)
- **Not Evaluated:** `track1_fact_sheet.json` (just tells system what you're doing)

---

## 🚀 Bottom Line

**The fact_sheet.json is NOT something you edit or compete on.**

It's simply:
- ✅ A required "form" that tells CodaBench what track you're in
- ✅ Mandatory for submission acceptance
- ✅ Unchanging (leave it as is)
- ✅ A one-time setup (copy and forget)

**Your actual competition is in:**
- 🏆 **Track 1:** Improving `track1_planning.py` (planning prompt & agent formatting)
- 🏆 **Track 2:** Improving `track2_execution.py` (execution logic & task revision)

The fact_sheet just labels which one you're competing in!

---

## ❓ FAQ

**Q: Can I add fields to fact_sheet.json?**
A: Not for the current competition. The organizers expect exactly the current structure.

**Q: What if I submit the wrong fact_sheet?**
A: Your submission may go to wrong evaluator or be rejected. Always double-check.

**Q: Can I use fact_sheet.json to improve my score?**
A: No. Scoring is based 100% on your workflow code (track1_planning.py or track2_execution.py), not metadata.

**Q: What if the fact_sheet has a syntax error?**
A: Submission will fail validation. Make sure it's valid JSON (CodaBench validates this).

**Q: Can I modify it "just to add a comment"?**
A: No. Don't touch it. Keep it exactly as provided.

**Q: Is there any way to use fact_sheet strategically?**
A: In future competitions, yes (as shown in extensions section). For this one, no—it's purely administrative.

---

**Created:** November 4, 2025
**For:** AssetOpsBench CODS Competition  
**Status:** ✅ Reference Guide Ready
