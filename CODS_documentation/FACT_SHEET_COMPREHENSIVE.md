# fact_sheet.json - Complete Analysis

## 📋 Executive Summary

**What is it?** A small JSON metadata file that accompanies your competition submission  
**Current size:** 2-3 lines  
**How important:** Required but not scored  
**Can you modify it?** **NO** - Leave it exactly as provided  

---

## 🔍 The Facts

### Current State

**Track 1 File:**
```json
{
    "Track": "Task Planning"
}
```

**Track 2 File:**
```json
{
    "Track": "Task Execution"
}
```

**Location:** `src/agent_hive/workflows/track*_fact_sheet.json`

---

## 🎯 Purpose

| Purpose | Details |
|---------|---------|
| **Track Identification** | Tells CodaBench which competition track this is for |
| **Submission Routing** | System uses this to route to correct evaluator |
| **Validation** | Ensures submission format is correct |
| **Organization** | Helps manage thousands of submissions |
| **Non-Scored Metadata** | Informational only, doesn't affect scoring |

---

## 🚀 How It's Used in Submission Process

```
Your Work:
├── Edit track1_planning.py (TODO sections)
└── Keep track1_fact_sheet.json unchanged
        ↓
Create ZIP:
├── submission_track1.zip
│   ├── track1_planning.py          ← YOUR CODE (SCORED)
│   └── track1_fact_sheet.json      ← METADATA (NOT SCORED)
        ↓
Upload to CodaBench:
    fact_sheet.json is read first
        ↓
    Tells system: "This is Task Planning"
        ↓
    Routes submission to Track 1 evaluator
        ↓
    Evaluator only scores track1_planning.py
        ↓
    Results posted to Track 1 leaderboard
```

---

## ⚙️ Technical Details

### What Happens When Submission is Received

```python
# Pseudocode of CodaBench's process
def process_submission(zip_file):
    # 1. Extract files
    files = zip_file.extract_all()
    
    # 2. Read fact_sheet.json
    metadata = json.load('track1_fact_sheet.json')
    track = metadata['Track']  # "Task Planning"
    
    # 3. Route to evaluator
    if track == "Task Planning":
        evaluator = Track1Evaluator()
    elif track == "Task Execution":
        evaluator = Track2Evaluator()
    else:
        raise InvalidTrackError()
    
    # 4. Evaluate only the code file
    if track == "Task Planning":
        score = evaluator.evaluate('track1_planning.py')
    
    # 5. fact_sheet.json never affects the score
    return score
```

### Why It Must Be Present

1. **Validation gate** - If missing, submission rejected
2. **Routing logic** - System needs to know which track
3. **Record keeping** - Organizers track which track has how many submissions
4. **Error prevention** - Wrong file in wrong track is caught immediately

---

## ✅ Submission Checklist

### Before Creating ZIP

- [ ] `track1_planning.py` is edited (TODO sections only)
- [ ] `track1_planning.py` is tested locally
- [ ] Both files exist in `src/agent_hive/workflows/`
- [ ] `track1_fact_sheet.json` is **UNMODIFIED** (exactly as provided)
- [ ] No other files in this directory should be zipped

### ZIP Contents

- [ ] ZIP contains exactly 2 files:
  - `track1_planning.py`
  - `track1_fact_sheet.json`
- [ ] ZIP filename is `submission_track1.zip`
- [ ] No subdirectories in ZIP
- [ ] No extra files added

### JSON Validation

- [ ] `track1_fact_sheet.json` is valid JSON (no syntax errors)
- [ ] Contains exactly the key `"Track"`
- [ ] Track value is exactly `"Task Planning"` (or "Task Execution" for Track 2)
- [ ] No extra fields added
- [ ] No modifications to original content

### Before Upload

- [ ] ZIP file is readable
- [ ] Both files verify correctly when extracted
- [ ] Ready to upload to CodaBench

---

## 🚫 Common Mistakes

| Mistake | Result | Fix |
|---------|--------|-----|
| Modify fact_sheet.json | ❌ May cause routing error | Don't edit it |
| Rename fact_sheet.json | ❌ Submission rejected | Keep original name |
| Change "Track" value | ❌ Wrong evaluator used | Keep original value |
| Add new fields | ⚠️ May fail JSON parsing | Don't add anything |
| Delete fact_sheet.json | ❌ Submission rejected | Must include it |
| Wrong JSON syntax | ❌ Validation fails | Don't touch the file |
| Forget to include it in ZIP | ❌ Submission rejected | Include both files |
| Add extra files to ZIP | ⚠️ May be rejected | Only 2 files needed |

---

## 📊 Comparison: Score vs. Metadata

| Component | Affects Score? | Editable? | Location |
|-----------|---|---|---|
| **track1_planning.py** | ✅ YES (100% of score) | ✅ YES (TODO sections) | Main file |
| **track1_fact_sheet.json** | ❌ NO | ❌ NO (admin only) | Metadata |

**Bottom line:** Your score depends entirely on your code edits, not metadata.

---

## 🔮 What Could Be Extended?

### Current (This Competition)
✓ Minimal - just track identification

### Possible Future Additions
- Team name (for attribution)
- Submission version (1.0, 1.1, etc.)
- Submission date (timestamp)
- Strategy description (freetext)
- Methodology list (array)
- Team size (integer)
- Hyperparameters (nested object)
- Performance predictions (decimal score)
- Local test results (metrics object)

### Why NOT Extended Now?
1. Keep it simple for baseline competition
2. Prevent gaming (can't claim high scores via metadata)
3. Gradual adoption (let organizers request later)
4. Backward compatible (easy to upgrade)
5. Fair for all teams (no strategic advantage)

### When It COULD Be Extended
- **Future competitions** using same infrastructure
- **Post-competition analysis** for research
- **Next year's iteration** of AssetOpsBench
- **Other benchmarks** inspired by this one

---

## 📝 Real Examples

### What You Might Add (but DON'T)

```json
// ❌ DON'T DO THIS - Will break submission
{
    "Track": "Task Planning",
    "Team": "My Team",              // Extra field
    "Version": "1.0",               // Extra field
    "Notes": "Improved prompting"   // Extra field
}
```

### What Should Be There

```json
// ✅ DO THIS - Use as-is
{
    "Track": "Task Planning"
}
```

### What Future Version COULD Look Like

```json
// 🔮 FUTURE (NOT NOW) - Next competition
{
    "Track": "Task Planning",
    "Team": "Alpha Squad",
    "Version": "2.1",
    "Submitted": "2025-11-04T15:30:00Z",
    "Methodology": [
        "Enhanced prompt structure",
        "Improved agent descriptions",
        "Added emoji formatting"
    ]
}
```

---

## ❓ Frequently Asked Questions

### Q: Why does fact_sheet.json exist if it's so minimal?

**A:** It's designed for extensibility. Right now it's minimal for simplicity, but the infrastructure allows adding more metadata later without breaking existing submissions.

### Q: Can I add a "Team" field for credit?

**A:** Not for this competition. The infrastructure doesn't support it yet. Team attribution might come in future versions.

### Q: What if I accidentally modify fact_sheet.json?

**A:** If it's minor (e.g., just formatting), it might still work if the JSON is valid. But the safe approach: restore from backup and don't modify it.

### Q: Does the order of JSON keys matter?

**A:** No. JSON is order-independent. But don't add keys you don't need.

### Q: Can I use fact_sheet.json to explain my strategy?

**A:** Not yet. That feature could come in future competitions. For now, it's just metadata.

### Q: What if I submit the wrong Track value?

**A:** Your submission goes to the wrong evaluator. If you're doing Track 1 but say "Task Execution", the scoring will fail or be incorrect.

### Q: Is fact_sheet.json used for tie-breaking?

**A:** No. Only your code score matters.

### Q: Can two teams have different fact_sheet.json files?

**A:** No. For this competition, all Track 1 submissions use the same `track1_fact_sheet.json`.

---

## 🎓 Key Insights

### Why This Design?

1. **Separation of concerns** - Code logic separate from metadata
2. **Future-proof** - Can add fields later without breaking old submissions
3. **Fair competition** - Metadata can't affect scoring
4. **Simplicity** - Start minimal, add complexity as needed

### What This Tells Us About Competition Design

- **Phase 1:** Focus on evaluating core logic (code)
- **Phase 2:** Add metadata tracking (later)
- **Phase 3:** Enable rich documentation (future)

### Strategic Implications

- **For this year:** Don't worry about fact_sheet.json
- **For next year:** Could use it for team credits
- **For research:** Could include methodology details
- **For industry:** Could enable reproducibility

---

## 📞 Support

### If Something Goes Wrong

| Issue | Action |
|-------|--------|
| JSON syntax error | Restore from original backup |
| Wrong Track value | Edit back to correct value |
| File missing | Copy from original repository |
| Unclear format | Refer to this guide |

### Where to Find Original

```bash
# Track 1
src/agent_hive/workflows/track1_fact_sheet.json

# Track 2
src/agent_hive/workflows/track2_fact_sheet.json

# Git (if you modified it)
git checkout src/agent_hive/workflows/track1_fact_sheet.json
```

---

## ✨ Summary Table

| Aspect | Detail |
|--------|--------|
| **File Name** | `track1_fact_sheet.json` or `track2_fact_sheet.json` |
| **Location** | `src/agent_hive/workflows/` |
| **Current Size** | 2-3 lines of JSON |
| **Fields** | 1 (just "Track") |
| **Editability** | ❌ NO - Don't modify |
| **Required?** | ✅ YES - Must include in ZIP |
| **Affects Score?** | ❌ NO - Not scored |
| **Validation?** | ✅ YES - Must be valid JSON |
| **Can be Extended?** | 🔮 YES - In future competitions |

---

## 🚀 Quick Action Items

1. **DO:** Include `track1_fact_sheet.json` in your ZIP submission
2. **DON'T:** Modify its contents
3. **DO:** Verify it's present before uploading
4. **DON'T:** Change the "Track" value
5. **DO:** Focus your efforts on editing `track1_planning.py` instead

---

## 📚 Related Documentation

- `COMPETITION_EDITABLE_SECTIONS.md` - What you CAN edit
- `CONNECTION_MAP.md` - How components work together
- `QUICK_REFERENCE.md` - Quick lookup guide
- `Submission_CODS.md` - Official submission guidelines

---

**Version:** 1.0  
**Created:** November 4, 2025  
**For:** AssetOpsBench CODS Competition  
**Status:** ✅ Complete Reference
