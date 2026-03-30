# Track 1 Submission Guidelines

This document provides the steps required to prepare, test, and submit your work to **CodaBench** for the **AssetOpsBench Competition**, more specific for **Track 1**.

---

## 0. Prerequisites for Track 1 Competition

Please ensure the competition code is ready on your local machine.  
Refer to the [README_CODS.md](https://github.com/IBM/AssetOpsBench/blob/Competition_CODS/benchmark/cods_track1/README_CODS.md) for details.

1. **Clone the Repository Locally**  
   - Use GitHub Desktop, CLI, or another tool to clone your fork of [IBM/AssetOpsBench](https://github.com/IBM/AssetOpsBench).  
   - Make sure you are on the **`Competition_CODS`** branch.  
   - Recommended local path:  
     ```
     <your_local_directory>/codabench/AssetOpsBench
     ```

2. **Verify Setup**  
   
   - Confirm that the branch is set to `Competition_CODS`.  
   - Your local files should match the competition branch.
   
3. **Code Editing Environment**  
   - You can use **Visual Studio Code** or any other IDE/text editor to view, modify, and test the code locally.



## 1. Access the Track 1 Competition  

### Access the CodaBench AssetOpsBench — Track 1 Page  

1. Open the CodaBench website:  
   👉 [https://www.codabench.org/](https://www.codabench.org/)  

2. Log in with your account credentials.  

3. Navigate to the **AssetOpsBench Competition** page:  
   👉 [https://www.codabench.org/competitions/10206](https://www.codabench.org/competitions/10206)  

4. In the left-hand menu, under **Get Started**, click **Track 1: Task Planning**.  
   
---

## 2. Understand Your Competition – Track 1  

- The challenge is about **better planning prompts** — all teams use the same fixed agents, ReAct agent, and Executor.  
- For **local testing**, you can use the baseline scaffold in **[`planning_review.py`](https://github.com/IBM/AssetOpsBench/blob/Competition_CODS/src/agent_hive/workflows/planning_review.py)**.  
  - Local path after cloning: `src/agent_hive/workflows/planning_review.py`  
  - This file lets you **practice and debug locally** with a few sample scenarios.  
  - It shows you what kind of output a planning prompt produces and helps verify your setup.  
  - It is **not submitted** — only for local testing.  
- For the **official submission**, you must edit **[`track1_planning.py`](https://github.com/IBM/AssetOpsBench/blob/Competition_CODS/src/agent_hive/workflows/track1_planning.py)**.  
  - Local path after cloning: `src/agent_hive/workflows/track1_planning.py`  
  - Only the marked **TODO** section can be modified to improve how agent information is collected and formatted.  
- Submissions are evaluated with **LLaMA-3-70B** on public scenarios (Phase 1) and unseen scenarios (Phase 2); final scores combine **task accomplishment** and **generalization**.  

---

## 3. What Need to Do

There are **two TODO sections** in `src/agent_hive/workflows/track1_planning.py`:

1. **Scenario 1 – Agent Info Formatting**  
   - **Where:** `generate_steps()` (lines 59–82)  
   - **Allowed:**  
     - Change numbering style or bullet points  
     - Include additional metadata (e.g., agent capabilities, tags)  
     - Provide examples in a different format  
     - Add emojis or formatting to make the prompt clearer  
     - More thinking  

2. **Scenario 2 – Prompt Template**  
   - **Where:** `get_prompt()` (lines 161–191)  
   - **Allowed:**  
     - Wording  
     - Structure  
     - Examples  
     - Emojis  

👉 To find them quickly, search in the file for:  
	TODO: Participants can edit this section ONLY

## 4. Local Test Before Submission

Before creating your submission, you should verify that your code runs correctly in a local environment.  
For Track 1, use the provided Docker Compose file under `benchmark/code_track1/`.

### Steps

1. **Run with Docker (Track 1)**
   ```bash
   docker-compose -f benchmark/code_track1/docker-compose.yml up

- This will start the competition environment and run your modified Track 1 workflow (`track1_planning.py`).

2. **Check the Logs**
- Watch the Docker terminal output.  
- Confirm that the service starts without errors.  
- **Tracking TO-DO changes**: any edits you make in the `# TODO` section of `track1_planning.py` will be picked up when the container runs.  
  - For debugging, add `print()` or `logger.info()` statements inside your changes.  
  - These messages will appear directly in the console output, confirming that your modifications are active.

3. **Iterate if Needed**
- If you modify code (e.g., `track1_planning.py`), stop the container and restart with the same command.  
- Repeat until the workflow runs smoothly and your changes display as expected.


---

## 5. Prepare Your Submission

1. Package your solution as a **ZIP file**.  
   - For Track 1, this means exactly **two files**:
     - **track1_planning.py**  
       - Path:  
         ```
         src/agent_hive/workflows/track1_planning.py
         ```
       - This is the main workflow file you modified (only edit the section marked with `# TODO`).
     - **track1_fact_sheet.json**  
       - Path:  
         ```
         src/agent_hive/workflows/track1_fact_sheet.json
         ```
       - Originally named `task_planning_fact_sheet.json`.  
         Rename it once to `track1_fact_sheet.json` before submission.  
         Do **not** modify its contents.

2. **Important rule**:  
   - No new files should be created.  
   - No file names can be changed  

3. Create the ZIP archive from your local project root:  
   ```bash
   cd src/agent_hive/workflows
   zip submission_track1.zip track1_planning.py track1_fact_sheet.json


---

## 6. Final Submission

1. Go to the [AssetOpsBench Competition page](https://www.codabench.org/competitions/10206).  
2. Click on the **“My Submissions”** tab or go directly to [this link](https://www.codabench.org/competitions/10206/#/participate-tab).  
3. Under **Submission upload**, make sure the **Track** is set to **Task Planning**.  
4. In **Submit as**, choose **Yourself** (or your team name if applicable).  
5. Click the **paperclip icon** to select and upload your prepared **ZIP file**.  
6. Once uploaded, press **Submit** to confirm.  

✅ You can make up to **100 submissions per day** (as shown on the page).  



## 7. After Submission
- Your job is automatically added to the competition pool — you do **not** need to create one manually.  
- After upload, the system will run your submission automatically.  
- You can track execution progress and view results on the **[My Submissions](https://www.codabench.org/competitions/10206/#/participate-tab)** page.  

---

✅ **Tip:** Keep a copy of your ZIP and logs for troubleshooting or future re-submissions.
