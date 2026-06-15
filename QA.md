# Q & A Page

**We create this Q&A page to share responses to participant questions and keep the competition fair. Please open GitHub issues or Kaggle discussions for questions. Thank you!**

What are the competition tracks?

> The competition has Track 1 and Track 2.

Do Track 1 and Track 2 use different input data filenames?

> No. Both tracks use the same input data filename on Kaggle. Download the data file from the Kaggle Data tab for the track you are entering and set the local path in the config file.

What should the Kaggle submission look like?

> Submit a CSV with `id` and `answer`. `answer` should be the selected option letter.

Can I change prompts during submission?

> Yes. You may choose your own prompting strategy as long as it follows the competition rules for the track you are entering.

Which files should I start with?

> Use `metadata_config_test.json` for a test submission run, `metadata_config_val.json` for a validation run, and `run.py` to generate `submission.csv` after setting the dataset and predictor paths.
