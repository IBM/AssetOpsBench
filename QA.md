# Q & A Page

**We use this page to share participant questions and answers publicly so everyone has the same information. Please use GitHub issues or the Kaggle discussion page for additional questions. Thank you!**

What are the tracks?

> The competition has Track 1 and Track 2.

Do Track 1 and Track 2 use different input data filenames?

> No. Both tracks use the same Kaggle input data filename. Download the data file from the Kaggle Data tab for the track you are entering and set its local path in the config file.

What should the Kaggle submission look like?

> Submit a CSV with `id` and `answer`. `answer` should be the selected option letter.

Which files should I start with?

> Use `metadata_config_test.json` for the test split, `metadata_config_val.json` for the validation split, and `run.py` to generate `competition_results/submission.csv` after setting the dataset and predictor paths.

Can I change prompts or model code?

> Yes. You may use your own prompting, model, and inference code as long as your submission follows the rules for the track you are entering.
