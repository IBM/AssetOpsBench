# Q & A Page

**We create this Q&A page to share responses to participant questions and keep the competition fair. Please open GitHub issues or Kaggle discussions for questions. Thank you!**

Can I use external tools in Track 1?

> No. Track 1 is internal model reasoning. The model should answer from its internal parameters during inference.

Can I use tools in Track 2?

> Yes. Track 2 is agentic tool-augmented reasoning. Tool usage should follow the competition rules and be documented in the submission metadata when applicable.

What should the Kaggle submission look like?

> The default submission CSV contains `id` and `answer`. `answer` should be the selected option letter.

Can I change prompts during submission?

> Yes. You may choose your own prompting strategy as long as it follows the rules for the track you are entering.

Which files should I start with?

> Use `metadata_config_test.json` for a test submission run, `metadata_config_val.json` for a validation run, and `run.py` to generate the submission package after setting the dataset and predictor paths.
