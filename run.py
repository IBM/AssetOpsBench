#!/usr/bin/env python3
"""Industrial Automation Challenge submission script."""

from __future__ import annotations

import logging

from eval_framework import CompetitionKit, build_predictor_from_args, create_metadata_parser, load_and_merge_config


def main() -> int:
    parser = create_metadata_parser()
    args = load_and_merge_config(parser.parse_args())

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    kit = CompetitionKit(config_path=args.config)
    if args.output_dir:
        kit.output_dir = kit.output_dir.__class__(args.output_dir)
        kit.output_dir.mkdir(parents=True, exist_ok=True)

    predictor = build_predictor_from_args(args)
    print("Industrial Automation Challenge - Submission Generation")
    kit.list_datasets()

    result = kit.run_predictions(
        predictor,
        subset_size=args.subset_size,
        dataset_path=args.dataset_path,
    )
    submission_path = kit.save_submission(result, filename=args.output_file)

    print(f"Processed scenarios: {len(result.predictions)}")
    print(f"Submission CSV: {submission_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
