# Scenario 4: the Chiller 6 triage, as a dialog

Scenario 3 asks for the same investigation in one prompt with six numbered
steps. This is the conversational form of it: the engineer arrives with a
vague first question and narrows down, the way the dialog paper describes real
O&M users behaving.

| Turn | Asks | Graded on |
| --- | --- | --- |
| 1 | Which MAIN work order is awaiting approval | `1000045` |
| 2 | Which asset and which measurement | registry name plus exact sensor |
| 3 | Retrieve the series, write it to disk, report its shape | 2876 observations, June 2020 |
| 4 | Run the detector **over the file from turn 3** | `anomalies_found` |
| 5 | The approver's paragraph | characteristic form |

## What each turn is for

Turn 1 is deliberately underspecified. The engineer says "work that has been
raised but not yet released", not `status == WAPPR`.

Turn 2 opens with "that work order". Nothing in the sentence identifies it, so
the turn is unanswerable without turn 1. It also separates the work order's
`assetnum` (`CHILLER6`) from the registry's name (`Chiller 6`), which is the
kind of identifier mismatch that produces confident wrong answers.

Turn 3 is the expensive turn and the one that leaves an artifact on disk.

Turn 4 is the measurement this dialog exists for. It names the file from the
previous turn and forbids re-retrieval. Under `m1` the agent reads the file
that turn 3 left behind. Under `m0` it cannot comply at all, because the file
is in a workspace it was never shown. A trajectory that retrieves the sensor
history again on turn 4 has failed the instruction even if the answer is right,
so artifact reuse is scored rather than assumed.

Turn 5 tests whether figures survive the dialog. Re-deriving them is a failure
mode as much as getting them wrong.

## Groundtruth provenance

Every value except turn 4 was computed from the files in `shared/` rather than
copied from scenario 3:

- turn 1: the only row in `workorders.csv` with `siteid=MAIN, status=WAPPR`
- turn 2: `asset_profile_sample.json`, matched on the work order's `location`
- turn 3: `chiller_6.json` holds 2896 records, of which 20 carry no value for
  `Chiller 6 Condenser Water Flow`, leaving 2876 over 2020-06-01T00:00:00 to
  2020-06-30T23:45:00
- turn 4: inherited from scenario 3's `anomalies_found`, since running the
  detector needs the TSFM server

Note that turn 3's `end` differs from scenario 3's. See the note below.

## A discrepancy in scenario 3

Scenario 3's groundtruth gives `end` as `2020-06-26T11:14:36`. That timestamp
is one of the 20 records that carry **no** value for the measurement, and
scenario 3's own prompt excludes those: "records where it is absent are not part
of the analysed series." Its `observations: 2876` already reflects the
exclusion, so the two fields disagree with each other.

The last timestamp of the analysed series is `2020-06-30T23:45:00`. This
scenario uses that. Scenario 3 looks like it needs the same correction.
