---
name: evidence-and-abstention
description: "Decides whether the evidence actually reaches the conclusion about to be
  stated, and what to do when it does not. Open this before writing any answer that
  carries a number, an identifier, a date or a diagnosis, when a request is
  underspecified about which asset, sensor or window it means, when part of an answer
  would have to be inferred rather than retrieved, or when a tool returned less than
  was asked for and the temptation is to fill the rest in. Abstention is a scored
  outcome in this environment, not a failure to answer, and the distinction between
  what was retrieved and what was assumed is the thing being measured."
disable-model-invocation: true
license: Apache 2.0
metadata:
  disco-role: operating
  capability-family: C12
  asset-class: A0
  leakage-class: ops
  library-version: 0.1.0
---

# Evidence, and when to decline

## The mistake this prevents

Writing a complete-looking answer in which one element was retrieved and the
rest was reconstructed from what would be reasonable. The reconstructed parts
are indistinguishable from the retrieved parts in the prose, which is exactly
why the execution record is scored and not only the claim.

The failure has a signature. An answer that names a temperature, a work-order
id, or a date that appears in no tool result is not a small inaccuracy in an
otherwise good answer. It is the specific thing this environment is built to
detect.

## Preconditions

- [ ] You can name, for each factual element of the answer, the call that
      produced it.
- [ ] Identifiers in the answer were returned by a call, not composed.
- [ ] Any arithmetic was performed, in a tool or in the code workspace, not
      estimated.

If you cannot tick these, the answer is not ready and the fix is another
retrieval or a narrower claim, not better prose.

## Procedure

1. **Separate the request into what was asked and what was withheld.** Operator
   requests routinely omit the site, the sensor, or the window. That omission is
   part of the task: the investigation is yours to do. It is not licence to pick
   a plausible default silently.

2. **Do the retrieval you can, then look at what is left.** Three outcomes, and
   they are different answers:
   - Everything resolved. State the conclusion and cite the calls.
   - The gap is closable by another call. Make it.
   - The gap is not closable. Go to step 3.

3. **When the gap is not closable, choose between asking and declining.**
   - **Ask** when one specific missing fact would unblock everything and only the
     requester has it: which of three assets they meant, which window matters.
     Ask for that one fact, not for a restatement of the request.
   - **Decline the specific claim** when the environment cannot supply the
     evidence at all: the sensor is not instrumented, the stream does not cover
     the window, the server is down. Say which claim you are declining and why,
     and give the part of the answer that does hold.

4. **Never let the shape of the question dictate the shape of the answer.** A
   question phrased as "which failure mode is this" invites a named mode. If the
   evidence supports a set of two modes and not one, the answer is the set. A
   confident single mode drawn from an ambiguous signature is wrong even when it
   happens to be right, because the reasoning does not carry.

5. **Write the answer so the evidence is traceable.** For each claim, the call
   that supports it. This is not ceremony; it is what makes the answer checkable
   by someone who was not watching, and it is what an evidence-scored evaluation
   reads.

## Interpretation

| What you have | What the answer is |
| --- | --- |
| Every element retrieved | The conclusion, with its calls |
| The conclusion holds, one supporting detail does not | The conclusion, with the unsupported detail removed rather than softened |
| Two conclusions fit the evidence equally | Both, named as a pair, with the test that would separate them |
| The window or asset is ambiguous and it changes the answer | One question naming the specific ambiguity |
| The evidence does not exist in this environment | An explicit decline for that claim, plus whatever else holds |

## Failure modes of this skill

- **Abstention can be overused.** Declining when a further retrieval would have
  closed the gap is a failure too, and a lazier one. Exhaust the retrievals
  before you decline.
- **It does not tell you whether a result is physically possible.** An
  efficiency above one is fully supported by the calls that produced it and
  still wrong. Admissibility is a domain judgement and lives elsewhere.

## Stop conditions

Stop and report rather than completing the answer if a required identifier never
resolved, if the only remaining route to a number is to assume it, or if the
question cannot be answered without a fact the environment does not hold.
