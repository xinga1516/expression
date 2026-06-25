# Drosophila Cell-Type Promoter and 3'UTR Model Transfer Guide

This is a self-contained Codex-to-Codex method-transfer guide. Its purpose is
to let a separate Codex-assisted Drosophila project borrow the current HsP
promoter and 3'UTR modeling strategy and build cell-type-specific
cis-regulatory sequence models from Drosophila single-cell expression data.

Assume the receiving project has no access to HsP files, paths, data, run IDs,
reports, or archived notes. HsP is used only as the source of method lessons,
guardrails, and starting hyperparameters. This document should describe the
current recommended workflow, not the historical path that produced it.

The filename is kept for compatibility with prior references, but the scope is
now broader than promoter-only modeling: it covers both promoter and 3'UTR /
downstream sequence models.

## Goal

For one Drosophila cell and one target gene, predict that target gene's
expression from:

- the same cell's expression state with the target gene value masked; and
- the target gene's strand-oriented local regulatory sequence.

Build two related sequence branches:

- promoter/TSS-proximal sequence model;
- stop-proximal 3'UTR/downstream sequence model.

The promoter model is expected to carry the stronger regulatory signal. The
3'UTR model should be treated as a complementary design component unless local
Drosophila evidence proves otherwise.

Headline claims must be made on unseen genes, not only unseen cells. A useful
model must beat both expression-only and matched non-regulatory sequence
controls on held-out genes.

## Success Milestones

Stage 1 minimum success is a clean promoter result on unseen test genes:

- expression-only;
- matched promoter with `contrastive_weight=0`;
- matched non-promoter/intergenic control with `contrastive_weight=0`.

The matched promoter model must beat both baselines under the same gene split,
feature panel, checkpoint rule, and frozen final test-cell panel.

Stage 2 transfer milestone is a fair promoter-vs-3'UTR comparison figure
equivalent to:

`promoter_best_vs_utr_best_frozen_test_cells.png`

This figure should compare the current best promoter model and current best
3'UTR model on the same frozen test-cell panel and the same held-out test genes.
Each point should be one frozen test cell or documented cell-type aggregate:

- x-axis: promoter model prediction correlation across held-out test genes;
- y-axis: 3'UTR model prediction correlation across the same held-out genes;
- dashed `x=0` and `y=0` reference lines;
- clear labels or annotations for notable outlier cells/cell types;
- caption listing the exact model IDs, gene split, frozen test-cell panel,
  target-gene count, and correlation definition.

This plot is not required for Stage 1, but it is the minimum visual checkpoint
that shows both sequence branches are trained, evaluated on the same fair panel,
and ready for promoter-vs-3'UTR complementarity analysis.

## Transfer Contract

Preserve these rules even if implementation details differ:

1. Split primarily by genes.
2. Keep train, validation, and test genes disjoint.
3. Mask the target gene expression value in the input cell-state vector.
4. Select checkpoints on unseen validation genes.
5. Report final metrics on unseen test genes and a frozen final test-cell
   panel.
6. Compare sequence models against expression-only and matched intergenic or
   non-regulatory controls.
7. Keep promoter and 3'UTR claims separate unless explicitly analyzing their
   complementarity.
8. Never mix models trained under different gene splits, feature panels, or
   final eval-cell panels in one current leaderboard.

## Project Source Of Truth

Create these project-local files before long GPU work:

- `CURRENT_STATE.md`: compact current protocol, active runs, current best model,
  caveats, and next actions.
- `docs/RUN_RESULTS_SUMMARY.md`: interpreted results and why comparisons are
  fair or not fair.
- `docs/DECISIONS.md`: durable defaults and active rules.
- `records/registry.tsv`: one row per run with run ID, status, split, command,
  checkpoint, metrics, and interpretation status.
- `records/run_interpretation_exclusions.tsv`: excluded or superseded runs with
  exact reasons.
- `configs/queues/*.json`: reproducible long-run queue definitions.
- `runs/<run_id>/command.sh`, `status.tsv`, `provenance.json`, `summary.json`:
  run-local truth for command, progress, provenance, and final metrics.
- `reports/decisions/<question>/`: regenerated filtered reports for current
  decisions, not hand-edited leaderboards.

Before answering "which model is best", verify run-local command/config,
checkpoint metadata, split files, and final evaluation metrics.

## Data Processing

Keep the preprocessing order fixed so model comparisons remain interpretable.

1. Audit the atlas.
   - Record atlas version, cell IDs, metadata columns, expression layer,
     normalization, feature IDs, sparse/dense layout, and excluded cells.
   - Decide whether to train on individual cells, pseudo-bulk profiles, or both.
     The HsP main task uses individual cells.

2. Audit genome and annotation compatibility.
   - Record genome FASTA release, annotation release, contig naming, gene IDs,
     transcript IDs, strand, coordinates, and missing contigs.
   - Normalize gene IDs across atlas features, annotation, and genome.

3. Define one modeling gene universe before splitting.
   - Use expressed, mappable, non-ambiguous Pol-II-like genes.
   - Exclude genes with missing sequence, ambiguous TSS/stop codon mapping,
     poor ID mapping, or unreliable target expression.
   - Record inclusion/exclusion status in a gene table.

4. Choose representative transcript landmarks.
   - Promoter: choose one TSS per gene and preserve transcriptional strand.
   - 3'UTR/downstream: choose one stop codon or representative 3'UTR anchor per
     gene. If using stop-proximal downstream sequence, ensure the extracted
     window starts after the stop codon and never includes CDS or the stop
     codon itself.
   - If multiple isoforms are biologically important, keep isoform choice
     deterministic and record the rule rather than mixing transcript policies.

5. Build leakage-aware gene splits.
   - Prefer chromosome/scaffold holdout if enough genes remain.
   - Otherwise use grouped gene splits that keep close paralogs, duplicated
     promoters, or highly similar local sequence windows in the same split.
   - Train/validation/test target genes must be disjoint.

6. Build the cell-state input panel.
   - Select input features from train-split genes only.
   - HsP uses 4096 input genes ranked by recurrent high variance, global train
     variance, detection fraction, and stable gene-ID tie-break. Use the same
     principle for Drosophila, then test panel size only if it becomes a real
     bottleneck.
   - During every prediction, mask or zero the target gene expression value if
     that target appears in the input panel.

7. Freeze evaluation cell policy.
   - Use rolling or seed-sampled validation cells for checkpoint screening.
   - Use a frozen final test-cell panel for current comparisons.
   - Start with 2048 validation and 2048 test cells if the atlas supports it;
     otherwise use the largest stable documented panels.
   - The final test cells do not have to be excluded from the broad training
     cell pool if the headline claim is unseen genes and target leakage is
     masked, but this policy must be recorded.

8. Build sequence assets.
   - Promoter: build a wider cache than the final crop if using shift
     augmentation. HsP uses an r420 cache for an r400 model crop with shift20.
   - 3'UTR/downstream: build a stop-proximal cache that can support the final
     crop plus shift augmentation while staying downstream of the stop codon.
     HsP-style r400 means an 801 nt model crop.
   - Store extraction status, strand, contig, start, end, radius/window length,
     and sequence row index.

9. Build matched controls.
   - Promoter control: matched intergenic or non-promoter windows.
   - 3'UTR control: matched downstream/intergenic windows that do not overlap
     the target gene's annotated 3'UTR, CDS, or stop codon, and should avoid
     known regulatory annotations whenever annotations are available.
   - Record matching criteria: chromosome/scaffold, length, strand/orientation
     handling, distance from genes, GC/dinucleotide match if practical, and
     extraction status.

10. Run a smoke test before long queues.
    - Load atlas shards, input/target panels, gene splits, eval cells, promoter
      assets, 3'UTR assets, control assets, and one train/eval batch.
    - Confirm tensor shapes, target masking, sequence orientation, and final
      metric writing.

Recommended tables:

- `genes.tsv`: `gene_id`, `gene_name`, `contig`, `strand`, `tss`,
  `stop_codon_end`, optional `biotype`, inclusion flags.
- `gene_splits.tsv`: `gene_id`, `split`, `status`, `reason`.
- `cells.tsv`: `cell_id`, dataset/batch, broad cell type, fine cell type,
  inclusion flags.
- `frozen_eval_cells.tsv`: `cell_id`, `panel` as `validation` or `test`.
- `promoter_windows.tsv`: `gene_id`, `contig`, `start`, `end`, `strand`,
  cache radius, model radius, extraction status.
- `utr_windows.tsv`: `gene_id`, `contig`, `start`, `end`, `strand`, anchor,
  cache length, model crop length, extraction status.
- `control_windows.tsv`: `gene_id`, `control_id`, `control_kind`, `contig`,
  `start`, `end`, `strand`, match criteria, extraction status.

## Model Contract

One supervised example predicts one `(cell, target_gene)` expression value.

Inputs:

- cell-state vector: selected expression features for that cell;
- target gene identifier: used only to look up the masked input position, target
  expression column, and sequence row/crop. Do not add a learned target-gene ID
  embedding for unseen-gene headline evaluation; if used, label it as a
  seen-gene diagnostic.
- sequence crop: promoter or 3'UTR/downstream sequence for that target gene.

Required masking:

- If the target gene is present in the cell-state input panel, mask or zero its
  input value before prediction.
- Randomly masking additional input genes is not a default. Test it only as a
  labeled ablation.

Starting architecture:

- cell encoder: dense projection from input expression features to a cell-state
  embedding;
- sequence encoder: `cnn_flatten`;
- fusion: concatenate cell-state and sequence embeddings;
- regressor: two hidden dense layers for promoter; one hidden dense layer is an
  acceptable conservative starting point for 3'UTR if data are limited;
- optional auxiliary heads such as validity/naturalness are filters or labeled
  ablations, not the default expression predictor.

Keep expression-only as the same cell-state encoder/regressor without sequence
input. Use the same gene split and eval-cell policy.

## Starting Training Defaults

Treat these as HsP-derived priors, not Drosophila facts. Run small matched
3-seed gates before promoting a default.

Promoter starting default:

- sequence kind: promoter;
- crop: r400 from a shift-compatible wider cache such as r420;
- encoder: `cnn_flatten`;
- input features: 4096 train-gene-derived features;
- EMA: `ema_decay=0.9999`;
- checkpointing: patience32 for screening; use patience64 or an epoch-aware
  adaptive extension for finalist confirmation;
- train-time shift: `promoter_shift_max=20`;
- clean evidence branch: `contrastive_weight=0`;
- enhancement branch: triplet `contrastive_weight=0.05` only after cw0 matched
  promoter beats expression-only and control; use only for matched
  real-promoter runs with documented intergenic negatives;
- controls: `contrastive_weight=0`;
- LR: `5e-4`;
- `eval_every_steps=512`;
- `max_train_cells_per_epoch=65536`;
- `targets_per_cell=128`;
- `pair_batch_size=512`;
- `eval_pair_batch_size=4096`;
- `regressor_hidden_layers=2`;
- `sequence_validity_weight=0`;
- rolling validation cells and frozen final test cells, 2048/2048 if possible.

3'UTR/downstream starting default:

- sequence kind: 3'UTR/downstream;
- crop: r400 / 801 nt stop-proximal model crop from a shift-compatible cache;
- ensure random shifts never cross into CDS or include the stop codon;
- encoder: `cnn_flatten`;
- input features: same 4096 cell-state feature panel;
- EMA: `ema_decay=0.9999`;
- checkpointing: patience32 for screening; use patience64 or adaptive extension
  for finalist confirmation;
- train-time shift: shift20 if valid within the downstream cache;
- contrastive: start with `contrastive_weight=0`;
- LR: start with `3e-4`;
- `eval_every_steps=512`;
- `targets_per_cell=32` as a conservative first pass; test 128 only as a
  labeled ablation if runtime and data support it;
- `pair_batch_size=512`;
- `eval_pair_batch_size=4096`;
- no validity head by default.

General training rules:

- no hard epoch cap unless debugging;
- use detached, recoverable queues for long GPU work;
- reserve enough CPU and memory for the machine; do not maximize GPU use by
  exhausting CPU workers;
- start with 3-seed gates, expand only when trends are close or promising;
- record exact commands and run-local summaries.

## Minimal Evidence Ladder

Use the same gene split, feature panel, validation policy, and frozen final test
panel for all rows in a comparison.

1. Data and leakage audit.
   - Verify gene IDs, contigs, TSS/stop anchors, sequence extraction, target
     masking, and non-overlap of train/validation/test target genes.

2. Smoke tests.
   - One expression-only train/eval batch.
   - One promoter train/eval batch.
   - One 3'UTR train/eval batch.
   - One matched-control train/eval batch.

3. Promoter clean 3-seed gate.
   - expression-only;
   - matched promoter with cw0;
   - matched non-promoter/intergenic control with cw0.

4. Promoter enhancement branch.
   - matched promoter with triplet cw0.05, only after the clean cw0 gate passes.

5. 3'UTR 3-seed gate.
   - expression-only or shared expression-only reference;
   - matched 3'UTR/downstream sequence;
   - matched intergenic/downstream control.

6. Only after matched sequence beats controls, test refinements.
   - EMA strength if curves are noisy;
   - shift augmentation versus fixed anchor;
   - LR;
   - eval cadence;
   - targets per cell;
   - regressor depth;
   - contrastive or validity heads as labeled ablations.

Report separately:

- matched-comparison result: only runs where the intended variable differs;
- operational default: the practical setting chosen after considering metrics,
  curve stability, runtime, and failure modes.

## Metrics And Model Selection

Checkpoint selection:

- select checkpoints by validation genes, not test genes;
- validation cells may be rolling or seed-sampled;
- final test-cell panel must be frozen for current comparisons.

Headline metrics:

- RMSE;
- global Pearson r across all evaluated `(cell, gene)` pairs;
- per-gene Pearson distribution on frozen test cells;
- per-cell or per-cell-type Pearson distribution across held-out genes;
- expressed-vs-zero AUROC/AP if the expression transform supports a meaningful
  zero/nonzero boundary;
- zero/nonzero RMSE when available.

Interpretation:

- promoter should beat expression-only and matched non-promoter control;
- 3'UTR should beat matched intergenic/downstream control before being used as
  a design signal;
- external datasets support domain-shift generalization but should not be
  treated as same-distribution RMSE evidence unless normalization is aligned.

## Generalization Experiments

Run these after the initial gates.

1. Frozen test-cell generalization.
   - Evaluate all current candidate models on the same held-out genes and frozen
     test cells.
   - Plot per-gene r and per-cell r.

2. External expression-profile generalization.
   - Prepare a Drosophila external dataset such as bulk RNA-seq, cell-line
     RNA-seq, perturbation RNA-seq, or independent single-cell atlas profiles.
   - Align it to the same input feature panel and target gene universe.
   - Fill missing input features with training-panel means or exclude them by a
     documented rule.
   - Use rank/correlation evidence first; do not over-interpret RMSE across
     different expression platforms.

3. Promoter-vs-3'UTR complementarity.
   - On the same held-out genes and cells, compare promoter per-gene r against
     3'UTR per-gene r.
   - Also compare promoter per-cell or per-cell-type r against 3'UTR per-cell
     or per-cell-type r.
   - Shared high-predictability genes suggest gene-level expression programs;
     promoter-dominant or 3'UTR-dominant outliers suggest sequence-branch
     specific biology worth inspecting.

4. Sequence-negative controls.
   - For promoter and 3'UTR separately, compare real sequence against matched
     intergenic/non-regulatory sequence using the same expression background and
     evaluation panel.
   - A nonzero intergenic signal is not automatically a failure; it may reflect
     genomic context or matching artifacts. The real sequence must still beat
     the control clearly enough to support sequence-specific claims.

## Recommended Figures

Use a consistent plotting style across model families.

- Training curves:
  - x-axis: global optimizer step;
  - plot train and validation losses independently min-max normalized if their
    scales differ;
  - mark full-epoch boundaries;
  - show best checkpoint step/eval.

- Frozen test per-gene scatter:
  - x-axis: promoter per-gene r;
  - y-axis: 3'UTR per-gene r;
  - add dashed `x=0` and `y=0` lines;
  - label outliers in all quadrants, including upper-right and lower-left;
  - avoid overlapping labels when possible.

- Frozen test per-cell or per-cell-type scatter:
  - canonical filename:
    `promoter_best_vs_utr_best_frozen_test_cells.png`;
  - each point is a cell or summarized cell type;
  - x-axis: promoter prediction r across held-out genes;
  - y-axis: 3'UTR prediction r across held-out genes;
  - add zero reference lines and annotate notable outliers.

- External profile scatter:
  - each point is an external cell line, tissue, perturbation, or pseudo-bulk
    profile;
  - compare promoter and 3'UTR ranking performance across target genes;
  - clearly label the external dataset and normalization.

- Control comparison plots:
  - matched real sequence versus intergenic/non-regulatory control;
  - show paired seed/group summaries, not only the best seed.

R/ggplot2 is a good default for final biological figures, but Python is fine
for automated diagnostics if the output is stable and readable.

## Generation-Stage Notes

The eventual application is to design cell-type-specific promoters and 3'UTRs
for transgene expression. Do not nominate candidates from a single model score.

Candidate nomination should combine:

- target-vs-background specificity score;
- ensemble mean and uncertainty penalty across seeds;
- robustness to small sequence perturbations;
- naturalness or validity filters;
- promoter-specific safety filters such as avoiding strong cryptic splice donor
  motifs when they could interfere with downstream coding sequence usage;
- basic sequence constraints such as homopolymers, repeats, restriction sites,
  synthesis constraints, and unwanted ORFs when relevant.

First run a small audited generation smoke: a few targets, short optimization,
known seed set, full provenance, and manual inspection of score components.

## Stop And Re-Audit If

- gene ID mapping is poor or ambiguous;
- train/validation/test genes are not disjoint;
- target expression leakage is detected;
- validation/test genes are too few for stable claims;
- 3'UTR windows include CDS or stop codon sequence by mistake;
- non-regulatory controls are weakly matched or overlap regulatory annotations;
- promoter or 3'UTR appears to win only on seen-cell holdout;
- conclusions depend on a single seed;
- old runs with different splits or eval panels re-enter current comparisons.

## What Not To Do

- Do not copy HsP run IDs, paths, or historical defaults.
- Do not start with broad architecture search.
- Do not treat BiLSTM, validity heads, random input masking, or redesigned
  contrastive losses as first-line defaults.
- Do not apply positive contrastive learning to control-sequence runs.
- Do not mix promoter and 3'UTR results into one claim unless the analysis is
  explicitly about complementarity.
- Do not compare models trained on different gene splits, feature panels, or
  final eval-cell panels as if they were one leaderboard.
