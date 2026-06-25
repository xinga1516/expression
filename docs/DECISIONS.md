# Decisions

## Promoter Stage 1

- Include only protein-coding genes in the Stage 1 clean gate.
- Infer protein-coding genes from GTF `mRNA` plus coding evidence (`CDS`, `start_codon`, or `stop_codon`).
- Record lncRNA-like annotations as `lncRNA_candidate`, but exclude them from Stage 1 because the local GTF symbol rule does not prove Pol II promoter status.
- Exclude lncRNA_candidate, tRNA, rRNA, miRNA, pre_miRNA, snoRNA, snRNA, pseudogene, ambiguous, and unknown classes.
- Use fixed train-gene input panel.
- Use `val_rmse` for Stage 1 checkpoint selection.
- Run standard test automatically after training when `--run-test-after-train` is set.
