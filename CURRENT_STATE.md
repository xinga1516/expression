# Current State

Active protocol: promoter Stage 1 clean gate.

Scope:
- ExpressionBaseline
- CNNFlattenPromoterModel with real promoter sequence
- CNNFlattenPromoterModel with matched intergenic control sequence

Current rules:
- Use gene-disjoint train/validation/test splits.
- Use train-gene-derived input expression panel.
- Use frozen validation/test cell panels from Stage 1 assets.
- Select best checkpoints by validation RMSE for Stage 1 runs.
- Do not include 3'UTR, shift augmentation, contrastive loss, or generation in Stage 1.

Next actions:
- Build `data/promoter_stage1_v1`.
- Run tiny smoke tests.
- Run three-seed Stage 1 gate.
