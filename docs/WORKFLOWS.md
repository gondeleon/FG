# Workflows (Git + development)

## Branching

- `main`: stable
- `feature/<topic>`: one deliverable per branch (e.g., `feature/ros2-source`)

Recommended pattern:

1. Create branch from `main`
2. Small commits, tests passing
3. Merge request into `main`
4. Tag stable milestones

## Tagging milestones

Use annotated tags:

```bash
git tag -a v0.1.0-baseline -m "Baseline working"
git push origin v0.1.0-baseline
```

## Commit hygiene

- Keep estimator/FG changes separate from I/O/streaming changes.
- Prefer commits that are individually testable.

## Testing

```bash
pytest
```

- End-to-end tests will be skipped if `gtsam` is missing.

## Reproducibility

- Keep `configs/` versioned.
- Keep raw datasets out of git (`data/`, `datasets/`).
- Consider Git LFS or DVC for large logs.
