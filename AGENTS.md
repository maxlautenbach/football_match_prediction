# Agent instructions

Before creating or changing an ML model, follow [`docs/MODELS.md`](docs/MODELS.md).

Hard rules:

- Do **not** write candidate training output directly to `artifacts/`.
- Train via `scripts/train.py --recipe ...` (temp bundle → MLflow → `@candidate`).
- Promote to production only via `scripts/promote_run.py`.
- Keep the root `model.py` adapter stable; put algorithm code under `models/<model_type>/`.
- Prefer registry name constants from `eval/mlflow_utils.py` over hardcoding.
