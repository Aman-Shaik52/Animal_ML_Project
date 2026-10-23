# Zoo Dataset MLOps Practicals 8–11

This template adapts the uploaded Student Result Prediction practical pattern to the existing Zoo animal classification project.

## Before running

Keep this existing trained model artifact in the project:
`artifacts/animal_pipeline.joblib`

It must be the full scikit-learn pipeline trained from `zoo3.csv`, with `animal_name` excluded and `class_type` as the target. Do not retrain the model for practicals 8–11.

Expected project layout:

```text
Animal_ML_Project/
├── app.py
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .github/
│   └── workflows/
│       └── docker-publish.yml
└── artifacts/
    └── animal_pipeline.joblib
```

Keep your existing `zoo3.csv`, `ml_pipeline.py`, Prefect files, `outputs/`, and `artifacts/` in the same project folder. Copy the files from this template into that folder; do not replace your existing model artifact.

## Practical 8 — Build and publish to GHCR

In PowerShell, from the project folder:

```powershell
docker --version
docker build -t animal-classification-api:latest .
docker run -d --name animal-api-local -p 8000:8000 animal-classification-api:latest
```

Test in browser:
- `http://localhost:8000/`
- `http://localhost:8000/health`
- `http://localhost:8000/docs`

Stop local test container:
```powershell
docker stop animal-api-local
docker rm animal-api-local
```

Create a GitHub Personal Access Token (classic) with `write:packages` and `read:packages` for manual publishing. Never paste the token into source code or screenshots. Then run, replacing placeholders only in your terminal:

```powershell
docker login ghcr.io -u YOUR_GITHUB_USERNAME
docker tag animal-classification-api:latest ghcr.io/YOUR_GITHUB_USERNAME/animal-classification-api:v1
docker tag animal-classification-api:latest ghcr.io/YOUR_GITHUB_USERNAME/animal-classification-api:latest
docker push ghcr.io/YOUR_GITHUB_USERNAME/animal-classification-api:v1
docker push ghcr.io/YOUR_GITHUB_USERNAME/animal-classification-api:latest
```

If the GHCR package is private, authenticate before pulling. Verify the published image:
```powershell
docker pull ghcr.io/YOUR_GITHUB_USERNAME/animal-classification-api:v1
```

## Practical 9 — GitHub Actions CI/CD

1. Put `.github/workflows/docker-publish.yml` in your GitHub repository.
2. Confirm the repository includes `app.py`, `requirements.txt`, `Dockerfile`, and `artifacts/animal_pipeline.joblib`.
3. Ensure the repository's Actions settings permit workflow runs and the existing package grants this repository write access if the package already exists.
4. Commit and push to `main`:
```powershell
git add app.py Dockerfile requirements.txt .dockerignore .github/workflows/docker-publish.yml artifacts/animal_pipeline.joblib
git commit -m "Add animal classification API and CI workflow"
git push origin main
```
5. Open GitHub → Actions and confirm build, `/health` test, and GHCR push pass.
6. The package name is `animal-classification-api`; workflow publishes `latest` and `sha-<commit>` tags.

## Practical 10 — Deploy the GHCR image to Render

1. Confirm the latest GitHub Actions run succeeded.
2. In Render, create a Web Service using **Existing Image**.
3. Image reference: `ghcr.io/YOUR_GITHUB_USERNAME/animal-classification-api:latest`.
4. If the package is private, add registry credentials with only the required read permission.
5. Set health-check path to `/health`. The image listens on `0.0.0.0` and uses `PORT` (defaults to 8000).
6. Deploy, then test the actual Render URL:
   - `/`
   - `/health`
   - `/docs`
7. To update after a new image is published, use Render → Manual Deploy → Deploy latest reference.

## Practical 11 — Logging and error tracking

The `app.py` already adds:
- model load success/failure logs;
- request ID, method, path, status, and duration logs;
- validation warnings for invalid request bodies;
- successful prediction and inference error logs;
- `X-Request-ID` response header.

Logs go to stdout so local terminal, Docker logs, and Render logs can capture them.

Local test:
```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/docs`. Example request for a mammal-like animal (16 features are required):

```json
{
  "hair": 1,
  "feathers": 0,
  "eggs": 0,
  "milk": 1,
  "airborne": 0,
  "aquatic": 0,
  "predator": 1,
  "toothed": 1,
  "backbone": 1,
  "breathes": 1,
  "venomous": 0,
  "fins": 0,
  "legs": 4,
  "tail": 1,
  "domestic": 0,
  "catsize": 1
}
```

This is a feature example for testing the endpoint, not a guaranteed label for a particular animal. Use the same JSON in `POST /predict`.

After building/running the container, inspect logs with:
```powershell
docker logs animal-api-local
```

To produce a validation warning, send a request missing a required field or with an invalid value such as `legs: 99`. Avoid logging private data or secrets.

## Important notes

- This is a multi-class classification API: response key is `predicted_class_type`, and valid labels are the `class_type` classes from your Zoo dataset.
- Do not send `animal_name` or `class_type` in the request; the API expects the 16 model features.
- Ensure the model artifact is committed and available to Docker build. If it is excluded by `.gitignore`, Docker/GitHub Actions will not be able to package it until you intentionally include it.
- If the model was serialized with a different scikit-learn version, use the same scikit-learn version used when creating the artifact to avoid compatibility warnings or errors.
