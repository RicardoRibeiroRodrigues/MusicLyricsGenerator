# Model Deployment

To run locally:
1. Run one of the Docker versions for deploying the backend:

To use CPU for inference:

```sh
docker run -p 8000:8000 ricardorr7/deploy-api-musicgen:latest-cpu
```

To use GPU for inference:

```sh
docker run -p 8000:8000 ricardorr7/deploy-api-musicgen:latest-gpu
```

2. Then, with Docker running, start an HTTP server for the frontend:
```sh
python -m http.server 8080 --directory ./frontend
```

(You can also simply drag the .html file into your web browser.)
