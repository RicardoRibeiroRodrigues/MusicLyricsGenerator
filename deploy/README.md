# Deploy dos modelos

Para rodar local:
1. Rode uma das versões do docker do deploy do backend:

Para usar a CPU nas inferências:

```sh
docker run -p 8000:8000 ricardorr7/deploy-api-musicgen:latest-cpu
```

Para usar a GPU nas inferências:

```sh
docker run -p 8000:8000 ricardorr7/deploy-api-musicgen:latest-gpu
```

2. Em seguida, com o docker rodando, suba um servidor http para o frontend:
```sh
python -m http.server 8080 --directory ./frontend
```

(Funciona apenas arrastando o arquivo .html para o navegador também)

