PREVIOUS_IMAGE=$(docker inspect yolo-api --format '{{.Config.Image}}' 2>/dev/null)

echo "[1/4] Baixando nova imagem..."
docker compose pull

echo "[2/4] Subindo nova versão..."
docker compose up -d

echo "[3/4] Aguardando health check (30s)..."
sleep 30

echo "[4/4] Verificando saúde do serviço..."

if curl -sf http://localhost:8000/health > /dev/null; then
    echo "[OK] Deploy bem-sucedido."
else
    echo "[ERRO] Health check falhou. Executando rollback..."

    docker compose down

    # Retorna para a imagem anterior
    IMAGE=$PREVIOUS_IMAGE docker compose up -d

    echo "[OK] Rollback concluído para: $PREVIOUS_IMAGE"

    exit 1
fi
