#!/usr/bin/env python3
"""
stream/mjpeg_server.py — Serve o stream YOLO anotado como MJPEG via HTTP.

Acesse no navegador:
    http://<IP_DO_RASPBERRY>:5000/stream

Execução:
    python3 stream/mjpeg_server.py --device 0 --port 5000
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response

sys.path.insert(0, str(Path(__file__).parent.parent))

from stream.v3_optimized import OptimizedCamera, RealtimeDetector


# ── Estado global do servidor ────────────────────────────────

app = Flask(__name__)

_camera = None
_detector = None

_lock = threading.Lock()

# Último frame JPEG comprimido.
_latest_jpg: bytes = b""


def _frame_producer():
    """
    Thread que captura, processa e comprime frames continuamente.
    """
    global _latest_jpg

    while True:
        if _camera is None or _detector is None:
            time.sleep(0.1)
            continue

        frame = _camera.read(timeout=2.0)

        if frame is None:
            continue

        annotated = _detector.process(frame)

        # Comprime para JPEG.
        # Qualidade 80 oferece um bom equilíbrio entre
        # tamanho e qualidade visual.
        ok, jpg = cv2.imencode(
            ".jpg",
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, 80],
        )

        if ok:
            with _lock:
                _latest_jpg = jpg.tobytes()


def _generate_mjpeg():
    """
    Generator que produz o stream multipart para o cliente HTTP.
    """
    while True:
        with _lock:
            jpg = _latest_jpg

        if not jpg:
            time.sleep(0.01)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(jpg)).encode() + b"\r\n"
            b"\r\n"
            + jpg
            + b"\r\n"
        )

        # Limita a taxa de envio para aproximadamente 30 FPS.
        time.sleep(0.033)


@app.route("/")
def index():
    """Página HTML simples com o stream embutido."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>YOLO — Stream em Tempo Real</title>

        <style>
            body {
                background: #111;
                color: #eee;
                font-family: sans-serif;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding-top: 30px;
            }

            h1 {
                margin-bottom: 12px;
                font-size: 1.4rem;
            }

            img {
                border: 2px solid #444;
                border-radius: 4px;
                max-width: 100%;
                height: auto;
            }

            p {
                color: #888;
                font-size: 0.85rem;
                margin-top: 10px;
            }
        </style>
    </head>

    <body>
        <h1>YOLOv8 — Raspberry Pi 5 — Tempo Real</h1>

        <img
            src="/stream"
            alt="Stream YOLO em tempo real"
        >

        <p>
            Stream MJPEG com inferência YOLO e anotações
            em tempo real.
        </p>
    </body>
    </html>
    """


@app.route("/stream")
def stream():
    """Endpoint MJPEG: resposta multipart contínua."""
    return Response(
        _generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/health")
def health():
    """
    Health check compatível com o padrão da API de inferência.
    """
    status = {
        "status": "ok",
        "stream": "active",
        "frame_count": (
            _detector._frame_idx
            if _detector
            else 0
        ),
    }

    return Response(
        json.dumps(status),
        mimetype="application/json",
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Índice da câmera",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Largura da captura",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Altura da captura",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS da câmera",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="models/yolov8n.pt",
        help="Caminho para o modelo YOLO",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence mínima das detecções",
    )

    parser.add_argument(
        "--infer-every",
        type=int,
        default=3,
        help="Executa YOLO a cada N frames",
    )

    parser.add_argument(
        "--infer-size",
        type=int,
        default=320,
        help="Resolução usada na inferência",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Porta HTTP do servidor",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Interface de rede onde o servidor escuta",
    )

    return parser.parse_args()


def main():
    global _camera, _detector

    args = parse_args()

    print("[INFO] Inicializando câmera...")

    _camera = OptimizedCamera(
        args.device,
        args.width,
        args.height,
        args.fps,
    ).start()

    print("[INFO] Inicializando detector YOLO...")

    _detector = RealtimeDetector(
        args.model,
        args.conf,
        args.infer_every,
        args.infer_size,
    )

    # Thread de processamento dos frames.
    producer = threading.Thread(
        target=_frame_producer,
        daemon=True,
        name="FrameProducer",
    )

    producer.start()

    # Aguarda o primeiro frame.
    time.sleep(1.0)

    print("[INFO] Servidor MJPEG iniciado.")
    print(
        f"[INFO] Página principal: "
        f"http://<IP_DO_RASPBERRY>:{args.port}/"
    )
    print(
        f"[INFO] Stream direto: "
        f"http://<IP_DO_RASPBERRY>:{args.port}/stream"
    )
    print(
        f"[INFO] Health check: "
        f"http://<IP_DO_RASPBERRY>:{args.port}/health"
    )

    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=False,
            threaded=True,
            use_reloader=False,
        )

    except KeyboardInterrupt:
        print("\n[INFO] Servidor encerrado pelo usuário.")

    finally:
        if _camera is not None:
            _camera.stop()

        print("[INFO] Recursos liberados.")


if __name__ == "__main__":
    main()
