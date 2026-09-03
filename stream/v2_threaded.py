#!/usr/bin/env python3
"""
stream/v2_threaded.py — Captura e inferência em threads separadas.

Buffer de 1 frame elimina o acúmulo e garante processamento
do frame mais atual disponível.

Execução:
    python3 stream/v2_threaded.py --device 0
"""

import argparse
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# Compatibilidade com versões do PyTorch que usam weights_only=True
# como comportamento padrão.
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Classe de captura em thread dedicada ────────────────────

class CameraCapture:
    """
    Captura frames em thread separada.

    Mantém sempre o frame mais recente disponível.
    O buffer de tamanho 1 descarta frames antigos automaticamente.
    """

    def __init__(
        self,
        device: int,
        width: int,
        height: int,
        fps: int = 30,
    ):
        self._cmd = [
            "rpicam-vid",
            "-t",
            "0",
            "-n",
            "--codec",
            "mjpeg",
            "--camera",
            str(device),
            "--width",
            str(width),
            "--height",
            str(height),
            "--framerate",
            str(fps),
            "-o",
            "-",
        ]

        self._proc = None

        # Buffer de 1 frame:
        # o producer sobrescreve o frame antigo.
        self._buffer = queue.Queue(maxsize=1)

        self._running = threading.Event()
        self._running.set()

        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="CaptureThread",
        )

        # Métricas de captura
        self.frames_captured = 0
        self.frames_dropped = 0

    def start(self):
        """Inicia o processo da câmera e a thread de captura."""
        self._proc = subprocess.Popen(
            self._cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        self._thread.start()

        print(
            f"[CameraCapture] rpicam-vid iniciado "
            f"(pid={self._proc.pid}) — buffer maxsize=1"
        )

        return self

    def _capture_loop(self):
        """Loop interno da thread de captura."""
        raw = b""

        while self._running.is_set():
            chunk = self._proc.stdout.read(4096)

            if not chunk:
                break

            raw += chunk

            # Mantém apenas o último frame JPEG completo;
            # os anteriores são descartados.
            end = raw.rfind(b"\xff\xd9")

            if end == -1:
                continue

            start = raw.rfind(b"\xff\xd8", 0, end)

            if start == -1:
                continue

            jpg = raw[start:end + 2]

            # Tudo antes do JPEG encontrado é descartado.
            raw = raw[end + 2:]

            frame = cv2.imdecode(
                np.frombuffer(jpg, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )

            if frame is None:
                continue

            # Se o buffer está cheio, descarta o frame antigo.
            if self._buffer.full():
                try:
                    self._buffer.get_nowait()
                    self.frames_dropped += 1
                except queue.Empty:
                    pass

            try:
                self._buffer.put_nowait(frame)
                self.frames_captured += 1
            except queue.Full:
                # Outro frame chegou antes de conseguirmos inserir.
                self.frames_dropped += 1

    def read(self, timeout: float = 1.0):
        """
        Retorna o frame mais recente.

        Bloqueia até `timeout` segundos se nenhum frame estiver disponível.
        """
        try:
            return self._buffer.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """Encerra a thread e o processo da câmera."""
        self._running.clear()

        if self._proc:
            self._proc.terminate()

            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()

        # Aguarda a thread terminar.
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

        print(
            f"[CameraCapture] Encerrada — "
            f"capturados: {self.frames_captured}, "
            f"descartados: {self.frames_dropped}"
        )


# ── Classe de inferência com métricas ───────────────────────

class YOLOInference:
    """Wrapper do YOLO com métricas de latência acumuladas."""

    def __init__(self, model_path: str, conf: float = 0.4):
        print(f"[YOLOInference] Carregando: {model_path}")

        self.model = YOLO(model_path)
        self.conf = conf

        self.count = 0
        self.total_ms = 0.0

    def run(self, frame):
        """
        Executa inferência.

        Retorna:
            (frame_anotado, número_de_deteções, latência_ms)
        """
        t0 = time.perf_counter()

        results = self.model(
            frame,
            conf=self.conf,
            verbose=False,
        )

        elapsed = (time.perf_counter() - t0) * 1000

        self.count += 1
        self.total_ms += elapsed

        # Anota bounding boxes diretamente no frame.
        annotated = results[0].plot()
        n_det = len(results[0].boxes)

        return annotated, n_det, elapsed

    @property
    def avg_ms(self) -> float:
        """Retorna a latência média de inferência."""
        return (
            self.total_ms / self.count
            if self.count > 0
            else 0.0
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
        help="FPS solicitado à câmera",
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
        "--frames",
        type=int,
        default=100,
        help="Número de frames processados",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    camera = CameraCapture(
        args.device,
        args.width,
        args.height,
        args.fps,
    )

    yolo = YOLOInference(
        args.model,
        args.conf,
    )

    camera.start()

    try:
        # Aguarda a câmera estabilizar.
        time.sleep(0.5)

        print(
            f"[INFO] Processando {args.frames} frames "
            "com threading..."
        )

        print(
            f"{'Frame':>6} | "
            f"{'Inferência':>10} | "
            f"{'FPS inst.':>9} | "
            f"{'Detecções':>9}"
        )

        print("-" * 48)

        t_start = time.perf_counter()
        frame_count = 0

        while frame_count < args.frames:
            frame = camera.read(timeout=2.0)

            if frame is None:
                print("[AVISO] Timeout na leitura do frame.")
                continue

            annotated, n_det, infer_ms = yolo.run(frame)

            # Mantido para deixar explícito que o frame foi anotado.
            # Pode ser usado posteriormente para exibição/gravação.
            _ = annotated

            frame_count += 1

            elapsed_total = time.perf_counter() - t_start

            fps_avg = (
                frame_count / elapsed_total
                if elapsed_total > 0
                else 0.0
            )

            if frame_count % 10 == 0:
                print(
                    f"{frame_count:>6} | "
                    f"{infer_ms:>9.1f}ms | "
                    f"{fps_avg:>8.1f} | "
                    f"{n_det:>9}"
                )

        total_time = time.perf_counter() - t_start

    finally:
        camera.stop()

    # ── Relatório final ─────────────────────────────────────

    print("\n" + "=" * 58)
    print("RELATÓRIO — Threading com buffer de 1 frame")
    print("=" * 58)

    if frame_count == 0:
        print(" Nenhum frame foi processado.")
        print("=" * 58)
        return

    fps_sustained = (
        frame_count / total_time
        if total_time > 0
        else 0.0
    )

    drop_percent = (
        100 * camera.frames_dropped
        / max(camera.frames_captured, 1)
    )

    print(f" Frames processados   : {frame_count}")
    print(f" Tempo total          : {total_time:.1f} s")
    print(f" FPS médio sustentado : {fps_sustained:.1f} FPS")
    print(f" Inferência média     : {yolo.avg_ms:.1f} ms")
    print(f" Frames capturados    : {camera.frames_captured}")
    print(
        f" Frames descartados   : "
        f"{camera.frames_dropped} ({drop_percent:.0f}%)"
    )

    print("=" * 58)


if __name__ == "__main__":
    main()
