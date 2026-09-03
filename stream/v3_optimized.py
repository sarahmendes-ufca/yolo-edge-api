#!/usr/bin/env python3
"""
stream/v3_optimized.py — Pipeline otimizado para tempo real no
Raspberry Pi 5.

Combina:
- threading;
- buffer de 1 frame;
- frame skip;
- resolução adaptativa para inferência;
- OSD.

Saída:
    Stream de vídeo anotado exibido via cv2.imshow()
    ou salvo em arquivo.

Execução:
    python3 stream/v3_optimized.py --device 0 --infer-every 3
"""

import argparse
import queue
import subprocess
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# Compatibilidade com versões do PyTorch que usam
# weights_only=True como comportamento padrão.
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False

    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


class OptimizedCamera:
    """
    Câmera com MJPEG nativo, FPS fixo e buffer de 1 frame.

    O formato MJPEG evita a conversão YUYV -> BGR na câmera,
    mantendo o processamento do frame sob controle da aplicação.
    """

    def __init__(
        self,
        device: int,
        width: int,
        height: int,
        fps: int = 30,
        use_mjpeg: bool = True,
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
        self._raw = b""

        self._width = width
        self._height = height
        self._fps = fps
        self._use_mjpeg = use_mjpeg

        print(
            f"[OptimizedCamera] Resolução solicitada: "
            f"{width}x{height} @ {fps} FPS"
        )

        # Buffer de apenas 1 frame.
        self._buf = queue.Queue(maxsize=1)

        self._running = threading.Event()
        self._running.set()

        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="CamThread",
        )

        self.frames_in = 0
        self.frames_out = 0
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
            f"[OptimizedCamera] rpicam-vid iniciado "
            f"(pid={self._proc.pid}) — buffer maxsize=1"
        )

        return self

    def _loop(self):
        """Loop da thread dedicada à captura."""
        while self._running.is_set():
            chunk = self._proc.stdout.read(4096)

            if not chunk:
                break

            self._raw += chunk

            # Procura o último JPEG completo disponível.
            end = self._raw.rfind(b"\xff\xd9")

            if end == -1:
                continue

            start = self._raw.rfind(
                b"\xff\xd8",
                0,
                end,
            )

            if start == -1:
                continue

            jpg = self._raw[start:end + 2]

            # Descarta dados anteriores ao JPEG selecionado.
            self._raw = self._raw[end + 2:]

            frame = cv2.imdecode(
                np.frombuffer(jpg, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )

            if frame is None:
                continue

            self.frames_in += 1

            # Se o buffer estiver cheio, descarta o frame antigo.
            if self._buf.full():
                try:
                    self._buf.get_nowait()
                    self.frames_dropped += 1
                except queue.Empty:
                    pass

            try:
                self._buf.put_nowait(frame)
            except queue.Full:
                self.frames_dropped += 1

    def read(self, timeout: float = 1.0):
        """
        Retorna o frame mais recente disponível.

        Bloqueia por no máximo `timeout` segundos.
        """
        try:
            frame = self._buf.get(timeout=timeout)
            self.frames_out += 1
            return frame
        except queue.Empty:
            return None

    def stop(self):
        """Encerra a captura e o processo rpicam-vid."""
        self._running.clear()

        if self._proc:
            self._proc.terminate()

            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()

        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

        print(
            f"[OptimizedCamera] Encerrada — "
            f"entrada: {self.frames_in}, "
            f"saída: {self.frames_out}, "
            f"descartados: {self.frames_dropped}"
        )


class RealtimeDetector:
    """
    Detector YOLO com frame skip e OSD.

    Mantém o último resultado de inferência para exibição
    nos frames intermediários.
    """

    def __init__(
        self,
        model_path: str,
        conf: float,
        infer_every: int,
        infer_size: int,
    ):
        if infer_every < 1:
            raise ValueError(
                "infer_every deve ser maior ou igual a 1"
            )

        if infer_size < 32:
            raise ValueError(
                "infer_size deve ser maior ou igual a 32"
            )

        print(f"[RealtimeDetector] Modelo: {model_path}")
        print(
            f"[RealtimeDetector] Inferência a cada "
            f"{infer_every} frames"
        )
        print(
            f"[RealtimeDetector] Tamanho de inferência: "
            f"{infer_size}px"
        )

        self.model = YOLO(model_path)
        self.conf = conf
        self.infer_every = infer_every
        self.infer_size = infer_size

        self._frame_idx = 0

        # [(label, conf, x1, y1, x2, y2), ...]
        self._last_boxes = []

        self._last_infer_ms = 0.0

        # FPS calculado sobre janela deslizante de 30 frames.
        self._fps_window = deque(maxlen=30)
        self._t_last = time.perf_counter()

        self.inference_count = 0

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Processa um frame.

        A cada `infer_every` frames:
            executa YOLO e atualiza `_last_boxes`.

        Nos demais frames:
            reutiliza `_last_boxes`.

        Retorna o frame com bounding boxes e OSD.
        """
        self._frame_idx += 1

        # ── Atualiza FPS ─────────────────────────────────────
        now = time.perf_counter()
        dt = now - self._t_last
        self._t_last = now

        if dt > 0:
            self._fps_window.append(dt)

        # ── Inferência a cada N frames ───────────────────────
        is_infer_frame = (
            self._frame_idx % self.infer_every == 0
        )

        if is_infer_frame:
            h, w = frame.shape[:2]

            # Redimensiona para acelerar a inferência.
            small = cv2.resize(
                frame,
                (self.infer_size, self.infer_size),
                interpolation=cv2.INTER_LINEAR,
            )

            t0 = time.perf_counter()

            results = self.model(
                small,
                conf=self.conf,
                verbose=False,
            )

            self._last_infer_ms = (
                time.perf_counter() - t0
            ) * 1000

            self.inference_count += 1

            # Reescala coordenadas para a resolução original.
            sx = w / self.infer_size
            sy = h / self.infer_size

            self._last_boxes = []

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id]
                    conf = float(box.conf[0])

                    self._last_boxes.append(
                        (
                            label,
                            conf,
                            int(x1 * sx),
                            int(y1 * sy),
                            int(x2 * sx),
                            int(y2 * sy),
                        )
                    )

        # ── Desenha bounding boxes ───────────────────────────
        output = frame.copy()

        for (
            label,
            conf,
            x1,
            y1,
            x2,
            y2,
        ) in self._last_boxes:

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            caption = f"{label} {conf:.0%}"

            (tw, th), _ = cv2.getTextSize(
                caption,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                1,
            )

            # Evita que a caixa do texto fique com coordenada
            # Y negativa.
            text_y1 = max(0, y1 - th - 8)
            text_y2 = max(th + 8, y1)

            cv2.rectangle(
                output,
                (x1, text_y1),
                (x1 + tw + 4, text_y2),
                (0, 255, 0),
                -1,
            )

            cv2.putText(
                output,
                caption,
                (x1 + 2, max(th + 2, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
            )

        # ── OSD: métricas sobrepostas ────────────────────────
        fps_display = (
            len(self._fps_window)
            / sum(self._fps_window)
            if self._fps_window
            else 0.0
        )

        osd_lines = [
            f"FPS: {fps_display:.1f}",
            f"Infer: {self._last_infer_ms:.0f}ms",
            f"Det: {len(self._last_boxes)}",
            f"Frame: {self._frame_idx}",
        ]

        for i, line in enumerate(osd_lines):
            y = 28 + i * 26

            color = (
                (0, 255, 255)
                if is_infer_frame
                else (200, 200, 200)
            )

            cv2.putText(
                output,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        return output


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
        "--infer-every",
        type=int,
        default=3,
        help="Executa YOLO a cada N frames (padrão: 3)",
    )

    parser.add_argument(
        "--infer-size",
        type=int,
        default=320,
        help="Resolução de inferência em px (padrão: 320)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Salva o stream anotado em arquivo .avi (opcional)",
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Desativa cv2.imshow() (modo headless/SSH)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    camera = OptimizedCamera(
        args.device,
        args.width,
        args.height,
        args.fps,
    )

    detector = RealtimeDetector(
        args.model,
        args.conf,
        args.infer_every,
        args.infer_size,
    )

    writer = None

    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")

        writer = cv2.VideoWriter(
            args.output,
            fourcc,
            args.fps,
            (args.width, args.height),
        )

        if not writer.isOpened():
            raise RuntimeError(
                f"Não foi possível abrir o arquivo de saída: "
                f"{args.output}"
            )

        print(
            f"[INFO] Gravando saída em: {args.output}"
        )

    camera.start()

    try:
        # Aguarda a câmera estabilizar.
        time.sleep(0.5)

        print(
            "[INFO] Stream iniciado. "
            "Pressione Ctrl+C para encerrar."
        )

        if not args.no_display:
            print(
                "[INFO] Pressione 'q' na janela "
                "para encerrar."
            )

        while True:
            frame = camera.read(timeout=2.0)

            if frame is None:
                print(
                    "[AVISO] Timeout na leitura."
                )
                continue

            annotated = detector.process(frame)

            if writer is not None:
                writer.write(annotated)

            if not args.no_display:
                cv2.imshow(
                    "YOLO - Tempo Real (pressione q para sair)",
                    annotated,
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Encerrado pelo usuário.")

    finally:
        camera.stop()

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

        print(
            f"[INFO] Frames processados: "
            f"{detector._frame_idx}"
        )

        print(
            f"[INFO] Inferências realizadas: "
            f"{detector.inference_count}"
        )

        print(
            f"[INFO] Frames descartados: "
            f"{camera.frames_dropped}"
        )


if __name__ == "__main__":
    main()
