import React, { useCallback, useEffect, useRef, useState } from "react";
import { replay } from "../game/rules.js";
import {
  XMarkIcon,
  PauseIcon,
  PlayIcon,
  ArrowDownTrayIcon,
} from "@heroicons/react/24/outline";
import "./ClipDialog.css";

const WIDTH = 540,
  HEIGHT = 960,
  MAX_MOVES = 100;
const colors = ["#ef4444", "#3b82f6"];
const MIME_TYPES = [
  "video/mp4;codecs=avc1.42E01E,mp4a.40.2",
  "video/mp4;codecs=avc1.42E01E",
  "video/mp4",
  "video/webm;codecs=vp9,opus",
  "video/webm;codecs=vp8,opus",
  "video/webm",
];
const supportedTypes = () =>
  typeof MediaRecorder === "undefined"
    ? []
    : MIME_TYPES.filter((type) => MediaRecorder.isTypeSupported(type));

function draw(ctx, state, previous, progress, flipped) {
  ctx.fillStyle = "#09090b";
  ctx.fillRect(0, 0, WIDTH, HEIGHT);
  const x = 36,
    y = 224,
    size = 468,
    pad = 12,
    cell = (size - 2 * pad) / 11;
  const step = cell * 1.25;
  ctx.fillStyle = "#e2e8f0";
  ctx.beginPath();
  ctx.roundRect(x, y, size, size, 24);
  ctx.fill();
  ctx.save();
  if (flipped) {
    ctx.translate(WIDTH, 2 * y + size);
    ctx.rotate(Math.PI);
  }
  const at = (r, c) => [x + pad + c * step, y + pad + r * step];
  for (let r = 0; r < 9; r++)
    for (let c = 0; c < 9; c++) {
      const [px, py] = at(r, c);
      ctx.fillStyle = r === 0 ? "#feeaea" : r === 8 ? "#e5f0ff" : "#f9fafb";
      ctx.beginPath();
      ctx.roundRect(px, py, cell, cell, cell * 0.16);
      ctx.fill();
    }
  for (const [kind, r, c] of state.walls) {
    const [px, py] = at(r, c);
    ctx.fillStyle = "#27272a";
    ctx.beginPath();
    if (kind === "h") ctx.roundRect(px, py + cell, cell * 2.25, cell * 0.25, 3);
    else ctx.roundRect(px + cell, py, cell * 0.25, cell * 2.25, 3);
    ctx.fill();
  }
  for (let player = 0; player < 2; player++) {
    const [r, c] = state.pawns[player],
      [pr, pc] = previous?.pawns[player] || [r, c];
    const [px, py] = at(pr + (r - pr) * progress, pc + (c - pc) * progress);
    ctx.fillStyle = colors[player];
    ctx.beginPath();
    ctx.arc(px + cell / 2, py + cell / 2, cell * 0.275, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
  ctx.font = "bold 24px system-ui, sans-serif";
  ctx.textAlign = "left";
  for (let i = 0; i < 2; i++) {
    const bx = WIDTH * (i === 0 ? 0.27 : 0.57);
    ctx.fillStyle = "#27272a";
    ctx.beginPath();
    ctx.roundRect(bx, 135, 88, 48, 24);
    ctx.fill();
    ctx.fillStyle = colors[i];
    ctx.beginPath();
    ctx.roundRect(bx + 14, 155, 25, 8, 4);
    ctx.fill();
    ctx.fillStyle = "#fafafa";
    ctx.fillText(String(state.remaining[i]), bx + 48, 168);
  }
}

export default function ClipDialog({
  record,
  flipped: initialFlipped = false,
  onClose,
}) {
  const moves = record.moves.slice(0, MAX_MOVES);
  const states = useRef([]);
  const dialog = useRef(null);
  const closeRef = useRef(onClose);
  const busyRef = useRef(false);
  const [format, setFormat] = useState(() =>
    supportedTypes()[0]?.startsWith("video/mp4") ? "MP4" : "WebM",
  );
  const canvas = useRef(null),
    raf = useRef(null),
    recorder = useRef(null),
    audio = useRef(null),
    lastSound = useRef(-1);
  const previewAudio = useRef(null),
    previewMove = useRef(0),
    mounted = useRef(true);
  const [playing, setPlaying] = useState(true),
    [progress, setProgress] = useState(0),
    [speed, setSpeed] = useState(2);
  const [flipped, setFlipped] = useState(initialFlipped),
    [sound, setSound] = useState(true),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const [recordingProgress, setRecordingProgress] = useState(0);
  const progressRef = useRef(0),
    playingRef = useRef(true),
    speedRef = useRef(2),
    flipRef = useRef(initialFlipped),
    soundRef = useRef(true);
  const duration = Math.max(1, moves.length * 1000 + 1000);
  closeRef.current = onClose;
  busyRef.current = busy;
  useEffect(() => {
    const previousFocus = document.activeElement;
    dialog.current?.querySelector("button:not([disabled])")?.focus();
    const handleKey = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        if (!busyRef.current) closeRef.current();
      } else if (event.key === "Tab") {
        const controls = [
          ...(dialog.current?.querySelectorAll(
            'button:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])',
          ) || []),
        ];
        if (!controls.length) {
          event.preventDefault();
          return;
        }
        const first = controls[0],
          last = controls[controls.length - 1];
        if (
          event.shiftKey &&
          (document.activeElement === first ||
            !dialog.current?.contains(document.activeElement))
        ) {
          event.preventDefault();
          last.focus();
        } else if (
          !event.shiftKey &&
          (document.activeElement === last ||
            !dialog.current?.contains(document.activeElement))
        ) {
          event.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener("keydown", handleKey, true);
    return () => {
      document.removeEventListener("keydown", handleKey, true);
      if (previousFocus instanceof HTMLElement && previousFocus.isConnected)
        previousFocus.focus();
    };
  }, []);
  useEffect(() => {
    if (busy)
      dialog.current?.querySelector(".clip-download:not([disabled])")?.focus();
  }, [busy]);
  useEffect(() => {
    states.current = Array.from({ length: moves.length + 1 }, (_, i) =>
      replay(moves.slice(0, i)),
    );
  }, [record]);
  useEffect(() => {
    speedRef.current = speed;
  }, [speed]);
  useEffect(() => {
    playingRef.current = playing;
  }, [playing]);
  useEffect(() => {
    flipRef.current = flipped;
  }, [flipped]);
  useEffect(() => {
    soundRef.current = sound;
  }, [sound]);
  const paint = useCallback(
    (position) => {
      if (
        !canvas.current ||
        !states.current.length ||
        !Number.isFinite(position)
      )
        return;
      const index = Math.max(
        0,
        Math.min(states.current.length - 1, Math.floor(position / 1000)),
      );
      const frac =
        index === 0 || position >= moves.length * 1000 + 250
          ? 1
          : Math.min(1, (position % 1000) / 250);
      draw(
        canvas.current.getContext("2d"),
        states.current[index],
        states.current[Math.max(0, index - 1)],
        frac,
        flipRef.current,
      );
    },
    [moves.length],
  );
  useEffect(() => {
    mounted.current = true;
    let last = performance.now();
    const tick = (now) => {
      const delta = Math.min(100, now - last);
      last = now;
      if (playingRef.current && !recorder.current) {
        progressRef.current =
          (progressRef.current + delta * speedRef.current) % duration;
        setProgress((progressRef.current / duration) * 1000);
        const nextMove = Math.min(
          moves.length,
          Math.floor(progressRef.current / 1000),
        );
        if (
          nextMove > previewMove.current &&
          soundRef.current &&
          previewAudio.current?.state === "running"
        ) {
          beep(previewAudio.current, previewAudio.current.destination);
        }
        previewMove.current = nextMove;
      }
      paint(progressRef.current);
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => {
      mounted.current = false;
      cancelAnimationFrame(raf.current);
      recorder.current?.cancelExport?.();
      audio.current?.close().catch(() => {});
      previewAudio.current?.close().catch(() => {});
    };
  }, [duration, paint]);
  const seek = (e) => {
    const p = Number(e.target.value);
    progressRef.current = (p / 1000) * duration;
    previewMove.current = Math.floor(progressRef.current / 1000);
    setProgress(p);
    paint(progressRef.current);
  };
  const enablePreviewAudio = () => {
    if (!window.AudioContext) return;
    if (!previewAudio.current || previewAudio.current.state === "closed")
      previewAudio.current = new AudioContext();
    if (previewAudio.current.state === "suspended")
      previewAudio.current.resume().catch(() => {});
  };
  const beep = (context, destination) => {
    const oscillator = context.createOscillator(),
      gain = context.createGain();
    oscillator.frequency.value = 500;
    gain.gain.setValueAtTime(0.06, context.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, context.currentTime + 0.08);
    oscillator.connect(gain).connect(destination);
    oscillator.start();
    oscillator.stop(context.currentTime + 0.08);
  };
  const download = async () => {
    if (!canvas.current?.captureStream || !window.MediaRecorder) {
      setError("此浏览器不支持视频录制，请使用最新版 Chrome 或 Edge。");
      return;
    }
    setError("");
    setBusy(true);
    setRecordingProgress(0);
    setPlaying(false);
    playingRef.current = false;
    const stream = canvas.current.captureStream(30);
    let context, destination;
    try {
      if (soundRef.current && window.AudioContext) {
        context = new AudioContext();
        audio.current = context;
        destination = context.createMediaStreamDestination();
        destination.stream
          .getAudioTracks()
          .forEach((track) => stream.addTrack(track));
      }
      const candidates = supportedTypes();
      let media, mime;
      for (const type of candidates) {
        try {
          media = new MediaRecorder(stream, {
            mimeType: type,
            videoBitsPerSecond: 6000000,
          });
          mime = media.mimeType || type;
          break;
        } catch {
          // Some browsers advertise a format that they cannot instantiate.
        }
      }
      if (!media) throw new Error("此浏览器不支持 MP4 或 WebM 视频录制。");
      setFormat(mime.startsWith("video/mp4") ? "MP4" : "WebM");
      const chunks = [];
      recorder.current = media;
      let cancelled = false,
        frame;
      media.ondataavailable = (event) => {
        if (event.data.size) chunks.push(event.data);
      };
      media.onerror = () => {
        cancelled = true;
        if (mounted.current) setError("视频录制失败，请重试。");
        if (media.state !== "inactive") media.stop();
      };
      media.onstop = () => {
        cancelAnimationFrame(frame);
        stream.getTracks().forEach((track) => track.stop());
        context?.close().catch(() => {});
        audio.current = null;
        if (!cancelled && chunks.length) {
          const url = URL.createObjectURL(new Blob(chunks, { type: mime }));
          const a = document.createElement("a");
          a.href = url;
          a.download = mime.startsWith("video/mp4")
            ? "barricade-replay.mp4"
            : "barricade-replay.webm";
          a.click();
          setTimeout(() => URL.revokeObjectURL(url), 60000);
        }
        recorder.current = null;
        if (mounted.current) setBusy(false);
      };
      media.cancelExport = () => {
        cancelled = true;
        if (media.state !== "inactive") media.stop();
      };
      progressRef.current = 0;
      lastSound.current = 0;
      paint(0);
      media.start(250);
      const started = performance.now();
      const render = (now) => {
        if (media.state === "inactive") return;
        const t = Math.min(duration, (now - started) * speedRef.current);
        progressRef.current = t;
        paint(t);
        setProgress((t / duration) * 1000);
        setRecordingProgress((t / duration) * 100);
        const index = Math.min(moves.length, Math.floor(t / 1000));
        if (context && destination && index > lastSound.current)
          beep(context, destination);
        lastSound.current = index;
        if (t >= duration) media.stop();
        else frame = requestAnimationFrame(render);
      };
      frame = requestAnimationFrame(render);
    } catch (e) {
      stream.getTracks().forEach((track) => track.stop());
      context?.close().catch(() => {});
      recorder.current = null;
      setBusy(false);
      setError(e.message || "视频导出失败");
    }
  };
  return (
    <div
      className="modal-backdrop clip-backdrop"
      role="presentation"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget && !busy) onClose();
      }}
    >
      <section
        ref={dialog}
        className="modal clip-modal"
        onPointerDown={() => {
          if (soundRef.current) enablePreviewAudio();
        }}
        role="dialog"
        aria-modal="true"
        aria-label="下载片段"
      >
        <header className="clip-header">
          <h2>下载片段</h2>
          <button
            className="icon-button"
            aria-label="关闭"
            disabled={busy}
            onClick={onClose}
          >
            <XMarkIcon className="clip-icon" />
          </button>
        </header>
        <div className="clip-body">
          <canvas
            ref={canvas}
            width={WIDTH}
            height={HEIGHT}
            className="clip-preview"
            aria-label="视频预览"
          />
          <div className="clip-controls">
            <div className="clip-timeline">
              <button
                className="icon-button"
                disabled={busy}
                aria-label={playing ? "暂停" : "播放"}
                onClick={() => {
                  enablePreviewAudio();
                  setPlaying(!playing);
                }}
              >
                {playing ? (
                  <PauseIcon className="clip-icon" />
                ) : (
                  <PlayIcon className="clip-icon" />
                )}
              </button>
              <input
                type="range"
                min="0"
                max="1000"
                value={progress}
                onChange={seek}
                disabled={busy}
                aria-label="视频进度"
              />
            </div>
            <div className="clip-row">
              <span>速度</span>
              <select
                aria-label="速度"
                value={speed}
                disabled={busy}
                onChange={(e) => setSpeed(Number(e.target.value))}
              >
                {[1, 2, 3].map((s) => (
                  <option key={s} value={s}>
                    {s}×
                  </option>
                ))}
              </select>
            </div>
            <div className="clip-row">
              <span>视角</span>
              <button disabled={busy} onClick={() => setFlipped(!flipped)}>
                {flipped ? "玩家 2" : "玩家 1"}
              </button>
            </div>
            <label className="clip-row">
              <span>声音</span>
              <input
                type="checkbox"
                checked={sound}
                disabled={busy}
                onChange={(e) => {
                  setSound(e.target.checked);
                  if (e.target.checked) enablePreviewAudio();
                }}
              />
            </label>
            {record.moves.length > MAX_MOVES && (
              <p className="clip-note">仅导出前 100 步。</p>
            )}
            {error && (
              <p className="clip-error" role="alert">
                {error}
              </p>
            )}
            {busy ? (
              <>
                <p className="clip-note">
                  正在录制 {Math.round(recordingProgress)}%
                </p>
                <button
                  className="clip-download"
                  onClick={() => recorder.current?.cancelExport()}
                >
                  取消导出
                </button>
              </>
            ) : (
              <button className="clip-download" onClick={download}>
                <ArrowDownTrayIcon className="clip-icon" /> 下载
              </button>
            )}
            <p className="clip-note">
              {format === "MP4"
                ? "视频将以 MP4 格式保存。"
                : "此浏览器将以 WebM 格式保存视频。"}
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
