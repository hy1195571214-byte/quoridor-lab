import { useLanguage } from "../i18n.jsx";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { replay } from "../game/rules.js";
import { preloadGameSounds, playSoundInto } from "../game/preferences.js";
import {
  XMarkIcon,
  PauseIcon,
  PlayIcon,
  ArrowDownTrayIcon,
} from "@heroicons/react/24/outline";
import "./ClipDialog.css";

const WIDTH = 720,
  HEIGHT = 1280,
  MAX_MOVES = 100;
const colors = ["#E5484D", "#3E63DD"];
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
  // Geometry and colors follow the original 9:16 canvas layout.
  const size = 672,
    x = 24,
    unit = size / 47;
  const cell = 4 * unit,
    gap = unit,
    pad = 1.5 * unit;
  const badgeHeight = (size * 96) / 1032,
    badgeWidth = (size * 220) / 1032;
  const y = (HEIGHT - size) / 2 + (badgeHeight + 28) / 2;
  const badgeY = y - 28 - badgeHeight;
  const badgePositions = [WIDTH / 2 - 12 - badgeWidth, WIDTH / 2 + 12];
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, WIDTH, HEIGHT);
  for (let i = 0; i < 2; i++) {
    const player = flipped ? 1 - i : i;
    const bx = badgePositions[i],
      barWidth = 0.55 * badgeHeight;
    const barHeight = 0.18 * badgeHeight,
      barGap = 0.2 * badgeHeight;
    ctx.fillStyle = "#232326";
    ctx.beginPath();
    ctx.roundRect(bx, badgeY, badgeWidth, badgeHeight, badgeHeight / 2);
    ctx.fill();
    ctx.font = `700 ${0.56 * badgeHeight}px system-ui, sans-serif`;
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    const text = String(state.remaining[player]),
      measure = ctx.measureText(text);
    const start = bx + (badgeWidth - barWidth - barGap - measure.width) / 2;
    ctx.fillStyle = colors[player];
    ctx.beginPath();
    ctx.roundRect(
      start,
      badgeY + (badgeHeight - barHeight) / 2,
      barWidth,
      barHeight,
      barHeight / 2,
    );
    ctx.fill();
    ctx.fillStyle = "#F4F4F5";
    ctx.fillText(
      text,
      start + barWidth + barGap,
      badgeY +
        badgeHeight / 2 +
        (measure.actualBoundingBoxAscent - measure.actualBoundingBoxDescent) /
          2,
    );
  }
  ctx.fillStyle = "#E4E4E7";
  ctx.beginPath();
  ctx.roundRect(x, y, size, size, 0.053 * size);
  ctx.fill();
  ctx.save();
  ctx.beginPath();
  ctx.roundRect(x, y, size, size, 0.053 * size);
  ctx.clip();
  ctx.translate(x, y);
  if (flipped) {
    ctx.translate(size / 2, size / 2);
    ctx.rotate(Math.PI);
    ctx.translate(-size / 2, -size / 2);
  }
  const at = (r, c) => [pad + c * (cell + gap), pad + r * (cell + gap)];
  const winner =
    state.winner && progress >= 1 ? state.pawns[state.winner - 1] : null;
  for (let r = 0; r < 9; r++)
    for (let c = 0; c < 9; c++) {
      const [px, py] = at(r, c);
      const winning = winner && winner[0] === r && winner[1] === c;
      ctx.fillStyle = winning
        ? "#FFFFFF"
        : r === 0
          ? "#FBECEC"
          : r === 8
            ? "#ECEFFB"
            : "#FAFAFA";
      ctx.beginPath();
      ctx.roundRect(px, py, cell, cell, 4);
      ctx.fill();
      if (winning) {
        ctx.strokeStyle = "#F59E0B";
        ctx.lineWidth = Math.max(2, 0.05 * cell);
        const inset = ctx.lineWidth / 2;
        ctx.beginPath();
        ctx.roundRect(
          px + inset,
          py + inset,
          cell - 2 * inset,
          cell - 2 * inset,
          4,
        );
        ctx.stroke();
      }
    }
  const wallOwners = new Map();
  state.history.forEach((move, i) => {
    if (move.length === 3) wallOwners.set(move, i % 2);
  });
  for (const [kind, r, c] of state.walls) {
    const [px, py] = at(r, c);
    const key = `${kind}${String.fromCharCode(97 + c)}${8 - r}`;
    ctx.fillStyle = colors[wallOwners.get(key) ?? 0];
    ctx.beginPath();
    if (kind === "h") ctx.roundRect(px, py + cell, 2 * cell + gap, gap, 3);
    else ctx.roundRect(px + cell, py, gap, 2 * cell + gap, 3);
    ctx.fill();
  }
  for (let player = 0; player < 2; player++) {
    const [r, c] = state.pawns[player],
      [pr, pc] = previous?.pawns[player] || [r, c];
    let row = pr + (r - pr) * progress,
      col = pc + (c - pc) * progress;
    // A diagonal jump passes through the opponent's square in the source.
    if (previous && Math.abs(r - pr) === 1 && Math.abs(c - pc) === 1) {
      const [or, oc] = previous.pawns[1 - player];
      if (progress < 0.5) {
        row = pr + (or - pr) * progress * 2;
        col = pc + (oc - pc) * progress * 2;
      } else {
        row = or + (r - or) * (progress * 2 - 1);
        col = oc + (c - oc) * (progress * 2 - 1);
      }
    }
    const [px, py] = at(row, col);
    ctx.fillStyle = colors[player];
    ctx.beginPath();
    ctx.arc(px + cell / 2, py + cell / 2, cell * 0.275, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

export default function ClipDialog({
  record,
  flipped: initialFlipped = false,
  onClose,
}) {
  const { t } = useLanguage();
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
  const exportGeneration = useRef(0);
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
  // The end hold is 500 ms of real time at every playback speed.
  const duration = Math.max(1, moves.length * 1000 + 500 * speed);
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
        Math.min(states.current.length - 1, Math.floor(position / 1000) + 1),
      );
      const frac =
        index <= 1 || position >= moves.length * 1000
          ? 1
          : Math.min(1, (position % 1000) / (250 * speedRef.current));
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
        const priorPosition = progressRef.current;
        progressRef.current =
          (progressRef.current + delta * speedRef.current) % duration;
        // A loop starts a new first move, so it needs a fresh sound event.
        if (progressRef.current < priorPosition) previewMove.current = 0;
        setProgress(
          Math.min(
            1000,
            (progressRef.current / Math.max(1, moves.length * 1000)) * 1000,
          ),
        );
        const nextMove = Math.min(
          moves.length,
          Math.floor(progressRef.current / 1000) + 1,
        );
        if (
          nextMove > previewMove.current &&
          soundRef.current &&
          previewAudio.current?.state === "running"
        ) {
          playSoundInto(
            previewAudio.current,
            previewAudio.current.destination,
            moves[nextMove - 1]?.length === 3,
          ).catch(() => {});
        }
        previewMove.current = nextMove;
      }
      paint(progressRef.current);
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf.current);
    };
  }, [duration, paint]);
  useEffect(() => {
    // StrictMode may run setup again after a development-only cleanup.
    mounted.current = true;
    return () => {
      mounted.current = false;
      exportGeneration.current++;
      recorder.current?.cancelExport?.();
      audio.current?.close().catch(() => {});
      previewAudio.current?.close().catch(() => {});
    };
  }, []);
  const seek = (e) => {
    const p = Number(e.target.value);
    progressRef.current = (p / 1000) * moves.length * 1000;
    previewMove.current = Math.min(
      moves.length,
      Math.floor(progressRef.current / 1000) + 1,
    );
    if (
      p < 1000 &&
      moves.length &&
      soundRef.current &&
      previewAudio.current?.state === "running"
    )
      playSoundInto(
        previewAudio.current,
        previewAudio.current.destination,
        moves[Math.floor(progressRef.current / 1000)]?.length === 3,
      ).catch(() => {});
    setProgress(p);
    paint(progressRef.current);
  };
  const enablePreviewAudio = () => {
    if (!window.AudioContext) return;
    if (!previewAudio.current || previewAudio.current.state === "closed")
      previewAudio.current = new AudioContext();
    if (previewAudio.current.state === "suspended")
      previewAudio.current.resume().catch(() => {});
    preloadGameSounds(previewAudio.current).catch(() => {});
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
    const generation = ++exportGeneration.current;
    let context, destination, stream;
    try {
      if (soundRef.current && window.AudioContext) {
        // Render the recording independently of an available speaker device.
        // Browsers without silent-sink support ignore this optional setting.
        context = new AudioContext({ sinkId: { type: "none" } });
        audio.current = context;
        // Resume before the first await, while the download click is active.
        const resumed = context.resume();
        await Promise.all([resumed, preloadGameSounds(context)]);
        if (!mounted.current || generation !== exportGeneration.current) {
          await context.close().catch(() => {});
          return;
        }
        destination = context.createMediaStreamDestination();
      }
      stream = canvas.current.captureStream(30);
      if (destination) {
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
        if (audio.current === context) audio.current = null;
        if (!cancelled && chunks.length) {
          const url = URL.createObjectURL(new Blob(chunks, { type: mime }));
          const a = document.createElement("a");
          a.href = url;
          a.download = mime.startsWith("video/mp4")
            ? "barricade-replay.mp4"
            : "barricade-replay.webm";
          a.click();
          setTimeout(() => URL.revokeObjectURL(url), 60000);
        } else if (!cancelled && mounted.current) {
          setError("没有录制到视频，请重试或更换浏览器。");
        }
        if (recorder.current === media) {
          recorder.current = null;
          if (mounted.current) setBusy(false);
        }
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
        setProgress(
          Math.min(1000, (t / Math.max(1, moves.length * 1000)) * 1000),
        );
        setRecordingProgress((t / duration) * 100);
        const index = Math.min(moves.length, Math.floor(t / 1000) + 1);
        if (context && destination && index > 0 && index > lastSound.current)
          playSoundInto(
            context,
            destination,
            moves[index - 1]?.length === 3,
          ).catch(() => {});
        lastSound.current = index;
        if (t >= duration) media.stop();
        else frame = requestAnimationFrame(render);
      };
      frame = requestAnimationFrame(render);
    } catch (e) {
      stream?.getTracks().forEach((track) => track.stop());
      destination?.stream.getTracks().forEach((track) => track.stop());
      context?.close().catch(() => {});
      if (mounted.current && generation === exportGeneration.current) {
        recorder.current = null;
        if (audio.current === context) audio.current = null;
        setBusy(false);
        setError(e.message || "视频导出失败");
      }
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
        aria-label={t("下载片段")}
      >
        <header className="clip-header">
          <h2>{t("下载片段")}</h2>
          <button
            className="icon-button"
            aria-label={t("关闭")}
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
            aria-label={t("视频预览")}
          />
          <div className="clip-timeline">
            <button
              className="icon-button"
              disabled={busy}
              aria-label={playing ? t("暂停") : t("播放")}
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
              aria-label={t("视频进度")}
            />
          </div>
          <div className="clip-controls">
            <div className="clip-row">
              <span>{t("速度")}</span>
              <select
                aria-label={t("速度")}
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
              <span>{t("视角")}</span>
              <button disabled={busy} onClick={() => setFlipped(!flipped)}>
                {flipped ? t("玩家2") : t("玩家1")}
              </button>
            </div>
            <label className="clip-row">
              <span>{t("声音")}</span>
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
              <p className="clip-note">{t("仅导出前 100 步。")}</p>
            )}
            {error && (
              <p className="clip-error" role="alert">
                {t(error)}
              </p>
            )}
            {busy && (
              <p className="clip-note">
                {t("正在录制 {percent}%", {
                  percent: Math.round(recordingProgress),
                })}
              </p>
            )}
            {format === "WebM" && (
              <p className="clip-note">
                {t("此浏览器将以 WebM 格式保存视频。")}
              </p>
            )}
          </div>
        </div>
        <footer className="clip-footer">
          {busy ? (
            <button
              className="clip-download"
              onClick={() => {
                exportGeneration.current++;
                if (recorder.current) recorder.current.cancelExport();
                else {
                  audio.current?.close().catch(() => {});
                  audio.current = null;
                  setBusy(false);
                }
              }}
            >
              {t("取消导出")}
            </button>
          ) : (
            <button className="clip-download" onClick={download}>
              <ArrowDownTrayIcon className="clip-icon" /> {t("下载")}
            </button>
          )}
        </footer>
      </section>
    </div>
  );
}
