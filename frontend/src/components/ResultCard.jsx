import { useLanguage } from "../i18n.jsx";
import React, { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import confetti from "canvas-confetti";
import "./ResultCard.css";

export default function ResultCard({
  title,
  description,
  onNewGame,
  onHome,
  onAnalyze,
  onSave,
}) {
  const { t } = useLanguage();
  const panel = useRef(null);
  const [mobile, setMobile] = useState(
    () => matchMedia("(max-width: 1023px)").matches,
  );
  useEffect(() => {
    const media = matchMedia("(max-width: 1023px)");
    const update = () => setMobile(media.matches);
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  useEffect(() => {
    const previous = document.activeElement;
    panel.current?.querySelector("button")?.focus({ preventScroll: true });
    const canvas = document.createElement("canvas");
    canvas.className = "result-confetti";
    const overlay = panel.current?.parentElement;
    overlay?.append(canvas);
    const celebrate = confetti.create(canvas, {
      resize: true,
      useWorker: false,
    });
    celebrate({
      particleCount: 90,
      spread: 70,
      startVelocity: 28,
      gravity: 0.9,
      origin: { x: 0.5, y: 0.35 },
      colors: ["#f59e0b", "#ef4444", "#3b82f6", "#ffffff"],
      disableForReducedMotion: true,
    });
    return () => {
      celebrate.reset();
      canvas.remove();
      if (previous?.isConnected) previous.focus();
    };
  }, [mobile]);

  function keepFocus(event) {
    if (event.key !== "Tab") return;
    const controls = [...panel.current.querySelectorAll("button")];
    const first = controls[0],
      last = controls.at(-1);
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  const card = (
    <section
      className="result-card"
      ref={panel}
      role="dialog"
      aria-modal="true"
      aria-labelledby="game-result-title"
      onKeyDown={keepFocus}
    >
      <h1 id="game-result-title">
        {title} <span aria-hidden="true">🏆</span>
      </h1>
      <p>{description}</p>
      <div className="result-actions">
        <button className="result-play" onClick={onNewGame}>
          {t("再玩一局")}
        </button>
        <div className="result-secondary">
          <button onClick={onHome}>{t("回到大厅")}</button>
          <button onClick={onAnalyze}>{t("分析对局")}</button>
        </div>
        <div className="result-record">
          <div className="result-record-copy">
            <strong>{t("对局记录")}</strong>
            <span>{t("完整棋谱与胜负结果")}</span>
          </div>
          <button onClick={onSave}>{t("保存棋谱")}</button>
        </div>
      </div>
    </section>
  );
  return mobile
    ? createPortal(
        <div className="mobile-result-layer">{card}</div>,
        document.body,
      )
    : card;
}
