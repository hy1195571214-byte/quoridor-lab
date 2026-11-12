import { useLanguage, LanguageSelector } from "../i18n.jsx";
import React from "react";
import {
  ArrowRightIcon,
  ComputerDesktopIcon,
  PlayIcon,
  UsersIcon,
} from "@heroicons/react/24/outline";
import "./Lobby.css";

const entryCards = [
  {
    title: "人机对战",
    description: "自定义策略与基线",
    icon: ComputerDesktopIcon,
    path: "/computer",
  },
  {
    title: "本地对战",
    description: "同一设备",
    icon: UsersIcon,
    path: "/local",
  },
];

export default function Lobby({ navigate }) {
  const { t } = useLanguage();
  return (
    <main className="lobby-page">
      <div className="lobby-content">
        <section className="lobby-hero lobby-card">
          <div className="lobby-hero-head">
            <div>
              <h1>
                {t("玩")} <span>Quoridor Lab</span>
              </h1>
              <p>
                {t(
                  "抢在对手之前穿过棋盘，用路障挡住他们的路 — 但别把自己困住。",
                )}
              </p>
            </div>
            <div className="lobby-hero-aside">
              <LanguageSelector className="lobby-language" />
              <button onClick={() => navigate("/rules")}>
                {t("新手？60 秒学会")} <ArrowRightIcon className="lobby-icon" />
              </button>
            </div>
          </div>
          <button
            className="lobby-play"
            onClick={() => navigate("/computer/intro")}
          >
            <PlayIcon className="lobby-icon" />
            <span>
              <strong>{t("立即开玩")}</strong>
            </span>
          </button>
          <div className="lobby-divider" />
          <div className="lobby-entries">
            {entryCards.map(({ title, description, icon: Icon, path }) => (
              <button key={title} onClick={() => navigate(path)}>
                <span className="lobby-entry-icon">
                  <Icon className="lobby-icon" />
                </span>
                <span>
                  <strong>{t(title)}</strong>
                  <small>{t(description)}</small>
                </span>
                <ArrowRightIcon className="lobby-arrow" />
              </button>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}
