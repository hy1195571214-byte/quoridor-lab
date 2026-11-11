import React, { useState } from "react";
import {
  ArrowDownTrayIcon,
  ArrowRightIcon,
  ComputerDesktopIcon,
  DevicePhoneMobileIcon,
  PlayIcon,
  PuzzlePieceIcon,
  TrophyIcon,
  UserGroupIcon,
  XMarkIcon,
  UsersIcon,
} from "@heroicons/react/24/outline";
import "./Lobby.css";

const entryCards = [
  {
    title: "和好友一起玩",
    description: "发送链接",
    icon: UserGroupIcon,
    path: "/friends",
  },
  {
    title: "人机对战",
    description: "自定义策略与基线",
    icon: ComputerDesktopIcon,
    path: "/computer",
  },
  {
    title: "山丘之王",
    description: "4 人 · 休闲",
    icon: TrophyIcon,
    path: "/king",
  },
  {
    title: "本地对战",
    description: "同一设备",
    icon: UsersIcon,
    path: "/local",
  },
];

export default function Lobby({ navigate, onImport, onRecords }) {
  const [roomCode, setRoomCode] = useState("");
  const [notice, setNotice] = useState("");
  function unavailable(message) {
    setNotice(message);
  }
  return (
    <main className="lobby-page">
      <div className="lobby-app-banner">
        <div className="lobby-banner-inner">
          <DevicePhoneMobileIcon className="lobby-icon lobby-banner-icon" />
          <div>
            <strong>在手机上玩 Barricade</strong>
            <span>iPhone 和 Android 免费下载</span>
          </div>
          <a href="https://barricade.gg/app" target="_blank" rel="noreferrer">
            获取
          </a>
        </div>
      </div>
      <div className="lobby-content">
        <section className="lobby-hero lobby-card">
          <div className="lobby-hero-head">
            <div>
              <h1>
                玩 <span>Barricade</span>
              </h1>
              <p>抢在对手之前穿过棋盘，用路障挡住他们的路 — 但别把自己困住。</p>
            </div>
            <div className="lobby-hero-aside">
              <span className="lobby-offline">实时服务未连接</span>
              <button onClick={() => navigate("/rules")}>
                新手？60 秒学会 <ArrowRightIcon className="lobby-icon" />
              </button>
            </div>
          </div>
          <button
            className="lobby-play"
            onClick={() => navigate("/computer/intro")}
          >
            <PlayIcon className="lobby-icon" />
            <span>
              <strong>立即开玩</strong>
              <small>免费 — 无需账号</small>
            </span>
          </button>
          <div className="lobby-divider" />
          <div className="lobby-entries">
            {entryCards.map(({ title, description, icon: Icon, path }) => (
              <button
                key={title}
                onClick={() =>
                  path === "/friends" || path === "/king"
                    ? unavailable("此模式需要连接平台服务。")
                    : navigate(path)
                }
              >
                <span className="lobby-entry-icon">
                  <Icon className="lobby-icon" />
                </span>
                <span>
                  <strong>{title}</strong>
                  <small>{description}</small>
                </span>
                <ArrowRightIcon className="lobby-arrow" />
              </button>
            ))}
          </div>
        </section>
        {notice && (
          <div role="status" className="lobby-notice">
            {notice}
            <button aria-label="关闭提示" onClick={() => setNotice("")}>
              <XMarkIcon className="lobby-icon" />
            </button>
          </div>
        )}
        <div className="lobby-columns">
          <section className="lobby-card lobby-rooms">
            <div className="lobby-section-heading">
              <div>
                <h2>实时大厅</h2>
                <p>寻找对手，加入一场对局</p>
              </div>
              <span className="lobby-offline">未连接</span>
            </div>
            <div className="lobby-room-actions">
              <button onClick={() => unavailable("创建房间需要连接平台服务。")}>
                创建房间
              </button>
              <div className="lobby-code">
                <input
                  aria-label="房间码"
                  value={roomCode}
                  onChange={(event) => setRoomCode(event.target.value)}
                  placeholder="输入房间码"
                />
                <button
                  onClick={() =>
                    unavailable(
                      roomCode.trim()
                        ? "房间服务未连接，暂时无法加入。"
                        : "请输入房间码。",
                    )
                  }
                >
                  加入
                </button>
              </div>
            </div>
            <div className="lobby-table-title">
              <strong>开放房间</strong>
              <button
                aria-label="刷新开放房间"
                onClick={() => unavailable("实时大厅服务未连接。")}
              >
                刷新
              </button>
            </div>
            <div className="lobby-table-head">
              <span>房间</span>
              <span>玩家</span>
              <span>模式</span>
            </div>
            <div className="lobby-empty">
              实时大厅未连接，无法获取开放房间。
            </div>
          </section>
          <aside className="lobby-side">
            <button
              className="lobby-card lobby-side-card"
              onClick={() => navigate("/puzzle")}
            >
              <span className="lobby-side-icon amber">
                <PuzzlePieceIcon className="lobby-icon" />
              </span>
              <span>
                <strong>每日谜题</strong>
                <small>挑战你的棋局技巧</small>
              </span>
              <ArrowRightIcon className="lobby-arrow" />
            </button>
            <button
              className="lobby-card lobby-side-card"
              onClick={() => navigate("/tv")}
            >
              <span className="lobby-side-icon red">
                <PlayIcon className="lobby-icon" />
              </span>
              <span>
                <strong>Barricade TV</strong>
                <small>观看精彩对局</small>
              </span>
              <ArrowRightIcon className="lobby-arrow" />
            </button>
            <button
              className="lobby-card lobby-side-card"
              onClick={() => navigate("/leaderboard")}
            >
              <span className="lobby-side-icon purple">
                <TrophyIcon className="lobby-icon" />
              </span>
              <span>
                <strong>排行榜</strong>
                <small>查看顶尖玩家</small>
              </span>
              <ArrowRightIcon className="lobby-arrow" />
            </button>
            <button className="lobby-import" onClick={onRecords}>
              本地对局记录
            </button>
            <button className="lobby-import" onClick={onImport}>
              <ArrowDownTrayIcon className="lobby-icon" /> 导入棋谱
            </button>
          </aside>
        </div>
      </div>
    </main>
  );
}
