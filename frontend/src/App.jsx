import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  ArrowDownTrayIcon,
  ArrowLeftIcon,
  ArrowPathIcon,
  ArrowsUpDownIcon,
  BackwardIcon,
  BookOpenIcon,
  ChartBarIcon,
  CheckIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  EllipsisHorizontalIcon,
  FlagIcon,
  ForwardIcon,
  HomeIcon,
  LinkIcon,
  PauseIcon,
  PlayIcon,
  PuzzlePieceIcon,
  SpeakerWaveIcon,
  SpeakerXMarkIcon,
  TrashIcon,
  TrophyIcon,
  TvIcon,
  XMarkIcon,
  Cog6ToothIcon,
} from "@heroicons/react/24/outline";
import Board from "./components/Board.jsx";
import ClipDialog from "./components/ClipDialog.jsx";
import Lobby from "./components/Lobby.jsx";
import { applyAction, notation, replay } from "./game/rules.js";
import { agentRegistry, createAgentRunner, httpAgent } from "./game/agents.js";
import {
  createRecord,
  decodeShare,
  encodeShare,
  exportRecord,
  parseRecord,
} from "./game/record.js";
import {
  playMoveSound,
  readPreference,
  writePreference,
} from "./game/preferences.js";
import "./styles.css";

for (const id of ["dl", "rl", "search"])
  agentRegistry.register(id, httpAgent(`/api/agents/${id}/move`));
const strategyNames = {
  shortest: "最短路基线",
  random: "随机基线",
  dl: "深度学习",
  rl: "强化学习",
  search: "搜索型 AI",
};
const recordsKey = "barricade.saved-records.v1";
function currentRoute() {
  return location.protocol === "file:"
    ? location.hash.slice(1) || "/"
    : location.pathname + location.search;
}
function routeParts() {
  const route = currentRoute();
  const at = route.indexOf("?");
  return {
    pathname: at < 0 ? route : route.slice(0, at),
    search: at < 0 ? "" : route.slice(at),
  };
}
function shareUrl(route) {
  return location.protocol === "file:"
    ? `${location.href.split("#")[0]}#${route}`
    : new URL(route, location.origin).href;
}
function readRecords() {
  try {
    const list = JSON.parse(localStorage.getItem(recordsKey) || "[]");
    return Array.isArray(list)
      ? list.flatMap((item) => {
          try {
            return [{ ...item, record: parseRecord(item.record) }];
          } catch {
            return [];
          }
        })
      : [];
  } catch {
    return [];
  }
}
const errorMessages = {
  crossing: "路障不能交叉。",
  overlap: "路障不能重叠。",
  "blocks-path": "必须给双方保留一条通往终点的路径。",
  "no-walls": "没有剩余路障了。",
  "game-over": "对局已经结束。",
};
const emptySelection = { cursor: -1, branchId: null, branchCursor: -1 };

function IconButton({ icon: Icon, label, ...props }) {
  return (
    <button className="icon-button" title={label} aria-label={label} {...props}>
      <Icon />
    </button>
  );
}

function Modal({ title, onClose, children, wide = false }) {
  const panel = useRef(null);
  useEffect(() => {
    const previous = document.activeElement;
    panel.current?.querySelector("button, input, textarea, select")?.focus();
    function handleKey(event) {
      if (event.key === "Escape") onClose();
      if (event.key !== "Tab") return;
      const elements = [
        ...panel.current.querySelectorAll(
          'button:not(:disabled), input, textarea, select, [tabindex="0"]',
        ),
      ];
      if (!elements.length) return;
      if (event.shiftKey && document.activeElement === elements[0]) {
        event.preventDefault();
        elements.at(-1).focus();
      }
      if (!event.shiftKey && document.activeElement === elements.at(-1)) {
        event.preventDefault();
        elements[0].focus();
      }
    }
    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("keydown", handleKey);
      previous?.focus();
    };
  }, []);
  return (
    <div
      className="modal-backdrop"
      onMouseDown={(event) => event.target === event.currentTarget && onClose()}
    >
      <section
        ref={panel}
        className={`modal ${wide ? "wide-modal" : ""}`}
        role="dialog"
        aria-modal="true"
        aria-label={title}
      >
        <header className="modal-header">
          <h2>{title}</h2>
          <IconButton icon={XMarkIcon} label="关闭" onClick={onClose} />
        </header>
        <div className="modal-body">{children}</div>
      </section>
    </div>
  );
}

function Sidebar({ navigate, path }) {
  return (
    <aside className="sidebar">
      <a
        className="brand"
        href="/"
        onClick={(event) => {
          event.preventDefault();
          navigate("/");
        }}
      >
        <img src="/source-assets/favicon.svg" alt="" />
        <strong>Barricade</strong>
      </a>
      <nav>
        {[
          [HomeIcon, "对局", "/"],
          [PuzzlePieceIcon, "谜题", "/puzzle"],
          [TvIcon, "观战", "/tv"],
          [TrophyIcon, "排行榜", "/leaderboard"],
        ].map(([Icon, text, href]) => (
          <a
            key={href}
            className={path === href ? "active" : ""}
            href={href}
            onClick={(event) => {
              event.preventDefault();
              navigate(href);
            }}
          >
            <Icon />
            {text}
          </a>
        ))}
      </nav>
      <div className="account-links">
        <button onClick={() => navigate("/signup")}>注册</button>
        <button onClick={() => navigate("/login")}>登录</button>
      </div>
      <nav className="sidebar-bottom">
        <a
          href="/rules"
          onClick={(event) => {
            event.preventDefault();
            navigate("/rules");
          }}
        >
          <BookOpenIcon />
          玩法
        </a>
        <a
          href="/premium"
          onClick={(event) => {
            event.preventDefault();
            navigate("/premium");
          }}
        >
          Premium
        </a>
        <span className="language">简体中文</span>
        <a
          href="/settings"
          onClick={(event) => {
            event.preventDefault();
            navigate("/settings");
          }}
        >
          <Cog6ToothIcon />
          设置
        </a>
      </nav>
    </aside>
  );
}

function PlayerCard({ player, state, name, highlighted }) {
  return (
    <div className={`player-card ${highlighted ? "highlighted" : ""}`}>
      <div className="player-name">
        <span className={`player-dot player-${player}`} />
        {name}
        {highlighted && <span className="sr-only">当前回合</span>}
      </div>
      <div className="wall-count">
        <span className="wall-ticks">
          {Array.from({ length: 10 }, (_, index) => (
            <i
              key={index}
              className={
                index < state.remaining[player - 1] ? `player-${player}` : ""
              }
            />
          ))}
        </span>
        <span>{state.remaining[player - 1]}/10</span>
      </div>
    </div>
  );
}

function MoveList({
  record,
  selection,
  onMain,
  onBranch,
  onDeleteBranch,
  replaying,
}) {
  const end = useRef(null);
  useEffect(() => {
    if (!replaying) end.current?.scrollIntoView({ block: "nearest" });
  }, [record.moves.length, replaying]);
  return (
    <div className="move-list" aria-label="棋谱">
      {!record.moves.length && <p className="empty-moves">还没有走棋</p>}
      {Array.from(
        { length: Math.ceil(record.moves.length / 2) },
        (_, round) => (
          <React.Fragment key={round}>
            <div className="move-row">
              <span className="move-number">{round + 1}.</span>
              {[round * 2, round * 2 + 1].map((index) => (
                <button
                  key={index}
                  disabled={!record.moves[index]}
                  className={`${index % 2 ? "blue-move" : "red-move"} ${selection.branchId === null && selection.cursor === index ? "selected-move" : ""}`}
                  onClick={() => onMain(index)}
                >
                  {record.moves[index] || ""}
                </button>
              ))}
            </div>
            {record.variations.map(
              (branch, branchId) =>
                Math.floor(branch.anchor / 2) === round && (
                  <div
                    key={branchId}
                    className={`variation ${selection.branchId === branchId ? "selected-variation" : ""}`}
                  >
                    <span className="variation-label">变化</span>
                    <div>
                      {branch.moves.map((move, index) => (
                        <button
                          key={index}
                          onClick={() => onBranch(branchId, index)}
                          className={
                            selection.branchId === branchId &&
                            selection.branchCursor === index
                              ? "selected-move"
                              : ""
                          }
                        >
                          {move}
                        </button>
                      ))}
                    </div>
                    <IconButton
                      icon={TrashIcon}
                      label={`删除变化 ${branchId + 1}`}
                      onClick={() => onDeleteBranch(branchId)}
                    />
                  </div>
                ),
            )}
          </React.Fragment>
        ),
      )}
      {record.variations.map(
        (branch, branchId) =>
          branch.anchor === -1 && (
            <div key={branchId} className="variation">
              <span className="variation-label">开局变化</span>
              <div>
                {branch.moves.map((move, index) => (
                  <button
                    key={index}
                    className={
                      selection.branchId === branchId &&
                      selection.branchCursor === index
                        ? "selected-move"
                        : ""
                    }
                    onClick={() => onBranch(branchId, index)}
                  >
                    {move}
                  </button>
                ))}
              </div>
              <IconButton
                icon={TrashIcon}
                label={`删除变化 ${branchId + 1}`}
                onClick={() => onDeleteBranch(branchId)}
              />
            </div>
          ),
      )}
      <div ref={end} />
    </div>
  );
}

function ImportForm({ onImport }) {
  const [text, setText] = useState("");
  const [error, setError] = useState("");
  const submit = (value) => {
    try {
      let record;
      const input = value.trim();
      if (input.startsWith("{")) record = parseRecord(input);
      else if (input.includes("moves=") || input.includes("var="))
        record = decodeShare(input);
      else record = createRecord(input.split(/[\s,]+/).filter(Boolean));
      onImport(record);
    } catch (failure) {
      setError(`棋谱无法载入。${failure.message}`);
    }
  };
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        submit(text);
      }}
    >
      <label className="field-label" htmlFor="record-input">
        粘贴棋谱、回放链接或 JSON
      </label>
      <textarea
        id="record-input"
        value={text}
        onChange={(event) => setText(event.target.value)}
        placeholder="e2,e8,he4"
        rows={6}
        required
      />
      <label className="file-input">
        选择棋谱文件
        <input
          type="file"
          accept=".json,application/json"
          onChange={async (event) => {
            const file = event.target.files?.[0];
            if (!file) return;
            if (file.size > 128000) {
              setError("棋谱文件过大。");
              return;
            }
            submit(await file.text());
          }}
        />
      </label>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      <button className="primary full-width" type="submit">
        载入棋谱
      </button>
    </form>
  );
}

export default function App() {
  const [locationKey, setLocationKey] = useState(currentRoute());
  const path = locationKey.split("?")[0];
  const isReplay = path === "/analysis";
  const [record, setRecord] = useState(() => {
    try {
      return routeParts().pathname === "/analysis"
        ? decodeShare(routeParts().search)
        : createRecord();
    } catch {
      return createRecord();
    }
  });
  const [selection, setSelection] = useState(() => ({
    ...emptySelection,
    cursor: record.moves.length - 1,
    branchId: record.variations.length ? record.variations.length - 1 : null,
    branchCursor: (record.variations.at(-1)?.moves.length ?? 0) - 1,
  }));
  const [muted, setMuted] = useState(() => readPreference("muted", false));
  const [light, setLight] = useState(() => readPreference("light", false));
  const [rack, setRack] = useState(() => readPreference("rack", false));
  const [flipped, setFlipped] = useState(false);
  const [menu, setMenu] = useState(false);
  const [modal, setModal] = useState(null);
  const [notice, setNotice] = useState("");
  const [confirmResign, setConfirmResign] = useState(false);
  const [computerStarted, setComputerStarted] = useState(false);
  const [savedRecords, setSavedRecords] = useState(readRecords);
  const savedId = useRef(null);
  const [endpoints, setEndpoints] = useState(() =>
    readPreference("agentEndpoints", {
      dl: `${location.protocol === "file:" ? "http://127.0.0.1:8765" : "/api"}/agents/dl/move`,
      rl: `${location.protocol === "file:" ? "http://127.0.0.1:8765" : "/api"}/agents/rl/move`,
      search: `${location.protocol === "file:" ? "http://127.0.0.1:8765" : "/api"}/agents/search/move`,
    }),
  );
  const [human, setHuman] = useState(1);
  const [strategy, setStrategy] = useState("shortest");
  const [thinking, setThinking] = useState(false);
  const [aiError, setAiError] = useState("");
  const [retry, setRetry] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [dragKind, setDragKind] = useState(null);
  const [dragPosition, setDragPosition] = useState(null);
  const [dragTarget, setDragTarget] = useState(null);
  const [suggestion, setSuggestion] = useState(null);
  const runner = useRef(createAgentRunner(agentRegistry));
  const analysisRunner = useRef(createAgentRunner(agentRegistry));
  const resignTimer = useRef(null);
  const dragTimer = useRef(null);
  const dragCleanup = useRef(null);
  const computerRoute = path === "/computer" || path === "/computer/intro";
  const isComputer = computerRoute && computerStarted;
  const isGame = path === "/local" || isComputer || isReplay;
  const branch =
    selection.branchId === null ? null : record.variations[selection.branchId];
  const displayedMoves = isReplay
    ? branch
      ? [
          ...record.moves.slice(0, branch.anchor + 1),
          ...branch.moves.slice(0, selection.branchCursor + 1),
        ]
      : record.moves.slice(0, selection.cursor + 1)
    : record.moves;
  const state = useMemo(
    () => replay(displayedMoves),
    [displayedMoves.join(",")],
  );
  const finished = !isReplay && Boolean(record.result);
  const locked =
    finished ||
    thinking ||
    Boolean(aiError) ||
    (isComputer && state.turn !== human);
  const names = isComputer
    ? {
        [human]: "你",
        [3 - human]: `电脑 · ${strategyNames[strategy] || strategy}`,
      }
    : { 1: "玩家 1", 2: "玩家 2" };
  const topPlayer = flipped ? 1 : 2;
  const bottomPlayer = 3 - topPlayer;

  useEffect(() => {
    for (const id of ["dl", "rl", "search"])
      if (endpoints[id]?.trim())
        agentRegistry.register(id, httpAgent(endpoints[id].trim()));
  }, [endpoints]);
  useEffect(() => {
    writePreference("muted", muted);
  }, [muted]);
  useEffect(() => {
    writePreference("light", light);
  }, [light]);
  useEffect(() => {
    writePreference("rack", rack);
  }, [rack]);
  useEffect(() => {
    if (notice) {
      const timer = setTimeout(() => setNotice(""), 4000);
      return () => clearTimeout(timer);
    }
  }, [notice]);
  useEffect(() => {
    const pop = () => {
      savedId.current = null;
      setModal(null);
      setMenu(false);
      setLocationKey(currentRoute());
      setComputerStarted(false);
      loadLocation();
    };
    addEventListener("popstate", pop);
    if (location.protocol === "file:") addEventListener("hashchange", pop);
    return () => {
      removeEventListener("popstate", pop);
      if (location.protocol === "file:") removeEventListener("hashchange", pop);
      runner.current.cancel();
      analysisRunner.current.cancel();
      clearTimeout(resignTimer.current);
      dragCleanup.current?.();
    };
  }, []);

  useEffect(() => {
    loadLocation();
  }, []);

  useEffect(() => {
    if (
      !(path === "/local" || isComputer) ||
      (!record.moves.length && !record.result && !savedId.current)
    )
      return;
    try {
      const clean = parseRecord(record);
      const id =
        savedId.current ||
        (globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`);
      savedId.current = id;
      const entry = {
        id,
        updatedAt: Date.now(),
        mode: isComputer ? "computer" : "local",
        human,
        strategy,
        endpoints,
        record: clean,
      };
      const next = [entry, ...readRecords().filter((item) => item.id !== id)];
      localStorage.setItem(recordsKey, JSON.stringify(next));
      setSavedRecords(next);
    } catch {
      notify("对局记录保存失败，请检查浏览器存储空间。");
    }
  }, [record, path, isComputer]);
  function restoreRecord(entry) {
    try {
      const clean = parseRecord(entry.record);
      navigate(entry.mode === "computer" ? "/computer" : "/local");
      savedId.current = entry.id;
      if (entry.mode === "computer") {
        const id = strategyNames[entry.strategy] ? entry.strategy : "shortest";
        const urls = entry.endpoints || endpoints;
        setHuman(entry.human === 2 ? 2 : 1);
        setStrategy(id);
        setEndpoints(urls);
        if (["dl", "rl", "search"].includes(id) && urls[id])
          agentRegistry.register(id, httpAgent(urls[id]));
        setComputerStarted(true);
      }
      setRecord(clean);
      setSelection({ ...emptySelection, cursor: clean.moves.length - 1 });
      notify("已恢复对局");
    } catch {
      notify("记录无效，无法恢复。");
    }
  }
  function deleteSaved(id) {
    const next = savedRecords.filter((item) => item.id !== id);
    try {
      localStorage.setItem(recordsKey, JSON.stringify(next));
      setSavedRecords(next);
      if (savedId.current === id) savedId.current = null;
    } catch {
      notify("删除失败，请检查浏览器存储。");
    }
  }

  function loadLocation() {
    dragCleanup.current?.();
    runner.current.cancel();
    analysisRunner.current.cancel();
    setPlaying(false);
    setConfirmResign(false);
    setAiError("");
    setSuggestion(null);
    setThinking(false);
    try {
      const route = routeParts();
      const next =
        route.pathname === "/analysis"
          ? decodeShare(route.search)
          : createRecord();
      if (
        route.pathname === "/analysis" &&
        new URLSearchParams(route.search).has("game")
      )
        throw new Error(
          "此链接使用原站服务器对局编号，请导入其棋谱或 moves 回放链接。",
        );
      setRecord(next);
      setSelection({
        ...emptySelection,
        cursor: next.moves.length - 1,
        branchId: next.variations.length ? next.variations.length - 1 : null,
        branchCursor: (next.variations.at(-1)?.moves.length ?? 0) - 1,
      });
    } catch (error) {
      setRecord(createRecord());
      setSelection(emptySelection);
      setNotice(error.message);
    }
  }
  function navigate(url) {
    dragCleanup.current?.();
    savedId.current = null;
    if (location.protocol === "file:") history.pushState(null, "", `#${url}`);
    else history.pushState(null, "", url);
    setLocationKey(currentRoute());
    setModal(null);
    setMenu(false);
    setComputerStarted(false);
    loadLocation();
  }
  function newGame() {
    savedId.current = null;
    dragCleanup.current?.();
    runner.current.cancel();
    setRecord(createRecord());
    setSelection(emptySelection);
    setAiError("");
    setThinking(false);
    setConfirmResign(false);
    setSuggestion(null);
    setRetry((value) => value + 1);
  }
  function notify(message) {
    setNotice("");
    requestAnimationFrame(() => setNotice(message));
  }
  function selectMain(cursor) {
    setPlaying(false);
    setSelection({ cursor, branchId: null, branchCursor: -1 });
    setSuggestion(null);
  }
  function selectBranch(branchId, branchCursor) {
    setPlaying(false);
    setSelection({
      cursor: record.variations[branchId].anchor,
      branchId,
      branchCursor,
    });
    setSuggestion(null);
  }
  function deleteBranch(branchId) {
    const anchor = record.variations[branchId].anchor;
    setRecord((previous) => ({
      ...previous,
      variations: previous.variations.filter((_, index) => index !== branchId),
    }));
    selectMain(anchor);
  }
  function handleAction(action) {
    if (!isReplay && locked) return;
    try {
      const next = applyAction(state, action);
      const move = notation(action);
      setConfirmResign(false);
      clearTimeout(resignTimer.current);
      setSuggestion(null);
      playMoveSound(muted, action.kind !== "move");
      if (!isReplay) {
        setRecord((previous) => ({
          ...previous,
          moves: next.history,
          result: next.winner ? { winner: next.winner, reason: "goal" } : null,
        }));
        setSelection({ ...emptySelection, cursor: next.history.length - 1 });
        return;
      }
      setPlaying(false);
      if (!branch && record.moves[selection.cursor + 1] === move) {
        selectMain(selection.cursor + 1);
        return;
      }
      if (branch && branch.moves[selection.branchCursor + 1] === move) {
        selectBranch(selection.branchId, selection.branchCursor + 1);
        return;
      }
      const prefix = branch
        ? branch.moves.slice(0, selection.branchCursor + 1)
        : [];
      const anchor = branch ? branch.anchor : selection.cursor;
      const extended = { anchor, moves: [...prefix, move] };
      if (branch && selection.branchCursor === branch.moves.length - 1) {
        setRecord((previous) => ({
          ...previous,
          variations: previous.variations.map((value, index) =>
            index === selection.branchId ? extended : value,
          ),
        }));
        setSelection((previous) => ({
          ...previous,
          branchCursor: previous.branchCursor + 1,
        }));
      } else {
        const branchId = record.variations.length;
        setRecord((previous) => ({
          ...previous,
          variations: [...previous.variations, extended],
        }));
        setSelection({
          cursor: anchor,
          branchId,
          branchCursor: extended.moves.length - 1,
        });
      }
    } catch (error) {
      if (error.code !== "illegal-move")
        notify(errorMessages[error.code] || error.message);
    }
  }
  function undo() {
    if (!record.moves.length || finished || thinking) return;
    const moves = record.moves.slice(0, -1);
    setRecord(createRecord(moves));
    setSelection({ ...emptySelection, cursor: moves.length - 1 });
    setConfirmResign(false);
  }
  function resign() {
    if (locked) return;
    if (!confirmResign) {
      setConfirmResign(true);
      clearTimeout(resignTimer.current);
      resignTimer.current = setTimeout(() => setConfirmResign(false), 4000);
      return;
    }
    setRecord((previous) => ({
      ...previous,
      result: { winner: 3 - state.turn, reason: "resign" },
    }));
    setConfirmResign(false);
  }

  useEffect(() => {
    if (
      !isComputer ||
      record.result ||
      state.winner ||
      state.turn === human ||
      aiError
    )
      return;
    let cancelled = false;
    const controller = new AbortController();
    setThinking(true);
    const timer = setTimeout(async () => {
      try {
        const decision = await runner.current.choose(strategy, {
          state,
          signal: controller.signal,
          requestId: `${record.moves.length}-${retry}`,
        });
        if (cancelled) return;
        const next = applyAction(state, decision.action);
        setRecord((previous) => ({
          ...previous,
          moves: next.history,
          result: next.winner ? { winner: next.winner, reason: "goal" } : null,
        }));
        setSelection({ ...emptySelection, cursor: next.history.length - 1 });
        playMoveSound(muted, decision.action.kind !== "move");
      } catch (error) {
        if (!cancelled && error.name !== "AbortError")
          setAiError("策略未能应手。请检查本地接口配置，然后重试。");
      } finally {
        if (!cancelled) setThinking(false);
      }
    }, 350);
    return () => {
      cancelled = true;
      clearTimeout(timer);
      controller.abort();
      runner.current.cancel();
    };
  }, [
    isComputer,
    record.moves.join(","),
    Boolean(record.result),
    human,
    strategy,
    retry,
  ]);

  const canPrev = branch ? selection.branchCursor >= 0 : selection.cursor >= 0;
  const canNext = branch
    ? selection.branchCursor < branch.moves.length - 1
    : selection.cursor < record.moves.length - 1;
  const goFirst = useCallback(() => {
    setPlaying(false);
    setSelection(emptySelection);
  }, []);
  function goLast() {
    setPlaying(false);
    setSelection({ ...emptySelection, cursor: record.moves.length - 1 });
  }
  function step(direction) {
    setPlaying(false);
    if (branch && direction < 0 && selection.branchCursor <= 0) {
      selectMain(branch.anchor);
    } else if (branch)
      selectBranch(
        selection.branchId,
        Math.max(
          0,
          Math.min(branch.moves.length - 1, selection.branchCursor + direction),
        ),
      );
    else
      selectMain(
        Math.max(
          -1,
          Math.min(record.moves.length - 1, selection.cursor + direction),
        ),
      );
  }
  function togglePlay() {
    if (!canNext)
      setSelection(
        branch ? { ...selection, branchCursor: -1 } : emptySelection,
      );
    setPlaying((value) => !value);
  }
  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => {
      setSelection((previous) => {
        const line =
          previous.branchId === null
            ? null
            : record.variations[previous.branchId];
        const atEnd = line
          ? previous.branchCursor >= line.moves.length - 1
          : previous.cursor >= record.moves.length - 1;
        if (atEnd) {
          setPlaying(false);
          return previous;
        }
        const next = line
          ? { ...previous, branchCursor: previous.branchCursor + 1 }
          : { ...previous, cursor: previous.cursor + 1 };
        const reachesEnd = line
          ? next.branchCursor === line.moves.length - 1
          : next.cursor === record.moves.length - 1;
        if (reachesEnd) setPlaying(false);
        return next;
      });
    }, 1000 / speed);
    return () => clearInterval(timer);
  }, [playing, speed, record.moves.length, record.variations]);
  useEffect(() => {
    if (!isReplay || modal) return;
    function keyboard(event) {
      if (
        event.target.closest("input,textarea,select") ||
        (event.target.closest("button") && [" ", "Enter"].includes(event.key))
      )
        return;
      if (
        (event.ctrlKey || event.metaKey) &&
        event.key.toLowerCase() === "c" &&
        !getSelection()?.toString()
      ) {
        event.preventDefault();
        copyText(record.moves.join(","));
        return;
      }
      const actions = {
        ArrowLeft: () => step(-1),
        ArrowRight: () => step(1),
        ArrowUp: goFirst,
        ArrowDown: goLast,
        " ": togglePlay,
        End: togglePlay,
      };
      if (actions[event.key]) {
        event.preventDefault();
        actions[event.key]();
      }
    }
    addEventListener("keydown", keyboard);
    return () => removeEventListener("keydown", keyboard);
  });
  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      notify("已复制");
    } catch {
      setModal({ type: "copy", text });
    }
  }
  function saveRecord() {
    setMenu(false);
    const url = URL.createObjectURL(
      new Blob([exportRecord(record)], { type: "application/json" }),
    );
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "barricade-game.json";
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  async function analyze(id) {
    setSuggestion({ loading: true });
    setModal(null);
    try {
      const result = await analysisRunner.current.choose(id, {
        state,
        requestId: displayedMoves.join(","),
      });
      setSuggestion({ move: result.notation, source: strategyNames[id] });
    } catch (error) {
      if (error.name !== "AbortError")
        setSuggestion({
          error: "分析策略尚未配置或响应失败，请检查本地服务。",
        });
    }
  }
  useEffect(() => {
    analysisRunner.current.cancel();
    setSuggestion(null);
  }, [displayedMoves.join(","), path]);
  function startDrag(kind, event) {
    if (locked) return;
    dragCleanup.current?.();
    const pointerId = event.pointerId;
    const source = event.currentTarget;
    try {
      source.setPointerCapture(pointerId);
    } catch {
      /* Pointer may have already ended. */
    }
    const start = { x: event.clientX, y: event.clientY };
    let active = false;
    const activate = () => {
      active = true;
      setDragKind(kind);
      setDragPosition(start);
    };
    if (event.pointerType === "touch")
      dragTimer.current = setTimeout(activate, 250);
    else activate();
    const move = (pointer) => {
      if (pointer.pointerId !== pointerId) return;
      if (
        !active &&
        Math.hypot(pointer.clientX - start.x, pointer.clientY - start.y) > 8
      )
        clearTimeout(dragTimer.current);
      if (active) {
        pointer.preventDefault();
        setDragPosition({ x: pointer.clientX, y: pointer.clientY });
        const target = document
          .elementsFromPoint(pointer.clientX, pointer.clientY)
          .find(
            (element) => element.dataset.kind === kind && element.dataset.wall,
          );
        setDragTarget(target?.dataset.wall || null);
      }
    };
    const end = (pointer) => {
      if (pointer.pointerId !== pointerId) return;
      clearTimeout(dragTimer.current);
      if (active && pointer.type !== "pointercancel") {
        const slot = document
          .elementsFromPoint(pointer.clientX, pointer.clientY)
          .find(
            (element) => element.dataset.kind === kind && element.dataset.wall,
          );
        if (slot)
          handleAction({ kind, ...parseCoordinates(slot.dataset.wall) });
      }
      setDragKind(null);
      setDragPosition(null);
      setDragTarget(null);
      try {
        if (source.hasPointerCapture(pointerId))
          source.releasePointerCapture(pointerId);
      } catch {
        /* Already released. */
      }
      dragCleanup.current = null;
      document.removeEventListener("pointermove", move);
      document.removeEventListener("pointerup", end);
      document.removeEventListener("pointercancel", end);
    };
    dragCleanup.current = () => {
      clearTimeout(dragTimer.current);
      try {
        if (source.hasPointerCapture(pointerId))
          source.releasePointerCapture(pointerId);
      } catch {
        /* Already released. */
      }
      document.removeEventListener("pointermove", move);
      document.removeEventListener("pointerup", end);
      document.removeEventListener("pointercancel", end);
      setDragKind(null);
      setDragPosition(null);
      setDragTarget(null);
      dragCleanup.current = null;
    };
    document.addEventListener("pointermove", move, { passive: false });
    document.addEventListener("pointerup", end);
    document.addEventListener("pointercancel", end);
  }
  function statusPanel() {
    return (
      <>
        {(thinking || aiError) && (
          <div className="ai-status" role="status">
            {thinking ? (
              "思考中…"
            ) : (
              <>
                <p>{aiError}</p>
                <button
                  onClick={() => {
                    setAiError("");
                    setRetry((value) => value + 1);
                  }}
                >
                  重试
                </button>
                <button onClick={newGame}>重新开始</button>
              </>
            )}
          </div>
        )}
        {suggestion && (
          <div className="suggestion" role="status">
            {suggestion.loading
              ? "分析中…"
              : suggestion.error || (
                  <>
                    <span>{suggestion.source}建议</span>
                    <button
                      onClick={() =>
                        handleAction({
                          kind:
                            suggestion.move.length === 2
                              ? "move"
                              : suggestion.move[0],
                          ...parseCoordinates(suggestion.move),
                        })
                      }
                    >
                      {suggestion.move}
                    </button>
                  </>
                )}
          </div>
        )}
      </>
    );
  }
  function rackControls() {
    return (
      <div className={`wall-rack ${rack ? "" : "rack-collapsed"}`}>
        <button className="rack-title" onClick={() => setRack(!rack)}>
          拖放路障 <span>{state.remaining[state.turn - 1]}</span>
        </button>
        {rack && (
          <div className="rack-pieces">
            {["h", "v"].map((kind) => (
              <button
                key={kind}
                className={`rack-piece rack-${kind} player-${state.turn}`}
                aria-label={`拖放${kind === "h" ? "横" : "竖"}墙`}
                disabled={locked || state.remaining[state.turn - 1] === 0}
                onPointerDown={(event) => startDrag(kind, event)}
              >
                <i />
              </button>
            ))}
          </div>
        )}
      </div>
    );
  }
  function optionsMenu() {
    return (
      menu && (
        <>
          <button
            className="menu-dismiss"
            aria-label="关闭菜单"
            onClick={() => setMenu(false)}
          />
          <div className="options-menu">
            <button
              onClick={() => {
                setRack(!rack);
                setMenu(false);
              }}
            >
              {rack && <CheckIcon />}拖放路障
            </button>
            <button
              onClick={() => {
                setLight(!light);
                setMenu(false);
              }}
            >
              切换到{light ? "深色" : "浅色"}棋盘
            </button>
            <button
              onClick={() => {
                setModal("records");
                setMenu(false);
              }}
            >
              本地对局记录
            </button>
            {!isReplay && <button onClick={saveRecord}>导出棋谱 JSON</button>}
            {isReplay && (
              <>
                <button
                  onClick={() => {
                    setModal("import");
                    setMenu(false);
                  }}
                >
                  载入棋谱
                </button>
                <button onClick={saveRecord}>保存棋谱</button>
              </>
            )}
          </div>
        </>
      )
    );
  }
  function replayControls() {
    return (
      <div className="replay-controls">
        <div className="playback-buttons">
          <IconButton
            icon={BackwardIcon}
            label="首步"
            onClick={goFirst}
            disabled={selection.cursor < 0 && !branch}
          />
          <IconButton
            icon={ChevronLeftIcon}
            label="上一步"
            onClick={() => step(-1)}
            disabled={!canPrev}
          />
          <IconButton
            icon={playing ? PauseIcon : PlayIcon}
            label={playing ? "暂停" : "播放"}
            onClick={togglePlay}
            disabled={!record.moves.length && !branch}
          />
          <IconButton
            icon={ChevronRightIcon}
            label="下一步"
            onClick={() => step(1)}
            disabled={!canNext}
          />
          <IconButton
            icon={ForwardIcon}
            label="末步"
            onClick={goLast}
            disabled={selection.cursor === record.moves.length - 1 && !branch}
          />
          <select
            aria-label="回放速度"
            value={speed}
            onChange={(event) => setSpeed(Number(event.target.value))}
          >
            {[0.5, 1, 2, 3, 5].map((value) => (
              <option key={value} value={value}>
                {value}×
              </option>
            ))}
          </select>
        </div>
        <div className="analysis-actions">
          <button onClick={() => setModal("analyze")}>
            <ChartBarIcon />
            分析对局
          </button>
          <IconButton
            icon={ArrowDownTrayIcon}
            label="下载片段"
            onClick={() => setModal("clip")}
          />
          <IconButton
            icon={LinkIcon}
            label="分享回放"
            onClick={() => copyText(shareUrl(encodeShare(record)))}
          />
          <IconButton
            icon={ArrowsUpDownIcon}
            label="旋转棋盘"
            onClick={() => setFlipped(!flipped)}
          />
        </div>
      </div>
    );
  }

  return (
    <>
      {(isReplay || (!isGame && !computerRoute)) && (
        <Sidebar path={path} navigate={navigate} />
      )}
      {path === "/" ? (
        <div className="with-sidebar">
          <Lobby
            navigate={navigate}
            onImport={() => setModal("import")}
            onRecords={() => setModal("records")}
          />
        </div>
      ) : isGame ? (
        <main
          className={`game-stage ${isReplay ? "with-sidebar replay-stage" : ""}`}
        >
          <div className="game-layout">
            <div className="mobile-toolbar">
              <IconButton
                icon={ArrowLeftIcon}
                label="回到大厅"
                onClick={() => navigate("/")}
              />
              {!isReplay && (
                <>
                  <button
                    onClick={resign}
                    disabled={locked}
                    className={confirmResign ? "danger" : ""}
                  >
                    {confirmResign ? "确定？" : "认输"}
                  </button>
                  {!isComputer && (
                    <IconButton
                      icon={ArrowPathIcon}
                      label="悔棋"
                      disabled={!record.moves.length || finished}
                      onClick={undo}
                    />
                  )}
                </>
              )}
              <button
                className="mobile-move-summary"
                onClick={() => setModal("moves")}
              >
                {record.moves.length
                  ? `${Math.ceil(record.moves.length / 2)}. ${record.moves.slice(-2).join(" ")}`
                  : "棋谱"}
              </button>
              <div className="menu-anchor">
                <IconButton
                  icon={EllipsisHorizontalIcon}
                  label="更多"
                  onClick={() => setMenu(!menu)}
                />
                {optionsMenu()}
              </div>
            </div>
            <div className="mobile-top-card">
              <PlayerCard
                player={topPlayer}
                state={state}
                name={names[topPlayer]}
                highlighted={state.turn === topPlayer && !finished}
              />
            </div>
            <Board
              state={state}
              onAction={handleAction}
              disabled={!isReplay && locked}
              flipped={flipped}
              light={light}
              analysis={isReplay}
              dragKind={dragKind}
              dragTarget={dragTarget}
            >
              {finished && (
                <div className="result-card">
                  <h1>
                    {isComputer
                      ? record.result.winner === human
                        ? "你赢了"
                        : "你输了"
                      : `玩家 ${record.result.winner} 胜出`}
                  </h1>
                  <p>
                    {record.result.reason === "resign"
                      ? `${names[3 - record.result.winner]}认输`
                      : `${names[record.result.winner]}先抵达终点`}
                  </p>
                  <button className="primary" onClick={newGame}>
                    再玩一局
                  </button>
                  <button className="secondary" onClick={() => navigate("/")}>
                    回到大厅
                  </button>
                  <div className="result-links">
                    <button onClick={() => navigate(encodeShare(record))}>
                      分析对局
                    </button>
                    <button onClick={() => setModal("clip")}>下载片段</button>
                    <button onClick={saveRecord}>保存棋谱</button>
                  </div>
                </div>
              )}
            </Board>
            <div className="mobile-bottom-card">
              <PlayerCard
                player={bottomPlayer}
                state={state}
                name={names[bottomPlayer]}
                highlighted={state.turn === bottomPlayer && !finished}
              />
            </div>
            <div className="mobile-rack">
              {statusPanel()}
              {rackControls()}
              {isReplay && replayControls()}
            </div>
            <section
              className="game-panel"
              aria-label={isReplay ? "回放面板" : "对局面板"}
            >
              <header className="panel-toolbar">
                <button className="lobby-button" onClick={() => navigate("/")}>
                  <ArrowLeftIcon />
                  大厅
                </button>
                <IconButton
                  icon={muted ? SpeakerXMarkIcon : SpeakerWaveIcon}
                  label={muted ? "开启声音" : "关闭声音"}
                  onClick={() => setMuted(!muted)}
                />
                <IconButton
                  icon={ArrowsUpDownIcon}
                  label="旋转棋盘"
                  onClick={() => setFlipped(!flipped)}
                />
                <div className="menu-anchor">
                  <IconButton
                    icon={EllipsisHorizontalIcon}
                    label="更多"
                    onClick={() => setMenu(!menu)}
                  />
                  {optionsMenu()}
                </div>
              </header>
              <div className="panel-player">
                <PlayerCard
                  player={topPlayer}
                  state={state}
                  name={names[topPlayer]}
                  highlighted={state.turn === topPlayer && !finished}
                />
              </div>
              <MoveList
                record={record}
                selection={
                  isReplay
                    ? selection
                    : { ...emptySelection, cursor: record.moves.length - 1 }
                }
                onMain={isReplay ? selectMain : () => {}}
                onBranch={selectBranch}
                onDeleteBranch={deleteBranch}
                replaying={isReplay}
              />
              {statusPanel()}
              {rackControls()}
              <div className="panel-player bottom-player">
                <PlayerCard
                  player={bottomPlayer}
                  state={state}
                  name={names[bottomPlayer]}
                  highlighted={state.turn === bottomPlayer && !finished}
                />
              </div>
              {isReplay ? (
                replayControls()
              ) : (
                <footer className="game-actions">
                  {!isComputer && (
                    <button
                      onClick={undo}
                      disabled={!record.moves.length || finished || thinking}
                    >
                      悔棋
                    </button>
                  )}
                  <button
                    className={confirmResign ? "danger" : ""}
                    onClick={resign}
                    disabled={locked}
                  >
                    <FlagIcon />
                    {confirmResign ? "确定？" : "认输"}
                  </button>
                </footer>
              )}
            </section>
          </div>
        </main>
      ) : computerRoute ? (
        <main className="computer-page">
          <header>
            <a
              className="brand"
              href="/"
              onClick={(event) => {
                event.preventDefault();
                navigate("/");
              }}
            >
              <img src="/source-assets/favicon.svg" alt="" />
              Barricade
            </a>
            <button onClick={() => navigate("/")}>
              <HomeIcon />
              首页
            </button>
          </header>
          <section className="computer-card">
            <h1>与电脑对战</h1>
            <label className="field-label">执子</label>
            <div className="side-options">
              {[1, 2].map((player) => (
                <button
                  key={player}
                  className={human === player ? "selected-option" : ""}
                  onClick={() => setHuman(player)}
                >
                  <span className={`player-dot player-${player}`} />
                  {player === 1 ? "红方" : "蓝方"}
                </button>
              ))}
            </div>
            <p className="muted">{human === 1 ? "你先走" : "电脑先走"}</p>
            <label className="field-label" htmlFor="strategy">
              本地策略接口
            </label>
            <select
              id="strategy"
              value={strategy}
              onChange={(event) => setStrategy(event.target.value)}
            >
              {Object.entries(strategyNames).map(([id, name]) => (
                <option key={id} value={id}>
                  {name}
                  {["dl", "rl", "search"].includes(id) ? "（需配置服务）" : ""}
                </option>
              ))}
            </select>
            {["dl", "rl", "search"].includes(strategy) && (
              <label className="field-label">
                策略服务地址
                <input
                  aria-label="策略服务地址"
                  value={endpoints[strategy] || ""}
                  onChange={(event) =>
                    setEndpoints({
                      ...endpoints,
                      [strategy]: event.target.value,
                    })
                  }
                  placeholder="http://127.0.0.1:8000/move"
                />
              </label>
            )}
            <p className="integration-note">
              最短路和随机策略是基线测试；深度学习、强化学习和搜索策略需要接入你自己的服务。
            </p>
            <button
              className="primary full-width"
              onClick={() => {
                if (["dl", "rl", "search"].includes(strategy)) {
                  if (!endpoints[strategy]?.trim()) {
                    notify("请填写策略服务地址。");
                    return;
                  }
                  agentRegistry.register(
                    strategy,
                    httpAgent(endpoints[strategy].trim()),
                  );
                  writePreference("agentEndpoints", endpoints);
                }
                newGame();
                setComputerStarted(true);
              }}
            >
              开始对局
            </button>
          </section>
        </main>
      ) : (
        <main className="content-page with-sidebar">
          {path === "/rules" ? (
            <>
              <h1>玩法</h1>
              <section className="text-card">
                <h2>双人经典模式</h2>
                <p>
                  在 9×9 棋盘上，红方从 e1 出发，蓝方从 e9
                  出发。红方先行，率先到达对面任意一格的玩家获胜。
                </p>
                <h3>移动棋子</h3>
                <p>
                  每回合可以上下左右移动一格。与对方相邻且身后畅通时，可以直接跳过对方。对方身后有墙或位于边界时，可以从两侧斜跳。
                </p>
                <h3>放置路障</h3>
                <p>
                  每人拥有 10
                  道长两格的路障。每回合选择移动或放墙，路障对双方生效。路障不能重叠或交叉，必须为双方各保留至少一条到终点的路径。两道墙首尾相接的接缝允许另一道墙穿过。
                </p>
                <button className="primary" onClick={() => navigate("/local")}>
                  开始本地对战
                </button>
              </section>
            </>
          ) : path === "/settings" ? (
            <>
              <h1>设置</h1>
              <section className="text-card settings-card">
                <label>
                  声音
                  <input
                    type="checkbox"
                    checked={!muted}
                    onChange={(event) => setMuted(!event.target.checked)}
                  />
                </label>
                <label>
                  浅色棋盘
                  <input
                    type="checkbox"
                    checked={light}
                    onChange={(event) => setLight(event.target.checked)}
                  />
                </label>
                <label>
                  拖放路障
                  <input
                    type="checkbox"
                    checked={rack}
                    onChange={(event) => setRack(event.target.checked)}
                  />
                </label>
                <h2>AI 接口</h2>
                <p>
                  支持浏览器策略、Web Worker 和本地 HTTP
                  服务。深度学习、强化学习与搜索型 AI 共用 209
                  个动作编号和合法动作掩码。
                </p>
                <button
                  className="secondary"
                  onClick={() => navigate("/computer")}
                >
                  选择本地策略
                </button>
              </section>
            </>
          ) : (
            <>
              <h1>
                {{
                  "/puzzle": "谜题",
                  "/tv": "观战",
                  "/leaderboard": "排行榜",
                  "/login": "登录",
                  "/signup": "注册",
                  "/premium": "Premium",
                }[path] || "页面未找到"}
              </h1>
              <section className="text-card">
                <p>
                  此入口需要相应的平台服务。当前本地平台提供双人对战、AI
                  接口和棋谱回放。
                </p>
                <button className="primary" onClick={() => navigate("/local")}>
                  本地对战
                </button>
                <button className="secondary" onClick={() => navigate("/")}>
                  回到大厅
                </button>
              </section>
            </>
          )}
        </main>
      )}
      {notice && (
        <div className="toast" role="status">
          {notice}
        </div>
      )}
      {dragPosition && (
        <div
          className={`drag-ghost ghost-${dragKind} player-${state.turn}`}
          style={{ left: dragPosition.x, top: dragPosition.y }}
        />
      )}
      {modal === "clip" && (
        <ClipDialog
          record={record}
          flipped={flipped}
          onClose={() => setModal(null)}
        />
      )}
      {modal === "records" && (
        <Modal title="本地对局记录" onClose={() => setModal(null)} wide>
          <p className="muted">
            对局有行动后自动保存于此设备。刷新本地对战会开启新棋局，可在这里恢复记录。
          </p>
          {!savedRecords.length && <p>暂无保存的对局。</p>}
          <div className="saved-record-list">
            {savedRecords.map((entry) => (
              <div className="saved-record" key={entry.id}>
                <div>
                  <strong>
                    {entry.mode === "computer"
                      ? `人机 · ${strategyNames[entry.strategy] || entry.strategy}`
                      : "本地双人"}
                  </strong>
                  <p className="muted">
                    {new Date(entry.updatedAt).toLocaleString()} ·{" "}
                    {entry.record.moves.length} 步 ·{" "}
                    {entry.record.result ? "已结束" : "进行中"}
                  </p>
                </div>
                <div>
                  <button onClick={() => restoreRecord(entry)}>恢复</button>{" "}
                  <button onClick={() => navigate(encodeShare(entry.record))}>
                    回放
                  </button>{" "}
                  <button onClick={() => deleteSaved(entry.id)}>删除</button>
                </div>
              </div>
            ))}
          </div>
        </Modal>
      )}
      {modal === "import" && (
        <Modal title="载入棋谱" onClose={() => setModal(null)}>
          <ImportForm onImport={(next) => navigate(encodeShare(next))} />
        </Modal>
      )}
      {modal === "moves" && (
        <Modal title="棋谱" onClose={() => setModal(null)}>
          <MoveList
            record={record}
            selection={selection}
            onMain={(index) => {
              if (isReplay) selectMain(index);
              setModal(null);
            }}
            onBranch={selectBranch}
            onDeleteBranch={deleteBranch}
            replaying
          />
        </Modal>
      )}
      {modal === "analyze" && (
        <Modal title="分析对局" onClose={() => setModal(null)}>
          <p className="muted">选择已接入的策略分析当前局面。</p>
          <div className="analysis-strategies">
            {Object.entries(strategyNames)
              .filter(([id]) => id !== "random")
              .map(([id, name]) => (
                <button
                  className="secondary"
                  key={id}
                  onClick={() => analyze(id)}
                >
                  {name}
                </button>
              ))}
          </div>
          <p className="integration-note">
            学习与搜索模型需先配置本地服务。最短路基线仅给出走棋建议。
          </p>
        </Modal>
      )}
      {modal?.type === "copy" && (
        <Modal title="复制内容" onClose={() => setModal(null)}>
          <textarea
            readOnly
            value={modal.text}
            rows={5}
            onFocus={(event) => event.target.select()}
          />
        </Modal>
      )}
    </>
  );
}

function parseCoordinates(move) {
  const square = move.length === 2 ? move : move.slice(1);
  return {
    row: (move.length === 2 ? 9 : 8) - Number(square[1]),
    col: square.charCodeAt(0) - 97,
  };
}
