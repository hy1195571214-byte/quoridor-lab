import React, { useEffect, useMemo, useRef, useState } from "react";
import { ArrowTopRightOnSquareIcon, ChevronLeftIcon, ChevronRightIcon,
  PauseIcon, PlayIcon, ArrowPathIcon } from "@heroicons/react/24/outline";
import Board from "./Board.jsx";
import Phase1Training from "./Phase1Training.jsx";
import TrainingStartPanel from "./TrainingStartPanel.jsx";
import TrainingProgress from "./TrainingProgress.jsx";
import TrainingRoadmap from "./TrainingRoadmap.jsx";
import { replay } from "../game/rules.js";
import { chooseTrainingRun, trainingReplayUrl, validatePrediction } from "../game/training.js";
import { useLanguage } from "../i18n.jsx";
import "./TrainingDashboard.css";

const api = location.protocol === "file:" ? "http://127.0.0.1:8765" : "/api";

async function request(path, signal, body) {
  const response = await fetch(`${api}${path}`, {
    signal,
    ...(body && { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body) }),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : `HTTP ${response.status}`);
  return result;
}

function LossChart({ rows, label }) {
  const points = rows.filter(row => Number.isFinite(row.train_loss) && Number.isFinite(row.validation_loss));
  if (!points.length) return <p className="training-empty">{label}</p>;
  const maximum = Math.max(...points.flatMap(row => [row.train_loss, row.validation_loss]), 0.01);
  const polyline = key => points.map((row, index) =>
    `${32 + index * 536 / Math.max(1, points.length - 1)},${166 - row[key] / maximum * 140}`,
  ).join(" ");
  return <svg className="training-chart" viewBox="0 0 600 200" role="img" aria-label={label}>
    <line x1="32" y1="166" x2="568" y2="166" stroke="#3f3f46" />
    <text x="32" y="192" fill="#a1a1aa" fontSize="12">{points[0].epoch}</text>
    <text x="552" y="192" fill="#a1a1aa" fontSize="12">{points.at(-1).epoch}</text>
    <text x="4" y="27" fill="#a1a1aa" fontSize="11">{maximum.toFixed(1)}</text>
    <polyline points={polyline("train_loss")} fill="none" stroke="#93c5fd" strokeWidth="2" />
    <polyline points={polyline("validation_loss")} fill="none" stroke="#fca5a5" strokeWidth="2" />
    {points.length === 1 && <circle cx="32" cy={166 - points[0].validation_loss / maximum * 140} r="3" fill="#fca5a5" />}
  </svg>;
}

export default function TrainingDashboard({ navigate }) {
  const { t } = useLanguage();
  const [runs, setRuns] = useState([]);
  const [phase1Demonstrations, setPhase1Demonstrations] = useState([]);
  const [runId, setRunId] = useState(() => {
    const url = location.protocol === "file:"
      ? new URL(location.hash.slice(1) || "/", "http://localhost")
      : new URL(location.href);
    return url.searchParams.get("run") || "";
  });
  const [run, setRun] = useState(null);
  const [connectionError, setConnectionError] = useState("");
  const [resuming, setResuming] = useState(false);
  const [resumeError, setResumeError] = useState("");
  const [pausing, setPausing] = useState(false);
  const [trainingConfig, setTrainingConfig] = useState(null);
  const [episodes, setEpisodes] = useState([]);
  const [episodeOffset, setEpisodeOffset] = useState(0);
  const [episodeTotal, setEpisodeTotal] = useState(0);
  const [split, setSplit] = useState("train");
  const [episodeId, setEpisodeId] = useState("");
  const [episode, setEpisode] = useState(null);
  const [episodeError, setEpisodeError] = useState("");
  const [ply, setPly] = useState(0);
  const [follow, setFollow] = useState(true);
  const [prediction, setPrediction] = useState(null);
  const [predictionError, setPredictionError] = useState("");
  const [predicting, setPredicting] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const [view, setView] = useState("reference");
  const [predictionPly, setPredictionPly] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const predictionRequest = useRef(0);
  const manualRunSelection = useRef(Boolean(runId));
  const phase1Selected = runId.startsWith("phase1:");
  const revision = run?.id === runId ? run.checkpoint?.revision : null;

  useEffect(() => {
    const controller = new AbortController();
    let timer;
    async function poll() {
      try {
        const result = await request("/training/runs", controller.signal);
        if (controller.signal.aborted) return;
        setRuns(result.runs);
        setPhase1Demonstrations(result.phase1_demonstrations ?? []);
        setConnectionError("");
        setRunId(previous => chooseTrainingRun(result.runs, previous, manualRunSelection.current));
      } catch (error) {
        if (error.name !== "AbortError") setConnectionError(t("训练服务连接失败，请检查本地 API 服务。"));
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(poll, 2000);
      }
    }
    poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [t]);

  useEffect(() => {
    setRun(null);
    setResumeError("");
    if (!runId) return;
    const controller = new AbortController();
    let timer;
    async function poll() {
      try {
        const result = await request(`/training/runs/${encodeURIComponent(runId)}`, controller.signal);
        if (!controller.signal.aborted) setRun(result);
      } catch (error) {
        if (error.name !== "AbortError") setConnectionError(error.message);
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(poll, 1500);
      }
    }
    poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [runId]);

  useEffect(() => {
    if (phase1Selected || !runId) return;
    const controller = new AbortController();
    setEpisodeError("");
    request(`/training/episodes?split=${split}&offset=${episodeOffset}&limit=200&run=${encodeURIComponent(runId)}`, controller.signal).then(result => {
      if (controller.signal.aborted) return;
      setEpisodes(result.episodes);
      setEpisodeTotal(result.total ?? result.episodes.length);
      setEpisodeId(previous => result.episodes.some(item => item.episode_id === previous)
        ? previous : result.episodes[0]?.episode_id ?? "");
    }).catch(error => { if (error.name !== "AbortError") setEpisodeError(error.message); });
    return () => controller.abort();
  }, [split, episodeOffset, phase1Selected, runId]);

  useEffect(() => {
    if (!follow || playing || !run?.sample) return;
    setSplit("train");
    setEpisodeId(run.sample.episode_id);
    setPly(run.sample.ply);
  }, [follow, playing, episodes, run?.sample?.episode_id, run?.sample?.ply]);

  useEffect(() => {
    setEpisode(null);
    setPlaying(false);
    if (!episodeId || !runId || phase1Selected) return;
    const controller = new AbortController();
    request(`/training/episodes/${encodeURIComponent(episodeId)}?run=${encodeURIComponent(runId)}`, controller.signal).then(result => {
      if (controller.signal.aborted) return;
      replay(result.moves);
      setEpisode(result);
      setEpisodeError("");
      setPly(previous => Math.min(previous, result.moves.length));
    }).catch(error => { if (error.name !== "AbortError") setEpisodeError(error.message); });
    return () => controller.abort();
  }, [episodeId, phase1Selected, runId]);

  const moves = useMemo(() => episode?.moves.slice(0, ply) ?? [], [episode, ply]);
  const positionKey = moves.join(",");
  useEffect(() => {
    const sequence = ++predictionRequest.current;
    setPrediction(null);
    setPredictionError("");
    setPredictionPly(0);
    setView("reference");
    setPredicting(false);
    if (view === "prediction") setPlaying(false);
    if (!episode || episode.episode_id !== episodeId || !revision || replay(moves).winner || moves.length >= 256) return;
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      setPredicting(true);
      try {
        const result = await request(`/training/runs/${encodeURIComponent(runId)}/predict`, controller.signal,
          { moves, revision, rollout_plies: 8 });
        if (controller.signal.aborted || sequence !== predictionRequest.current) return;
        setPrediction(validatePrediction(result, { runId, revision, moves }));
      } catch (error) {
        if (error.name !== "AbortError" && sequence === predictionRequest.current) setPredictionError(error.message);
      } finally {
        if (sequence === predictionRequest.current) setPredicting(false);
      }
    }, 200);
    return () => { controller.abort(); clearTimeout(timer); };
  }, [runId, revision, episodeId, Boolean(episode), positionKey, refresh]);

  const playbackMaximum = view === "prediction"
    ? prediction?.predicted_moves.length ?? 0 : episode?.moves.length ?? 0;
  useEffect(() => {
    if (!playing) return;
    const timer = setTimeout(() => {
      const current = view === "prediction" ? predictionPly : ply;
      if (view === "prediction") setPredictionPly(Math.min(playbackMaximum, current + 1));
      else setPly(Math.min(playbackMaximum, current + 1));
      if (current + 1 >= playbackMaximum) setPlaying(false);
    }, 1000 / speed);
    return () => clearTimeout(timer);
  }, [playing, view, ply, predictionPly, speed, playbackMaximum]);

  const displayedMoves = view === "prediction" && prediction
    ? [...moves, ...prediction.predicted_moves.slice(0, predictionPly)] : moves;
  const state = useMemo(() => replay(displayedMoves), [displayedMoves.join(",")]);
  const metric = run?.metrics?.at(-1);
  const percentage = value => Number.isFinite(value) ? `${(100 * value).toFixed(1)}%` : "—";
  const status = { preparing: "准备数据", training: "训练中", completed: "已完成", failed: "训练失败", stopped: "已停止" };
  const interrupted = run && ["failed", "stopped"].includes(run.status);
  const running = run && ["preparing", "training"].includes(run.status);
  const accelerationAvailable = running && trainingConfig?.input_pipeline && run.input_pipeline !== trainingConfig.input_pipeline;

  useEffect(() => {
    if (!running) setPausing(false);
  }, [Boolean(running), runId]);

  async function pauseTraining() {
    if (!running || pausing) return;
    setPausing(true);
    setResumeError("");
    try {
      await request(`/training/runs/${encodeURIComponent(runId)}/pause`, undefined, {});
    } catch (error) {
      setResumeError(error.message);
      setPausing(false);
    }
  }

  async function resumeTraining() {
    if (!run?.checkpoint || resuming) return;
    setResuming(true);
    setResumeError("");
    try {
      await request(`/training/runs/${encodeURIComponent(runId)}/resume`, undefined, {});
      manualRunSelection.current = false;
      setFollow(true);
    } catch (error) {
      setResumeError(error.message);
    } finally {
      setResuming(false);
    }
  }

  function browse(nextPly) {
    setFollow(false);
    setPlaying(false);
    if (view === "prediction") setPredictionPly(Math.max(0, Math.min(prediction.predicted_moves.length, nextPly)));
    else setPly(Math.max(0, Math.min(episode?.moves.length ?? 0, nextPly)));
  }

  function selectRun(id) {
    manualRunSelection.current = true;
    setRunId(id);
    setEpisodeId("");
    setEpisodeOffset(0);
    setPly(0);
    setFollow(false);
  }

  const phase1Runs = runs.filter(item => item.kind === "phase1");
  const historicalRuns = runs.filter(item => item.kind !== "phase1");

  return <main className="training-page with-sidebar">
    <header className="training-heading">
      <div><a href="/" onClick={event => { event.preventDefault(); navigate("/"); }}>Quoridor Lab</a>
        <h1>{t("训练看板")}{run?.is_smoke && <small className="training-smoke-label">{t("试运行")}</small>}</h1></div>
      <label className="training-run-select">{t("训练运行")}
        <select value={runId} onChange={event => selectRun(event.target.value)} disabled={!runs.length}>
          {!runs.length && <option value="">{t("尚未开始训练")}</option>}
          {[["Phase 1 · 本地搜索", phase1Runs], ["历史棋谱训练", historicalRuns]].map(([label, items]) => items.length > 0 &&
            <optgroup key={label} label={t(label)}>{items.map(item => <option value={item.id} key={item.id}>
              {item.display_name ?? item.id}
            </option>)}</optgroup>)}
        </select>
      </label>
    </header>
    {connectionError && <p className="training-error" role="alert">{connectionError}</p>}
    <div className="training-dashboard-layout">
      <div className="training-dashboard-content">
      {phase1Selected ? <Phase1Training run={run?.id === runId ? run : null} navigate={navigate} t={t} /> : <>
    {run?.error && run.status !== "stopped" && <p className="training-error" role="alert">{
      run.failure_reason === "data_unavailable" || run.error.includes("[Errno 5]")
        ? t("训练因数据读取中断而停止，请确认数据盘已连接。") : run.error
    }</p>}
    {run?.checkpoint_error && <p className="training-error" role="alert">{t("最近一次保存未成功，继续时将使用上一次检查点。")}</p>}
    {interrupted && <div className="training-resume-panel">
      <p>{t(run.checkpoint ? "已保留最近保存的模型，点击后可继续训练。" : "这个运行没有保存模型，请在下方“新训练设置”中重新开始。")}</p>
      {run.checkpoint && <button onClick={resumeTraining} disabled={resuming || historicalRuns.some(item => ["preparing", "training"].includes(item.status))}>{t(resuming ? "正在恢复…" : "继续训练")}</button>}
    </div>}
    {resumeError && <p className="training-error" role="alert">{resumeError}</p>}
    {running && <div className="training-resume-panel">
      <p>{t(accelerationAvailable ? "加速版本已就绪。先暂停保存，再点击“继续训练”即可使用。" : "可暂停并保留最近安全保存的模型，之后手动继续。")}</p>
      <button onClick={pauseTraining} disabled={pausing || resuming}>{t(pausing ? "正在保存并暂停…" : "暂停训练")}</button>
    </div>}
    <TrainingProgress run={run} t={t} />
    <section className="training-stats" aria-label={t("训练进度")}>
      <div><span>{t("状态")}</span><strong>{run ? t(status[run.status] ?? run.status) : t("尚未开始训练")}</strong></div>
      <div><span>{t("训练设备")}</span><strong>{run?.device === "mps" ? "Apple GPU" : run?.device?.toUpperCase() ?? "—"}</strong></div>
      <div><span>{t("验证准确率")}</span><strong className={!metric ? "training-stat-pending" : undefined}>{metric ? percentage(metric.validation_action_accuracy) : t("首轮结束后显示")}</strong></div>
      <div><span>{t("已保存模型")}</span><strong className={!run?.checkpoint ? "training-stat-pending" : undefined}>{run?.checkpoint ? (
        run.checkpoint.epoch_complete === false ? t("第 {epoch} 轮 · 第 {batch} 批", { epoch: run.checkpoint.epoch, batch: run.checkpoint.batch })
          : t("第 {epoch} 轮", { epoch: run.checkpoint.epoch })
      ) : t(run?.checkpoint_every ? "完成首批后保存" : "此运行尚无保存模型")}</strong></div>
    </section>
    <details className="training-launch-details" open={!historicalRuns.some(item => ["preparing", "training"].includes(item.status))}>
      <summary>{t("新训练设置")}</summary>
      <TrainingStartPanel request={request} t={t} runState={runs.map(item => `${item.id}:${item.status}`).join(",")}
        onConfiguration={setTrainingConfig}
        onStarted={id => { manualRunSelection.current = false; setRunId(id); setFollow(true); }} />
    </details>
    <div className="training-workspace">
      <section className="training-replay-panel">
        <div className="training-panel-heading"><h2>{t("当前棋谱")}</h2>
          <label className="training-follow"><input type="checkbox" checked={follow} onChange={event => setFollow(event.target.checked)} />{t("跟随训练样本")}</label>
        </div>
        <div className="training-episode-select">
          <select aria-label={t("数据分区")} value={split} onChange={event => { setSplit(event.target.value); setEpisodeOffset(0); setFollow(false); setPly(0); }}>
            <option value="train">{t("训练集")}</option><option value="validation">{t("验证集")}</option>
          </select>
          <select aria-label={t("选择棋谱")} value={episodeId} onChange={event => { setEpisodeId(event.target.value); setFollow(false); setPly(0); }} disabled={!episodes.length}>
            {episodeId && !episodes.some(item => item.episode_id === episodeId) && <option value={episodeId}>{t("当前训练样本")}</option>}
            {episodes.map((item, index) => <option key={item.episode_id} value={item.episode_id}>
              {episodeOffset + index + 1} · {t(({ "expert-vs-expert": "专家对局", "quoridorgg-pgn": "公开对局", "quoridorcpp-pgn": "QuoridorCPP 棋谱" })[item.source] ?? item.source)} · {item.plies} {t("手")}
            </option>)}
          </select>
        </div>
        {episodeTotal > 200 && <div className="training-episode-pages">
          <button disabled={episodeOffset === 0} onClick={() => { setFollow(false); setEpisodeOffset(value => Math.max(0, value - 200)); }}>{t("上一页")}</button>
          <span>{episodeOffset + 1}–{Math.min(episodeOffset + 200, episodeTotal)} / {episodeTotal.toLocaleString()}</span>
          <button disabled={episodeOffset + 200 >= episodeTotal} onClick={() => { setFollow(false); setEpisodeOffset(value => value + 200); }}>{t("下一页")}</button>
        </div>}
        {episodeError && <p className="training-error" role="alert">{episodeError}</p>}
        <div className="training-board-context"><span>{t(view === "prediction" ? "模型预测" : "原始棋谱")}</span>
          <span>{state.winner ? t(state.winner === 1 ? "红方获胜" : "蓝方获胜") : t(state.turn === 1 ? "红方行动" : "蓝方行动")}</span></div>
        <div className="training-board"><Board state={state} disabled analysis /></div>
        <div className="training-playback">
          <button aria-label={t("上一步")} disabled={!episode || (view === "prediction" ? predictionPly === 0 : ply === 0)} onClick={() => browse((view === "prediction" ? predictionPly : ply) - 1)}><ChevronLeftIcon /></button>
          <button onClick={() => { setFollow(false); setView("reference"); setPlaying(view === "reference" ? !playing : true); if (ply >= (episode?.plies ?? 0)) setPly(0); }} disabled={!episode}>
            {playing && view === "reference" ? <PauseIcon /> : <PlayIcon />}{t("播放棋谱")}</button>
          <button onClick={() => { setFollow(false); setView("prediction"); setPlaying(view === "prediction" ? !playing : true); if (predictionPly >= (prediction?.predicted_moves.length ?? 0)) setPredictionPly(0); }} disabled={!prediction?.predicted_moves.length}>
            {playing && view === "prediction" ? <PauseIcon /> : <PlayIcon />}{t("播放模型预测")}</button>
          <button aria-label={t("下一步")} disabled={!episode || (view === "prediction" ? predictionPly >= (prediction?.predicted_moves.length ?? 0) : ply >= episode.plies)} onClick={() => browse((view === "prediction" ? predictionPly : ply) + 1)}><ChevronRightIcon /></button>
          <select aria-label={t("回放速度")} value={speed} onChange={event => setSpeed(Number(event.target.value))}>{[0.5, 1, 2, 3].map(value => <option key={value} value={value}>{value}×</option>)}</select>
        </div>
        <label className="training-timeline">{view === "prediction" ? `${t("预测续走")} ${predictionPly} / ${prediction?.predicted_moves.length ?? 0}` : `${t("已走")} ${ply} / ${episode?.plies ?? 0}`}
          <input aria-label={t("回放步数")} type="range" min="0" max={view === "prediction" ? prediction?.predicted_moves.length ?? 0 : episode?.plies ?? 0} value={view === "prediction" ? predictionPly : ply} onChange={event => browse(Number(event.target.value))} />
        </label>
        <button className="training-replay-link" disabled={!episode} onClick={() => navigate(trainingReplayUrl(episode.moves, ply, view === "prediction" ? prediction : null, predictionPly))}>
          <ArrowTopRightOnSquareIcon />{t(view === "prediction" ? "在完整回放中打开预测分支" : "在完整回放中打开棋谱")}</button>
      </section>
      <aside className="training-inspector">
        <section className="training-panel">
          <div className="training-panel-heading"><h2>{t("下一手预测")}</h2><button aria-label={t("更新预测")} onClick={() => setRefresh(value => value + 1)} disabled={!revision || predicting}><ArrowPathIcon /></button></div>
          <p className="training-reference-move">{t("棋谱下一手")}<strong>{episode?.moves[ply] ?? "—"}</strong></p>
          {prediction && <><p className="training-checkpoint-label">{prediction.architecture} · Epoch {prediction.checkpoint_epoch} · {t("步")} {prediction.checkpoint_step}</p>
            <ol className="training-candidates">{prediction.top_actions.map(item => <li key={item.action_id}><strong>{item.notation}</strong><span>{percentage(item.probability)}</span><div style={{ width: `${100 * item.probability}%` }} /></li>)}</ol>
            <p className="training-help">{t("概率表示模型对合法走法的偏好。续走由同一检查点交替预测双方行动。")}</p></>}
          {!prediction && !predictionError && <p className="training-empty">{t(predicting ? "正在计算预测…" : state.winner ? "当前棋局已结束" : !revision ? "保存首个检查点后，这里会显示模型预测。" : "选择局面查看预测。")}</p>}
          {predictionError && <p className="training-error" role="alert">{predictionError}</p>}
        </section>
        <section className="training-panel"><h2>{t("损失曲线")}</h2><div className="training-legend"><span>{t("训练")}</span><span>{t("验证")}</span></div><LossChart rows={run?.metrics ?? []} label={t("完成一个训练轮次后显示损失曲线。")} /></section>
        <section className="training-panel training-run-info"><h2>{t("运行信息")}</h2><dl>
          <div><dt>{t("架构")}</dt><dd>{run?.architecture ?? "—"}</dd></div>
          <div><dt>{t("随机种子")}</dt><dd>{run?.seed ?? "—"}</dd></div>
          <div><dt>{t("学习率")}</dt><dd>{metric?.learning_rate ?? run?.learning_rate ?? "—"}</dd></div>
          <div><dt>{t("设备")}</dt><dd>{run?.device ?? "—"}</dd></div>
        </dl></section>
      </aside>
    </div>
      </>}
      </div>
      <TrainingRoadmap runs={runs} demonstrations={phase1Demonstrations} onSelectRun={selectRun} t={t} />
    </div>
  </main>;
}
