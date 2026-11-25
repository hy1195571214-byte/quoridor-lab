import React from "react";
import { ArrowTopRightOnSquareIcon } from "@heroicons/react/24/outline";
import { trainingReplayUrl } from "../game/training.js";

const format = value => Number.isFinite(value) ? value.toLocaleString() : "—";
const fixed = value => Number.isFinite(value) ? value.toFixed(4) : "—";
const fraction = (done, total) => total == null ? format(done) : `${format(done)} / ${format(total)}`;
const percent = value => Number.isFinite(value) ? `${(100 * value).toFixed(1)}%` : "—";
const opponentLabel = { random: "Random", bfs: "BFS", level1: "Level 1", level1_pool: "Level 1 70% · BFS 20% · 随机化搜索 10%" };
const poolLabel = { ...opponentLabel, randomized_search: "随机化搜索", level2: "Level 2", level3: "Level 3",
  level3_practice: "三层搜索与 BFS 练习", history: "历史模型" };
const rolloutOutcomes = ["wins", "losses", "truncated"];
const rolloutOpenings = { standard: "普通开局", curriculum: "专项开局" };

function recentLevel3Rollouts(ppo, unit) {
  let trainingGames = 0;
  const scores = { standard: [0, 0, 0], curriculum: [0, 0, 0] };
  for (let i = ppo.length - 1; i >= 0 && trainingGames < unit; i--) {
    const row = ppo[i];
    const counts = row.by_opponent?.level3 ?? {};
    if (!Object.keys(counts).some(key => /_(standard|curriculum)$/.test(key))) continue;
    trainingGames += rolloutOutcomes.reduce((sum, outcome) => sum + (row.rollout_results?.[outcome] ?? 0), 0);
    for (const [source, totals] of Object.entries(scores)) {
      rolloutOutcomes.forEach((outcome, index) => { totals[index] += counts[`${outcome}_${source}`] ?? 0; });
    }
  }
  return { trainingGames, scores };
}

function AdjustedQualification({ evidence, t }) {
  const { original, adjusted } = evidence;
  const rounds = [
    [t("正式复评"), evidence.formal_eval],
    [t("独立开局复测"), evidence.independent_gate_bank_1],
  ];
  return <section className="training-panel training-phase-panel">
    <div className="training-panel-heading">
      <h2>{t("调整后通过")}</h2>
      <span className="training-phase-meta">{t("第 {update} 次 PPO 更新", { update: evidence.update })}</span>
    </div>
    <dl className="training-phase-facts">
      <div><dt>{t("Random 原门槛")}</dt>
        <dd>{t("胜率 ≥ {win}，截断率 < {cap}", {
          win: percent(original.wins.random), cap: percent(original.truncation_limit),
        })}</dd></div>
      <div><dt>{t("Random 批准门槛")}</dt>
        <dd>{t("胜率 ≥ {win}，截断率 < {cap}", {
          win: percent(adjusted.wins.random), cap: percent(adjusted.truncation_limit),
        })}</dd></div>
    </dl>
    <div className="training-table-scroll"><table className="training-metrics-table">
      <thead><tr>{["验收记录", "对手", "胜率", "胜", "负", "截断", "非法"].map(label =>
        <th key={label} scope="col">{t(label)}</th>)}</tr></thead>
      <tbody>{rounds.flatMap(([label, scores]) => Object.entries(scores ?? {}).map(([name, result]) =>
        <tr key={`${label}-${name}`}>
          <th scope="row">{label}</th>
          <td>{t(poolLabel[name] ?? name)}</td><td>{percent(result.win_rate)}</td>
          <td>{format(result.wins)}</td><td>{format(result.losses)}</td>
          <td>{percent(result.truncation_rate)} ({format(result.truncated)}/{format(result.games)})</td>
          <td>{format(result.illegal)}</td>
        </tr>
      ))}</tbody>
    </table></div>
  </section>;
}

export default function Phase1Training({ run, course, navigate, t }) {
  if (!run) return <section className="training-panel" aria-live="polite">{t("正在读取训练记录…")}</section>;
  const status = { completed: "试跑已完成", training: "训练中", failed: "训练失败", stopped: "已停止", partial: "已有训练记录" };
  const latestImitation = run.imitation?.at(-1);
  const adjustments = (run.ppo ?? []).flatMap(row => row.tuning ? [row.tuning] : []);
  const latestAdjustment = adjustments.at(-1);
  const evaluation = run.evaluation;
  const stage = course?.stages?.find(item => item.runs?.[run.architecture] === run.id);
  const adjustment = stage?.promotion?.[run.architecture]?.adjustment;
  const qualification = adjustment?.kind === "adjusted" && adjustment.adjusted_passed ? adjustment : null;
  const showSeats = (run.phase1_level ?? 1) >= 2;
  const requiredPasses = run.required_passes ?? evaluation?.required_passes ?? 3;
  const opponentPool = run.ppo?.at(-1)?.training_opponent_pool ?? run.opponent_pool;
  const pool = opponentPool && Object.entries(opponentPool)
    .map(([name, ratio]) => `${t(poolLabel[name] ?? name)} ${Math.round(100 * ratio)}%`).join(" · ");
  const rollout = run.phase1_level === 3 && recentLevel3Rollouts(run.ppo ?? [], run.game_units?.size ?? 500);
  const evaluations = evaluation?.formal_eval ? Object.entries(evaluation.formal_eval) : evaluation ? [
    [evaluation.opponent, evaluation.development_eval], ["shortest", evaluation.shortest_eval],
  ].filter(([, result]) => result) : [];
  const currentStatus = run.level1 && run.status === "completed"
    ? (run.qualified ? (run.gate_adjustment_reason ? "调整后通过" : "已达标") : run.comparison_arm ? "对照实验已完成" : "Level 1 结束 · 未达标")
    : run.status === "stopped" && stage?.level === course?.active_level && course?.status === "paused" ? "已暂停"
    : run.level1 && run.status === "training" && run.activity === "evaluating" ? "正在正式评估" : status[run.status] ?? run.status;
  return <>
    <section className="training-stats" aria-label={t("训练进度")}>
      <div><span>{t("状态")}</span><strong>{t(currentStatus)}</strong></div>
      <div><span>{t(run.comparison_arm || run.stage_source_update != null ? "起点模型的 PPO 更新数" : run.level1 ? "继承的试跑轮数" : "模仿暖启动")}</span>
        <strong>{run.level1 ? format(run.stage_source_update ?? run.comparison_source_update ?? run.pilot_updates) : fraction(run.imitation_done, run.imitation_total)}</strong></div>
      <div><span>{t("PPO 更新")}</span><strong>{fraction(run.ppo_done, run.ppo_total)}</strong></div>
      <div><span>{t("学习方行动")}</span><strong>{format(run.steps)}</strong></div>
    </section>
    <section className="training-panel training-phase-panel">
      <div className="training-panel-heading"><h2>{t("运行信息")}</h2><span className="training-phase-meta">
        {run.architecture?.toUpperCase()} · seed {run.seed ?? "—"} · {run.device === "mps" ? "Apple GPU" : run.device?.toUpperCase() ?? "—"}
      </span></div>
      {run.error && <p className="training-error" role="alert">{run.error}</p>}
      {run.gate_adjustment_reason && <p className="training-error">{run.gate_adjustment_reason}</p>}
      <dl className="training-phase-facts">
        <div><dt>{t("当前对手")}</dt><dd>{pool || t(opponentLabel[run.opponent] ?? run.opponent ?? "—")}</dd></div>
        {run.level1 ? <div><dt>{t("训练效率 · 动作/秒")}</dt><dd>{format(run.actions_per_second)}</dd></div>
          : <div><dt>{t("暖启动验证准确率")}</dt><dd>{percent(latestImitation?.val_accuracy)}</dd></div>}
        {run.level1 && run.pbrs_scale > 0 && <div><dt>{t("BFS 势函数系数")}</dt><dd>{format(run.pbrs_scale)}</dd></div>}
        {run.game_units && <div><dt>{t(run.qualified ? "续训已结算对局" : "续训对局 / 下次评估")}</dt>
          <dd>{fraction(run.game_units.completed, run.qualified ? null : run.game_units.next_evaluation)}</dd></div>}
        {latestAdjustment && <div><dt>{t("最近训练调整")}</dt>
          <dd>{t("第 {update} 次 PPO 更新", { update: latestAdjustment.from_update })} · {latestAdjustment.reason}</dd></div>}
        <div><dt>{t("已保存模型")}</dt><dd>{[run.warm_start_saved && "warm_start.pt", run.checkpoint_saved && "latest.pt", run.best_saved && "best.pt"].filter(Boolean).join(" · ") || "—"}</dd></div>
      </dl>
      {adjustments.length > 1 && <details className="training-phase-replays training-phase-adjustments">
        <summary>{t("查看 {count} 条训练调整", { count: adjustments.length })}</summary>
        <ul>{[...adjustments].reverse().map(change => <li key={change.from_update}>
          <strong>{t("第 {update} 次 PPO 更新", { update: change.from_update })}</strong>
          <span>{change.reason}</span>
        </li>)}</ul>
      </details>}
      {rollout && rollout.trainingGames > 0 && <div className="training-table-scroll">
        <table className="training-metrics-table">
          <caption>{t("最近 {games} 盘训练中的完整三层搜索对局", { games: format(rollout.trainingGames) })}</caption>
          <thead><tr>{["开局来源", "胜", "负", "截断"].map(label => <th scope="col" key={label}>{t(label)}</th>)}</tr></thead>
          <tbody>{Object.entries(rollout.scores).map(([source, scores]) => <tr key={source}>
            <th scope="row">{t(rolloutOpenings[source])}</th>
            {scores.map((value, index) => <td key={rolloutOutcomes[index]}>{format(value)}</td>)}
          </tr>)}</tbody>
        </table>
      </div>}
    </section>
    {qualification && <AdjustedQualification evidence={qualification} t={t} />}
    <section className="training-panel training-phase-panel">
      <div className="training-panel-heading"><h2>{qualification ? t("最近一次正式评估")
        : run.level1 ? t("Level {level} 正式评估", { level: run.phase1_level ?? 1 }) : t("开发评估")}</h2>
        {evaluation && <span className="training-phase-meta">{t("第 {update} 次 PPO 更新", { update: evaluation.update })}</span>}
      </div>
      {evaluation?.formal_eval && <dl className="training-phase-facts">
        <div><dt>{t("本机 Arena Elo")}</dt><dd>{format(evaluation.elo)}</dd></div>
        <div><dt>{t(requiredPasses === 1 ? "正式评估达标" : "连续达标")}</dt>
          <dd>{fraction(run.qualified && !qualification ? requiredPasses
            : Math.min(evaluation.qualifying_streak, requiredPasses), requiredPasses)}</dd></div>
      </dl>}
      {evaluations.length ? <div className="training-table-scroll"><table className="training-metrics-table">
        <thead><tr>{(run.level1 ? ["对手", "胜率", "胜", ...(showSeats ? ["红方胜", "蓝方胜"] : []), "负", "截断", "非法"]
          : ["对手", "胜", "负", "截断"]).map(label => <th scope="col" key={label}>{t(label)}</th>)}</tr></thead>
        <tbody>{evaluations.map(([opponent, result]) => <tr key={opponent}>
          <th scope="row">{t(poolLabel[opponent] ?? opponent)}</th>{run.level1 && <td>{percent(result.win_rate)}</td>}
          <td>{format(result.wins)}</td>{showSeats && <><td>{format(result.wins_red)}</td><td>{format(result.wins_blue)}</td></>}
          <td>{format(result.losses)}</td><td>{format(result.truncated)}</td>{run.level1 && <td>{format(result.illegal)}</td>}
        </tr>)}</tbody>
      </table></div> : <p className="training-empty">{run.game_units
        ? t("每 {games} 盘训练对局评估一次，每个对手 200 局", { games: run.game_units.size })
        : t(run.level1 ? "每 25 轮评估，每个对手 200 局" : "等待开发评估记录")}</p>}
      {run.games?.length > 0 && <details className="training-phase-replays">
        <summary>{t("查看 {games} 局对战棋谱", { games: run.games.length })}</summary>
        <ul>{run.games.map((game, index) => <li key={`${game.opponent}-${index}`}>
          <button type="button" onClick={() => navigate(trainingReplayUrl(game.moves, 0))}>
            <span>{index + 1} · {game.opponent} · {t(game.learner_player === 1 ? "模型执红" : "模型执蓝")} · {t(({ wins: "胜", losses: "负", truncated: "截断" })[game.result] ?? game.result)}</span>
            <ArrowTopRightOnSquareIcon aria-hidden="true" />
          </button>
        </li>)}</ul>
      </details>}
    </section>
    <section className="training-panel training-phase-panel"><h2>{t("PPO 更新记录")}</h2>
      {run.ppo?.length ? <div className="training-table-scroll"><table className="training-metrics-table">
        <thead><tr>{["更新", "学习方行动", "策略损失", "价值损失", ...(run.level1 ? ["KL", "Entropy"] : []), "胜 / 负 / 截断"].map(label => <th scope="col" key={label}>{t(label)}</th>)}</tr></thead>
        <tbody>{[...run.ppo].reverse().slice(0, run.level1 ? 25 : undefined).map(row => <tr key={row.update}>
          <th scope="row">{row.update}</th><td>{format(row.steps)}</td><td>{fixed(row.policy_loss)}</td><td>{fixed(row.value_loss)}</td>
          {run.level1 && <><td>{fixed(row.approx_kl)}</td><td>{fixed(row.entropy)}</td></>}
          <td>{[row.rollout_results?.wins, row.rollout_results?.losses, row.rollout_results?.truncated].map(format).join(" / ")}</td>
        </tr>)}</tbody>
      </table></div> : <p className="training-empty">{t("等待首条 PPO 更新记录")}</p>}
    </section>
  </>;
}
