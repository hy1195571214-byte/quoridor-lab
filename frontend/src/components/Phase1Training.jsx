import React from "react";
import { ArrowTopRightOnSquareIcon } from "@heroicons/react/24/outline";
import { trainingReplayUrl } from "../game/training.js";

const format = value => Number.isFinite(value) ? value.toLocaleString() : "—";
const fixed = value => Number.isFinite(value) ? value.toFixed(4) : "—";
const fraction = (done, total) => total == null ? format(done) : `${format(done)} / ${format(total)}`;
const percent = value => Number.isFinite(value) ? `${(100 * value).toFixed(1)}%` : "—";
const opponentLabel = { random: "Random", bfs: "BFS", level1: "Level 1", level1_pool: "Level 1 70% · BFS 20% · 随机化搜索 10%" };

export default function Phase1Training({ run, navigate, t }) {
  if (!run) return <section className="training-panel" aria-live="polite">{t("正在读取训练记录…")}</section>;
  const status = { completed: "试跑已完成", training: "训练中", failed: "训练失败", stopped: "已停止", partial: "已有训练记录" };
  const latestImitation = run.imitation?.at(-1);
  const evaluation = run.evaluation;
  const requiredPasses = run.required_passes ?? evaluation?.required_passes ?? 3;
  const evaluations = evaluation?.formal_eval ? Object.entries(evaluation.formal_eval) : evaluation ? [
    [evaluation.opponent, evaluation.development_eval], ["shortest", evaluation.shortest_eval],
  ].filter(([, result]) => result) : [];
  const currentStatus = run.level1 && run.status === "completed"
    ? (run.qualified ? "Level 1 已达标" : "Level 1 结束 · 未达标")
    : run.level1 && run.status === "training" && run.activity === "evaluating" ? "正在正式评估" : status[run.status] ?? run.status;
  return <>
    <section className="training-stats" aria-label={t("训练进度")}>
      <div><span>{t("状态")}</span><strong>{t(currentStatus)}</strong></div>
      <div><span>{t(run.level1 ? "继承的试跑轮数" : "模仿暖启动")}</span><strong>{run.level1 ? format(run.pilot_updates) : fraction(run.imitation_done, run.imitation_total)}</strong></div>
      <div><span>{t("PPO 更新")}</span><strong>{fraction(run.ppo_done, run.ppo_total)}</strong></div>
      <div><span>{t("学习方行动")}</span><strong>{format(run.steps)}</strong></div>
    </section>
    <section className="training-panel training-phase-panel">
      <div className="training-panel-heading"><h2>{t("运行信息")}</h2><span className="training-phase-meta">
        {run.architecture?.toUpperCase()} · seed {run.seed ?? "—"} · {run.device === "mps" ? "Apple GPU" : run.device?.toUpperCase() ?? "—"}
      </span></div>
      {run.error && <p className="training-error" role="alert">{run.error}</p>}
      <dl className="training-phase-facts">
        <div><dt>{t("当前对手")}</dt><dd>{t(opponentLabel[run.opponent] ?? run.opponent ?? "—")}</dd></div>
        {run.level1 ? <div><dt>{t("训练效率 · 动作/秒")}</dt><dd>{format(run.actions_per_second)}</dd></div>
          : <div><dt>{t("暖启动验证准确率")}</dt><dd>{percent(latestImitation?.val_accuracy)}</dd></div>}
        {run.level1 && run.pbrs_scale > 0 && <div><dt>{t("BFS 势函数系数")}</dt><dd>{format(run.pbrs_scale)}</dd></div>}
        <div><dt>{t("已保存模型")}</dt><dd>{[run.warm_start_saved && "warm_start.pt", run.checkpoint_saved && "latest.pt", run.best_saved && "best.pt"].filter(Boolean).join(" · ") || "—"}</dd></div>
      </dl>
    </section>
    <section className="training-panel training-phase-panel">
      <div className="training-panel-heading"><h2>{t(run.level1 ? "Level 1 正式评估" : "开发评估")}</h2>
        {evaluation && <span className="training-phase-meta">{t("第 {update} 次 PPO 更新", { update: evaluation.update })}</span>}
      </div>
      {evaluation?.formal_eval && <dl className="training-phase-facts">
        <div><dt>{t("本机 Arena Elo")}</dt><dd>{format(evaluation.elo)}</dd></div>
        <div><dt>{t(requiredPasses === 1 ? "正式评估达标" : "连续达标")}</dt>
          <dd>{fraction(run.qualified ? requiredPasses : Math.min(evaluation.qualifying_streak, requiredPasses), requiredPasses)}</dd></div>
      </dl>}
      {evaluations.length ? <div className="training-table-scroll"><table className="training-metrics-table">
        <thead><tr>{(run.level1 ? ["对手", "胜率", "胜", "负", "截断", "非法"] : ["对手", "胜", "负", "截断"]).map(label => <th scope="col" key={label}>{t(label)}</th>)}</tr></thead>
        <tbody>{evaluations.map(([opponent, result]) => <tr key={opponent}>
          <th scope="row">{opponentLabel[opponent] ?? opponent}</th>{run.level1 && <td>{percent(result.win_rate)}</td>}
          <td>{format(result.wins)}</td><td>{format(result.losses)}</td><td>{format(result.truncated)}</td>{run.level1 && <td>{format(result.illegal)}</td>}
        </tr>)}</tbody>
      </table></div> : <p className="training-empty">{t(run.level1 ? "每 25 轮评估，每个对手 200 局" : "等待开发评估记录")}</p>}
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
