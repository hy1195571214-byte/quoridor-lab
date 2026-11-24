import React from "react";
import { CheckCircleIcon, ClockIcon, MinusCircleIcon } from "@heroicons/react/24/outline";

function Stage({ title, tasks, t }) {
  const completed = tasks.filter(task => task.state === "done").length;
  return <section className="training-stage" aria-label={title}>
    <div className="training-stage-heading"><h3>{title}</h3>
      <span>{t("{done}/{total} 项完成", { done: completed, total: tasks.length })}</span>
    </div>
    <progress value={completed} max={tasks.length} aria-label={title} />
    <ul>{tasks.map(({ label, detail, state = "pending" }) => {
      const Icon = { done: CheckCircleIcon, active: ClockIcon, pending: MinusCircleIcon }[state];
      return <li key={label} className={`training-task is-${state}`}>
        <Icon aria-hidden="true" />
        <div><strong>{label}</strong>{detail && <small>{detail}</small>}</div>
        <span className="training-task-status">{t({done: "已完成", active: "进行中", pending: "待开始"}[state])}</span>
      </li>;
    })}</ul>
  </section>;
}

export default function TrainingRoadmap({ runs, demonstrations, onSelectRun, t }) {
  const groups = new Map();
  for (const run of runs.filter(item => item.kind === "phase1" && !item.level1)) {
    const group = groups.get(run.batch) ?? [];
    group.push(run);
    groups.set(run.batch, group);
  }
  const both = (group, predicate) => ["mlp", "cnn32"].every(architecture =>
    group.some(run => run.architecture === architecture && predicate(run)));
  const ready = run => run.status === "completed" && run.ppo_done > 0 && run.checkpoint_saved;
  const groupsByBatch = Array.from(groups.values());
  const pilot = groupsByBatch.find(group => both(group, ready));
  const warm = groupsByBatch.some(group => both(group, run => run.warm_start_saved));
  const active = runs.some(run => run.kind === "phase1" && !run.level1 && run.status === "training");
  const level1 = runs.filter(run => run.level1);
  const demo = demonstrations.find(item => item.batch === pilot?.[0]?.batch) ?? demonstrations[0];
  const pilotRuns = pilot ? ["mlp", "cnn32"].map(architecture =>
    pilot.find(run => run.architecture === architecture && ready(run))) : [];
  const stage1 = [
    { label: t("生成本地搜索示范"), state: demo ? "done" : "pending",
      detail: demo && t("{games} 局 · 训练 {train} · 验证 {validation}", demo) },
    { label: t("MLP / CNN32 模仿暖启动"), state: warm ? "done" : active ? "active" : "pending" },
    { label: t("MLP / CNN32 PPO 试跑"), state: pilot ? "done" : active ? "active" : "pending",
      detail: pilotRuns.length > 0 && <span className="training-task-links">
        {pilotRuns.map(run => <button key={run.id} type="button" onClick={() => onSelectRun(run.id)}>
          {run.architecture.toUpperCase()} · {run.ppo_done}/{run.ppo_total ?? run.ppo_done}
        </button>)}
      </span> },
    { label: t("正式训练与多种子比较"), state: level1.some(run => run.status === "training") ? "active" : "pending",
      detail: level1.length ? <span className="training-task-links">
        {level1.map(run => <button key={run.id} type="button" onClick={() => onSelectRun(run.id)}>
          {run.architecture.toUpperCase()} · {run.ppo_done}/{run.ppo_total ?? "—"}
        </button>)}
      </span> : t("两种架构各 3 个独立种子") },
    { label: t("固定 Arena 与 Elo 评估") },
  ];
  return <aside className="training-panel training-roadmap" aria-label={t("训练任务清单")}>
    <h2>{t("训练任务清单")}</h2>
    <p className="training-roadmap-intro">{t("阶段完成进度")}</p>
    <Stage title={t("Phase 1 · 搜索示范与 PPO")} tasks={stage1} t={t} />
    <Stage title={t("Phase 2 · MCTS 推理")} t={t} tasks={[
      { label: t("冻结 PPO 权重并接入 MCTS") },
      { label: t("比较 Elo 增益与决策耗时") },
    ]} />
    <Stage title={t("Phase 3 · AlphaZero 自我对弈")} t={t} tasks={[
      { label: t("生成搜索自我对弈数据") },
      { label: t("训练两种模型并评估 Elo") },
    ]} />
  </aside>;
}
