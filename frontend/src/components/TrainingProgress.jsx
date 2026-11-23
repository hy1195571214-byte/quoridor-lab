import React, { useEffect, useState } from "react";
import { trainingBatchProgress } from "../game/training.js";

function elapsedLabel(seconds, t) {
  const value = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor(value % 3600 / 60);
  return hours ? t("{hours}小时 {minutes}分钟", { hours, minutes })
    : t("{minutes}分 {seconds}秒", { minutes, seconds: value % 60 });
}

export default function TrainingProgress({ run, t }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);
  if (!run) return null;
  const running = ["preparing", "training"].includes(run.status);
  const progress = trainingBatchProgress(run);
  const updated = Date.parse(run.updated_at);
  const age = Number.isFinite(updated) ? Math.max(0, (now - updated) / 1000) : 0;
  const elapsed = (Number(run.total_seconds ?? run.session_seconds) || 0) + (running ? age : 0);
  const number = value => value.toLocaleString();
  const percent = progress.percentage.toFixed(progress.percentage < 1 ? 2 : 1);
  return <section className="training-panel training-live-progress" aria-label={t("实时训练进度")}>
    <div className="training-panel-heading"><div>
      <h2>{t("本轮训练进度")}</h2>
      <p className="training-live-name">{run.display_name ?? run.id}</p>
    </div><strong className="training-live-percent">{percent}%</strong></div>
    <progress aria-label={t("本轮训练进度")} max="100" value={progress.percentage} />
    <div className="training-live-facts">
      <div><span>{t("训练轮次")}</span><strong>{run.epoch ?? 0} / {run.total_epochs ?? "—"}</strong></div>
      <div><span>{t("已完成批次")}</span><strong>{number(progress.batch)} / {number(progress.batches)}</strong></div>
      <div><span>{t("当前损失")}</span><strong>{Number.isFinite(run.batch_loss) ? run.batch_loss.toFixed(4) : "—"}</strong></div>
      <div><span>{t("已用时间")}</span><strong>{elapsedLabel(elapsed, t)}</strong></div>
    </div>
    <div className="training-live-footer">
      <span>{t("本轮已学习 {done} / {total} 个样本", { done: number(progress.processedSamples), total: number(progress.totalSamples) })}</span>
      <span>{t("最后更新：{seconds}秒前", { seconds: Math.floor(age) })}</span>
    </div>
    {running && <p className="training-help">{t("数据按批准备，进度会分段前进。每 {batches} 批自动保存；验证准确率在整轮结束后更新。", { batches: run.checkpoint_every ?? 100 })}</p>}
  </section>;
}
