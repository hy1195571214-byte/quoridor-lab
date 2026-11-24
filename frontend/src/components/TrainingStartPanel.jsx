import React, { useEffect, useState } from "react";
import { PlayIcon } from "@heroicons/react/24/outline";

export default function TrainingStartPanel({ request, t, runState, onStarted, onConfiguration }) {
  const [config, setConfig] = useState(null);
  const [name, setName] = useState("");
  const [epochs, setEpochs] = useState(40);
  const [batchSize, setBatchSize] = useState(256);
  const [architecture, setArchitecture] = useState("cnn32");
  const [seed, setSeed] = useState(0);
  const [device, setDevice] = useState("auto");
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");
  const [configError, setConfigError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    let timer;
    async function poll() {
      try {
        const result = await request("/training/config", controller.signal);
        if (controller.signal.aborted) return;
        setConfig(result);
        onConfiguration?.(result);
        setName(previous => previous || result.suggested_name);
        setConfigError("");
      } catch (reason) {
        if (reason.name !== "AbortError") { setConfig(null); onConfiguration?.(null); setConfigError(reason.message); }
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(poll, 5000);
      }
    }
    poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [runState, onConfiguration]);

  async function start(event) {
    event.preventDefault();
    if (!config || starting || config.active) return;
    setStarting(true);
    setError("");
    try {
      const result = await request("/training/start", undefined, {
        name, epochs: Number(epochs), batch_size: Number(batchSize), architecture,
        seed: Number(seed), device,
      });
      setConfig(previous => ({ ...previous, active: true }));
      setName("");
      onStarted(result.id);
    } catch (reason) {
      setError(reason.message);
    } finally {
      setStarting(false);
    }
  }

  const number = value => Number(value ?? 0).toLocaleString();
  return <section className="training-panel training-start-panel" aria-label={t("开始新训练")}>
    <div className="training-panel-heading"><h2>{t("开始新训练")}</h2>
      <span className="training-start-device">{t(!config ? "正在检测设备…" : config.devices.mps ? "Apple GPU 可用" : config.devices.cuda ? "GPU 可用" : "CPU 训练")}</span>
    </div>
    <p className="training-help">{config?.dataset.name ?? t("正在读取训练数据…")}</p>
    {config && <p className="training-data-summary">
      {t("训练集")} {number(config.dataset.splits.train)} · {t("验证集")} {number(config.dataset.splits.validation)} · {t("测试集")} {number(config.dataset.splits.test)}
    </p>}
    <form onSubmit={start}>
      <div className="training-start-fields">
        <label>{t("训练名称")}<input value={name} onChange={event => setName(event.target.value)} maxLength={60} required /></label>
        <label>{t("训练轮数")}<input type="number" min="1" max="200" value={epochs} onChange={event => setEpochs(event.target.value)} required /></label>
        <label>{t("每批样本数")}<select value={batchSize} onChange={event => setBatchSize(Number(event.target.value))}>
          {[64, 128, 256, 512].map(value => <option key={value} value={value}>{value}</option>)}
        </select></label>
      </div>
      <details className="training-more-settings"><summary>{t("更多设置")}</summary>
        <div className="training-start-fields">
          <label>{t("模型")}<select value={architecture} onChange={event => setArchitecture(event.target.value)}>
            <option value="cnn32">{t("卷积模型（CNN）")}</option><option value="mlp">{t("基础模型（MLP）")}</option>
          </select></label>
          <label>{t("随机种子")}<input type="number" min="0" max="2147483647" value={seed} onChange={event => setSeed(event.target.value)} required /></label>
          <label>{t("训练设备")}<select value={device} onChange={event => setDevice(event.target.value)}>
            <option value="auto">{t("自动选择")}</option>
            {config?.devices.mps && <option value="mps">Apple GPU</option>}
            {config?.devices.cuda && <option value="cuda">NVIDIA GPU</option>}
            <option value="cpu">CPU</option>
          </select></label>
        </div>
      </details>
      <div className="training-start-footer"><p className="training-help">{t(config?.active ? "已有训练正在运行。" : "点击“开始训练”后才会启动。")}</p>
        <button className="training-start-button" type="submit" disabled={!config || starting || config.active}>
          <PlayIcon />{t(starting ? "正在启动…" : "开始训练")}
        </button>
      </div>
      {error && <p className="training-error" role="alert">{error}</p>}
      {configError && <p className="training-error" role="alert">{configError}</p>}
      <p className="training-help">{t("每 100 批自动保存，异常中断后可从最近保存处继续。")}</p>
    </form>
  </section>;
}
