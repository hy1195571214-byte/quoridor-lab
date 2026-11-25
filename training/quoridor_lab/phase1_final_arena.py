"""Run the held-out Phase 1 Arena and export the course review."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import fcntl
from itertools import combinations
import gzip
import json
import math
from pathlib import Path
from shutil import copyfile
from statistics import median

import torch

from .dataset_import import reflection_group
from .level1_arena import DEEP_SEARCH, MAX_PLIES, POLICIES, estimate_elo, evaluation_games, play_evaluation, score_games
from .phase1_curriculum import LEVEL_GATES, Opponents
from .phase1_monitor import read_json, read_rows
from .rules import State
from .train_level1 import sha256
from .training_monitor import atomic_json


ANCHORS = ("bfs", "random", "level1", "level2", "level3")
ARENA_SEED = 4_100_000
arena = resolve = output = None
anchor_cache = None


def prepare_worker(frozen, models, folder, cached_anchors=None):
    global arena, resolve, output, anchor_cache
    torch.set_num_threads(1)
    arena = frozen
    resolve = Opponents(models)
    output = folder
    anchor_cache = cached_anchors


def verify_match_replays(rows, openings, opponent):
    """Recover the score from complete, seat-swapped replays of the frozen openings."""
    expected = {(ARENA_SEED + index, player): opening
                for index, opening in enumerate(openings) for player in (1, 2)}
    for row in rows:
        opening = expected.pop((row["seed"], row["learner_player"]), None)
        moves = tuple(row["moves"])
        if (opening is None or row["opponent"] != opponent or
            tuple(row["opening"]) != opening or moves[:len(opening)] != opening or len(moves) > MAX_PLIES):
            raise ValueError("Game replay does not match its Arena opening, opponent, or seat")
        state = State.from_moves(moves)
        if state.winner is not None:
            result = "wins" if state.winner == row["learner_player"] else "losses"
        elif len(moves) == MAX_PLIES:
            result = "truncated"
        elif row["result"] == "illegal" and state.turn == row["learner_player"]:
            result = "illegal"
        else:
            raise ValueError("Arena replay ended before a recorded terminal condition")
        if row["result"] != result:
            raise ValueError("Arena replay result disagrees with the board")
    if expected:
        raise ValueError("Arena match is missing one or more paired games")
    return score_games(rows)


def cached_result(folder, pair, openings):
    left, right = pair
    target = folder / f"{left}--{right}.json"
    replay = folder / f"{left}--{right}.jsonl.gz"
    if not replay.is_file() or not (scores := read_json(target)):
        return None
    try:
        with gzip.open(replay, "rt", encoding="utf-8") as stream:
            verified = verify_match_replays([json.loads(line) for line in stream], openings, right)
    except (OSError, EOFError, ValueError, KeyError, TypeError):
        return None
    return (replay, verified) if verified == scores else None


def arena_match(pair):
    left, right = pair
    target = output / f"{left}--{right}.json"
    replay_file = output / f"{left}--{right}.jsonl.gz"
    temporary = replay_file.with_name(f".{replay_file.name}.tmp")
    openings = [tuple(opening) for opening in arena["final_openings"]]
    if cached := cached_result(output, pair, openings):
        return pair, cached[1]
    if anchor_cache is not None and left in ANCHORS and right in ANCHORS:
        if cached := cached_result(anchor_cache, pair, openings):
            replay, scores = cached
            copyfile(replay, temporary)
            temporary.replace(replay_file)
            atomic_json(target, scores)
            return pair, scores
    games = evaluation_games(openings, right, ARENA_SEED, resolve)
    if left in resolve.models:
        model, balance = resolve.models[left]
        rows = play_evaluation(games, "cpu", model=model, family_balance=balance, resolve=resolve)
    else:
        rows = play_evaluation(games, "cpu", bot=left, resolve=resolve)
    scores = verify_match_replays(rows, openings, right)
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(replay_file)
    atomic_json(target, scores)
    return pair, scores


def win_interval(wins, games):
    z = 1.96
    rate = wins / games
    center = (rate + z * z / (2 * games)) / (1 + z * z / games)
    radius = z * math.sqrt((rate * (1 - rate) + z * z / (4 * games)) / games) / (1 + z * z / games)
    return max(0., center - radius), min(1., center + radius)


def write_review(root, stages, ratings, matches, plan):
    index = dict(ratings)
    lines = ["# Quoridor Phase 1 训练回顾", "",
             "五个关卡按低深度、中深度、高深度、混合搜索与历史模型的顺序完成。"
             "每种架构独立保存最佳和通过检查点。表中标为调整的关卡仍保留原门槛未通过的记录。", "",
             "## 最终 Arena", "",
             "每个配对 200 局，100 个保留开局各交换一次先后手。Arena 采用所有阶段最佳模型"
             "和固定搜索对手循环赛。胜率基于全部局数。Elo 用自然胜负拟合，截断和非法单列，BFS 固定为 1500。"
             "全部棋谱、逐对成绩和开局位于同一训练课程目录。", "",
             f"共 {len(matches)} 个配对、{sum(s['games'] for s in matches.values()):,} 局。"
             f"Arena 配置 SHA-256 为 `{plan['arena_sha256']}`。", "",
             "| 排名 | 模型或基准 | 最终 Elo |", "|---:|---|---:|"]
    lines.extend(f"| {position} | {name} | {elo:.1f} |" for position, (name, elo) in enumerate(ratings, 1))
    lines.extend(["", "## 各关卡最佳模型", "",
                  "| 关卡 | 模型 | 最佳 PPO 更新 | 晋级 PPO 更新 | 晋级记录 | 最终 Elo |",
                  "|---|---|---:|---:|---|---:|"])
    for stage in stages:
        level = stage["level"]
        for architecture, item in stage["promotion"].items():
            qualified, best = item["qualified"], item["best"]
            label = "调整后通过" if item["adjustment"] else "原门槛通过"
            lines.append(f"| Level {level} | {architecture.upper()} | {best['update']} | {qualified['update']} "
                         f"| {label} | {index[f'{architecture}-level{level}']:.1f} |")
    for stage in stages:
        level = stage["level"]
        lines.extend(["", f"### Level {level} · {stage['name']}"])
        for architecture, item in stage["promotion"].items():
            best, accepted, adjustment = item["best"], item["qualified"], item["adjustment"]
            lines.extend(["", f"**{architecture.upper()}** 第 {best['update']} 次更新，"
                          f"训练阶段 Arena Elo {best['elo']:.1f}。权重 SHA-256 为 `{best['sha256']}`。", "",
                          "| 对手 | 胜率 | 95% 区间 | 红方胜 | 蓝方胜 | 负 | 截断 | 非法 |",
                          "|---|---:|---:|---:|---:|---:|---:|---:|"])
            for name, stats in best["scores"].items():
                lo, hi = win_interval(stats["wins"], stats["games"])
                lines.append(f"| {name} | {stats['win_rate']:.1%} | {lo:.1%}–{hi:.1%} | {stats['wins_red']} | {stats['wins_blue']} | "
                             f"{stats['losses']} | {stats['truncated']} | {stats['illegal']} |")
            if adjustment:
                policy = adjustment if isinstance(adjustment, dict) else {"reason": adjustment,
                                                                            "original": {"wins": LEVEL_GATES[level],
                                                                                         "truncation_limit": .05},
                                                                            "adjusted": accepted["gate_policy"]}
                lines.extend(["", policy["reason"],
                              f"原门槛为 `{json.dumps(policy['original'], ensure_ascii=False)}`。",
                              f"本次晋级采用 `{json.dumps(policy['adjusted'], ensure_ascii=False)}`。"])
                if independent := policy.get("independent_gate_bank_1"):
                    random = independent["random"]
                    lines.append(f"独立开局复测对 Random 胜率 {random['win_rate']:.1%}、"
                                 f"截断率 {random['truncation_rate']:.1%}。")
                    earlier = read_json(root / "level1-mlp-verification.json")
                    earlier_random = earlier.get("independent_gate_bank_1", {}).get("random")
                    if earlier_random and earlier["update"] != accepted["update"]:
                        lines.append(f"较早的第 {earlier['update']} 次更新也做过独立复测。"
                                     f"对 Random 胜率 {earlier_random['win_rate']:.1%}、"
                                     f"截断率 {earlier_random['truncation_rate']:.1%}。")
            if series := item.get("qualification_series"):
                lines.extend(["", "连续三轮晋级证据", ""])
                for evidence in series:
                    scores = "、".join(f"{name} {result['win_rate']:.1%} 胜率 / "
                                      f"{result['truncation_rate']:.1%} 截断率"
                                      for name, result in evidence["scores"].items())
                    original = "原门槛通过" if evidence["original_gate_passed"] else "原门槛未通过"
                    lines.append(f"- 第 {evidence['update']} 次更新，开局组 {evidence['opening_bank']}，{scores}，{original}。")
            run_dir = Path(accepted["checkpoint"]).parent
            records = read_rows(run_dir / "metrics.jsonl")
            series = [r for r in records if r.get("formal_eval")]
            if level == 1 and any(row["update"] == accepted["update"] and row.get("required_passes") == 1
                                  for row in series):
                lines.extend(["", "Level 1 当时采用单轮正式评估的晋级流程。后续关卡采用三组开局连续达标。"])
            if series:
                first, last = series[0], series[-1]
                name = "random" if level == 1 else "history" if level == 5 else f"level{min(level, 3)}"
                if name in first["formal_eval"] and name in last["formal_eval"]:
                    begin, finish = first["formal_eval"][name], last["formal_eval"][name]
                    lines.extend(["", f"成长轨迹采用 {name}。首轮第 {first['update']} 次更新为 "
                                  f"{begin['win_rate']:.1%} 胜率、{begin['truncation_rate']:.1%} 截断率。"
                                  f"最后一轮第 {last['update']} 次更新为 "
                                  f"{finish['win_rate']:.1%} 胜率、{finish['truncation_rate']:.1%} 截断率。"
                                  f"共记录 {len(series)} 次正式评估。"])
                config = read_json(run_dir / "config.json")
                if config.get("opponent_pool"):
                    pool = "、".join(f"{name} {share:.0%}" for name, share in config["opponent_pool"].items())
                    lines.append(f"该运行的训练对手池为 {pool}。")
                ppo = [row for row in records if row.get("phase") == "ppo" and row.get("actions_per_second")]
                if ppo:
                    throughput = median(row["actions_per_second"] for row in ppo)
                    latest = ppo[-1]
                    lines.append(f"训练吞吐中位数为 {throughput:.1f} 个学习方动作/秒。"
                                 f"末次更新的总损失为 {latest['loss']:.4f}，近似 KL 为 "
                                 f"{latest['approx_kl']:.4f}，策略熵为 {latest['entropy']:.4f}。")
                for row in records:
                    if change := row.get("tuning"):
                        lines.append(f"第 {change['from_update']} 次更新起，"
                                     f"`{change['parameter']}` 从 {change['before']} 调为 {change['after']}。"
                                     f"{change['reason']}")
    lines.extend(["", "## 策略与局限", "",
                  "- Level 1 针对 Random 布墙和模型折返增加了 Random 对局。对照与专项从相同权重分支，"
                  "两组使用同样的 PBRS 0.02。专项后续沿用同一配比。",
                  "- 后续每累计 500 盘训练局，在完整 PPO 更新后评估。超出边界的局数进入下一单元。"
                  "中、高深度沿用本地冻结的搜索配置。历史网络全部来自本机先前 Level 的检查点。",
                  "- 每种架构本次使用一个种子。阶段 Arena 用于过关和选模。最终排名使用独立保留开局，"
                  "仍需结合原始截断率、非法率和对手分项解读。"])
    first_level = stages[0]["promotion"]
    special = Path(first_level["mlp"]["qualified"]["checkpoint"]).parent.parent
    control = special.with_name(special.name.replace("-special-", "-control-"))
    if control != special and control.is_dir():
        lines.extend(["", "## Level 1 首轮 500 次更新的专项对照", "",
                      "| 架构 | 对照 Random 胜率 | 专项 Random 胜率 | 对照截断率 | 专项截断率 |",
                      "|---|---:|---:|---:|---:|"])
        for architecture in first_level:
            series = []
            for branch in (control, special):
                records = read_rows(branch / f"{architecture}-seed0" / "metrics.jsonl")
                series.append(next((r["formal_eval"]["random"] for r in records
                                    if r.get("update") == 500 and r.get("formal_eval")), None))
            if all(series):
                baseline, trained = series
                lines.append(f"| {architecture.upper()} | {baseline['win_rate']:.1%} | {trained['win_rate']:.1%} | "
                             f"{baseline['truncation_rate']:.1%} | {trained['truncation_rate']:.1%} |")
    diagnostic = read_json(root / "level2-baseline-diagnostic.json")
    if diagnostic.get("scores"):
        results = "，".join(f"对 {name} {score['wins']}/{score['games']} 胜"
                          for name, score in diagnostic["scores"].items())
        lines.extend(["", "## 中深度搜索难度复核", "",
                      f"固定开局子集中的配对实测显示，中深度搜索{results}。"
                      "该子集用于解释 Level 1 到 Level 2 的棋力差距。"])
    seat_diagnostic = read_json(root / "level2-mlp-seat-diagnostic.json")
    if seat_diagnostic.get("sides"):
        reds = seat_diagnostic["sides"]["red"]
        blues = seat_diagnostic["sides"]["blue"]
        updates = "、".join(map(str, seat_diagnostic["checkpoint_updates"]))
        lines.extend(["", "## Level 2 MLP 先后手诊断", "",
                      f"复盘第 {updates} 次更新的正式对局，执红 {reds['wins']}/{reds['games']} 胜，"
                      f"有一步取胜机会的对局为 {reds['games_with_finish']} 局。"
                      f"执蓝 {blues['wins']}/{blues['games']} 胜，"
                      f"有一步取胜机会的对局为 {blues['games_with_finish']} 局。"
                      "该诊断用于定位开局至中局的走棋瓶颈。"])
    archived = read_json(root / "level2-mlp-archived-best-diagnostic.json")
    if archived.get("results"):
        results = "、".join(f"开局组 {row['opening_bank']} 为 {row['win_rate']:.1%}"
                           for row in archived["results"])
        red_wins = sum(row["wins_red"] for row in archived["results"])
        red_games = sum(row["red_games"] for row in archived["results"])
        lines.append(f"另外复测了第 {archived['update']} 次更新的历史最佳检查点。"
                     f"对中深度搜索的胜率为 {results}，执红 {red_wins}/{red_games} 胜。"
                     "这组数据用于比较训练前后的棋力和先后手差异。")
    walls = read_json(root / "level3-wall-diagnostic.json")
    if walls.get("models"):
        lines.extend(["", "## Level 3 布墙诊断", "",
                      "逐步复盘正式棋局。若一堵墙让对手到终点的 BFS 距离增加量超过己方，"
                      "就计为改善双方距离差。", "",
                      "| 模型 | PPO 更新 | 模型改善距离差的墙 | 三层搜索改善距离差的墙 |",
                      "|---|---:|---:|---:|"])
        for architecture, item in walls["models"].items():
            ratios = []
            for player in ("learner", "opponent"):
                seats = item["wall_changes"][player]
                placed = sum(side.get("walls", 0) for side in seats.values())
                improved = sum(side.get("positive", 0) for side in seats.values())
                ratios.append(f"{improved}/{placed}（{improved / placed:.1%}）" if placed else "0/0")
            lines.append(f"| {architecture.upper()} | {item['update']} | {' | '.join(ratios)} |")
    repetition = read_json(root / "level3-cnn32-repetition-diagnostic.json")
    if trials := repetition.get("evaluations"):
        total = sum(row["truncated"] for row in trials)
        blue = sum(row["counts"]["learner_blue"] for row in trials)
        stocked = sum(row["counts"]["learner_has_walls"] for row in trials)
        loops = sum(row["counts"]["at_most_eight_positions"] for row in trials)
        lines.extend(["", f"CNN32 五次正式评估共复盘 {total} 局被截断的对局。{blue} 局为执蓝，"
                      f"{stocked} 局结束时仍有墙可用，{loops} 局最后 {repetition['last_plies']} 手"
                      "只出现至多 8 种棋盘状态。"])
    checkpoints = read_json(root / "level3-checkpoint-comparison.json").get("results", {}).get("cnn32", {})
    if "level3_before_pbrs" in checkpoints and "level3_recent" in checkpoints:
        older, recent = (checkpoints[key] for key in ("level3_before_pbrs", "level3_recent"))
        first, second = older["scores"], recent["scores"]
        lines.extend(["", f"同一批 {first['games']} 局训练专用普通开局中，CNN32 第 {older['update']} 次更新"
                      f"为 {first['wins']} 胜、{first['truncated']} 局截断，第 {recent['update']} 次更新"
                      f"为 {second['wins']} 胜、{second['truncated']} 局截断。"
                      "对比结果用于选择接续训练的检查点。"])
    recovery = read_json(root / "level3-cnn32-second-recovery-diagnostic.json")
    recovery_by_update = {row["update"]: row["banks"] for row in recovery.get("results", [])}
    if 102 in recovery_by_update and 263 in recovery_by_update:
        lines.extend(["", "CNN32 另用两组独立开局比较了用于第二次恢复的检查点。"])
        for update in (102, 263):
            scores = [bank["scores"] for bank in recovery_by_update[update]]
            wins = sum(score["wins"] for score in scores)
            games = sum(score["games"] for score in scores)
            caps = sum(score["truncated"] for score in scores)
            lines.append(f"第 {update} 次更新为 {wins}/{games} 胜，截断 {caps} 局。")
    loop_test = read_json(root / "level3-cnn32-loop-escape-test.json")
    if cases := loop_test.get("audits"):
        total = sum(len(case["reference_games"]) for case in cases)
        lines.extend(["", f"从训练专用开局复现了 CNN32 的 {len(cases)} 个折返局面。"
                      f"本地搜索从循环出现时或八手前开始续弈，{total} 次参考对局取得 "
                      f"{loop_test['reference_counts'].get('win', 0)} 次自然胜利。"])
    repeat_temperature = read_json(root / "level3-cnn32-repeat-temperature-diagnostic.json")
    if repeat_temperature.get("cases"):
        lines.append(f"对第 {repeat_temperature['update']} 次更新的 {repeat_temperature['cases']} 个折返样本，"
                     f"把 PPO 采样温度调到 {repeat_temperature['temperature']} 后，已知能脱离循环的动作概率"
                     f"中位数由 {repeat_temperature['median_known_exit_mass_before']:.2%} "
                     f"升至 {repeat_temperature['median_known_exit_mass_after']:.2%}。")
    teacher_pilot = read_json(root / "level3-local-teacher-pilot" / "comparison.json")
    if teacher_pilot.get("after"):
        before, after = teacher_pilot["before"], teacher_pilot["after"]
        epoch = next(row for row in teacher_pilot["validation"] if row["epoch"] == teacher_pilot["selected_epoch"])
        lines.extend(["", "## Level 3 本地搜索示范对照", "",
                      f"MLP 副本的所选轮次在保留示范中的走法匹配率为 {epoch['teacher_top1']:.1%}。"
                      f"同批独立普通开局对完整三层搜索，补习前 {before['wins']}/{before['games']} 胜，"
                      f"补习后 {after['wins']}/{after['games']} 胜。"])
        extended = read_json(root / "level3-local-teacher-pilot" / "extended-comparison.json")
        if extra := extended.get("after"):
            best = next(row for row in extended["validation"] if row["epoch"] == extended["selected_epoch"])
            lines.append(f"延长到第 {best['epoch']} 轮后，走法匹配率达到 {best['teacher_top1']:.1%}，"
                         f"同批实战为 {extra['wins']}/{extra['games']} 胜。")
        corrected = read_json(root / "level3-policy-state-teacher-pilot" / "comparison.json")
        if after_correction := corrected.get("after"):
            positions = sum(row["new_teacher_positions"] for row in corrected["rounds"])
            lines.append(f"另在 MLP 自己访问的 {positions:,} 个局面上用本地三层搜索提供训练标签。"
                         f"这项对照的独立实战为 {after_correction['wins']}/{after_correction['games']} 胜。")
        expanded = read_json(root / "level3-500-local-teacher-pilot" / "comparison.json")
        if expanded.get("after"):
            before, after = expanded["before"], expanded["after"]
            selected = next(row for row in expanded["validation"] if row["epoch"] == expanded["selected_epoch"])
            lines.append("500 盘本地搜索示范试验使用同一批独立开局复测。"
                         f"保留示范中的走法匹配率为 {selected['val_accuracy']:.1%}。"
                         f"对三层搜索从 {before['level3']['wins']}/{before['level3']['games']} 胜变为 "
                         f"{after['level3']['wins']}/{after['level3']['games']} 胜，"
                         f"对中深度搜索从 {before['level2']['win_rate']:.1%} 变为 "
                         f"{after['level2']['win_rate']:.1%}。")
        probe = read_json(root / "level3-500-local-teacher-pilot" / "ppo-probe-comparison.json")
        if probe.get("after"):
            result = probe["after"]["level3"]
            lines.append(f"示范副本随后结算 {probe['games']} 盘，执行 {probe['updates']} 次 Masked PPO 更新。"
                         f"同一批诊断开局对三层搜索为 {result['wins']}/{result['games']} 胜，"
                         f"截断 {result['truncated']} 局。")
        practice = read_json(root / "level3-500-local-teacher-pilot" / "practice-calibration.json")
        if practice.get("scores"):
            blend = practice["scores"]["bfs-85"]
            full = practice["scores"]["level3"]
            lines.append(f"MLP 第 {practice['source_update']} 次更新在训练诊断开局中，"
                         f"对每手有 85% 概率使用 BFS 的练习对手赢 {blend['wins']}/{blend['games']} 局，"
                         f"对完整三层搜索赢 {full['wins']}/{full['games']} 局。")
            bridge = read_json(root / "level3-500-local-teacher-pilot" / "bridge-probe-comparison.json")
            if bridge.get("after"):
                result = bridge["after"]["level3"]
                lines.append(f"副本使用该练习对手占 10% 的配比结算 {bridge['games']} 盘。"
                             f"复测完整三层搜索为 {result['wins']}/{result['games']} 胜。")
        stock = read_json(root / "level3-500-local-teacher-pilot" / "wall-stock-probe-comparison.json")
        if stock.get("after"):
            result = stock["after"]["level3"]
            weight = stock["config"]["wall_stock_potential_weight"]
            lines.append(f"墙体资源差以 {weight:g} 的权重加入 BFS 势函数。"
                         f"副本结算 {stock['games']} 盘后，"
                         f"复测完整三层搜索为 {result['wins']}/{result['games']} 胜。")
        correction = read_json(root / "level3-500-local-teacher-pilot" / "state-correction" / "comparison.json")
        if correction.get("after"):
            before, after = (correction[key]["level3"] for key in ("before", "after"))
            lines.append("在 500 盘 MLP 实际训练局的前 16 个决策上取得本地三层搜索标签，并与原示范一起训练。"
                         f"独立实战由 {before['wins']}/{before['games']} 升至 {after['wins']}/{after['games']}，"
                         f"截断 {after['truncated']} 局。")
        verification = read_json(root / "level3-500-local-teacher-pilot" / "independent-confirmation.json")
        if verified := verification.get("results", {}).get("mlp-corrected"):
            result = verified["scores"]
            lines.append(f"另一批全新开局复测为 {result['wins']}/{result['games']} 胜、"
                         f"{result['truncated']} 局截断。")
        corrected_ppo = read_json(root / "level3-500-local-teacher-pilot" / "corrected-ppo-probe-comparison.json")
        if corrected_ppo.get("after"):
            result = corrected_ppo["after"]["level3"]
            lines.append(f"纠错副本接续 Masked PPO {corrected_ppo['updates']} 次，结算 {corrected_ppo['games']} 盘。"
                         f"同批新开局对三层搜索为 {result['wins']}/{result['games']} 胜。")
        regression = read_json(root / "level3-mlp-ppo-regression-early-moves.json")
        if lost := regression.get("lost_winning_opening_pairs"):
            before, after = regression["updates"]
            lines.append(f"第 {before} 次更新在同一开局组的 {len(lost)} 局胜利在第 {after} 次更新时全部消失。"
                         f"首次选招分歧的中位位置为开局后第 {regression['median_first_change_ply']:g} 手。")
        low_lr = read_json(root / "level3-500-local-teacher-pilot" / "low-lr-probe-comparison.json")
        if low_lr.get("after"):
            before, after = (low_lr[key]["level3"] for key in ("before", "after"))
            lr = low_lr["config"]["hyperparameters"]["lr"]
            lines.append(f"学习率 {lr:g} 的续训对照结算 {low_lr['games']} 盘，"
                         f"同组开局从 {before['wins']}/{before['games']} 胜变为 {after['wins']}/{after['games']} 胜。")
        for name in ("reference-ppo-probe", "reference-ppo-probe-8"):
            anchored = read_json(root / "level3-500-local-teacher-pilot" / f"{name}-comparison.json")
            if anchored.get("after"):
                before, after = (anchored[key]["level3"] for key in ("before", "after"))
                coefficient = anchored["config"]["hyperparameters"]["reference_kl_coef"]
                lines.append(f"参考策略 KL 系数为 {coefficient:g} 时，结算 {anchored['games']} 盘，"
                             f"同组开局从 {before['wins']}/{before['games']} 胜变为 {after['wins']}/{after['games']} 胜。")
    symmetry = read_json(root / "level3-symmetric-policy-diagnostic.json")
    if models := symmetry.get("results"):
        lines.extend(["", "## Level 3 棋盘对称性试验", "",
                      "| 模型 | 对称变换 | 胜 | 负 | 截断 |",
                      "|---|---|---:|---:|---:|"])
        names = {"direct": "直接选招", "left_right": "左右镜像合并",
                 "left_right_and_color_reversal": "左右镜像与红蓝换边合并"}
        for architecture, evidence in models.items():
            for mode, scores in evidence["results"].items():
                lines.append(f"| {architecture.upper()} | {names.get(mode, mode)} | {scores['wins']} | "
                             f"{scores['losses']} | {scores['truncated']} |")
    policy_test = read_json(root / "level3-ppo-policy-selection-diagnostic.json")
    if policies := policy_test.get("results"):
        lines.extend(["", "## Level 3 模型选招对照", "",
                      "| 模型 | 选招方式 | 胜 | 负 | 截断 | 棋局 |",
                      "|---|---|---:|---:|---:|---:|"])
        selection = {"greedy": "最高概率动作", "sample_raw": "合法动作概率采样",
                     "sample_on_policy": "采样并在重复局面提高温度"}
        for architecture, result in policies.items():
            for mode, trials in result["modes"].items():
                totals = {key: sum(row["scores"][key] for row in trials)
                          for key in ("wins", "losses", "truncated", "games")}
                lines.append(f"| {architecture.upper()} | {selection.get(mode, mode)} | {totals['wins']} | "
                             f"{totals['losses']} | {totals['truncated']} | {totals['games']} |")
    early_walls = read_json(root / "level3-mlp-early-wall-audit.json")
    if early_walls.get("counts"):
        learner, opponent = (early_walls["counts"][side] for side in ("learner", "opponent"))
        n = early_walls["first_decisions_per_side"]
        lines.extend(["", f"第 {early_walls['source_update']} 次更新后的正式评估中，"
                      f"统计各方前 {n} 手。MLP 的 {learner['walls']:,} 次布墙有 "
                      f"{learner['walls_with_positive_net_distance']:,} 次给对手增加的 BFS 距离大于己方。"
                      f"三层搜索对手为 {opponent['walls_with_positive_net_distance']:,}/{opponent['walls']:,} 次。"])
    post_recovery = read_json(root / "level3-post-recovery-progress-audit.json")
    if cnn_audit := post_recovery.get("cnn32"):
        shared = "、".join(str(item["shared_winning_opening_seat_pairs"])
                          for _, item in sorted(cnn_audit["by_bank"].items()))
        lines.extend(["", f"CNN32 在第 296 至 329 次更新间有 {cnn_audit['changed_elements']:,} 个权重数值变动。"
                      f"把每组开局多次评估的胜局按开局和先后手匹配后，三组分别有 {shared} 个始终获胜的组合。"])
    if mlp_audit := post_recovery.get("mlp", {}).get("checkpoints"):
        before, after = (mlp_audit[update]["counts"]["learner"] for update in ("234", "277"))
        lines.append(f"MLP 开局布墙改善距离差的比例从 {before['positive_wall_gain'] / before['walls']:.1%} "
                     f"变为 {after['positive_wall_gain'] / after['walls']:.1%}。")
    history_audit = read_json(root / "level5-history-truncation-audit.json")
    if history_audit.get("models"):
        lines.extend(["", "## Level 5 历史模型对局诊断", ""])
        for architecture, record in history_audit["models"].items():
            findings = record["by_model"].values()
            caps = sum(row.get("truncated", 0) for row in findings)
            repeats = sum(row.get("repeats_at_cap", 0) for row in findings)
            lines.append(f"{architecture.upper()} 第 {record['update']} 次更新的正式历史池对局中，"
                         f"{caps} 局截断，{repeats} 局结束时仍在重复走子。")
        history_folder = root / "level5-local-history-diagnostics"
        selection = read_json(history_folder / "mode-comparison.json")
        for architecture, record in selection.get("results", {}).items():
            greedy, sampled = (record["modes"][mode] for mode in ("greedy", "sample-on-repeat"))
            lines.append(f"{architecture.upper()} 在独立诊断开局中的直接选招成绩为 "
                         f"{greedy['wins']}/{greedy['games']} 胜、{greedy['truncated']} 局截断。"
                         f"重复时增加采样探索后为 {sampled['wins']}/{sampled['games']} 胜、"
                         f"{sampled['truncated']} 局截断。正式评估使用直接选招。")
        for architecture in ("mlp", "cnn32"):
            coaching = read_json(history_folder / f"{architecture}-coach" / "comparison.json")
            if coaching.get("after"):
                before, after = (coaching[key]["history"] for key in ("before", "after"))
                lines.append(f"{architecture.upper()} 副本用本机历史对局中的实际局面做搜索示范训练后，"
                             f"独立实战从 {before['wins']}/{before['games']} 胜、{before['truncated']} 局截断，"
                             f"变为 {after['wins']}/{after['games']} 胜、{after['truncated']} 局截断。")
    operational_changes = []
    for stage in stages:
        item = next(iter(stage["promotion"].values()))
        batch = Path(item["qualified"]["checkpoint"]).parent.parent
        for event in read_rows(batch / "runner-events.jsonl"):
            if event.get("action") == "resume_with_parallel_search":
                operational_changes.append(f"- Level {stage['level']} · {event['architecture'].upper()} 从第 "
                                           f"{event['checkpoint_update']} 次更新接续并行对手搜索，"
                                           f"交接时已结算 {event['completed_training_games']:,} 盘训练局。")
            elif event.get("action") == "resume_after_parallel_evaluation_fix":
                operational_changes.append(f"- Level {stage['level']} 的并行评估记录错误已修复，"
                                           "两种模型从已保存的 PPO 检查点恢复。")
            elif event.get("action") == "resume_with_bitset_wall_path_filter":
                operational_changes.append(f"- Level {stage['level']} · {event['architecture'].upper()} 从第 "
                                           f"{event['checkpoint_update']} 次更新接续位集搜索。"
                                           "候选墙顺序与对手选招已对照验证，验收标准未变。")
    if operational_changes:
        lines.extend(["", "## 运行记录", "", *operational_changes])
    target = root / "PHASE1_REVIEW.md"
    temporary = target.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def load_frozen_arena(root):
    arena_path = root / "arena.json"
    frozen = read_json(arena_path)
    final_openings = {reflection_group(opening) for opening in frozen["final_openings"]}
    prior_openings = {reflection_group(opening) for opening in
                      (*frozen["training_openings"], *frozen["curriculum_openings"],
                       *(opening for bank in frozen["gate_openings"] for opening in bank))}
    if len(frozen["final_openings"]) != 100 or len(final_openings) != 100 or final_openings & prior_openings:
        raise ValueError("Final Arena needs 100 distinct held-out openings")
    if frozen["search"] != {name: asdict(preset) for name, preset in DEEP_SEARCH.items()}:
        raise ValueError("The final Arena search configuration does not match its frozen settings")
    return frozen, sha256(arena_path)


def anchor_plan(arena_sha256):
    return {"protocol": "phase1-final-anchor-cache-v1", "arena_sha256": arena_sha256,
            "games_per_pair": 200, "seed": ARENA_SEED, "max_plies": MAX_PLIES, "policies": POLICIES,
            "pairs": [list(pair) for pair in combinations(ANCHORS, 2)]}


def run_matches(frozen, models, folder, workers, pairs, progress, cached_anchors=None):
    matches = {}
    atomic_json(progress, {"completed_pairs": 0, "total_pairs": len(pairs)})
    with ProcessPoolExecutor(max_workers=workers, initializer=prepare_worker,
                             initargs=(frozen, models, folder, cached_anchors)) as pool:
        for future in as_completed(pool.submit(arena_match, pair) for pair in pairs):
            pair, result = future.result()
            matches[pair] = result
            atomic_json(progress, {"completed_pairs": len(matches), "total_pairs": len(pairs)})
    return {pair: matches[pair] for pair in pairs}


def precompute_anchors(root, workers):
    frozen, arena_sha256 = load_frozen_arena(root)
    folder = root / "final-arena-anchors"
    folder.mkdir(exist_ok=True)
    with (folder / ".writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = anchor_plan(arena_sha256)
        plan_path = folder / "plan.json"
        if plan_path.exists() and read_json(plan_path) != plan:
            raise ValueError("Anchor cache belongs to a different frozen Arena")
        atomic_json(plan_path, plan)
        pairs = [tuple(pair) for pair in plan["pairs"]]
        return run_matches(frozen, {}, folder, workers, pairs, folder / "progress.json")


def run_final_arena(root, stages, workers):
    frozen, arena_sha256 = load_frozen_arena(root)
    models = {f"{architecture}-level{stage['level']}": item["best"]
              for stage in stages for architecture, item in stage["promotion"].items()}
    pairs = list(combinations((*ANCHORS, *models), 2))
    plan = {"protocol": "phase1-final-arena-v1", "arena_sha256": arena_sha256,
            "games_per_pair": 200, "anchors": ANCHORS, "models": models, "pairs": pairs}
    folder = root / "final-arena"
    folder.mkdir(exist_ok=True)
    plan_path = folder / "plan.json"
    if plan_path.exists() and read_json(plan_path) != json.loads(json.dumps(plan)):
        raise ValueError("Final Arena resume requires the original model and opening roster")
    atomic_json(plan_path, plan)
    cached = root / "final-arena-anchors"
    if read_json(cached / "plan.json") != anchor_plan(arena_sha256):
        cached = None
    matches = run_matches(frozen, models, folder, workers, pairs, root / "arena-progress.json", cached)
    ratings = sorted(estimate_elo(matches).items(), key=lambda item: (-item[1], item[0]))
    target = root / "ARENA_RESULTS.json"
    atomic_json(target, {"protocol": plan["protocol"], "arena_sha256": plan["arena_sha256"],
                        "games_per_pair": plan["games_per_pair"], "ratings": ratings,
                        "games": sum(result["games"] for result in matches.values()),
                        "matches": [{"players": pair, **result} for pair, result in matches.items()]})
    report = write_review(root, stages, ratings, matches, plan)
    return {"report": str(report.resolve()), "arena_results": str(target.resolve()), "games": len(pairs) * 200,
            "ratings": ratings}


if __name__ == "__main__":
    parser = ArgumentParser(description="Precompute frozen Arena games between fixed baselines.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    scores = precompute_anchors(args.root, args.workers)
    print(f"Verified {len(scores)} baseline pairings and {sum(s['games'] for s in scores.values()):,} games.", flush=True)
