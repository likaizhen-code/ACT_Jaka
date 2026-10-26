"""episode 一键体检：python check_episode.py <episode目录>"""
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
meta = json.loads((root / "episode.json").read_text(encoding="utf-8"))

print("=" * 50)
print("状态:", meta["status"], "  任务:", meta.get("task"))
if meta.get("error"):
    print("错误:", meta["error"])

EXPORT_FPS = 10


def stats(name, images):
    rows = [
        json.loads(line)
        for line in (root / f"{name}.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    t = np.asarray([r["t"] for r in rows])
    dt = np.diff(t)
    missing = sum(
        1 for r in rows if images and not (root / r["path"]).exists()
    )
    fps = (len(rows) - 1) / (t[-1] - t[0])
    print(
        f"{name}: {len(rows)} 条, 时长 {t[-1]-t[0]:.1f}s, "
        f"{fps:.1f}fps, 最大间隔 {dt.max()*1000:.0f}ms, 缺图 {missing}"
    )
    return rows, dt


rows_r, dt_r = stats("robot", False)
rows_f, dt_f = stats("front", True)
rows_w, dt_w = stats("wrist", True)

q = np.asarray([r["q"] for r in rows_r])
travel = q.max(axis=0) - q.min(axis=0)
print("各关节行程(rad):", np.round(travel, 3).tolist())
moved = travel.max() > 0.05

issues = []
if meta["status"] != "complete":
    issues.append(f"状态是 {meta['status']}，导出会被拒绝")
if not moved:
    issues.append("机器人全程几乎没动，没有训练价值")
if dt_r.max() > 0.15:
    issues.append(f"robot 断流 {dt_r.max()*1000:.0f}ms 超 150ms，导出会拒绝")
for name, rows, dt in (
    ("front", rows_f, dt_f), ("wrist", rows_w, dt_w)
):
    eff = (len(rows) - 1) / max(dt.sum(), 1e-9)
    if eff < 2 * EXPORT_FPS:
        issues.append(
            f"{name} 有效帧率 {eff:.1f}fps 低于 {2*EXPORT_FPS}，导出可能报帧率不足"
        )
    if dt.max() > 0.1:
        issues.append(f"{name} 存在 {dt.max()*1000:.0f}ms 断流")

print("=" * 50)
if issues:
    print("结论: 有问题")
    for it in issues:
        print(" -", it)
else:
    print("结论: 健康，可以导出")
