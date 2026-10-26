"""把 raw_episodes 下的 episode 批量转换为一个 lerobot v3.0 训练数据集。

用法（在 lerobot 0.4.3 环境运行）：
    python export_dataset.py                          # 全量转换
    python export_dataset.py --min-travel 0           # 关闭行程过滤
    python export_dataset.py --only episode_20261006_xxx [episode_...]
    python export_dataset.py --output E:\\...\\my_dataset --repo-id local/jaka_zu3_act

行为：
    - 只转换 status=complete 且关节行程达标的条目
    - 已转换过的条目自动跳过（断点续转，见 conversion_manifest.json）
    - 单条对齐失败只跳过该条，不中断整批
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from collect_data import (  # noqa: E402
    CAMERA_PAIR_TOLERANCE,
    CAMERA_TOLERANCE,
    EXPORT_FPS,
    ROBOT_MAX_GAP,
    ROBOT_MAX_QUERY,
    nearest_indices,
    read_stream,
)

SAVE_ROOT = Path(r"E:\jaka\SDK v2.2.2")
DEFAULT_INPUT = SAVE_ROOT / "raw_episodes"
DEFAULT_OUTPUT = SAVE_ROOT / "dataset_jaka_zu3"


class SkipEpisode(Exception):
    """结构性不合格（未完成/没录数据），记入 skipped 而非 failed"""


def align_episode(root, min_travel):
    """对齐一条 episode，返回逐帧数据。失败抛 ValueError。"""
    metadata = json.loads(
        (root / "episode.json").read_text(encoding="utf-8")
    )

    if metadata["status"] != "complete":
        raise SkipEpisode(f"status={metadata['status']}")

    streams = {
        name: read_stream(root / f"{name}.jsonl")
        for name in ("robot", "front", "wrist")
    }

    # 共同覆盖区间只看三路数据流本身：旧版本数据里主进程的
    # record_start/record_end 与 worker 时间轴存在系统性错位，
    # 不能参与区间计算
    begin = max(s[1][0] for s in streams.values())
    end = min(s[1][-1] for s in streams.values())

    frame_count = int(np.floor((end - begin) * EXPORT_FPS)) + 1
    if frame_count < 3:
        raise ValueError("共同覆盖的时间不足")

    grid = begin + np.arange(frame_count) / EXPORT_FPS

    robot_rows, robot_times = streams["robot"]
    joints = np.asarray([row["q"] for row in robot_rows])

    if joints.shape != (len(robot_rows), 6):
        raise ValueError("关节状态维度异常")
    if not np.isfinite(joints).all():
        raise ValueError("关节状态包含无效数值")

    right = np.clip(
        np.searchsorted(robot_times, grid, side="right"),
        1,
        len(robot_times) - 1,
    )
    left = right - 1
    gaps = robot_times[right] - robot_times[left]

    query_times = np.asarray([
        row["after"] - row["before"] for row in robot_rows
    ])

    if np.any(gaps > ROBOT_MAX_GAP):
        raise ValueError("机器人反馈间隔过大，拒绝跨越断流插值")
    if np.any(
        np.maximum(query_times[left], query_times[right])
        > ROBOT_MAX_QUERY
    ):
        raise ValueError("SDK 调用耗时过大，时间对齐不可靠")

    weights = (grid - robot_times[left]) / gaps
    states = (
        joints[left]
        + weights[:, None] * (joints[right] - joints[left])
    ).astype(np.float32)

    travel = float((joints.max(axis=0) - joints.min(axis=0)).max())
    if travel < min_travel:
        raise SkipEpisode(
            f"关节行程 {travel:.3f} rad 低于 {min_travel}，"
            "机器人基本没动，无训练价值"
        )

    # action = 下一统一时刻的实测位置；最后一帧没有未来，丢弃
    actions = states[1:].copy()
    frame_count = len(grid) - 1

    camera_indices = {}
    camera_offsets = {}

    for name in ("front", "wrist"):
        rows, times = streams[name]
        indices = nearest_indices(times, grid[:-1])
        offsets = times[indices] - grid[:-1]

        if np.any(np.abs(offsets) > CAMERA_TOLERANCE):
            raise ValueError(f"{name} 相机时间偏差超过阈值")

        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"{name} 原始帧率不足")

        camera_indices[name] = indices
        camera_offsets[name] = offsets

    pair_offsets = np.abs(
        camera_offsets["front"] - camera_offsets["wrist"]
    )
    if np.any(pair_offsets > CAMERA_PAIR_TOLERANCE):
        raise ValueError("两台相机的时间差超过阈值")

    stats = {
        "frames": frame_count,
        "task": metadata["task"],
        "joint_travel_rad": round(travel, 3),
        "front_max_offset_ms": round(
            float(np.max(np.abs(camera_offsets["front"])) * 1000), 1
        ),
        "wrist_max_offset_ms": round(
            float(np.max(np.abs(camera_offsets["wrist"])) * 1000), 1
        ),
        "camera_pair_max_offset_ms": round(
            float(np.max(pair_offsets) * 1000), 1
        ),
        "robot_max_interpolation_gap_ms": round(
            float(np.max(gaps) * 1000), 1
        ),
    }

    return {
        "streams": streams,
        "grid": grid,
        "states": states,
        "actions": actions,
        "camera_indices": camera_indices,
        "metadata": metadata,
        "stats": stats,
    }


def build_features(metadata):
    image_shape = (metadata["height"], metadata["width"], 3)

    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": [
                "q1_rad", "q2_rad", "q3_rad",
                "q4_rad", "q5_rad", "q6_rad",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": [
                "q1_target_rad", "q2_target_rad", "q3_target_rad",
                "q4_target_rad", "q5_target_rad", "q6_target_rad",
            ],
        },
        "observation.images.front": {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="批量转换 raw_episodes 为 lerobot 训练数据集"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-id", default="local/jaka_zu3_act")
    parser.add_argument("--min-travel", type=float, default=0.05,
                        help="关节行程低于该值(rad)的条目跳过；0 关闭")
    parser.add_argument("--only", nargs="*", default=None,
                        help="只转换指定的 episode 目录名")
    args = parser.parse_args()

    # 这些依赖只在转换时加载
    import cv2
    from lerobot.datasets.lerobot_dataset import (
        CODEBASE_VERSION,
        LeRobotDataset,
    )
    from importlib.metadata import version

    if version("lerobot") != "0.4.3" or CODEBASE_VERSION != "v3.0":
        raise RuntimeError("请使用 lerobot==0.4.3 的独立导出环境")

    manifest_path = args.output / "conversion_manifest.json"

    episode_dirs = sorted(
        d for d in args.input.iterdir()
        if d.is_dir() and (d / "episode.json").exists()
    )

    if args.only is not None:
        episode_dirs = [
            d for d in episode_dirs if d.name in set(args.only)
        ]

    if not episode_dirs:
        print("没有找到任何 episode 目录")
        return

    if args.output.exists():
        if not manifest_path.exists():
            raise SystemExit(
                f"输出目录已存在但没有 manifest：{args.output}"
            )
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        dataset = LeRobotDataset(
            repo_id=args.repo_id, root=args.output, video_backend="pyav"
        )
        if dataset.meta.total_episodes != len(manifest["converted"]):
            raise SystemExit(
                f"数据集 episode 数（{dataset.meta.total_episodes}）与 "
                f"manifest 记录（{len(manifest['converted'])}）不一致，"
                "索引可能损坏。请删除输出目录后重新全量转换。"
            )
        print(f"继续追加到已有数据集：{args.output} "
              f"（{dataset.meta.total_episodes} 条，"
              f"{len(dataset)} 帧）")
    else:
        manifest = {"converted": {}, "skipped": {}, "failed": {}}
        dataset = None

    aligned_dir = args.output / "alignment"
    report_dir = args.output / "alignment_reports"

    converted = skipped = failed = 0
    t0 = time.perf_counter()
    wrote_any = False
    dataset_is_new = False

    try:
        for root in episode_dirs:
            name = root.name

            if name in manifest["converted"]:
                print(f"[跳过] {name}（已转换）")
                skipped += 1
                continue

            try:
                aligned = align_episode(root, args.min_travel)
            except SkipEpisode as exc:
                manifest["skipped"][name] = str(exc)
                print(f"[跳过] {name}：{exc}")
                skipped += 1
                continue
            except (ValueError, OSError, KeyError) as exc:
                manifest["failed"][name] = str(exc)
                print(f"[失败] {name}：{exc}")
                failed += 1
                continue

            stats = aligned["stats"]
            metadata = aligned["metadata"]

            if dataset is None:
                dataset = LeRobotDataset.create(
                    repo_id=args.repo_id,
                    root=args.output,
                    fps=EXPORT_FPS,
                    robot_type="jaka_zu3",
                    features=build_features(metadata),
                    use_videos=True,
                    vcodec="h264",
                    video_backend="pyav",
                )
                dataset_is_new = True
                aligned_dir.mkdir()
                report_dir.mkdir()

            image_shape = (
                metadata["height"], metadata["width"], 3
            )

            try:
                for i, target_time in enumerate(aligned["grid"][:-1]):
                    frame = {
                        "observation.state": aligned["states"][i].copy(),
                        "action": aligned["actions"][i].copy(),
                        "task": metadata["task"],
                    }

                    for cam in ("front", "wrist"):
                        row = aligned["streams"][cam][0][
                            aligned["camera_indices"][cam][i]
                        ]
                        path = root / row["path"]

                        image = cv2.imdecode(
                            np.fromfile(str(path), dtype=np.uint8),
                            cv2.IMREAD_COLOR,
                        )

                        if image is None or image.shape != image_shape:
                            raise RuntimeError(f"图像读取失败：{path}")

                        frame[f"observation.images.{cam}"] = (
                            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        )

                    dataset.add_frame(frame)

                dataset.save_episode()
                wrote_any = True
                episode_index = dataset.meta.total_episodes - 1
            except BaseException as exc:
                # 丢弃写了一半的内存缓冲，重载数据集继续下一条
                manifest["failed"][name] = f"写入失败：{exc}"
                print(f"[失败] {name}：{exc}")
                failed += 1
                dataset = LeRobotDataset(
                    repo_id=args.repo_id,
                    root=args.output,
                    video_backend="pyav",
                )
                continue

            manifest["converted"][name] = {
                "dataset_episode_index": episode_index,
                **stats,
            }
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            with (report_dir / f"{name}.json").open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(
                    {"source": str(root), **stats},
                    f, ensure_ascii=False, indent=2,
                )

            with (aligned_dir / f"{name}.csv").open(
                "w", newline="", encoding="utf-8"
            ) as f:
                writer = csv.writer(f)
                writer.writerow(["frame_index", "timestamp", "host_time"])

                for i, t in enumerate(aligned["grid"][:-1]):
                    writer.writerow([i, i / EXPORT_FPS, t])

            converted += 1
            print(
                f"[完成] {name} -> episode {episode_index} "
                f"({stats['frames']} 帧, 行程 {stats['joint_travel_rad']} rad)"
            )

    finally:
        # 只读加载（全部跳过）时不要 finalize，避免重写元数据
        if dataset is not None and (wrote_any or dataset_is_new):
            dataset.finalize()

        if manifest_path.parent.exists():
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    print()
    print("========== 转换汇总 ==========")
    print(f"转换 {converted} 条，跳过 {skipped} 条，失败 {failed} 条，"
          f"耗时 {time.perf_counter()-t0:.0f}s")

    if dataset is not None:
        for name, reason in manifest["skipped"].items():
            print(f"  [跳过] {name}: {reason}")
        for name, reason in manifest["failed"].items():
            print(f"  [失败] {name}: {reason}")

        loaded = LeRobotDataset(
            repo_id=args.repo_id, root=args.output, video_backend="pyav"
        )
        total = len(loaded)
        expected = sum(
            v["frames"] for v in manifest["converted"].values()
        )
        expected_eps = len(manifest["converted"])
        ok_frames = total == expected
        ok_eps = (
            loaded.meta.total_episodes == expected_eps
        )
        print(f"数据集校验：{total} 帧/manifest {expected} 帧 "
              f"{'一致' if ok_frames else '不一致！'}；"
              f"{loaded.meta.total_episodes} 条 episode/manifest "
              f"{expected_eps} 条 {'一致' if ok_eps else '不一致！'}")
        print(f"训练入口：dataset.repo_id={args.repo_id} "
              f"dataset.root={args.output}")


if __name__ == "__main__":
    main()
