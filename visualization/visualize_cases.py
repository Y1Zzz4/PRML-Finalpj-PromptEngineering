import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_val_data(path="data/val.jsonl"):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_predictions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def draw_grid(ax, grid, title):
    ax.set_title(title, fontsize=12)
    ax.imshow(grid, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
    ax.grid(True, which='both', color='gray', linewidth=0.5, linestyle='-')
    ax.set_xticks(np.arange(-0.5, len(grid[0]), 1))
    ax.set_yticks(np.arange(-0.5, len(grid), 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            ax.text(j, i, str(grid[i][j]), ha="center", va="center", color="white", fontsize=10, fontweight="bold")


def visualize(strategy: str, task_id: int, save_path: Path | None = None):
    val_data = load_val_data()
    preds = load_predictions(f"results/{strategy}_val.json")

    if task_id >= len(val_data):
        raise ValueError(f"task_id {task_id} 超出范围，val.jsonl 只有 {len(val_data)} 个任务")

    task = val_data[task_id]
    pred_item = preds[task_id]

    test_input = task["test"][0]["input"]
    gt_output = task["test"][0]["output"]
    pred_output = pred_item.get("predicted_grid") or pred_item.get("prediction")

    if pred_output is None:
        print(f"警告：任务 {task_id} 的预测失败（predicted_grid 为 None）")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    draw_grid(axes[0], np.array(test_input), "Test Input")
    draw_grid(axes[1], np.array(pred_output) if pred_output else np.zeros_like(np.array(test_input)),
              f"Prediction ({strategy})" + ("\n(解析失败)" if pred_output is None else ""))
    draw_grid(axes[2], np.array(gt_output), "Ground Truth")

    plt.suptitle(f"Task {task_id} - Strategy: {strategy}", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 保存或显示
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
        plt.close(fig)  # 保存后关闭图，防止内存占用
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="可视化 ARC 任务的预测结果")
    parser.add_argument("--strategy", required=True,
                        help="策略名称，如 baseline, cot, structured 等")
    parser.add_argument("--task_id", type=int, required=True,
                        help="val.jsonl 中的任务索引 (0 ~ 29)")
    parser.add_argument("--save", action="store_true",
                        help="是否保存图片到文件")
    parser.add_argument("--output_dir", type=str, default="visuals_results",
                        help="保存图片的目录（默认 visuals/）")

    args = parser.parse_args()

    # 构建保存路径
    save_path = None
    if args.save:
        output_dir = Path(args.output_dir)
        save_path = output_dir / f"task_{args.task_id:02d}_{args.strategy}.png"

    visualize(args.strategy, args.task_id, save_path)


if __name__ == "__main__":
    main()