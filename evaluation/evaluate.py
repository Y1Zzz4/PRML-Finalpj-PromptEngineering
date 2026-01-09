import json
import argparse
from pathlib import Path


def load_val_data(path: str):
    """加载 val.jsonl，每行一个任务"""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_predictions(path: str):
    """加载推理结果 JSON 文件"""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def exact_match(pred, gt) -> bool:
    """完全匹配：两个网格必须完全相同（包括尺寸和每个元素）"""
    if pred is None:
        return False
    return pred == gt


def evaluate(val_path: str, pred_path: str):
    val_data = load_val_data(val_path)
    preds = load_predictions(pred_path)

    total = len(preds)
    assert total == len(val_data), f"任务数量不匹配！val: {len(val_data)}, pred: {total}"

    correct = 0
    wrong_tasks = []
    parse_failed = 0

    for idx, item in enumerate(preds):
        pred_grid = item.get("predicted_grid") 

        # 真实答案
        gt_grid = val_data[idx]["test"][0]["output"]  # 索引对齐

        if pred_grid is None:
            parse_failed += 1
            wrong_tasks.append(idx)
            continue

        if exact_match(pred_grid, gt_grid):
            correct += 1
        else:
            wrong_tasks.append(idx)

    acc = correct / total if total > 0 else 0

    print("====== Evaluation Result ======")
    print(f"数据集       : {Path(val_path).name}")
    print(f"策略文件      : {Path(pred_path).name}")
    print(f"总任务数      : {total}")
    print(f"正确任务数    : {correct}")
    print(f"解析失败数    : {parse_failed}")
    print(f"错误任务 ID   : {sorted(wrong_tasks)}")
    print(f"准确率 (Accuracy) : {acc:.4f} ({acc * 100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="评估 ARC 任务预测准确率")
    parser.add_argument("--val", default="data/val.jsonl",
                        help="val.jsonl 文件路径（默认 data/val.jsonl）")
    parser.add_argument("--pred", required=True,
                        help="预测结果 JSON 文件路径（如 results/baseline_val.json）")
    args = parser.parse_args()

    # 检查文件是否存在
    if not Path(args.val).exists():
        raise FileNotFoundError(f"val 文件不存在: {args.val}")
    if not Path(args.pred).exists():
        raise FileNotFoundError(f"预测文件不存在: {args.pred}")

    evaluate(args.val, args.pred)


if __name__ == "__main__":
    main()