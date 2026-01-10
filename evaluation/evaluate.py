import json
import argparse
from pathlib import Path


def load_val_data(path: str):
    """加载 val.jsonl 或 val_hard.jsonl，每行一个任务"""
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


def evaluate_single(val_data: list, preds: list, file_name: str) -> dict:
    """评估单个结果文件，返回统计信息"""
    total = len(preds)
    assert total == len(val_data), f"任务数量不匹配！val: {len(val_data)}, pred: {total}"

    correct = 0
    parse_failed = 0
    wrong_tasks = []

    for idx, item in enumerate(preds):
        pred_grid = item.get("predicted_grid")

        gt_grid = val_data[idx]["test"][0]["output"]

        if pred_grid is None:
            parse_failed += 1
            wrong_tasks.append(idx)
            continue

        if exact_match(pred_grid, gt_grid):
            correct += 1
        else:
            wrong_tasks.append(idx)

    acc = correct / total if total > 0 else 0
    return {
        "file": file_name,
        "total": total,
        "correct": correct,
        "parse_failed": parse_failed,
        "accuracy": acc,
        "wrong_tasks": sorted(wrong_tasks)
    }


def print_summary(results: list[dict]):
    """打印所有结果的汇总"""
    print("\n" + "="*80) 
    print("批量评估汇总（按准确率降序）")
    print("="*80)

    # 按准确率降序排序
    sorted_results = sorted(results, key=lambda x: x["accuracy"], reverse=True)

    for res in sorted_results:
        stem = res["file"].stem
        # 从右往左解析，或者匹配已知的数据集后缀
        if stem.endswith("_val_hard"):
            dataset = "val_hard"
            # 去掉后缀，剩下的就是策略名
            strategy = stem[:-9]
        elif stem.endswith("_val"):
            dataset = "val"
            strategy = stem[:-4]
        else:
            # 如果不是标准后缀，则假设最后一个下划线后是数据集
            parts = stem.rsplit("_", 1)
            if len(parts) == 2:
                strategy = parts[0]
                dataset = parts[1]
            else:
                strategy = stem
                dataset = "unknown"

        print(f"文件: {res['file'].name:<35} | "
              f"数据集: {dataset:<8} | " 
              f"策略: {strategy:<25} | " 
              f"准确率: {res['accuracy']*100:5.2f}% | "
              f"正确/总数: {res['correct']}/{res['total']}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="评估 ARC 任务预测准确率（支持 --pred all 批量）")
    parser.add_argument("--val", default="data/val.jsonl",
                        help="默认 val.jsonl；若评估 hard 数据集请手动指定（如 data/val_hard.jsonl）")
    parser.add_argument("--pred", type=str, default=None,
                        help="单个预测结果 JSON 文件路径，或 'all' 表示批量评估 results/ 目录下所有文件")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="结果目录（批量模式时使用，默认 results/）")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if args.pred == "all":
        if not results_dir.exists():
            raise FileNotFoundError(f"结果目录不存在: {results_dir}")

        pred_files = sorted(results_dir.glob("*.json"))
        if not pred_files:
            print("results/ 目录下没有找到任何 *.json 文件")
            return

        all_stats = []
        for pred_file in pred_files:
            print(f"\n正在评估: {pred_file.name}")

            val_path = "data/val_hard.jsonl" if "hard" in pred_file.name.lower() else args.val
            val_data = load_val_data(val_path)

            preds = load_predictions(pred_file)
            stats = evaluate_single(val_data, preds, pred_file)
            print(f"准确率: {stats['accuracy']*100:.2f}% ({stats['correct']}/{stats['total']})")
            if stats['parse_failed'] > 0:
                print(f"解析失败: {stats['parse_failed']} 个")

            all_stats.append(stats)

        print_summary(all_stats)

    elif args.pred:
        pred_path = Path(args.pred)
        if not pred_path.exists():
            raise FileNotFoundError(f"预测文件不存在: {pred_path}")

        val_data = load_val_data(args.val)
        preds = load_predictions(pred_path)
        stats = evaluate_single(val_data, preds, pred_path)

        print("\n====== 单个评估结果 ======")
        print(f"文件          : {pred_path.name}")
        print(f"总任务数      : {stats['total']}")
        print(f"正确任务数    : {stats['correct']}")
        print(f"解析失败数    : {stats['parse_failed']}")
        print(f"错误任务 ID   : {stats['wrong_tasks']}")
        print(f"准确率        : {stats['accuracy']:.4f} ({stats['accuracy']*100:.2f}%)")

    else:
        raise ValueError("请指定 --pred <文件路径> 或 --pred all 进行批量评估")


if __name__ == "__main__":
    main()