import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

from ortools.algorithms.python import knapsack_solver


GROUPS = [
    "00Uncorrelated",
    "01WeaklyCorrelated",
    "02StronglyCorrelated",
    "03InverseStronglyCorrelated",
    "04AlmostStronglyCorrelated",
    "05SubsetSum",
    "06UncorrelatedWithSimilarWeights",
    "07SpannerUncorrelated",
    "08SpannerWeaklyCorrelated",
    "09SpannerStronglyCorrelated",
    "10MultipleStronglyCorrelated",
    "11ProfitCeiling",
    "12Circle",
]

TIME_LIMIT_SEC = 300  # 5 minutes


def parse_kp_file(filepath: str):
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    n = int(lines[0])
    c = int(lines[1])
    profits = []
    weights = []
    for i in range(2, 2 + n):
        parts = lines[i].split()
        profits.append(int(parts[0]))
        weights.append(int(parts[1]))
    return n, c, profits, weights


def solve_instance(filepath: str):
    n, capacity, profits, weights = parse_kp_file(filepath)

    solver = knapsack_solver.KnapsackSolver(
        knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
        "KnapsackSolver",
    )
    solver.set_time_limit(TIME_LIMIT_SEC)
    solver.init(profits, [weights], [capacity])

    start_time = time.time()
    computed_value = solver.solve()
    solve_time = time.time() - start_time

    selected_indices = []
    total_weight = 0
    for i in range(n):
        if solver.best_solution_contains(i):
            selected_indices.append(i)
            total_weight += weights[i]

    is_optimal = solve_time < TIME_LIMIT_SEC
    status = "COMPLETED" if is_optimal else "TIME_LIMIT"

    capacity_utilization = total_weight / capacity if capacity > 0 else 0.0

    return {
        "n": n,
        "capacity": capacity,
        "computed_value": computed_value,
        "total_weight": total_weight,
        "solve_time": solve_time,
        "is_optimal": is_optimal,
        "status": status,
        "selected_indices": selected_indices,
        "capacity_utilization": capacity_utilization,
        "items_selected_count": len(selected_indices),
        "selection_density": len(selected_indices) / n if n > 0 else 0.0,
        "unused_capacity": capacity - total_weight,
    }


def build_result(filepath: str, result: dict, group_name: str, range_cat: str, R_value: int):
    rel_path = Path(filepath).relative_to(Path(__file__).parent / "kplib").as_posix()
    filename = os.path.basename(filepath)

    return {
        "test_case_name": filename,
        "instance_metadata": {
            "n_items": result["n"],
            "coefficient_range": R_value,
            "range_category": range_cat,
            "capacity": result["capacity"],
            "input_file_path": f"./kplib/{rel_path}",
        },
        "solver_configuration": {
            "solver_name": "OR-Tools KnapsackSolver",
            "algorithm": "KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER",
            "time_limit_sec": float(TIME_LIMIT_SEC),
            "language": "Python",
        },
        "performance_results": {
            "solve_time_sec": result["solve_time"],
            "is_optimal": result["is_optimal"],
            "total_profit": result["computed_value"],
            "total_weight": result["total_weight"],
            "status": result["status"],
        },
        "analytical_metrics": {
            "capacity_utilization_ratio": round(result["capacity_utilization"], 5),
            "unused_capacity": result["unused_capacity"],
            "items_selected_count": result["items_selected_count"],
            "selection_density": round(result["selection_density"], 6),
        },
        "solution_data": {
            "selected_indices": result["selected_indices"],
        },
    }


def solve_task(task: dict):
    """Worker function for multiprocessing. Solves one knapsack instance and writes result."""
    kp_file = task["kp_file"]
    out_file = task["out_file"]
    group_name = task["group_name"]
    range_cat = task["range_cat"]
    R_value = task["R_value"]
    kplib_rel = task["kplib_rel"]

    try:
        result = solve_instance(kp_file)
        output = build_result(kp_file, result, group_name, range_cat, R_value)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  [{group_name}] {kplib_rel} -> {result['status']} in {result['solve_time']:.4f}s, profit={result['computed_value']}")
        return {"file": kplib_rel, "status": result["status"]}
    except Exception as e:
        print(f"  [{group_name}] {kplib_rel} -> ERROR: {e}")
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        error_output = {"test_case_name": os.path.basename(kp_file), "error": str(e)}
        with open(out_file, "w") as f:
            json.dump(error_output, f, indent=2)
        return {"file": kplib_rel, "status": "ERROR", "error": str(e)}


def collect_tasks(groups: list[str], kplib_dir: Path, experiments_dir: Path, range_filter: str | None = None) -> list[dict]:
    """Collect all tasks to run across all specified groups."""
    tasks = []
    for group_name in groups:
        group_dir = kplib_dir / group_name
        if not group_dir.exists():
            print(f"Group directory not found: {group_dir}")
            continue

        for n_dir in sorted(group_dir.iterdir()):
            if not n_dir.is_dir():
                continue
            for r_dir in sorted(n_dir.iterdir()):
                if not r_dir.is_dir():
                    continue
                range_cat = r_dir.name
                if range_filter and range_cat != range_filter:
                    continue
                R_value = int(range_cat[1:])

                kp_files = sorted(r_dir.glob("*.kp"))
                for kp_file in kp_files:
                    rel_from_group = kp_file.relative_to(group_dir)
                    out_dir = experiments_dir / group_name / rel_from_group.parent
                    out_file = out_dir / (kp_file.stem + ".json")

                    if out_file.exists():
                        continue

                    kplib_rel = str(kp_file.relative_to(kplib_dir))
                    tasks.append({
                        "kp_file": str(kp_file),
                        "out_file": str(out_file),
                        "group_name": group_name,
                        "range_cat": range_cat,
                        "R_value": R_value,
                        "kplib_rel": kplib_rel,
                    })
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Run OR-Tools knapsack solver on kplib instances")
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        choices=GROUPS,
        help="Specific test group to run. If not specified, runs all groups.",
    )
    parser.add_argument(
        "--range",
        type=st
        default=None,
        choices=["R01000", "R10000"],
        dest="range_filter",
        help="Filter by range category (R01000 or R10000). If not specified, runs both.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers. Defaults to number of CPU cores.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    kplib_dir = base_dir / "kplib"
    experiments_dir = base_dir / "experiments"
    experiments_dir.mkdir(exist_ok=True)

    groups_to_run = [args.group] if args.group else GROUPS
    num_workers = args.workers or multiprocessing.cpu_count()

    tasks = collect_tasks(groups_to_run, kplib_dir, experiments_dir, args.range_filter)
    total = len(tasks)

    if total == 0:
        print("No new tasks to run (all results already exist or no matching files found).")
        return

    print(f"Found {total} instances to solve using {num_workers} workers.")
    print(f"{'='*60}")

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(solve_task, tasks)

    completed = sum(1 for r in results if r["status"] == "COMPLETED")
    timed_out = sum(1 for r in results if r["status"] == "TIME_LIMIT")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    print(f"\n{'='*60}")
    print(f"Done. Total: {total} | Completed: {completed} | Time-limited: {timed_out} | Errors: {errors}")


if __name__ == "__main__":
    main()
