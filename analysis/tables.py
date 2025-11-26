import numpy as np


def write_summary_md(results, save_path):
    """
    results = [
        (name, success_rewards, all_rewards, reflections),
        ...
    ]
    Produces a Markdown README with tables instead of plaintext,
    including rankings by successes and average success reward.
    """

    summary = []

    # Collect stats
    for name, success, rewards, refl in results:
        avg_reward = np.mean(rewards)
        avg_success_reward = np.mean(success) if len(success) > 0 else 0
        successes = len(success)
        total_refl = np.sum(refl)
        avg_refl = np.mean(refl)

        summary.append({
            "name": name,
            "avg_reward": avg_reward,
            "avg_success_reward": avg_success_reward,
            "successes": successes,
            "total_refl": total_refl,
            "avg_refl": avg_refl
        })

    # -----------------------------
    # Overview Table
    # -----------------------------
    lines = []
    lines.append("# Experiment Summary\n")
    lines.append("## Overview\n")

    lines.append("| Agent | Successes | Avg Reward | Avg Success Reward | Total Reflections | Avg Reflections/Episode |")
    lines.append("|-------|-----------|------------|--------------------|-------------------|--------------------------|")

    for s in summary:
        lines.append(
            f"| {s['name']} | {s['successes']} | {s['avg_reward']:.2f} | "
            f"{s['avg_success_reward']:.2f} | {s['total_refl']} | {s['avg_refl']:.2f} |"
        )

    # -----------------------------
    # Rankings Section
    # -----------------------------
    lines.append("\n## Rankings\n")

    # Sort by successes descending
    sorted_by_success = sorted(summary, key=lambda x: x["successes"], reverse=True)
    lines.append("### Ranked by Number of Successful Episodes\n")
    lines.append("| Rank | Agent | Successful Episodes |")
    lines.append("|------|-------|--------------------|")
    for i, s in enumerate(sorted_by_success, 1):
        lines.append(f"| {i} | {s['name']} | {s['successes']} |")

    # Sort by average success reward descending
    sorted_by_avg_success_reward = sorted(summary, key=lambda x: x["avg_success_reward"], reverse=True)
    lines.append("\n### Ranked by Average Reward of Successful Episodes\n")
    lines.append("| Rank | Agent | Avg Successful Reward |")
    lines.append("|------|-------|----------------------|")
    for i, s in enumerate(sorted_by_avg_success_reward, 1):
        lines.append(f"| {i} | {s['name']} | {s['avg_success_reward']:.2f} |")

    # Write file
    with open(save_path, "w") as f:
        f.write("\n".join(lines))