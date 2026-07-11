# analysis/plotting.py
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import subprocess
import shutil
import numpy as np


def smooth(y, window=20):
    if len(y) < window:
        return y
    gaussian = np.exp(-np.linspace(-2, 2, window)**2)
    gaussian /= gaussian.sum()
    return np.convolve(y, gaussian, mode='same')


def _build_grid_image(env, current_path=None):
    grid = np.zeros((env.size, env.size, 3))
    grid[:] = [1, 1, 1]

    for ox, oy in env.obstacles:
        grid[ox, oy] = [1, 0, 0]

    gx, gy = env.goal_pos
    grid[gx, gy] = [0, 1, 0]

    sx, sy = env.start_pos
    grid[sx, sy] = [0, 0, 1]

    if current_path and len(current_path) > 0:
        fx, fy = current_path[-1]
        grid[fx, fy] = [0.2, 0.2, 0.2]

    return grid


def plot_path_evolution_video(env, successful_paths, save=None, fps=6, interval=500, pause_frames=8):
    if not successful_paths:
        raise ValueError("No successful paths were recorded to animate.")

    fig, ax = plt.subplots(figsize=(6, 6))
    base_grid = _build_grid_image(env)
    image = ax.imshow(base_grid, interpolation="nearest")
    ax.set_title("Path Optimization Over Successful Episodes")
    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(env.size - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, env.size, 1))
    ax.set_yticks(np.arange(-0.5, env.size, 1))
    ax.grid(True, color="black", linewidth=0.5)

    first_line, = ax.plot([], [], color="red", linewidth=3, marker="o", markersize=6, zorder=5)
    last_line, = ax.plot([], [], color="cyan", linewidth=3, marker="o", markersize=6, zorder=5)
    current_line, = ax.plot([], [], color="blue", linewidth=2, marker="o", markersize=4, zorder=4)
    history_lines = []
    expanded_scatter = ax.scatter([], [], s=70, color="orange", edgecolor="black", alpha=0.75, zorder=3)
    text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                   fontsize=9, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    def update(frame):
        for line in history_lines:
            line.remove()
        history_lines.clear()

        visible_paths = successful_paths[: frame + 1]
        for idx, path in enumerate(visible_paths):
            if len(path) < 2:
                continue
            xs = [p[1] for p in path]
            ys = [p[0] for p in path]
            if idx == 0:
                color = "red"
                alpha = 0.25
                linewidth = 1.6
            elif idx == len(visible_paths) - 1:
                color = "cyan"
                alpha = 0.35
                linewidth = 1.8
            else:
                color = "gray"
                alpha = 0.12
                linewidth = 0.8
            line = ax.plot(xs, ys, color=color, alpha=alpha, linewidth=linewidth)
            history_lines.append(line[0])

        first_path = visible_paths[0] if visible_paths else None
        current_path = visible_paths[-1] if visible_paths else []
        last_path = current_path

        if first_path is not None and len(first_path) >= 2:
            xs = [p[1] for p in first_path]
            ys = [p[0] for p in first_path]
            first_line.set_data(xs, ys)
        else:
            first_line.set_data([], [])

        if last_path is not None and len(last_path) >= 2:
            xs = [p[1] for p in last_path]
            ys = [p[0] for p in last_path]
            last_line.set_data(xs, ys)
        else:
            last_line.set_data([], [])

        if len(current_path) >= 2:
            xs = [p[1] for p in current_path]
            ys = [p[0] for p in current_path]
            current_line.set_data(xs, ys)
        else:
            current_line.set_data([], [])

        expanded_nodes = set()
        for path in visible_paths:
            expanded_nodes.update(path)
        if expanded_nodes:
            nodes = np.array(list(expanded_nodes), dtype=float)
            expanded_scatter.set_offsets(np.column_stack((nodes[:, 1], nodes[:, 0])))
        else:
            expanded_scatter.set_offsets([])

        current_grid = _build_grid_image(env, current_path)
        image.set_array(current_grid)
        episode_number = frame + 1
        text.set_text(
            f"Episode {episode_number} | First path (red) | Latest path (cyan) | Explored nodes: {len(expanded_nodes)}"
        )
        return [image, first_line, last_line, current_line, expanded_scatter, text, *history_lines]

    frame_count = len(successful_paths)
    full_frames = list(range(frame_count))
    intro_frames = [-1] * pause_frames
    outro_frames = [frame_count] * pause_frames
    animation_frames = intro_frames + full_frames + outro_frames

    def update_with_pause(frame):
        if frame < 0:
            return update(0)
        if frame >= frame_count:
            return update(frame_count - 1)
        return update(frame)

    anim = FuncAnimation(fig, update_with_pause, frames=animation_frames, interval=interval, blit=False)

    if save:
        directory = os.path.dirname(save)
        if directory:
            os.makedirs(directory, exist_ok=True)

        save_lower = save.lower()
        if save_lower.endswith(".mp4"):
            ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path:
                try:
                    anim.save(save, writer="ffmpeg", fps=fps)
                except Exception:
                    gif_path = save.rsplit(".", 1)[0] + ".gif"
                    anim.save(gif_path, writer=PillowWriter(fps=fps))
                    print(f"ffmpeg failed; saved animation to {gif_path}")
            else:
                gif_path = save.rsplit(".", 1)[0] + ".gif"
                anim.save(gif_path, writer=PillowWriter(fps=fps))
                print(f"ffmpeg not installed; saved animation to {gif_path}")
        else:
            anim.save(save, writer=PillowWriter(fps=fps))

    plt.close(fig)
    return save


def plot_reflection_comparison_videos(env, runs, save_dir=None, fps=6, interval=500, pause_frames=8):
    """
    runs = [
        ("baseline", successful_paths_baseline),
        ("low", successful_paths_low),
        ("medium", successful_paths_medium),
        ("high", successful_paths_high),
    ]
    """
    if save_dir is None:
        save_dir = "results"

    os.makedirs(save_dir, exist_ok=True)
    for name, successful_paths in runs:
        output_path = os.path.join(save_dir, f"{name}_path_evolution.mp4")
        plot_path_evolution_video(
            env,
            successful_paths,
            save=output_path,
            fps=fps,
            interval=interval,
            pause_frames=pause_frames,
        )


# -------------------------------------------------
# Reward Over Time (scatter + smooth)
# -------------------------------------------------
def plot_reward_over_time(results, save=None):
    """
    results = [
        ("Baseline", rewards_list, smoothed_color, scatter_color),
        ("Adaptive Low", rewards_list, color, color),
        ...
    ]
    """

    plt.figure(figsize=(10,5))

    for name, rewards, color in results:
        # x = range(len(rewards))
        # plt.scatter(x, rewards, alpha=0.3, s=10, label=name, color=color)
        plt.plot(smooth(rewards), linewidth=2, color=color, label=name)

    plt.title("Reward Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout(rect=[0,0,0.93,1])

    if save:
        plt.savefig(save)
    plt.show()

# -------------------------------------------------
# Successful Reward Over Time (scatter + smooth)
# -------------------------------------------------
def plot_successful_rewards_over_time(results, save=None):
    """
    Plot rewards only for successful episodes over time.

    results = [
        ("Baseline", success_rewards_list, color),
        ("Adaptive Low", success_rewards_list, color),
        ...
    ]
    """

    plt.figure(figsize=(10,5))

    for name, success_rewards, color in results:
        # x = range(len(success_rewards))
        # plt.scatter(x, success_rewards, alpha=0.3, s=10, label=name, color=color)
        plt.plot(smooth(success_rewards), linewidth=2, color=color, label=name)

    plt.title("Successful Episode Rewards Over Time")
    plt.xlabel("Successful Episode Number")
    plt.ylabel("Reward")

    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout(rect=[0,0,0.93,1])

    if save:
        plt.savefig(save)
    plt.show()



# -------------------------------------------------
# Reflection Over Time
# -------------------------------------------------
def plot_reflections_over_time(results, save=None):
    """
    results = [
        ("Baseline", reflections_list, color),
        ("Adaptive Low", reflections_list, color),
        ...
    ]

    reflections_list must be length = number of episodes,
    each entry = reflections used in that episode.
    """

    plt.figure(figsize=(10,5))

    for name, refl, color in results:
        # x = range(len(refl))
        # plt.scatter(x, refl, alpha=0.3, s=10, color=color, label=name)
        plt.plot(smooth(refl), linewidth=2, color=color, label=name)

    plt.title("Reflections Used Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Reflections (this episode)")
    plt.yscale("log")

    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout(rect=[0,0,0.93,1])

    if save:
        plt.savefig(save)
    plt.show()


# -------------------------------------------------
# Successful episodes count bar chart
# -------------------------------------------------
def plot_success_counts(success_stats, save=None):
    names = [s[0] for s in success_stats]
    counts = [s[1] for s in success_stats]
    colors = [s[2] for s in success_stats]

    plt.figure(figsize=(10,5))
    plt.bar(names, counts, color=colors)
    plt.title("Successful Episodes")
    plt.ylabel("Count")
    plt.xticks(rotation=30)

    if save:
        plt.savefig(save)
    plt.show()

# -------------------------------------------------
# Alpha Over Time
# -------------------------------------------------
def plot_alpha_over_time(results, save=None):
    plt.figure(figsize=(10,5))

    for name, alphas, epsilons, color in results:
        plt.plot(smooth(alphas), label=f"{name} - Alpha", color=color)


    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title("Alpha Over Time")
    plt.legend()

    if save:
        plt.savefig(save)
    plt.show()

# -------------------------------------------------
# Epsilon Over Time
# -------------------------------------------------
def plot_epsilon_over_time(results, save=None):
    plt.figure(figsize=(10,5))

    for name, alphas, epsilons, color in results:
        plt.plot(smooth(epsilons), label=f"{name} - Epsilon", color=color)


    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title("Epsilon Over Time")
    plt.legend()

    if save:
        plt.savefig(save)
    plt.show()