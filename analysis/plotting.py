# analysis/plotting.py
import matplotlib.pyplot as plt
import numpy as np

def smooth(y, window=20):
    if len(y) < window:
        return y
    gaussian = np.exp(-np.linspace(-2, 2, window)**2)
    gaussian /= gaussian.sum()
    return np.convolve(y, gaussian, mode='same')


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