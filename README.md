# Self-Reflection Pathfinding

## Abstract

This project investigates whether a Q-learning agent can improve its pathfinding behaviour by adding an explicit self-reflection mechanism during training. The experiment compares a baseline agent with reflection-enabled agents that incur different reflection costs. The central question is whether self-reflection helps the agent reach the goal more reliably, adapt more efficiently, and produce better paths over time.

## Introduction

Pathfinding is a classic planning problem in reinforcement learning, but many agents still struggle when the environment is partially unfamiliar or contains obstacles that require adaptation. This experiment explores a simple hypothesis: if an agent is allowed to pause and reflect on its recent behaviour, it may learn to adjust its strategy more effectively than an agent that only reacts greedily to immediate rewards.

The motivation is to test whether reflection is beneficial in a controlled gridworld setting and to study how the cost of reflection changes the outcome. In practical terms, the experiment asks whether reflection is useful only when it is cheap, whether it becomes noisy when it is too frequent, or whether moderate reflection cost yields the best balance.

## Methods

### Environment

The environment is a gridworld with obstacles, a fixed start state, and a goal state. The agent must learn a policy that reaches the goal while avoiding blocked cells. The environment is implemented in [experiment/gridworld.py](experiment/gridworld.py).

### Agent

The learning agent is a Q-learning implementation in [agents/q_agent.py](agents/q_agent.py). The reflection-enabled variant uses an adaptive reflection strategy from [agents/reflection/adaptive_reflection.py](agents/reflection/adaptive_reflection.py), with reflection costs defined in [agents/reflection/reflection_costs.py](agents/reflection/reflection_costs.py).

### Experimental Conditions

The study compares four conditions:

- Baseline: no reflection
- Low-cost reflection
- Medium-cost reflection
- High-cost reflection

The main experiment loop is run through [main.py](main.py), which generates the environment, trains each agent, records rewards and reflections, and saves plots and summaries.

## Results

The saved experiment summaries in [results/](results/) show that reflection generally improves the agent’s ability to succeed in harder environments. The baseline agent performs poorly in comparison, while the reflection-enabled agents produce many more successful episodes.

### Summary of Findings

- Reflection improves task success relative to the baseline in the harder settings.
- Medium- and high-cost reflection often produce the strongest balance between success and efficiency.
- Low-cost reflection tends to use reflection very frequently, which can increase adaptation but also raise the cost of the process.
- The results suggest that reflection is helpful, but its usefulness depends on the price of using it.

### Visual Results

#### Reward and Success Plots

Representative plots from the experiment runs are stored under [results/](results/).

- [results/hard_1/summary.md](results/hard_1/summary.md)
- [results/hard_2/summary.md](results/hard_2/summary.md)
- [results/hard_3/summary.md](results/hard_3/summary.md)
- [results/hard_4a/summary.md](results/hard_4a/summary.md)
- [results/hard_4b/summary.md](results/hard_4b/summary.md)

#### Path Evolution Videos

The repository includes animated comparisons of the agent’s path evolution for the different reflection settings:

- [results/video_demo/baseline_path_evolution.gif](results/video_demo/baseline_path_evolution.gif)
- [results/video_demo/low_path_evolution.gif](results/video_demo/low_path_evolution.gif)
- [results/video_demo/medium_path_evolution.gif](results/video_demo/medium_path_evolution.gif)
- [results/video_demo/high_path_evolution.gif](results/video_demo/high_path_evolution.gif)

These animations highlight the first successful path, the latest successful path, and the nodes explored over time so that the improvement process is easier to observe.

## Discussion

The experiment suggests that reflection is not simply a free improvement. Instead, it behaves like a resource that must be used carefully. When reflection is too cheap, the agent may overuse it; when it is too expensive, the agent may underuse it. The most promising results appear in the middle of that spectrum, where reflection provides adaptation without overwhelming the learning process.

This supports the broader idea that self-reflection can be beneficial in reinforcement learning, provided that the reflection mechanism is controlled by a realistic cost and is used selectively.

## Limitations

This work uses a relatively simple gridworld environment and a basic Q-learning setup. As a result, the findings should be interpreted as an initial proof-of-concept rather than a definitive statement about reflection in all planning domains. Future work could extend this study to larger environments, different reward schemes, or more sophisticated reflection policies.

## Reproducibility

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the main experiment:

```bash
python main.py
```

Generate the comparison animations:

```bash
python generate_video_demo.py
```

All generated summaries and plots are stored under [results/](results/), and the animated outputs are saved in [results/video_demo/](results/video_demo/).

## Repository Structure

- [main.py](main.py): entry point for the experiment
- [experiment/](experiment/): environment and runner logic
- [agents/](agents/): Q-learning agent and reflection strategies
- [analysis/](analysis/): plotting and report generation
- [results/](results/): saved experiment summaries, plots, and videos
- [generate_video_demo.py](generate_video_demo.py): script for generating comparison animations
