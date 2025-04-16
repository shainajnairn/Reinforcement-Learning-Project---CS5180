import matplotlib.pyplot as plt
import pickle
import math
import numpy as np
from sklearn.linear_model import LinearRegression


def subsample_with_indices(lst, step=1000):
    """
    Return a tuple of the subsampled list and the corresponding indices from the original list.
    """
    indices = np.arange(0, len(lst), step)
    return lst[::step], indices


def compare_episode_rewards(
    rewards1,
    x1,
    rewards2,
    x2,
    rewards3,
    x3,
    label1="Agent 1",
    label2="Agent 2",
    label3="Agent 3",
    filename="episode_rewards.png",
):
    """
    Compare three sets of episodic rewards over training and plot a linear regression
    trend line for each model using the provided episode indices.
    Saves the figure to the specified filename.
    """
    fig, ax = plt.subplots()

    # Plot the raw episode rewards using the provided x-axis values
    ax.plot(x1, rewards1, label=label1)
    ax.plot(x2, rewards2, label=label2)
    ax.plot(x3, rewards3, label=label3)

    # Plot linear regression trends for Model 1
    if len(rewards1) >= 2:
        X1 = x1.reshape(-1, 1)
        y1 = np.array(rewards1).reshape(-1, 1)
        reg1 = LinearRegression().fit(X1, y1)
        x1_pred = np.linspace(x1[0], x1[-1], 100).reshape(-1, 1)
        y1_pred = reg1.predict(x1_pred)
        ax.plot(x1_pred, y1_pred, "--", label=f"{label1} trend")

    # Plot linear regression trends for Model 2
    if len(rewards2) >= 2:
        X2 = x2.reshape(-1, 1)
        y2 = np.array(rewards2).reshape(-1, 1)
        reg2 = LinearRegression().fit(X2, y2)
        x2_pred = np.linspace(x2[0], x2[-1], 100).reshape(-1, 1)
        y2_pred = reg2.predict(x2_pred)
        ax.plot(x2_pred, y2_pred, "--", label=f"{label2} trend")

    # Plot linear regression trends for Model 3
    if len(rewards3) >= 2:
        X3 = x3.reshape(-1, 1)
        y3 = np.array(rewards3).reshape(-1, 1)
        reg3 = LinearRegression().fit(X3, y3)
        x3_pred = np.linspace(x3[0], x3[-1], 100).reshape(-1, 1)
        y3_pred = reg3.predict(x3_pred)
        ax.plot(x3_pred, y3_pred, "--", label=f"{label3} trend")

    ax.set_title("Comparison of Episode Rewards")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()

    plt.savefig(filename)
    plt.close(fig)


def compare_episode_lengths(
    lengths1,
    x1,
    lengths2,
    x2,
    lengths3,
    x3,
    label1="Agent 1",
    label2="Agent 2",
    label3="Agent 3",
    filename="episode_lengths.png",
):
    """
    Compare three sets of episode lengths over training and plot a linear regression trend
    line for each model using the provided x-axis values.
    Saves the figure to the specified filename.
    """
    fig, ax = plt.subplots()

    ax.plot(x1, lengths1, label=label1)
    ax.plot(x2, lengths2, label=label2)
    ax.plot(x3, lengths3, label=label3)

    if len(lengths1) >= 2:
        X1 = x1.reshape(-1, 1)
        y1 = np.array(lengths1).reshape(-1, 1)
        reg1 = LinearRegression().fit(X1, y1)
        x1_pred = np.linspace(x1[0], x1[-1], 100).reshape(-1, 1)
        y1_pred = reg1.predict(x1_pred)
        ax.plot(x1_pred, y1_pred, "--", label=f"{label1} trend")

    if len(lengths2) >= 2:
        X2 = x2.reshape(-1, 1)
        y2 = np.array(lengths2).reshape(-1, 1)
        reg2 = LinearRegression().fit(X2, y2)
        x2_pred = np.linspace(x2[0], x2[-1], 100).reshape(-1, 1)
        y2_pred = reg2.predict(x2_pred)
        ax.plot(x2_pred, y2_pred, "--", label=f"{label2} trend")

    if len(lengths3) >= 2:
        X3 = x3.reshape(-1, 1)
        y3 = np.array(lengths3).reshape(-1, 1)
        reg3 = LinearRegression().fit(X3, y3)
        x3_pred = np.linspace(x3[0], x3[-1], 100).reshape(-1, 1)
        y3_pred = reg3.predict(x3_pred)
        ax.plot(x3_pred, y3_pred, "--", label=f"{label3} trend")

    ax.set_title("Comparison of Episode Lengths")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length")
    ax.legend()

    plt.savefig(filename)
    plt.close(fig)


def compare_best_rewards(
    best_reward_1,
    best_reward_2,
    best_reward_3,
    label1="Agent 1",
    label2="Agent 2",
    label3="Agent 3",
    filename="best_rewards.png",
):
    """
    Compare the best (maximum) single-episode rewards from three training runs.
    Saves the figure to the specified filename.
    """
    fig, ax = plt.subplots()

    x_positions = np.array([0, 1, 2])
    best_rewards = [best_reward_1, best_reward_2, best_reward_3]
    labels = [label1, label2, label3]

    ax.bar(x_positions, best_rewards, tick_label=labels)
    ax.set_title("Comparison of Best Episode Rewards")
    ax.set_ylabel("Best Reward")

    plt.savefig(filename)
    plt.close(fig)


def compare_convergence_timestep(
    conv_timestep_1,
    conv_timestep_2,
    conv_timestep_3,
    label1="Agent 1",
    label2="Agent 2",
    label3="Agent 3",
    filename="convergence_timestep.png",
):
    """
    Compare the timesteps at which each model converged.
    Saves the figure to the specified filename.
    """
    fig, ax = plt.subplots()

    x_positions = np.array([0, 1, 2])
    # Use 0 or another default if a convergence timestep is None
    val1 = conv_timestep_1 if conv_timestep_1 is not None else 0
    val2 = conv_timestep_2 if conv_timestep_2 is not None else 0
    val3 = conv_timestep_3 if conv_timestep_3 is not None else 0
    conv_values = [val1, val2, val3]
    labels = [label1, label2, label3]

    ax.bar(x_positions, conv_values, tick_label=labels)
    ax.set_title("Comparison of Convergence Timesteps")
    ax.set_ylabel("Timestep of Convergence")

    plt.savefig(filename)
    plt.close(fig)


# --- Load your data for three models ---
# Make sure that you have the appropriate metric files for all three models.
with open("saved_metrics/metrics_dqn.pkl", "rb") as f:
    metrics_model1 = pickle.load(f)
with open("saved_metrics/metrics_ppo_gae.pkl", "rb") as f:
    metrics_model2 = pickle.load(f)
with open("saved_metrics/metrics_ppo_safe.pkl", "rb") as f:
    metrics_model3 = pickle.load(f)

# Labels for your models
label1 = "DQN Model"
label2 = "PPO With GAE"
label3 = "Safe PPO"

# Optionally, print data lengths to check
print("Model 1 rewards length:", len(metrics_model1["episode_rewards"]))
print("Model 2 rewards length:", len(metrics_model2["episode_rewards"]))
print("Model 3 rewards length:", len(metrics_model3["episode_rewards"]))

# --- Subsample the episode rewards and lengths for smoother plots ---
# The x-values now reflect the true episode numbers from the original data
rewards1, x1 = subsample_with_indices(metrics_model1["episode_rewards"], 1000)
rewards2, x2 = subsample_with_indices(metrics_model2["episode_rewards"], 1000)
rewards3, x3 = subsample_with_indices(metrics_model3["episode_rewards"], 1000)

lengths1, xl1 = subsample_with_indices(metrics_model1["episode_lengths"], 1000)
lengths2, xl2 = subsample_with_indices(metrics_model2["episode_lengths"], 1000)
lengths3, xl3 = subsample_with_indices(metrics_model3["episode_lengths"], 1000)

# --- Generate and save each plot in its own file ---

compare_episode_rewards(
    rewards1,
    x1,
    rewards2,
    x2,
    rewards3,
    x3,
    label1=label1,
    label2=label2,
    label3=label3,
    filename="plot_episode_rewards.png",
)

compare_episode_lengths(
    lengths1,
    xl1,
    lengths2,
    xl2,
    lengths3,
    xl3,
    label1=label1,
    label2=label2,
    label3=label3,
    filename="plot_episode_lengths.png",
)

compare_best_rewards(
    best_reward_1=metrics_model1["best_reward"],
    best_reward_2=metrics_model2["best_reward"],
    best_reward_3=metrics_model3["best_reward"],
    label1=label1,
    label2=label2,
    label3=label3,
    filename="plot_best_rewards.png",
)

# You can uncomment the following section if you want to plot the convergence timesteps as well.
# compare_convergence_timestep(
#     conv_timestep_1=metrics_model1["convergence_timestep"],
#     conv_timestep_2=metrics_model2["convergence_timestep"],
#     conv_timestep_3=metrics_model3["convergence_timestep"],
#     label1=label1,
#     label2=label2,
#     label3=label3,
#     filename="plot_convergence_timestep.png",
# )
