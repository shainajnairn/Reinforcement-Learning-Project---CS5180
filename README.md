# Reinforcement-Learning-Project---CS5180
2D Rocket Landing Environment:  Reinforcement Learning Driven Rocket Landing  Simulation

## Introduction & Motivation
Rocket landing is a challenging control problem that demands precise maneuvering under potentially unpredictable conditions. Traditional feedback control systems often rely on carefully tuned controllers and linearized models, which may fail when dynamics become highly nonlinear or when unforeseen external factors occur (e.g., gusting winds, variable thrust). Reinforcement Learning (RL) offers a flexible framework for learning robust, adaptive policies directly from simulated experience.

## Author
**Nivedita Shainaj Nair**  
Northeastern University  
shainajnair.n@northeastern.edu

**Harshavardhan Manohar**  
Northeastern University  
m.h@northeastern.edu

## Project Objectives

### 1. Develop & Enhance a 2D Rocket Landing Environment
- Provide a simplified but realistic approximation of rigid-body rocket dynamics
- Include tasks such as hovering at a target altitude and performing safe landings

### 2. Compare Two RL Algorithms
- **Algorithm A**: Proximal Policy Optimization (PPO) - a strong baseline for continuous control
- **Algorithm B**: One of the following advanced methods, depending on research interest:
  - Safe (Constrained) PPO, or
  - Recurrent PPO
- Quantitatively evaluate their performance, sample efficiency, and robustness to environment variations

### 3. Incorporate Domain Randomization & Additional Features
- Improve the environment's generality by adding variability (e.g., wind, random initial states) and new landing conditions (e.g., multiple pads, variable terrain)

## Project Structure
