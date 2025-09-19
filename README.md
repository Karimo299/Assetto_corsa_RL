# Assetto Corsa F1 Bot 🏎️

A reinforcement learning agent built to drive an F1 car in Assetto Corsa.  
The bot uses ray-casting sensors, throttle and steering control, and a custom reward system to complete laps as efficiently as possible.  

Currently, the **Soft Actor-Critic (SAC)** algorithm achieves the best lap performance.  

## Demo Video 🎥
👉 [Watch the YouTube Demo](https://youtu.be/xvmljSweza0)

## Features
- Ray-cast sensor system for track awareness  
- Adaptive throttle and steering control  
- Reward shaping for speed, stability, and racing line  
- Trained with state-of-the-art RL algorithms (SAC, TD3, PPO comparisons)  

## Requirements
- Python 3.10+  
- PyTorch  
- Assetto Corsa + Custom Shaders Patch (for memory-mapped I/O)  

