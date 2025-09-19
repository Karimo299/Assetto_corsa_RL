# Assetto Corsa F1 Bot 🏎️

A reinforcement learning agent built to drive an F1 car in Assetto Corsa.  
The bot uses ray-casting sensors, throttle and steering control, and a custom reward system to complete laps as efficiently as possible.  

Currently, the **Soft Actor-Critic (SAC)** algorithm achieves the best lap performance.  

## Demo Video 🎥
👉 [Watch the YouTube Demo](https://youtube.com/your-video-link-here)

## Description
I wanted to make a reinforcement learning agent that doesn’t have an edge over a human in terms of inputs.  
The agent only sees what a dedicated human would — things like visible track distance and car state — with no hidden or unfair information.  
This forces it to learn the track, discover the optimal racing line, and improve lap times just like a human would.  

## Goals
- Experiment further with reward functions, hyperparameters, and RL algorithms to push lap times lower  
- Achieve a **1:10 lap time or better** to beat the built-in AI  
- Generalize performance so the bot can drive well on **any track**  

## Features
- Ray-cast sensor system for track awareness  
- Adaptive throttle and steering control  
- Reward shaping for speed, stability, and racing line  
- Trained with state-of-the-art RL algorithms (SAC, TD3, PPO comparisons)  

## Requirements
- Python 3.10+  
- PyTorch  
- Assetto Corsa + Custom Shaders Patch (for memory-mapped I/O)
