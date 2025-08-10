## 🚌 Overview

The Multi-Agent Urban Bus Simulation Environment is a reinforcement learning environment built upon real-world data from Salvador’s public transportation system (SUNT – Salvador Urban Network Transportation). It simulates the operation of multiple buses as independent agents navigating an actual city transit network, enabling the development and testing of intelligent control strategies for public transport.

This environment allows agents to be trained in realistic, data-driven scenarios with the aim of improving service efficiency and passenger experience.
The system focuses on optimizing bus operations by leveraging boarding, alighting, and travel time data from real operations.

---
## 🎯 Objective

Train public transportation agents (buses) capable of:

- Reducing passenger waiting time at bus stops;
- Regularizing bus headways (time intervals between buses);
- Maintaining an adequate vehicle occupancy level;
- Operating with energy efficiency and minimal maintenance degradation.

The goal is to achieve **better service coverage**, optimizing passenger service based on **real boarding, alighting, and travel time data**. By applying Multi-Agent Reinforcement Learning (MARL), buses act autonomously while implicitly cooperating through a shared global reward function. This enables them to coordinate in order to maintain service quality, adapt to demand patterns, and respond to operational challenges in real time.

---
## 📊 Observation Outputs

During training, the environment generates several key metrics as output.  
These metrics are designed to **aid in observation**, **monitor system performance**, and can also be used as components in the **reward function**.

- **avg_travel_time_AB**  
  Average travel time between two reference stops (A → B).  
  Helps evaluate operational speed and detect potential congestion or delays.

- **future_demand_at_B**  
  Estimated future passenger demand at stop B.  
  Allows agents to anticipate boarding needs and adjust operations proactively.

- **occupancy_rate**  
  Proportion of the bus’s seating/standing capacity currently in use.  
  Useful for balancing passenger comfort with operational efficiency.

- **uptime_normalized**  
  Normalized measure of the vehicle’s operational availability.  
  Reflects how consistently the bus remains active in service without downtime.

**Purpose:**  
These outputs provide the agent with essential feedback on system state,  
enabling better decision-making and facilitating the design of reward functions that align with real-world service goals.

---

---
## 🎮 Actions

At the beginning of each training episode (`reset`), the agent is assigned one of the valid routes registered in the `real_routes` file.  
Currently, the route selection is **random**, but once chosen, the agent remains on that route for the entire training episode — reflecting a real-world scenario where, with hundreds of buses in operation, each vehicle typically serves **one fixed route per day**.

The action space available to the agent at each training step has been expanded to three options:

- **WAIT**  
  The agent chooses to wait before starting or continuing its route.  
  This enables strategic timing decisions, such as holding to avoid vehicle clustering and improve headway regularity.

- **MOVE**  
  The default action where the agent moves to the next stop in its assigned route.

- **SERVICE_CENTER**  
  The agent diverts to a maintenance facility when its maintenance status is not `"OK"` or its fuel level is low.  
  This action helps ensure operational reliability and long-term availability.

**Purpose:**  
These actions allow agents to balance service efficiency, passenger satisfaction, and operational constraints, while simulating realistic decision-making in a multi-bus network.

---

---
## 🎯 Reward Function

The reward function is designed to guide agents toward providing **efficient, reliable, and passenger-focused service**.  
It combines multiple operational objectives into a single feedback signal, encouraging cooperative behavior among buses.

The main components of the reward are:

- **Passenger Service Quality**  
  Positive rewards for reducing passenger waiting time and meeting estimated demand at stops.  
  Encourages minimizing gaps (*headways*) and avoiding long waits for passengers.

- **Operational Efficiency**  
  Rewards for maintaining appropriate occupancy levels — avoiding both overcrowding and running nearly empty.  
  Includes incentives for completing routes within expected travel times.

- **Maintenance and Fuel Management**  
  Penalties for ignoring low fuel levels or maintenance needs.  
  Rewards for timely visits to the service center to ensure operational readiness.

- **Traffic Flow & Coordination**  
  Negative rewards for excessive idling or bunching with other buses on the same route.  
  Encourages better distribution of vehicles along the network.

**Goal:**  
By combining these factors, the reward function teaches agents to **balance passenger satisfaction, fleet efficiency, and operational sustainability** in a real-world inspired transit network.


---

---
## 🛠 Training Setup

The training environment is built using **Multi-Agent Reinforcement Learning (MARL)** principles, integrating:

- **[Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html)** — Scalable reinforcement learning framework used to handle multiple agents in parallel.
- **[PettingZoo](https://pettingzoo.farama.org/)** — Multi-agent environment API providing compatibility between RLlib and the simulation.
- **[SuperSuit](https://github.com/Farama-Foundation/SuperSuit)** — A collection of wrappers for preprocessing and transforming observations and actions.
- **[Gymnasium](https://gymnasium.farama.org/)** — Standardized API for RL environments, used as a base for action and observation space definitions.

### Environment Configuration
- **Agents**: Each bus is modeled as an independent agent operating in a shared environment.
- **Scenario**: Built on the real **SUNT** (Salvador Urban Network Transportation) dataset, using actual bus stops, routes, and demand data.
- **Episodes**: Each training episode represents a simulated day of operations.
- **Policy Setup**:  
  Agents are trained with **PPO (Proximal Policy Optimization)**, using a shared policy for coordinated behavior, while maintaining decentralized decision-making.

### Parallel Training
The framework supports **scaling to hundreds of agents** in simulation, leveraging Ray’s distributed training to speed up learning and allow experiments with large-scale bus networks.

---