# safe_traffic_weaving

ROS package accompanying ["On Infusing Reachability-Based Safety Assurance within Probabilistic Planning Frameworks for Human-Robot Vehicle Interactions"](http://asl.stanford.edu/wp-content/papercite-data/pdf/Leung.Schmerling.Chen.ea.ISER18.pdf) by Karen Leung\*, Edward Schmerling\*, Mo Chen, John Talbot, J. Christian Gerdes, Marco Pavone) published in the [2018 International Symposium on Experimental Robotics](http://www.iser2018.org/). An extended [journal version](http://asl.stanford.edu/wp-content/papercite-data/pdf/Leung.Schmerling.ea.IJRR19.pdf) is to appear in the International Journal of Robotics Research. 

This ROS package performs the prediction and planning for a pairwise traffic-weaving interaction detailed in our ICRA2018 paper, [Multimodal Probabilistic Model-Based Planning for Human-Robot Interaction](http://asl.stanford.edu/wp-content/papercite-data/pdf/Schmerling.Leung.Vollprecht.Pavone.ICRA18.pdf) by Edward Schmerling, Karen Leung, Wolf Vollprecht, Marco Pavone. In a nutshell, it predictions are made about what the human-driven car will do in the future, and the robot selects an optimal action sequence to perform a traffic-weaving maneuver with the human-driven car. A desired trajectory (ROS msg) is produced. In our experiments, this trajectory is fed into a lower-level tracking controller. We use [`Pigeon.jl`](https://github.com/StanfordASL/Pigeon.jl).

## Software Requirements


## Run
This code is tailored towards our experimental platform, X1, a full-scale test vehicle designed and developed by the [Stanford Dynamic Design Lab (DDL)](https://dynamicdesignlab.sites.stanford.edu/) directed by Professor Chris Gerdes. If you want to use this for your own platform, you will need to make some changes. In short, we take the vehicle's state published by the `from_autobox` topic, and translate that to a position, velocity, and acceleration message (see `x1_state_republisher.py`). The translated message is used in our model-based planner (see `traffic_weaving_translator.py`, `TrafficWeavingPlanner.jl`, and `HOP_action_sequence_score.py`) to produce desired action sequences.

Additionally, we performed experiments at [Thunderhill Raceway Park](https://www.thunderhill.com/) and on Stanford campus. The GPS coordinates used in the `roadways` folder correspond to these locations. These GPS coordinates are important as this maps the GPS (in world frame) coordinates into a local frame which is used for prediction and planning.

An xbox controller is needed to virutally control the human-driven car. We did not perform experiments with two full-scale test vehicles. 

To run on the car, first start a `roscore`, then

Terminal window #1: 

> roslaunch safe_traffic_weaving X1_traffic_weaving.launch

Terminal window #2: Go to the scripts folder and open Julia 0.6 repl
> include("XboxCarSim.jl")\
> XboxCarSim.run()


Terminal window #3: Go to the scripts folder and open Julia 0.6 repl
> include("TrafficWeavingPlanner.jl")\
> TrafficWeavingPlanner.run()

In our experiments, we had a fourth window that ran a the tracking controller (`Pigeon.jl`).


You can also do this in simulation by running 
> roslaunch safe_traffic_weaving X1_traffic_weaving_sim.launch

However, this requires another package called `osprey` which has a simulator for X1 and outputs `from_autobox` msgs. Unfortunately `osprey` is managed by the DDL and is not open-sourced. It would be possible to use your own vehicle simulator for your car, and a rosnode that translated the simulator outputs into a `from_autobox` msg.
