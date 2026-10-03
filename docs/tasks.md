# Tasks

Every task this package registers, with what its sources say about it, quoted directly,
and how this port differs. Each GIF shows 16 rollouts of one PPO policy trained on that
task here. Quotes keep their sources' wording; links lead to them.

Names follow one scheme: dashes join words, and slashes add variants. Any rendered task
`name` also exists as `name/pix`, observing 64×64 renders, and the robots as
`name/pix-prp`, observing renders and proprioception. Tasks have no time limit; the
limits below are Gymnasium's and Playground's.

## Classic control

### `cart-pole`

<img src="gifs/cart-pole.gif" alt="16 rollouts of a policy trained on cart-pole" loading="lazy">

> **Description.** This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
> ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
> A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
> The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
>  in the left and right direction on the cart.
>
> **Rewards.** Since the goal is to keep the pole upright for as long as possible, by default, a reward of `+1` is given for every step taken, including the termination step. The default reward threshold is 500 for v1 and 200 for v0 due to the time limit on the environment.
>
> If `sutton_barto_reward=True`, then a reward of `0` is awarded for every non-terminating step and `-1` for the terminating step. As a result, the reward threshold is 0 for v0 and v1.
>
> **Starting State.** All observations are assigned a uniformly random value in `(-0.05, 0.05)`
>
> **Episode End.** The episode ends if any one of the following occurs:
>
> 1. Termination: Pole Angle is greater than ±12°
> 2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
> 3. Truncation: Episode length is greater than 500 (200 for v0)
>
> — Gymnasium 1.3.0, [`gymnasium/envs/classic_control/cartpole.py`](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/classic_control/cartpole.py); [documentation](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

**Origin.** Barto, Sutton & Anderson (1983).

**Here.** Dynamics, rewards and observations match Gymnasium's to float64 precision. Gymnasium's time limit is 500 steps.

### `acrobot`

<img src="gifs/acrobot.gif" alt="16 rollouts of a policy trained on acrobot" loading="lazy">

> **Description.** The Acrobot environment is based on Sutton's work in
> ["Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding"](https://papers.nips.cc/paper/1995/hash/8f1d43620bc6bb580df6e80b0dc05c48-Abstract.html)
> and [Sutton and Barto's book](http://www.incompleteideas.net/book/the-book-2nd.html).
> The system consists of two links connected linearly to form a chain, with one end of
> the chain fixed. The joint between the two links is actuated. The goal is to apply
> torques on the actuated joint to swing the free end of the linear chain above a
> given height while starting from the initial state of hanging downwards.
>
> As seen in the **Gif**: two blue links connected by two green joints. The joint in
> between the two links is actuated. The goal is to swing the free end of the outer-link
> to reach the target height (black horizontal line above system) by applying torque on
> the actuator.
>
> **Rewards.** The goal is to have the free end reach a designated target height in as few steps as possible,
> and as such all steps that do not reach the goal incur a reward of -1.
> Achieving the target height results in termination with a reward of 0. The reward threshold is -100.
>
> **Starting State.** Each parameter in the underlying state (`theta1`, `theta2`, and the two angular velocities) is initialized
> uniformly between -0.1 and 0.1. This means both links are pointing downwards with some initial stochasticity.
>
> **Episode End.** The episode ends if one of the following occurs:
> 1. Termination: The free end reaches the target height, which is constructed as:
> `-cos(theta1) - cos(theta2 + theta1) > 1.0`
> 2. Truncation: Episode length is greater than 500 (200 for v0)
>
> — Gymnasium 1.3.0, [`gymnasium/envs/classic_control/acrobot.py`](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/classic_control/acrobot.py); [documentation](https://gymnasium.farama.org/environments/classic_control/acrobot/)

**Origin.** Sutton (1996); Sutton & Barto, *Reinforcement Learning: An Introduction*. Implementation from RLPy.

**Here.** Matches Gymnasium to float64 precision. Gymnasium's time limit is 500 steps.

### `mountain-car`

<img src="gifs/mountain-car.gif" alt="16 rollouts of a policy trained on mountain-car" loading="lazy">

> **Description.** The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically
> at the bottom of a sinusoidal valley, with the only possible actions being the accelerations
> that can be applied to the car in either direction. The goal of the MDP is to strategically
> accelerate the car to reach the goal state on top of the right hill. There are two versions
> of the mountain car domain in gymnasium: one with discrete actions and one with continuous.
> This version is the one with discrete actions.
>
> This MDP first appeared in [Andrew Moore's PhD Thesis (1990)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.pdf)
>
> ```
> @TECHREPORT{Moore90efficientmemory-based,
>     author = {Andrew William Moore},
>     title = {Efficient Memory-based Learning for Robot Control},
>     institution = {University of Cambridge},
>     year = {1990}
> }
> ```
>
> **Starting State.** The position of the car is assigned a uniform random value in *[-0.6 , -0.4]*.
> The starting velocity of the car is always assigned to 0.
>
> **Episode End.** The episode ends if either of the following happens:
> 1. Termination: The position of the car is greater than or equal to 0.5 (the goal position on top of the right hill)
> 2. Truncation: The length of the episode is 200.
>
> — Gymnasium 1.3.0, [`gymnasium/envs/classic_control/mountain_car.py`](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/classic_control/mountain_car.py); [documentation](https://gymnasium.farama.org/environments/classic_control/mountain_car/)

**Origin.** Andrew W. Moore, PhD thesis (1990).

**Here.** Matches Gymnasium to float64 precision. Gymnasium's time limit is 200 steps.

### `mountain-car/continuous`

<img src="gifs/mountain-car__continuous.gif" alt="16 rollouts of a policy trained on mountain-car/continuous" loading="lazy">

> **Description.** The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically
> at the bottom of a sinusoidal valley, with the only possible actions being the accelerations
> that can be applied to the car in either direction. The goal of the MDP is to strategically
> accelerate the car to reach the goal state on top of the right hill. There are two versions
> of the mountain car domain in gymnasium: one with discrete actions and one with continuous.
> This version is the one with continuous actions.
>
> This MDP first appeared in [Andrew Moore's PhD Thesis (1990)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.pdf)
>
> ```
> @TECHREPORT{Moore90efficientmemory-based,
>     author = {Andrew William Moore},
>     title = {Efficient Memory-based Learning for Robot Control},
>     institution = {University of Cambridge},
>     year = {1990}
> }
> ```
>
> **Starting State.** The position of the car is assigned a uniform random value in `[-0.6 , -0.4]`.
> The starting velocity of the car is always assigned to 0.
>
> **Episode End.** The episode ends if either of the following happens:
> 1. Termination: The position of the car is greater than or equal to 0.45 (the goal position on top of the right hill)
> 2. Truncation: The length of the episode is 999.
>
> — Gymnasium 1.3.0, [`gymnasium/envs/classic_control/continuous_mountain_car.py`](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/classic_control/continuous_mountain_car.py); [documentation](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/)

**Origin.** Andrew W. Moore, PhD thesis (1990).

**Here.** Matches Gymnasium to float64 precision. Gymnasium's time limit is 999 steps.

### `pendulum`

<img src="gifs/pendulum.gif" alt="16 rollouts of a policy trained on pendulum" loading="lazy">

> **Description.** The inverted pendulum swingup problem is based on the classic problem in control theory.
> The system consists of a pendulum attached at one end to a fixed point, and the other end being free.
> The pendulum starts in a random position and the goal is to apply torque on the free end to swing it
> into an upright position, with its center of gravity right above the fixed point.
>
> The diagram below specifies the coordinate system used for the implementation of the pendulum's
> dynamic equations.
>
> ![Pendulum Coordinate System](/_static/diagrams/pendulum.png)
>
> - `x-y`: cartesian coordinates of the pendulum's end in meters.
> - `theta` : angle in radians.
> - `tau`: torque in `N m`. Defined as positive _counter-clockwise_.
>
> **Rewards.** The reward function is defined as:
>
> *r = -(theta<sup>2</sup> + 0.1 * theta_dt<sup>2</sup> + 0.001 * torque<sup>2</sup>)*
>
> where `theta` is the pendulum's angle normalized between *[-pi, pi]* (with 0 being in the upright position).
> Based on the above equation, the minimum reward that can be obtained is
> *-(pi<sup>2</sup> + 0.1 * 8<sup>2</sup> + 0.001 * 2<sup>2</sup>) = -16.2736044*,
> while the maximum reward is zero (pendulum is upright with zero velocity and no torque applied).
>
> **Starting State.** The starting state is a random angle in *[-pi, pi]* and a random angular velocity in *[-1,1]*.
>
> — Gymnasium 1.3.0, [`gymnasium/envs/classic_control/pendulum.py`](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/classic_control/pendulum.py); [documentation](https://gymnasium.farama.org/environments/classic_control/pendulum/)

**Origin.** The classic inverted-pendulum swing-up; Gymnasium's implementation by Carlos Luis.

**Here.** Matches Gymnasium to float64 precision. Gymnasium's time limit is 200 steps.

## Box2D

### `lunar-lander`

<img src="gifs/lunar-lander.gif" alt="16 rollouts of a policy trained on lunar-lander" loading="lazy">

> **Description.** This environment is a classic rocket trajectory optimization problem.
> According to Pontryagin's maximum principle, it is optimal to fire the
> engine at full throttle or turn it off. This is the reason why this
> environment has discrete actions: engine on or off.
>
> There are two environment versions: discrete or continuous.
> The landing pad is always at coordinates (0,0). The coordinates are the
> first two numbers in the state vector.
> Landing outside of the landing pad is possible. Fuel is infinite, so an agent
> can learn to fly and then land on its first attempt.
>
> […]
>
> **Rewards.** After every step a reward is granted. The total reward of an episode is the
> sum of the rewards for all the steps within that episode.
>
> For each step, the reward:
> - is increased/decreased the closer/further the lander is to the landing pad.
> - is increased/decreased the slower/faster the lander is moving.
> - is decreased the more the lander is tilted (angle not horizontal).
> - is increased by 10 points for each leg that is in contact with the ground.
> - is decreased by 0.03 points each frame a side engine is firing.
> - is decreased by 0.3 points each frame the main engine is firing.
>
> The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively.
>
> An episode is considered a solution if it scores at least 200 points.
>
> **Starting State.** The lander starts at the top center of the viewport with a random initial
> force applied to its center of mass.
>
> — Gymnasium 1.3.0, [`gymnasium/envs/box2d/lunar_lander.py`](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/box2d/lunar_lander.py); [documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

**Origin.** Created by Oleg Klimov; Gymnasium version by Andrea Pierré.

**Here.** Simulated by `jax_gym.rigid2d`, a compact engine after Box2D, so it matches Gymnasium in distribution rather than step by step. `lunar-lander/continuous` is the continuous variant. Gymnasium's time limit is 1000 steps.

### `lunar-lander/continuous`

<img src="gifs/lunar-lander__continuous.gif" alt="16 rollouts of a policy trained on lunar-lander/continuous" loading="lazy">

> **Description.** This environment is a classic rocket trajectory optimization problem.
> According to Pontryagin's maximum principle, it is optimal to fire the
> engine at full throttle or turn it off. This is the reason why this
> environment has discrete actions: engine on or off.
>
> There are two environment versions: discrete or continuous.
> The landing pad is always at coordinates (0,0). The coordinates are the
> first two numbers in the state vector.
> Landing outside of the landing pad is possible. Fuel is infinite, so an agent
> can learn to fly and then land on its first attempt.
>
> […]
>
> **Rewards.** After every step a reward is granted. The total reward of an episode is the
> sum of the rewards for all the steps within that episode.
>
> For each step, the reward:
> - is increased/decreased the closer/further the lander is to the landing pad.
> - is increased/decreased the slower/faster the lander is moving.
> - is decreased the more the lander is tilted (angle not horizontal).
> - is increased by 10 points for each leg that is in contact with the ground.
> - is decreased by 0.03 points each frame a side engine is firing.
> - is decreased by 0.3 points each frame the main engine is firing.
>
> The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively.
>
> An episode is considered a solution if it scores at least 200 points.
>
> **Starting State.** The lander starts at the top center of the viewport with a random initial
> force applied to its center of mass.
>
> — Gymnasium 1.3.0, [`gymnasium/envs/box2d/lunar_lander.py`](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/box2d/lunar_lander.py); [documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

**Origin.** Created by Oleg Klimov; Gymnasium version by Andrea Pierré.

**Here.** Continuous throttles; otherwise as `lunar-lander`.

### `bipedal-walker`

<img src="gifs/bipedal-walker.gif" alt="16 rollouts of a policy trained on bipedal-walker" loading="lazy">

> **Description.** This is a simple 4-joint walker robot environment.
> There are two versions:
> - Normal, with slightly uneven terrain.
> - Hardcore, with ladders, stumps, pitfalls.
>
> To solve the normal version, you need to get 300 points in 1600 time steps.
> To solve the hardcore version, you need 300 points in 2000 time steps.
>
> A heuristic is provided for testing. It's also useful to get demonstrations
> to learn from. […]
>
> **Rewards.** Reward is given for moving forward, totaling 300+ points up to the far end.
> If the robot falls, it gets -100. Applying motor torque costs a small
> amount of points. A more optimal agent will get a better score.
>
> **Starting State.** The walker starts standing at the left end of the terrain with the hull
> horizontal, and both legs in the same position with a slight knee angle.
>
> — Gymnasium 1.3.0, [`gymnasium/envs/box2d/bipedal_walker.py`](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/box2d/bipedal_walker.py); [documentation](https://gymnasium.farama.org/environments/box2d/bipedal_walker/)

**Origin.** Created by Oleg Klimov; Gymnasium version by Andrea Pierré.

**Here.** Simulated by `jax_gym.rigid2d`; matches Gymnasium in distribution. Gymnasium's time limit is 1600 steps.

### `bipedal-walker/hardcore`

<img src="gifs/bipedal-walker__hardcore.gif" alt="16 rollouts of a policy trained on bipedal-walker/hardcore" loading="lazy">

> **Description.** This is a simple 4-joint walker robot environment.
> There are two versions:
> - Normal, with slightly uneven terrain.
> - Hardcore, with ladders, stumps, pitfalls.
>
> To solve the normal version, you need to get 300 points in 1600 time steps.
> To solve the hardcore version, you need 300 points in 2000 time steps.
>
> A heuristic is provided for testing. It's also useful to get demonstrations
> to learn from. […]
>
> **Rewards.** Reward is given for moving forward, totaling 300+ points up to the far end.
> If the robot falls, it gets -100. Applying motor torque costs a small
> amount of points. A more optimal agent will get a better score.
>
> **Starting State.** The walker starts standing at the left end of the terrain with the hull
> horizontal, and both legs in the same position with a slight knee angle.
>
> — Gymnasium 1.3.0, [`gymnasium/envs/box2d/bipedal_walker.py`](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/box2d/bipedal_walker.py); [documentation](https://gymnasium.farama.org/environments/box2d/bipedal_walker/)

**Origin.** Created by Oleg Klimov; Gymnasium version by Andrea Pierré.

**Here.** Stumps, stairs and pits; otherwise as `bipedal-walker`. Gymnasium's time limit is 2000 steps.

### `car-racing`

<img src="gifs/car-racing.gif" alt="16 rollouts of a policy trained on car-racing" loading="lazy">

> **Description.** The easiest control task to learn from pixels - a top-down
> racing environment. The generated track is random every episode.
>
> Some indicators are shown at the bottom of the window along with the
> state RGB buffer. From left to right: true speed, four ABS sensors,
> steering wheel position, and gyroscope.
> […]
>
> **Rewards.** The reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles
>  visited in the track. For example, if you have finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.
>
> **Starting State.** The car starts at rest in the center of the road.
>
> — Gymnasium 1.3.0, [`gymnasium/envs/box2d/car_racing.py`](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/box2d/car_racing.py); [documentation](https://gymnasium.farama.org/environments/box2d/car_racing/)

**Origin.** Created by Oleg Klimov after Chris Campbell's top-down car tutorial; Gymnasium version by Andrea Pierré.

**Here.** The car is one rigid body with Gymnasium's wheel friction model on Gymnasium's tracks. Observations are its 96×96 view; `car-racing/mkv` observes a vector. Gymnasium's time limit is 1000 steps.

### `car-racing/discrete`

CarRacing-v3 with `continuous=False`: five actions (nothing, left, right, gas, brake),
otherwise as `car-racing`. No policy was trained for it, so no GIF.

## MuJoCo

### `inverted-pendulum`

<img src="gifs/inverted-pendulum.gif" alt="16 rollouts of a policy trained on inverted-pendulum" loading="lazy">

> **Description.** This environment is the Cartpole environment, based on the work of Barto, Sutton, and Anderson in ["Neuronlike adaptive elements that can solve difficult learning control problems"](https://ieeexplore.ieee.org/document/6313077),
> just like in the classic environments, but now powered by the Mujoco physics simulator - allowing for more complex experiments (such as varying the effects of gravity).
> This environment consists of a cart that can be moved linearly, with a pole attached to one end and having another end free.
> The cart can be pushed left or right, and the goal is to balance the pole on top of the cart by applying forces to the cart.
>
> **Rewards.** The goal is to keep the inverted pendulum stand upright (within a certain angle limit) for as long as possible - as such, a reward of +1 is given for each timestep that the pole is upright.
>
> The pole is considered upright if:
> $|angle| < 0.2$.
>
> and `info` also contains the reward.
>
> **Starting State.** The initial position state is $\mathcal{U}_{[-reset\\_noise\\_scale \times I_{2}, reset\\_noise\\_scale \times I_{2}]}$.
> The initial velocity state is $\mathcal{U}_{[-reset\\_noise\\_scale \times I_{2}, reset\\_noise\\_scale \times I_{2}]}$.
>
> where $\mathcal{U}$ is the multivariate uniform continuous distribution.
>
> **Episode End.**
>
> **Termination.**
> The environment terminates when the Inverted Pendulum is unhealthy.
> The Inverted Pendulum is unhealthy if any of the following happens:
>
> 1. Any of the state space values is no longer finite.
> 2. The absolute value of the vertical angle between the pole and the cart is greater than 0.2 radians.
>
> **Truncation.**
> The default duration of an episode is 1000 timesteps.
>
> — Gymnasium 1.3.0, [`gymnasium/envs/mujoco/inverted_pendulum_v5.py`](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/mujoco/inverted_pendulum_v5.py); [documentation](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/)

**Origin.** Barto, Sutton & Anderson (1983); v5 by Kallinteris-Andreas.

**Here.** Gymnasium's model on MJX; double-precision rollouts match Gymnasium to 1e-8. Gymnasium's time limit is 1000 steps.

### `inverted-double-pendulum`

<img src="gifs/inverted-double-pendulum.gif" alt="16 rollouts of a policy trained on inverted-double-pendulum" loading="lazy">

> **Description.** This environment originates from control theory and builds on the cartpole environment based on the work of Barto, Sutton, and Anderson in ["Neuronlike adaptive elements that can solve difficult learning control problems"](https://ieeexplore.ieee.org/document/6313077),
> powered by the Mujoco physics simulator - allowing for more complex experiments (such as varying the effects of gravity or constraints).
> This environment involves a cart that can be moved linearly, with one pole attached to it and a second pole attached to the other end of the first pole (leaving the second pole as the only one with a free end).
> The cart can be pushed left or right, and the goal is to balance the second pole on top of the first pole, which is in turn on top of the cart, by applying continuous forces to the cart.
>
> **Rewards.** The total reward is: ***reward*** *=* *alive_bonus - distance_penalty - velocity_penalty*.
>
> - *alive_bonus*:
> Every timestep that the Inverted Pendulum is healthy (see definition in section "Episode End"),
> it gets a reward of fixed value `healthy_reward` (default is $10$).
> - *distance_penalty*:
> This reward is a measure of how far the *tip* of the second pendulum (the only free end) moves,
> and it is calculated as $0.01 x_{pole2-tip}^2 + (y_{pole2-tip}-2)^2$,
> where $x_{pole2-tip}, y_{pole2-tip}$ are the xy-coordinatesof the tip of the second pole.
> - *velocity_penalty*:
> A negative reward to penalize the agent for moving too fast.
> $10^{-3} \omega_1 + 5 \times 10^{-3} \omega_2$,
> where $\omega_1, \omega_2$ are the angular velocities of the hinges.
>
> `info` contains the individual reward terms.
>
> **Starting State.** The initial position state is $\mathcal{U}_{[-reset\\_noise\\_scale \times I_{3}, reset\\_noise\\_scale \times I_{3}]}$.
> The initial velocity state is $\mathcal{N}(0_{3}, reset\\_noise\\_scale^2 \times I_{3})$.
>
> where $\mathcal{N}$ is the multivariate normal distribution and $\mathcal{U}$ is the multivariate uniform continuous distribution.
>
> **Episode End.**
>
> **Termination.**
> The environment terminates when the Inverted Double Pendulum is unhealthy.
> The Inverted Double Pendulum is unhealthy if any of the following happens:
>
> 1.Termination: The y_coordinate of the tip of the second pole $\leq 1$.
>
> Note: The maximum standing height of the system is 1.2 m when all the parts are perpendicularly vertical on top of each other.
>
> **Truncation.**
> The default duration of an episode is 1000 timesteps.
>
> — Gymnasium 1.3.0, [`gymnasium/envs/mujoco/inverted_double_pendulum_v5.py`](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/mujoco/inverted_double_pendulum_v5.py); [documentation](https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/)

**Origin.** Builds on Barto, Sutton & Anderson (1983); v5 by Kallinteris-Andreas.

**Here.** Gymnasium's model on MJX; matches to 1e-8. Gymnasium's time limit is 1000 steps.

### `reacher`

<img src="gifs/reacher.gif" alt="16 rollouts of a policy trained on reacher" loading="lazy">

> **Description.** "Reacher" is a two-jointed robot arm.
> The goal is to move the robot's end effector (called *fingertip*) close to a target that is spawned at a random position.
>
> **Rewards.** The total reward is: ***reward*** *=* *reward_distance + reward_control*.
>
> - *reward_distance*:
> This reward is a measure of how far the *fingertip* of the reacher (the unattached end) is from the target,
> with a more negative value assigned if the reacher's *fingertip* is further away from the target.
> It is $-w_{near} \|(P_{fingertip} - P_{target})\|_2$.
> where $w_{near}$ is the `reward_near_weight` (default is $1$).
> - *reward_control*:
> A negative reward to penalize the walker for taking actions that are too large.
> It is measured as the negative squared Euclidean norm of the action, i.e. as $-w_{control} \|action\|_2^2$.
> where $w_{control}$ is the `reward_control_weight`. (default is $0.1$)
>
> `info` contains the individual reward terms.
>
> **Starting State.** The initial position state of the reacher arm is $\mathcal{U}_{[-0.1 \times I_{2}, 0.1 \times I_{2}]}$.
> The position state of the goal is (permanently) $\mathcal{S}(0.2)$.
> The initial velocity state of the Reacher arm is $\mathcal{U}_{[-0.005 \times 1_{2}, 0.005 \times 1_{2}]}$.
> The velocity state of the object is (permanently) $0_2$.
>
> where $\mathcal{U}$ is the multivariate uniform continuous distribution and $\mathcal{S}$ is the uniform continuous spherical distribution.
>
> The default frame rate is $2$, with each frame lasting $0.01$, so *dt = 5 * 0.01 = 0.02*.
>
> **Episode End.**
>
> **Termination.**
> The Reacher never terminates.
>
> **Truncation.**
> The default duration of an episode is 50 timesteps.
>
> — Gymnasium 1.3.0, [`gymnasium/envs/mujoco/reacher_v5.py`](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/mujoco/reacher_v5.py); [documentation](https://gymnasium.farama.org/environments/mujoco/reacher/)

**Origin.** OpenAI Gym's MuJoCo suite; v5 by Kallinteris-Andreas.

**Here.** Gymnasium's model on MJX; matches to 1e-8. Gymnasium's time limit is 50 steps.

### `pusher`

<img src="gifs/pusher.gif" alt="16 rollouts of a policy trained on pusher" loading="lazy">

> **Description.** "Pusher" is a multi-jointed robot arm that is very similar to a human arm.
> The goal is to move a target cylinder (called *object*) to a goal position using the robot's end effector (called *fingertip*).
> The robot consists of shoulder, elbow, forearm and wrist joints.
>
> **Rewards.** The total reward is: ***reward*** *=* *reward_dist + reward_ctrl + reward_near*.
>
> - *reward_dist*:
> This reward is a measure of how far the object is from the target goal position,
> with a more negative value assigned if the object is further away from the target.
> It is $-w_{dist} \|(P_{object} - P_{target})\|_2$.
> where $w_{dist}$ is the `reward_dist_weight` (default is $1$).
> - *reward_ctrl*:
> A negative reward to penalize the pusher for taking actions that are too large.
> It is measured as the negative squared Euclidean norm of the action, i.e. as $-w_{control} \|action\|_2^2$.
> where $w_{control}$ is the `reward_control_weight` (default is $0.1$).
> - *reward_near*:
> This reward is a measure of how far the *fingertip* of the pusher (the unattached end) is from the object,
> with a more negative value assigned for when the pusher's *fingertip* is further away from the target.
> It is $-w_{near} \|(P_{fingertip} - P_{target})\|_2$.
> where $w_{near}$ is the `reward_near_weight` (default is $0.5$).
>
> `info` contains the individual reward terms.
>
> **Starting State.** The initial position state of the Pusher arm is $0_{6}$.
> The initial position state of the object is $\mathcal{U}_{[[-0.3, -0.2], [0, 0.2]]}$.
> The position state of the goal is (permanently) $[0.45, -0.05, -0.323]$.
> The initial velocity state of the Pusher arm is $\mathcal{U}_{[-0.005 \times I_{6}, 0.005 \times I_{6}]}$.
> The initial velocity state of the object is $0_2$.
> The velocity state of the goal is (permanently) $0_3$.
>
> where $\mathcal{U}$ is the multivariate uniform continuous distribution.
>
> Note that the initial position state of the object is sampled until its distance to the goal is $ > 0.17 m$.
>
> The default frame rate is 5, with each frame lasting 0.01, so *dt = 5 * 0.01 = 0.05*.
>
> **Episode End.**
>
> **Termination.**
> The Pusher never terminates.
>
> **Truncation.**
> The default duration of an episode is 100 timesteps.
>
> — Gymnasium 1.3.0, [`gymnasium/envs/mujoco/pusher_v5.py`](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/mujoco/pusher_v5.py); [documentation](https://gymnasium.farama.org/environments/mujoco/pusher/)

**Origin.** OpenAI Gym's MuJoCo suite; v5 by Kallinteris-Andreas.

**Here.** Gymnasium's model on MJX; matches to 1e-8. Gymnasium's time limit is 100 steps.

### `swimmer`

<img src="gifs/swimmer.gif" alt="16 rollouts of a policy trained on swimmer" loading="lazy">

> **Description.** This environment corresponds to the Swimmer environment described in Rémi Coulom's PhD thesis ["Reinforcement Learning Using Neural Networks, with Applications to Motor Control"](https://tel.archives-ouvertes.fr/tel-00003985/document).
> The environment aims to increase the number of independent state and control variables compared to classical control environments.
> The swimmers consist of three or more segments ('***links***') and one less articulation joints ('***rotors***') - one rotor joint connects exactly two links to form a linear chain.
> The swimmer is suspended in a two-dimensional pool and always starts in the same position (subject to some deviation drawn from a uniform distribution),
> and the goal is to move as fast as possible towards the right by applying torque to the rotors and using fluid friction.
>
> **Rewards.** The total reward is: ***reward*** *=* *forward_reward - ctrl_cost*.
>
> - *forward_reward*:
> A reward for moving forward,
> this reward would be positive if the Swimmer moves forward (in the positive $x$ direction / in the right direction).
> $w_{forward} \times \frac{dx}{dt}$, where
> $dx$ is the displacement of the (front) "tip" ($x_{after-action} - x_{before-action}$),
> $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is 4),
> and `frametime` which is $0.01$ - so the default is $dt = 4 \times 0.01 = 0.04$,
> $w_{forward}$ is the `forward_reward_weight` (default is $1$).
> - *ctrl_cost*:
> A negative reward to penalize the Swimmer for taking actions that are too large.
> $w_{control} \times \|action\|_2^2$,
> where $w_{control}$ is `ctrl_cost_weight` (default is $10^{-4}$).
>
> `info` contains the individual reward terms.
>
> **Starting State.** The initial position state is $\mathcal{U}_{[-reset\\_noise\\_scale \times I_{5}, reset\\_noise\\_scale \times I_{5}]}$.
> The initial velocity state is $\mathcal{U}_{[-reset\\_noise\\_scale \times I_{5}, reset\\_noise\\_scale \times I_{5}]}$.
>
> where $\mathcal{U}$ is the multivariate uniform continuous distribution.
>
> **Episode End.**
>
> **Termination.**
> The Swimmer never terminates.
>
> **Truncation.**
> The default duration of an episode is 1000 timesteps.
>
> — Gymnasium 1.3.0, [`gymnasium/envs/mujoco/swimmer_v5.py`](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/mujoco/swimmer_v5.py); [documentation](https://gymnasium.farama.org/environments/mujoco/swimmer/)

**Origin.** Rémi Coulom, PhD thesis (2002); v5 by Kallinteris-Andreas and Rushiv Arora.

**Here.** Gymnasium's model on MJX; matches to 1e-8. Gymnasium's time limit is 1000 steps.

### `half-cheetah`

<img src="gifs/half-cheetah.gif" alt="16 rollouts of a policy trained on half-cheetah" loading="lazy">

> **Description.** This environment is based on the work of P. Wawrzyński in ["A Cat-Like Robot Real-Time Learning to Run"](http://staff.elka.pw.edu.pl/~pwawrzyn/pub-s/0812_LSCLRR.pdf).
> The HalfCheetah is a 2-dimensional robot consisting of 9 body parts and 8 joints connecting them (including two paws).
> The goal is to apply torque to the joints to make the cheetah run forward (right) as fast as possible, with a positive reward based on the distance moved forward and a negative reward for moving backward.
> The cheetah's torso and head are fixed, and torque can only be applied to the other 6 joints over the front and back thighs (which connect to the torso), the shins (which connect to the thighs), and the feet (which connect to the shins).
>
> **Rewards.** The total reward is: ***reward*** *=* *forward_reward - ctrl_cost*.
>
> - *forward_reward*:
> A reward for moving forward,
> this reward would be positive if the Half Cheetah moves forward (in the positive $x$ direction / in the right direction).
> $w_{forward} \times \frac{dx}{dt}$, where
> $dx$ is the displacement of the "tip" ($x_{after-action} - x_{before-action}$),
> $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is $5$),
> and `frametime` which is $0.01$ - so the default is $dt = 5 \times 0.01 = 0.05$,
> $w_{forward}$ is the `forward_reward_weight` (default is $1$).
> - *ctrl_cost*:
> A negative reward to penalize the Half Cheetah for taking actions that are too large.
> $w_{control} \times \|action\|_2^2$,
> where $w_{control}$ is `ctrl_cost_weight` (default is $0.1$).
>
> `info` contains the individual reward terms.
>
> **Starting State.** The initial position state is $\mathcal{U}_{[-reset\\_noise\\_scale \times I_{9}, reset\\_noise\\_scale \times I_{9}]}$.
> The initial velocity state is $\mathcal{N}(0_{9}, reset\\_noise\\_scale^2 \times I_{9})$.
>
> where $\mathcal{N}$ is the multivariate normal distribution and $\mathcal{U}$ is the multivariate uniform continuous distribution.
>
> **Episode End.**
>
> **Termination.**
> The Half Cheetah never terminates.
>
> **Truncation.**
> The default duration of an episode is 1000 timesteps.
>
> — Gymnasium 1.3.0, [`gymnasium/envs/mujoco/half_cheetah_v5.py`](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/mujoco/half_cheetah_v5.py); [documentation](https://gymnasium.farama.org/environments/mujoco/half_cheetah/)

**Origin.** Paweł Wawrzyński (2009); v5 by Kallinteris-Andreas and Rushiv Arora.

**Here.** Gymnasium's model on MJX; matches to 1e-8. Gymnasium's time limit is 1000 steps.

### `hopper`

<img src="gifs/hopper.gif" alt="16 rollouts of a policy trained on hopper" loading="lazy">

> **Description.** This environment is based on the work of Erez, Tassa, and Todorov in ["Infinite Horizon Model Predictive Control for Nonlinear Periodic Tasks"](http://www.roboticsproceedings.org/rss07/p10.pdf).
> The environment aims to increase the number of independent state and control variables compared to classical control environments.
> The hopper is a two-dimensional one-legged figure consisting of four main body parts - the torso at the top, the thigh in the middle, the leg at the bottom, and a single foot on which the entire body rests.
> The goal is to make hops that move in the forward (right) direction by applying torque to the three hinges that connect the four body parts.
>
> **Rewards.** The total reward is: ***reward*** *=* *healthy_reward + forward_reward - ctrl_cost*.
>
> - *healthy_reward*:
> Every timestep that the Hopper is healthy (see definition in section "Episode End"),
> it gets a reward of fixed value `healthy_reward` (default is $1$).
> - *forward_reward*:
> A reward for moving forward,
> this reward would be positive if the Hopper moves forward (in the positive $x$ direction / in the right direction).
> $w_{forward} \times \frac{dx}{dt}$, where
> $dx$ is the displacement of the "torso" ($x_{after-action} - x_{before-action}$),
> $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is $4$),
> and `frametime` which is $0.002$ - so the default is $dt = 4 \times 0.002 = 0.008$,
> $w_{forward}$ is the `forward_reward_weight` (default is $1$).
> - *ctrl_cost*:
> A negative reward to penalize the Hopper for taking actions that are too large.
> $w_{control} \times \|action\|_2^2$,
> where $w_{control}$ is `ctrl_cost_weight` (default is $10^{-3}$).
>
> `info` contains the individual reward terms.
>
> **Starting State.** The initial position state is $[0, 1.25, 0, 0, 0, 0] + \mathcal{U}_{[-reset\\_noise\\_scale \times I_{6}, reset\\_noise\\_scale \times I_{6}]}$.
> The initial velocity state is $\mathcal{U}_{[-reset\\_noise\\_scale \times I_{6}, reset\\_noise\\_scale \times I_{6}]}$.
>
> where $\mathcal{U}$ is the multivariate uniform continuous distribution.
>
> Note that the z-coordinate is non-zero so that the hopper can stand up immediately.
>
> **Episode End.**
>
> **Termination.**
> If `terminate_when_unhealthy is True` (the default), the environment terminates when the Hopper is unhealthy.
> The Hopper is unhealthy if any of the following happens:
>
> 1. An element of `observation[1:]` (if  `exclude_current_positions_from_observation=True`, otherwise `observation[2:]`) is no longer contained in the closed interval specified by the `healthy_state_range` argument (default is $[-100, 100]$).
> 2. The height of the hopper (`observation[0]` if  `exclude_current_positions_from_observation=True`, otherwise `observation[1]`) is no longer contained in the closed interval specified by the `healthy_z_range` argument (default is $[0.7, +\infty]$) (usually meaning that it has fallen).
> 3. The angle of the torso (`observation[1]` if  `exclude_current_positions_from_observation=True`, otherwise `observation[2]`) is no longer contained in the closed interval specified by the `healthy_angle_range` argument (default is $[-0.2, 0.2]$).
>
> **Truncation.**
> The default duration of an episode is 1000 timesteps.
>
> — Gymnasium 1.3.0, [`gymnasium/envs/mujoco/hopper_v5.py`](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/mujoco/hopper_v5.py); [documentation](https://gymnasium.farama.org/environments/mujoco/hopper/)

**Origin.** Erez, Tassa & Todorov (2011); v5 by Kallinteris-Andreas.

**Here.** Gymnasium's model on MJX; matches to 1e-8. Gymnasium's time limit is 1000 steps.

### `walker2d`

<img src="gifs/walker2d.gif" alt="16 rollouts of a policy trained on walker2d" loading="lazy">

> **Description.** This environment builds on the [hopper](https://gymnasium.farama.org/environments/mujoco/hopper/) environment by adding another set of legs that allow the robot to walk forward instead of hop.
> Like other MuJoCo environments, this environment aims to increase the number of independent state and control variables compared to classical control environments.
> The walker is a two-dimensional bipedal robot consisting of seven main body parts - a single torso at the top (with the two legs splitting after the torso), two thighs in the middle below the torso, two legs below the thighs, and two feet attached to the legs on which the entire body rests.
> The goal is to walk in the forward (right) direction by applying torque to the six hinges connecting the seven body parts.
>
> **Rewards.** The total reward is: ***reward*** *=* *healthy_reward bonus + forward_reward - ctrl_cost*.
>
> - *healthy_reward*:
> Every timestep that the Walker2d is alive, it receives a fixed reward of value `healthy_reward` (default is $1$),
> - *forward_reward*:
> A reward for moving forward,
> this reward would be positive if the Walker2d moves forward (in the positive $x$ direction / in the right direction).
> $w_{forward} \times \frac{dx}{dt}$, where
> $dx$ is the displacement of the (front) "tip" ($x_{after-action} - x_{before-action}$),
> $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is $4$),
> and `frametime` which is $0.002$ - so the default is $dt = 4 \times 0.002 = 0.008$,
> $w_{forward}$ is the `forward_reward_weight` (default is $1$).
> - *ctrl_cost*:
> A negative reward to penalize the Walker2d for taking actions that are too large.
> $w_{control} \times \|action\|_2^2$,
> where $w_{control}$ is `ctrl_cost_weight` (default is $10^{-3}$).
>
> `info` contains the individual reward terms.
>
> **Starting State.** The initial position state is $[0, 1.25, 0, 0, 0, 0, 0, 0, 0] + \mathcal{U}_{[-reset\\_noise\\_scale \times I_{9}, reset\\_noise\\_scale \times I_{9}]}$.
> The initial velocity state is $\mathcal{U}_{[-reset\\_noise\\_scale \times I_{9}, reset\\_noise\\_scale \times I_{9}]}$.
>
> where $\mathcal{U}$ is the multivariate uniform continuous distribution.
>
> Note that the z-coordinate is non-zero so that the Walker2d can stand up immediately.
>
> **Episode End.**
>
> **Termination.**
> If `terminate_when_unhealthy is True` (which is the default), the environment terminates when the Walker2d is unhealthy.
> The Walker2d is unhealthy if any of the following happens:
>
> 1. Any of the state space values is no longer finite
> 2. The z-coordinate of the torso (the height) is **not** in the closed interval given by the `healthy_z_range` argument (default to $[0.8, 2.0]$).
> 3. The absolute value of the angle (`observation[1]` if `exclude_current_positions_from_observation=False`, else `observation[2]`) is ***not*** in the closed interval specified by the `healthy_angle_range` argument (default is $[-1, 1]$).
>
> **Truncation.**
> The default duration of an episode is 1000 timesteps.
>
> — Gymnasium 1.3.0, [`gymnasium/envs/mujoco/walker2d_v5.py`](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/mujoco/walker2d_v5.py); [documentation](https://gymnasium.farama.org/environments/mujoco/walker2d/)

**Origin.** Erez, Tassa & Todorov (2011), extending Hopper; v5 by Kallinteris-Andreas.

**Here.** Gymnasium's model on MJX; matches to 1e-8. Gymnasium's time limit is 1000 steps.

### `ant`

<img src="gifs/ant.gif" alt="16 rollouts of a policy trained on ant" loading="lazy">

> **Description.** This environment is based on the one introduced by Schulman, Moritz, Levine, Jordan, and Abbeel in ["High-Dimensional Continuous Control Using Generalized Advantage Estimation"](https://arxiv.org/abs/1506.02438).
> The ant is a 3D quadruped robot consisting of a torso (free rotational body) with four legs attached to it, where each leg has two body parts.
> The goal is to coordinate the four legs to move in the forward (right) direction by applying torque to the eight hinges connecting the two body parts of each leg and the torso (nine body parts and eight hinges).
>
> Note: Although the robot is called "Ant", it is actually 75cm tall and weighs 910.88g, with the torso being 327.25g and each leg being 145.91g.
>
> **Rewards.** The total reward is ***reward*** *=* *healthy_reward + forward_reward - ctrl_cost - contact_cost*.
>
> - *healthy_reward*:
> Every timestep that the Ant is healthy (see definition in section "Episode End"),
> it gets a reward of fixed value `healthy_reward` (default is $1$).
> - *forward_reward*:
> A reward for moving forward,
> this reward would be positive if the Ant moves forward (in the positive $x$ direction / in the right direction).
> $w_{forward} \times \frac{dx}{dt}$, where
> $dx$ is the displacement of the `main_body` ($x_{after-action} - x_{before-action}$),
> $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is $5$),
> and `frametime`, which is $0.01$ - so the default is $dt = 5 \times 0.01 = 0.05$,
> $w_{forward}$ is the `forward_reward_weight` (default is $1$).
> - *ctrl_cost*:
> A negative reward to penalize the Ant for taking actions that are too large.
> $w_{control} \times \|action\|_2^2$,
> where $w_{control}$ is `ctrl_cost_weight` (default is $0.5$).
> - *contact_cost*:
> A negative reward to penalize the Ant if the external contact forces are too large.
> $w_{contact} \times \|F_{contact}\|_2^2$, where
> $w_{contact}$ is `contact_cost_weight` (default is $5\times10^{-4}$),
> $F_{contact}$ are the external contact forces clipped by `contact_force_range` (see `cfrc_ext` section on Observation Space).
>
> `info` contains the individual reward terms.
>
> But if `use_contact_forces=False` on `v4`
> The total reward returned is ***reward*** *=* *healthy_reward + forward_reward - ctrl_cost*.
>
> **Starting State.** The initial position state is $[0.0, 0.0, 0.75, 1.0, 0.0, ... 0.0] + \mathcal{U}_{[-reset\\_noise\\_scale \times I_{15}, reset\\_noise\\_scale \times I_{15}]}$.
> The initial velocity state is $\mathcal{N}(0_{14}, reset\\_noise\\_scale^2 \times I_{14})$.
>
> where $\mathcal{N}$ is the multivariate normal distribution and $\mathcal{U}$ is the multivariate uniform continuous distribution.
>
> Note that the z- and x-coordinates are non-zero so that the ant can immediately stand up and face forward (x-axis).
>
> **Episode End.**
>
> **Termination.**
> If `terminate_when_unhealthy is True` (the default), the environment terminates when the Ant is unhealthy.
> the Ant is unhealthy if any of the following happens:
>
> 1. Any of the state space values is no longer finite.
> 2. The z-coordinate of the torso (the height) is **not** in the closed interval given by the `healthy_z_range` argument (default is $[0.2, 1.0]$).
>
> **Truncation.**
> The default duration of an episode is 1000 timesteps.
>
> — Gymnasium 1.3.0, [`gymnasium/envs/mujoco/ant_v5.py`](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/mujoco/ant_v5.py); [documentation](https://gymnasium.farama.org/environments/mujoco/ant/)

**Origin.** Schulman, Moritz, Levine, Jordan & Abbeel (2016); v5 by Kallinteris-Andreas.

**Here.** Gymnasium's model on MJX; matches to 1e-8. Gymnasium's time limit is 1000 steps.

### `humanoid`

<img src="gifs/humanoid.gif" alt="16 rollouts of a policy trained on humanoid" loading="lazy">

> **Description.** This environment is based on the environment introduced by Tassa, Erez and Todorov in ["Synthesis and stabilization of complex behaviors through online trajectory optimization"](https://ieeexplore.ieee.org/document/6386025).
> The 3D bipedal robot is designed to simulate a human.
> It has a torso (abdomen) with a pair of legs and arms, and a pair of tendons connecting the hips to the knees.
> The legs each consist of three body parts (thigh, shin, foot), and the arms consist of two body parts (upper arm, forearm).
> The goal of the environment is to walk forward as fast as possible without falling over.
>
> **Rewards.** The total reward is: ***reward*** *=* *healthy_reward + forward_reward - ctrl_cost - contact_cost*.
>
> - *healthy_reward*:
> Every timestep that the Humanoid is alive (see definition in section "Episode End"),
> it gets a reward of fixed value `healthy_reward` (default is $5$).
> - *forward_reward*:
> A reward for moving forward,
> this reward would be positive if the Humanoid moves forward (in the positive $x$ direction / in the right direction).
> $w_{forward} \times \frac{dx}{dt}$, where
> $dx$ is the displacement of the center of mass ($x_{after-action} - x_{before-action}$),
> $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is $5$),
> and `frametime` which is $0.001$ - so the default is $dt = 5 \times 0.003 = 0.015$,
> $w_{forward}$ is the `forward_reward_weight` (default is $1.25$).
> - *ctrl_cost*:
> A negative reward to penalize the Humanoid for taking actions that are too large.
> $w_{control} \times \|action\|_2^2$,
> where $w_{control}$ is `ctrl_cost_weight` (default is $0.1$).
> - *contact_cost*:
> A negative reward to penalize the Humanoid if the external contact forces are too large.
> $w_{contact} \times clamp(contact\\_cost\\_range, \|F_{contact}\|_2^2)$, where
> $w_{contact}$ is `contact_cost_weight` (default is $5\times10^{-7}$),
> $F_{contact}$ are the external contact forces (see `cfrc_ext` section on observation).
>
> `info` contains the individual reward terms.
>
> **Note:** There is a bug in the `Humanoid-v4` environment that causes *contact_cost* to always be 0.
>
> **Starting State.** The initial position state is $[0.0, 0.0, 1.4, 1.0, 0.0, ... 0.0] + \mathcal{U}_{[-reset\\_noise\\_scale \times I_{24}, reset\\_noise\\_scale \times I_{24}]}$.
> The initial velocity state is $\mathcal{U}_{[-reset\\_noise\\_scale \times I_{23}, reset\\_noise\\_scale \times I_{23}]}$.
>
> where $\mathcal{U}$ is the multivariate uniform continuous distribution.
>
> Note that the z- and x-coordinates are non-zero so that the humanoid can immediately stand up and face forward (x-axis).
>
> **Episode End.**
>
> **Termination.**
> If `terminate_when_unhealthy is True` (the default), the environment terminates when the Humanoid is unhealthy.
> The Humanoid is said to be unhealthy if any of the following happens:
>
> 1. The z-coordinate of the torso (the height) is **not** in the closed interval given by the `healthy_z_range` argument (default is $[1.0, 2.0]$).
>
> **Truncation.**
> The default duration of an episode is 1000 timesteps.
>
> — Gymnasium 1.3.0, [`gymnasium/envs/mujoco/humanoid_v5.py`](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/mujoco/humanoid_v5.py); [documentation](https://gymnasium.farama.org/environments/mujoco/humanoid/)

**Origin.** Tassa, Erez & Todorov (2012); v5 by Kallinteris-Andreas.

**Here.** Brax's simplified model for MJX: Euler integration, Newton with 8 iterations, only the feet touch the floor. Over 5 steps, positions stay within 2% of Gymnasium's. Gymnasium's time limit is 1000 steps.

### `humanoid-standup`

<img src="gifs/humanoid-standup.gif" alt="16 rollouts of a policy trained on humanoid-standup" loading="lazy">

> **Description.** This environment is based on the environment introduced by Tassa, Erez and Todorov in ["Synthesis and stabilization of complex behaviors through online trajectory optimization"](https://ieeexplore.ieee.org/document/6386025).
> The 3D bipedal robot is designed to simulate a human.
> It has a torso (abdomen) with a pair of legs and arms, and a pair of tendons connecting the hips to the knees.
> The legs each consist of three body parts (thigh, shin, foot), and the arms consist of two body parts (upper arm, forearm).
> The environment starts with the humanoid laying on the ground, and then the goal of the environment is to make the humanoid stand up and then keep it standing by applying torques to the various hinges.
>
> **Rewards.** The total reward is: ***reward*** *=* *uph_cost + 1 - quad_ctrl_cost - quad_impact_cost*.
>
> - *uph_cost*:
> A reward for moving up (trying to stand up).
> This is not a relative reward, measuring how far up the robot has moved since the last timestep,
> but an absolute reward measuring how far up the Humanoid has moved up in total.
> It is measured as $w_{uph} \times \frac{z_{after\\_action} - 0}{dt}$,
> where $z_{after\\_action}$ is the z coordinate of the torso after taking an action,
> and $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is $5$),
> and `frametime`, which is $0.01$ - so the default is $dt = 5 \times 0.01 = 0.05$,
> and $w_{uph}$ is `uph_cost_weight` (default is $1$).
> - *quad_ctrl_cost*:
> A negative reward to penalize the Humanoid for taking actions that are too large.
> $w_{quad\\_control} \times \|action\|_2^2$,
> where $w_{quad\\_control}$ is `ctrl_cost_weight` (default is $0.1$).
> - *impact_cost*:
> A negative reward to penalize the Humanoid if the external contact forces are too large.
> $w_{impact} \times clamp(impact\\_cost\\_range, \|F_{contact}\|_2^2)$, where
> $w_{impact}$ is `impact_cost_weight` (default is $5\times10^{-7}$),
> $F_{contact}$ are the external contact forces (see `cfrc_ext` section on Observation Space).
>
> `info` contains the individual reward terms.
>
> **Starting State.** The initial position state is $[0.0, 0.0, 1.4, 1.0, 0.0, ... 0.0] + \mathcal{U}_{[-reset\\_noise\\_scale \times I_{24}, reset\\_noise\\_scale \times I_{24}]}$.
> The initial velocity state is $\mathcal{U}_{[-reset\\_noise\\_scale \times I_{23}, reset\\_noise\\_scale \times I_{23}]}$.
>
> where $\mathcal{U}$ is the multivariate uniform continuous distribution.
>
> Note that the z- and x-coordinates are non-zero so that the humanoid immediately lies down and faces forward (x-axis).
>
> **Episode End.**
>
> **Termination.**
> The Humanoid never terminates.
>
> **Truncation.**
> The default duration of an episode is 1000 timesteps.
>
> — Gymnasium 1.3.0, [`gymnasium/envs/mujoco/humanoidstandup_v5.py`](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/mujoco/humanoidstandup_v5.py); [documentation](https://gymnasium.farama.org/environments/mujoco/humanoid_standup/)

**Origin.** Tassa, Erez & Todorov (2012); v5 by Kallinteris-Andreas.

**Here.** Brax's simplified model for MJX, as for `humanoid`. Gymnasium's time limit is 1000 steps.

## Tiger

### `tiger`

The tiger problem of Leslie Pack Kaelbling, Michael L. Littman and Anthony R. Cassandra,
"Planning and acting in partially observable stochastic domains", *Artificial
Intelligence* 101, 1998. It has no render, so no GIF.

## MuJoCo Playground

Registered by `import jax_gym.playground` (the `playground` extra) under Playground's names in dashed lower case. Dynamics, rewards and observations are Playground's own; each config's `episode_length` is its time limit.

### DeepMind Control Suite

#### Acrobot

> Acrobot (4, 1, 6): The underactuated double pendulum, torque applied to the second joint. The goal is to swing up and balance. Despite being low-dimensional, this is not an easy control problem. The physical model conforms to (Coulom, 2002) rather than the earlier (Spong, 1995). Both swingup and swingup_sparse tasks with smooth and sparse rewards, respectively.
>
> — Tassa et al., [DeepMind Control Suite](https://arxiv.org/abs/1801.00690), 2018, section 3

##### `playground/acrobot-swingup`

<img src="gifs/playground__acrobot-swingup.gif" alt="16 rollouts of a policy trained on playground/acrobot-swingup" loading="lazy">

> Acrobot environment.
>
> — MuJoCo Playground 0.2.0, [`_src/dm_control_suite/acrobot.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/dm_control_suite/acrobot.py) (`Balance`)

##### `playground/acrobot-swingup-sparse`

<img src="gifs/playground__acrobot-swingup-sparse.gif" alt="16 rollouts of a policy trained on playground/acrobot-swingup-sparse" loading="lazy">

> Acrobot environment.
>
> — MuJoCo Playground 0.2.0, [`_src/dm_control_suite/acrobot.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/dm_control_suite/acrobot.py) (`Balance`)

#### Ball in cup

> Ball in cup (8, 2, 8): A planar ball-in-cup task. An actuated planar receptacle can translate in the vertical plane in order to swing and catch a ball attached to its bottom. The catch task has a sparse reward: 1 when the ball is in the cup, 0 otherwise.
>
> — Tassa et al., [DeepMind Control Suite](https://arxiv.org/abs/1801.00690), 2018, section 3

##### `playground/ball-in-cup`

<img src="gifs/playground__ball-in-cup.gif" alt="16 rollouts of a policy trained on playground/ball-in-cup" loading="lazy">

> Ball in cup environment.
>
> — MuJoCo Playground 0.2.0, [`_src/dm_control_suite/ball_in_cup.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/dm_control_suite/ball_in_cup.py) (`BallInCup`)

#### Cart-pole

> Cart-pole (4, 1, 5): Swing up and balance an unactuated pole by applying forces to a cart at its base. The physical model conforms to (Barto et al., 1983). Four benchmarking tasks: in swingup and swingup_sparse the pole starts pointing down while in balance and balance_sparse the pole starts near the upright.
>
> — Tassa et al., [DeepMind Control Suite](https://arxiv.org/abs/1801.00690), 2018, section 3

##### `playground/cartpole-balance`

<img src="gifs/playground__cartpole-balance.gif" alt="16 rollouts of a policy trained on playground/cartpole-balance" loading="lazy">

> Cartpole environment.
>
> Cartpole environment with balance task.
>
> — MuJoCo Playground 0.2.0, [`_src/dm_control_suite/cartpole.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/dm_control_suite/cartpole.py) (`Balance`)

##### `playground/cartpole-balance-sparse`

<img src="gifs/playground__cartpole-balance-sparse.gif" alt="16 rollouts of a policy trained on playground/cartpole-balance-sparse" loading="lazy">

> Cartpole environment.
>
> Cartpole environment with balance task.
>
> — MuJoCo Playground 0.2.0, [`_src/dm_control_suite/cartpole.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/dm_control_suite/cartpole.py) (`Balance`)

##### `playground/cartpole-swingup`

<img src="gifs/playground__cartpole-swingup.gif" alt="16 rollouts of a policy trained on playground/cartpole-swingup" loading="lazy">

> Cartpole environment.
>
> Cartpole environment with balance task.
>
> — MuJoCo Playground 0.2.0, [`_src/dm_control_suite/cartpole.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/dm_control_suite/cartpole.py) (`Balance`)

##### `playground/cartpole-swingup-sparse`

<img src="gifs/playground__cartpole-swingup-sparse.gif" alt="16 rollouts of a policy trained on playground/cartpole-swingup-sparse" loading="lazy">

> Cartpole environment.
>
> Cartpole environment with balance task.
>
> — MuJoCo Playground 0.2.0, [`_src/dm_control_suite/cartpole.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/dm_control_suite/cartpole.py) (`Balance`)

#### Cheetah

> Cheetah (18, 6, 17): A running planar biped based on (Wawrzyński, 2009). The reward r is linearly proportional to the forward velocity v up to a maximum of 10m/s i.e. r(v) = max(0, min(v/10, 1)).
>
> — Tassa et al., [DeepMind Control Suite](https://arxiv.org/abs/1801.00690), 2018, section 3

##### `playground/cheetah-run`

<img src="gifs/playground__cheetah-run.gif" alt="16 rollouts of a policy trained on playground/cheetah-run" loading="lazy">

> Cheetah environment.
>
> Cheetah running environment.
>
> — MuJoCo Playground 0.2.0, [`_src/dm_control_suite/cheetah.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/dm_control_suite/cheetah.py) (`Run`)

#### Finger

> Finger (6, 2, 12): A 3-DoF toy manipulation problem based on (Tassa and Todorov, 2010). A planar ‘finger’ is required to rotate a body on an unactuated hinge. In the turn_easy and turn_hard tasks, the tip of the free body must overlap with a target (the target is smaller for the turn_hard task). In the spin task, the body must be continually rotated.
>
> — Tassa et al., [DeepMind Control Suite](https://arxiv.org/abs/1801.00690), 2018, section 3

##### `playground/finger-spin`

<img src="gifs/playground__finger-spin.gif" alt="16 rollouts of a policy trained on playground/finger-spin" loading="lazy">

> Finger environment.
>
> Changes from the dm_control implementation:
>
> - Changed integrator to implicitfast.
> - Reduced the timestep to 0.005 (from 0.01).
>
> Spin environment.
>
> — MuJoCo Playground 0.2.0, [`_src/dm_control_suite/finger.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/dm_control_suite/finger.py) (`Spin`)

##### `playground/finger-turn-easy`

<img src="gifs/playground__finger-turn-easy.gif" alt="16 rollouts of a policy trained on playground/finger-turn-easy" loading="lazy">

> Finger environment.
>
> Changes from the dm_control implementation:
>
> - Changed integrator to implicitfast.
> - Reduced the timestep to 0.005 (from 0.01).
>
> Turn environment.
>
> — MuJoCo Playground 0.2.0, [`_src/dm_control_suite/finger.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/dm_control_suite/finger.py) (`Turn`)

##### `playground/finger-turn-hard`

<img src="gifs/playground__finger-turn-hard.gif" alt="16 rollouts of a policy trained on playground/finger-turn-hard" loading="lazy">

> Finger environment.
>
> Changes from the dm_control implementation:
>
> - Changed integrator to implicitfast.
> - Reduced the timestep to 0.005 (from 0.01).
>
> Turn environment.
>
> — MuJoCo Playground 0.2.0, [`_src/dm_control_suite/finger.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/dm_control_suite/finger.py) (`Turn`)

#### Fish

> Fish (26, 5, 24): A fish is required to swim to a target. This domain relies on MuJoCo’s simplified fluid dynamics. Two tasks: in the upright task, the fish is rewarded only for righting itself with respect to the vertical, while in the swim task it is also rewarded for swimming to the target.
>
> — Tassa et al., [DeepMind Control Suite](https://arxiv.org/abs/1801.00690), 2018, section 3

##### `playground/fish-swim`

<img src="gifs/playground__fish-swim.gif" alt="16 rollouts of a policy trained on playground/fish-swim" loading="lazy">

> Fish environment.
>
> — MuJoCo Playground 0.2.0, [`_src/dm_control_suite/fish.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/dm_control_suite/fish.py) (`Swim`)

#### Hopper

> Hopper (14, 4, 15): The planar one-legged hopper introduced in (Lillicrap et al., 2015), initialised in a random configuration. In the stand task it is rewarded for bringing its torso to a minimal height. In the hop task it is rewarded for torso height and forward velocity.
>
> — Tassa et al., [DeepMind Control Suite](https://arxiv.org/abs/1801.00690), 2018, section 3

##### `playground/hopper-hop`

<img src="gifs/playground__hopper-hop.gif" alt="16 rollouts of a policy trained on playground/hopper-hop" loading="lazy">

> Hopper environment.
>
> — MuJoCo Playground 0.2.0, [`_src/dm_control_suite/hopper.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/dm_control_suite/hopper.py) (`Hopper`)

##### `playground/hopper-stand`

<img src="gifs/playground__hopper-stand.gif" alt="16 rollouts of a policy trained on playground/hopper-stand" loading="lazy">

> Hopper environment.
>
> — MuJoCo Playground 0.2.0, [`_src/dm_control_suite/hopper.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/dm_control_suite/hopper.py) (`Hopper`)

#### Humanoid

> Humanoid (54, 21, 67): A simplified humanoid with 21 joints, based on the model in (Tassa et al., 2012). Three tasks: stand, walk and run are differentiated by the desired horizontal speed of 0, 1 and 10m/s, respectively. Observations are in an egocentric frame and many movement styles are possible solutions e.g. running backwards or sideways. This facilitates exploration of local optima.
>
> — Tassa et al., [DeepMind Control Suite](https://arxiv.org/abs/1801.00690), 2018, section 3

##### `playground/humanoid-stand`

**Here.** Training diverged, so no GIF.

> Humanoid environment.
>
> — MuJoCo Playground 0.2.0, [`_src/dm_control_suite/humanoid.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/dm_control_suite/humanoid.py) (`Humanoid`)

##### `playground/humanoid-walk`

**Here.** Training diverged, so no GIF.

> Humanoid environment.
>
> — MuJoCo Playground 0.2.0, [`_src/dm_control_suite/humanoid.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/dm_control_suite/humanoid.py) (`Humanoid`)

##### `playground/humanoid-run`

**Here.** Training diverged, so no GIF.

> Humanoid environment.
>
> — MuJoCo Playground 0.2.0, [`_src/dm_control_suite/humanoid.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/dm_control_suite/humanoid.py) (`Humanoid`)

#### Pendulum

> Pendulum (2, 1, 3): The classic inverted pendulum. The torque-limited actuator is 1/6th as strong as required to lift the mass from motionless horizontal, necessitating several swings to swing up and balance. The swingup task has a simple sparse reward: 1 when the pole is within 30° of the vertical and 0 otherwise.
>
> — Tassa et al., [DeepMind Control Suite](https://arxiv.org/abs/1801.00690), 2018, section 3

##### `playground/pendulum-swingup`

<img src="gifs/playground__pendulum-swingup.gif" alt="16 rollouts of a policy trained on playground/pendulum-swingup" loading="lazy">

> Pendulum environment.
>
> Swingup environment.
>
> — MuJoCo Playground 0.2.0, [`_src/dm_control_suite/pendulum.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/dm_control_suite/pendulum.py) (`SwingUp`)

#### Point-mass

> Point-mass (4, 2, 4): A planar point-mass receives a reward of 1 when within a target at the origin. In the easy task, one of simplest in the suite, the 2 actuators correspond to the global x and y axes. In the hard task the gain matrix from the controls to the axes is randomised for each episode, making it impossible to solve by memory-less agents; this task is not in the benchmarking set.
>
> — Tassa et al., [DeepMind Control Suite](https://arxiv.org/abs/1801.00690), 2018, section 3

##### `playground/point-mass`

<img src="gifs/playground__point-mass.gif" alt="16 rollouts of a policy trained on playground/point-mass" loading="lazy">

> Point mass environment.
>
> — MuJoCo Playground 0.2.0, [`_src/dm_control_suite/point_mass.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/dm_control_suite/point_mass.py) (`PointMass`)

#### Reacher

> Reacher (4, 2, 7): The simple two-link planar reacher with a randomised target location. The reward is one when the end effector penetrates the target sphere. In the easy task the target sphere is bigger than on the hard task (shown on the left).
>
> — Tassa et al., [DeepMind Control Suite](https://arxiv.org/abs/1801.00690), 2018, section 3

##### `playground/reacher-easy`

<img src="gifs/playground__reacher-easy.gif" alt="16 rollouts of a policy trained on playground/reacher-easy" loading="lazy">

> Reacher environment.
>
> — MuJoCo Playground 0.2.0, [`_src/dm_control_suite/reacher.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/dm_control_suite/reacher.py) (`Reacher`)

##### `playground/reacher-hard`

<img src="gifs/playground__reacher-hard.gif" alt="16 rollouts of a policy trained on playground/reacher-hard" loading="lazy">

> Reacher environment.
>
> — MuJoCo Playground 0.2.0, [`_src/dm_control_suite/reacher.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/dm_control_suite/reacher.py) (`Reacher`)

#### Swimmer

> Swimmer (2k+4, k−1, 4k+1): This procedurally generated k-link planar swimmer is based on (Coulom, 2002) but using MuJoCo’s high-Reynolds fluid drag model. A reward of 1 is provided when the nose is inside the target and decreases smoothly with distance like a Lorentzian. The two instantiations provided in the benchmarking set are the 6-link and 15-link swimmers.
>
> — Tassa et al., [DeepMind Control Suite](https://arxiv.org/abs/1801.00690), 2018, section 3

##### `playground/swimmer-swimmer6`

<img src="gifs/playground__swimmer-swimmer6.gif" alt="16 rollouts of a policy trained on playground/swimmer-swimmer6" loading="lazy">

> Swimmer environment.
>
> — MuJoCo Playground 0.2.0, [`_src/dm_control_suite/swimmer.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/dm_control_suite/swimmer.py) (`Swim`)

#### Walker

> Walker (18, 6, 24): An improved planar walker based on the one introduced in (Lillicrap et al., 2015). In the stand task reward is a combination of terms encouraging an upright torso and some minimal torso height. The walk and run tasks include a component encouraging forward velocity.
>
> — Tassa et al., [DeepMind Control Suite](https://arxiv.org/abs/1801.00690), 2018, section 3

##### `playground/walker-stand`

<img src="gifs/playground__walker-stand.gif" alt="16 rollouts of a policy trained on playground/walker-stand" loading="lazy">

> Walker environment.
>
> A planar walker task.
>
> — MuJoCo Playground 0.2.0, [`_src/dm_control_suite/walker.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/dm_control_suite/walker.py) (`PlanarWalker`)

##### `playground/walker-walk`

<img src="gifs/playground__walker-walk.gif" alt="16 rollouts of a policy trained on playground/walker-walk" loading="lazy">

> Walker environment.
>
> A planar walker task.
>
> — MuJoCo Playground 0.2.0, [`_src/dm_control_suite/walker.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/dm_control_suite/walker.py) (`PlanarWalker`)

##### `playground/walker-run`

<img src="gifs/playground__walker-run.gif" alt="16 rollouts of a policy trained on playground/walker-run" loading="lazy">

> Walker environment.
>
> A planar walker task.
>
> — MuJoCo Playground 0.2.0, [`_src/dm_control_suite/walker.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/dm_control_suite/walker.py) (`PlanarWalker`)

### Locomotion

> Locomotion environments in MuJoCo Playground are implemented for multiple quadrupeds and bipeds (Figure 1 left). The quadrupeds include the Unitree Go1, Boston Dynamics Spot, and Google Barkour [6], while the humanoids include the Berkeley Humanoid [35], Unitree H1 and G1, Booster T1, and the Robotis OP3. For each robot embodiment, we implement a joystick environment that learns to track a velocity command consisting of base linear velocities in both the forward and lateral directions, as well as a desired yaw rate.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), section II-B

#### `playground/apollo-joystick-flat-terrain`

Playground's `ApolloJoystickFlatTerrain`.

<img src="gifs/playground__apollo-joystick-flat-terrain.gif" alt="16 rollouts of a policy trained on playground/apollo-joystick-flat-terrain" loading="lazy">

> Joystick task for Apollo.
>
> Track a joystick command.
>
> — MuJoCo Playground 0.2.0, [`_src/locomotion/apollo/joystick.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/apollo/joystick.py) (`Joystick`)

> We implement a joystick locomotion task as in [50, 25], where the command is specified by three values indicating the desired forward velocity, lateral velocity, and turning rate of the robot’s root body.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), section IV-B

> For humanoid locomotion tasks, a phase variable [55] is introduced to shape the gait.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), appendix B.21

#### `playground/barkour-joystick`

Playground's `BarkourJoystick`.

<img src="gifs/playground__barkour-joystick.gif" alt="16 rollouts of a policy trained on playground/barkour-joystick" loading="lazy">

> Joystick environment for Barkour.
>
> — MuJoCo Playground 0.2.0, [`_src/locomotion/barkour/joystick.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/barkour/joystick.py) (`Joystick`)

> We implement a joystick locomotion task as in [50, 25], where the command is specified by three values indicating the desired forward velocity, lateral velocity, and turning rate of the robot’s root body.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), section IV-B

#### `playground/berkeley-humanoid-joystick-flat-terrain`

Playground's `BerkeleyHumanoidJoystickFlatTerrain`.

<img src="gifs/playground__berkeley-humanoid-joystick-flat-terrain.gif" alt="16 rollouts of a policy trained on playground/berkeley-humanoid-joystick-flat-terrain" loading="lazy">

> Joystick task for Berkeley Humanoid.
>
> Track a joystick command.
>
> — MuJoCo Playground 0.2.0, [`_src/locomotion/berkeley_humanoid/joystick.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/berkeley_humanoid/joystick.py) (`Joystick`)

> We implement a joystick locomotion task as in [50, 25], where the command is specified by three values indicating the desired forward velocity, lateral velocity, and turning rate of the robot’s root body.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), section IV-B

> For humanoid locomotion tasks, a phase variable [55] is introduced to shape the gait.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), appendix B.21

#### `playground/berkeley-humanoid-joystick-rough-terrain`

Playground's `BerkeleyHumanoidJoystickRoughTerrain`.

<img src="gifs/playground__berkeley-humanoid-joystick-rough-terrain.gif" alt="16 rollouts of a policy trained on playground/berkeley-humanoid-joystick-rough-terrain" loading="lazy">

> Joystick task for Berkeley Humanoid.
>
> Track a joystick command.
>
> — MuJoCo Playground 0.2.0, [`_src/locomotion/berkeley_humanoid/joystick.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/berkeley_humanoid/joystick.py) (`Joystick`)

> We implement a joystick locomotion task as in [50, 25], where the command is specified by three values indicating the desired forward velocity, lateral velocity, and turning rate of the robot’s root body.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), section IV-B

> For humanoid locomotion tasks, a phase variable [55] is introduced to shape the gait.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), appendix B.21

> The rough terrain is modeled as a heightfield generated from Perlin noise.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), appendix B.25

#### `playground/g1-joystick-flat-terrain`

Playground's `G1JoystickFlatTerrain`.

<img src="gifs/playground__g1-joystick-flat-terrain.gif" alt="16 rollouts of a policy trained on playground/g1-joystick-flat-terrain" loading="lazy">

> Joystick task for Unitree G1.
>
> Track a joystick command.
>
> — MuJoCo Playground 0.2.0, [`_src/locomotion/g1/joystick.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/g1/joystick.py) (`Joystick`)

> We implement a joystick locomotion task as in [50, 25], where the command is specified by three values indicating the desired forward velocity, lateral velocity, and turning rate of the robot’s root body.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), section IV-B

> For humanoid locomotion tasks, a phase variable [55] is introduced to shape the gait.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), appendix B.21

#### `playground/g1-joystick-rough-terrain`

Playground's `G1JoystickRoughTerrain`.

<img src="gifs/playground__g1-joystick-rough-terrain.gif" alt="16 rollouts of a policy trained on playground/g1-joystick-rough-terrain" loading="lazy">

> Joystick task for Unitree G1.
>
> Track a joystick command.
>
> — MuJoCo Playground 0.2.0, [`_src/locomotion/g1/joystick.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/g1/joystick.py) (`Joystick`)

> We implement a joystick locomotion task as in [50, 25], where the command is specified by three values indicating the desired forward velocity, lateral velocity, and turning rate of the robot’s root body.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), section IV-B

> For humanoid locomotion tasks, a phase variable [55] is introduced to shape the gait.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), appendix B.21

> The rough terrain is modeled as a heightfield generated from Perlin noise.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), appendix B.25

#### `playground/go1-joystick-flat-terrain`

Playground's `Go1JoystickFlatTerrain`.

<img src="gifs/playground__go1-joystick-flat-terrain.gif" alt="16 rollouts of a policy trained on playground/go1-joystick-flat-terrain" loading="lazy">

> Joystick task for Go1.
>
> Track a joystick command.
>
> — MuJoCo Playground 0.2.0, [`_src/locomotion/go1/joystick.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/go1/joystick.py) (`Joystick`)

> We implement a joystick locomotion task as in [50, 25], where the command is specified by three values indicating the desired forward velocity, lateral velocity, and turning rate of the robot’s root body.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), section IV-B

#### `playground/go1-joystick-rough-terrain`

Playground's `Go1JoystickRoughTerrain`.

<img src="gifs/playground__go1-joystick-rough-terrain.gif" alt="16 rollouts of a policy trained on playground/go1-joystick-rough-terrain" loading="lazy">

> Joystick task for Go1.
>
> Track a joystick command.
>
> — MuJoCo Playground 0.2.0, [`_src/locomotion/go1/joystick.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/go1/joystick.py) (`Joystick`)

> We implement a joystick locomotion task as in [50, 25], where the command is specified by three values indicating the desired forward velocity, lateral velocity, and turning rate of the robot’s root body.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), section IV-B

> The rough terrain is modeled as a heightfield generated from Perlin noise.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), appendix B.25

#### `playground/go1-getup`

Playground's `Go1Getup`.

**Here.** No policy was trained for it, so no GIF.

> Fall recovery task for the Go1.
>
> Recover from a fall and stand up.
>
> Observation space:
>     - Gyroscope readings (3)
>     - Gravity vector (3)
>     - Joint angles (12)
>     - Last action (12)
>
> Action space: Joint angles (12) scaled by a factor and added to the current
> joint angles. We tried using the same action space used in the joystick task
> where the output of the policy is added to the nominal "home" pose but it
> didn't work as well as adding to the current joint configuration. I suspect
> this is because the latter gives the policy a wider initial range of motion.
>
> Reward function:
>     - Orientation: The torso should be upright.
>     - Torso height: The torso should be at a desired height. This is to
>         prevent the robot from flipping over and just lying on the ground.
>     - Posture: The robot should be in the neural pose. This reward is only
>         given when the robot is upright and at the desired height.
>     - Stand still: Policy outputs should be zero once the robot is upright
>         and at the desired height. This minimizes jittering.
>     The next two rewards aren't really needed but promote better sim2real
>         transfer (in theory):
>     - Torques: Minimize joint torques.
>     - Action rate: Minimize the first and second derivative of actions.
>
> — MuJoCo Playground 0.2.0, [`_src/locomotion/go1/getup.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/go1/getup.py) (`Getup`)

> For fall recovery, we follow [30, 58], enabling the robot to return to a stable “home” posture from arbitrary fallen configurations.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), section IV-B

#### `playground/go1-handstand`

Playground's `Go1Handstand`.

<img src="gifs/playground__go1-handstand.gif" alt="16 rollouts of a policy trained on playground/go1-handstand" loading="lazy">

> Handstand task for Go1.
>
> — MuJoCo Playground 0.2.0, [`_src/locomotion/go1/handstand.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/go1/handstand.py) (`Handstand`)

> Additionally, we design policies for handstand and footstand tasks, in which the robot balances on the front or hind legs, respectively, while minimizing actuator torque.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), section IV-B

#### `playground/go1-footstand`

Playground's `Go1Footstand`.

<img src="gifs/playground__go1-footstand.gif" alt="16 rollouts of a policy trained on playground/go1-footstand" loading="lazy">

> Handstand task for Go1.
>
> Footstand task for Go1.
>
> — MuJoCo Playground 0.2.0, [`_src/locomotion/go1/handstand.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/go1/handstand.py) (`Footstand`)

> Additionally, we design policies for handstand and footstand tasks, in which the robot balances on the front or hind legs, respectively, while minimizing actuator torque.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), section IV-B

#### `playground/h1-inplace-gait-tracking`

Playground's `H1InplaceGaitTracking`.

<img src="gifs/playground__h1-inplace-gait-tracking.gif" alt="16 rollouts of a policy trained on playground/h1-inplace-gait-tracking" loading="lazy">

> Inplace gait tracking for H1.
>
> — MuJoCo Playground 0.2.0, [`_src/locomotion/h1/inplace_gait_tracking.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/h1/inplace_gait_tracking.py) (`InplaceGaitTracking`)

> For humanoid locomotion tasks, a phase variable [55] is introduced to shape the gait.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), appendix B.21

#### `playground/h1-joystick-gait-tracking`

Playground's `H1JoystickGaitTracking`.

<img src="gifs/playground__h1-joystick-gait-tracking.gif" alt="16 rollouts of a policy trained on playground/h1-joystick-gait-tracking" loading="lazy">

> Joystick gait tracking for H1.
>
> A class for tracking joystick-controlled gait in a simulated environment.
>
> — MuJoCo Playground 0.2.0, [`_src/locomotion/h1/joystick_gait_tracking.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/h1/joystick_gait_tracking.py) (`JoystickGaitTracking`)

> We implement a joystick locomotion task as in [50, 25], where the command is specified by three values indicating the desired forward velocity, lateral velocity, and turning rate of the robot’s root body.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), section IV-B

> For humanoid locomotion tasks, a phase variable [55] is introduced to shape the gait.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), appendix B.21

#### `playground/op3-joystick`

Playground's `Op3Joystick`.

<img src="gifs/playground__op3-joystick.gif" alt="16 rollouts of a policy trained on playground/op3-joystick" loading="lazy">

> Joystick for OP3.
>
> Track a joystick command.
>
> — MuJoCo Playground 0.2.0, [`_src/locomotion/op3/joystick.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/op3/joystick.py) (`Joystick`)

> We implement a joystick locomotion task as in [50, 25], where the command is specified by three values indicating the desired forward velocity, lateral velocity, and turning rate of the robot’s root body.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), section IV-B

> For humanoid locomotion tasks, a phase variable [55] is introduced to shape the gait.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), appendix B.21

#### `playground/spot-flat-terrain-joystick`

Playground's `SpotFlatTerrainJoystick`.

<img src="gifs/playground__spot-flat-terrain-joystick.gif" alt="16 rollouts of a policy trained on playground/spot-flat-terrain-joystick" loading="lazy">

> Joystick task for Spot.
>
> Track a joystick command.
>
> — MuJoCo Playground 0.2.0, [`_src/locomotion/spot/joystick.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/spot/joystick.py) (`Joystick`)

> We implement a joystick locomotion task as in [50, 25], where the command is specified by three values indicating the desired forward velocity, lateral velocity, and turning rate of the robot’s root body.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), section IV-B

#### `playground/spot-getup`

Playground's `SpotGetup`.

<img src="gifs/playground__spot-getup.gif" alt="16 rollouts of a policy trained on playground/spot-getup" loading="lazy">

> Fall recovery task for Spot.
>
> Recover from a fall and stand up.
>
> — MuJoCo Playground 0.2.0, [`_src/locomotion/spot/getup.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/spot/getup.py) (`Getup`)

> For fall recovery, we follow [30, 58], enabling the robot to return to a stable “home” posture from arbitrary fallen configurations.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), section IV-B

#### `playground/spot-joystick-gait-tracking`

Playground's `SpotJoystickGaitTracking`.

<img src="gifs/playground__spot-joystick-gait-tracking.gif" alt="16 rollouts of a policy trained on playground/spot-joystick-gait-tracking" loading="lazy">

> Joystick task with gait control for Spot.
>
> Joystick task with gait control.
>
> — MuJoCo Playground 0.2.0, [`_src/locomotion/spot/joystick_gait_tracking.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/spot/joystick_gait_tracking.py) (`JoystickGaitTracking`)

> We implement a joystick locomotion task as in [50, 25], where the command is specified by three values indicating the desired forward velocity, lateral velocity, and turning rate of the robot’s root body.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), section IV-B

#### `playground/t1-joystick-flat-terrain`

Playground's `T1JoystickFlatTerrain`.

<img src="gifs/playground__t1-joystick-flat-terrain.gif" alt="16 rollouts of a policy trained on playground/t1-joystick-flat-terrain" loading="lazy">

> Joystick task for Booster T1.
>
> Track a joystick command.
>
> — MuJoCo Playground 0.2.0, [`_src/locomotion/t1/joystick.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/t1/joystick.py) (`Joystick`)

> We implement a joystick locomotion task as in [50, 25], where the command is specified by three values indicating the desired forward velocity, lateral velocity, and turning rate of the robot’s root body.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), section IV-B

> For humanoid locomotion tasks, a phase variable [55] is introduced to shape the gait.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), appendix B.21

#### `playground/t1-joystick-rough-terrain`

Playground's `T1JoystickRoughTerrain`.

<img src="gifs/playground__t1-joystick-rough-terrain.gif" alt="16 rollouts of a policy trained on playground/t1-joystick-rough-terrain" loading="lazy">

> Joystick task for Booster T1.
>
> Track a joystick command.
>
> — MuJoCo Playground 0.2.0, [`_src/locomotion/t1/joystick.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/t1/joystick.py) (`Joystick`)

> We implement a joystick locomotion task as in [50, 25], where the command is specified by three values indicating the desired forward velocity, lateral velocity, and turning rate of the robot’s root body.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), section IV-B

> For humanoid locomotion tasks, a phase variable [55] is introduced to shape the gait.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), appendix B.21

> The rough terrain is modeled as a heightfield generated from Perlin noise.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), appendix B.25

### Manipulation

> With the Leap Hand [56] robot, we demonstrate contact-rich dexterous re-orientation of a block. Using the Franka Emika Panda and Robotiq gripper, we show re-orientation of a yoga block using high frequency torque control.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), section II-C

#### `playground/aloha-hand-over`

Playground's `AlohaHandOver`.

<img src="gifs/playground__aloha-hand-over.gif" alt="16 rollouts of a policy trained on playground/aloha-hand-over" loading="lazy">

> Handover task for ALOHA.
>
> Single peg insertion task for ALOHA.
>
> — MuJoCo Playground 0.2.0, [`_src/manipulation/aloha/handover.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/manipulation/aloha/handover.py) (`HandOver`)

> A few additional environments, such as bi-arm peg-insertion with the Aloha robot [2], are also available.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), section II-C

#### `playground/aloha-single-peg-insertion`

Playground's `AlohaSinglePegInsertion`.

<img src="gifs/playground__aloha-single-peg-insertion.gif" alt="16 rollouts of a policy trained on playground/aloha-single-peg-insertion" loading="lazy">

> Peg insertion task for ALOHA.
>
> Single peg insertion task for ALOHA.
>
> — MuJoCo Playground 0.2.0, [`_src/manipulation/aloha/single_peg_insertion.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/manipulation/aloha/single_peg_insertion.py) (`SinglePegInsertion`)

> A few additional environments, such as bi-arm peg-insertion with the Aloha robot [2], are also available.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), section II-C

#### `playground/panda-pick-cube`

Playground's `PandaPickCube`.

<img src="gifs/playground__panda-pick-cube.gif" alt="16 rollouts of a policy trained on playground/panda-pick-cube" loading="lazy">

> Bring a box to a target and orientation.
>
> Bring a box to a target.
>
> — MuJoCo Playground 0.2.0, [`_src/manipulation/franka_emika_panda/pick.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/manipulation/franka_emika_panda/pick.py) (`PandaPickCube`)

#### `playground/panda-pick-cube-orientation`

Playground's `PandaPickCubeOrientation`.

<img src="gifs/playground__panda-pick-cube-orientation.gif" alt="16 rollouts of a policy trained on playground/panda-pick-cube-orientation" loading="lazy">

> Bring a box to a target and orientation.
>
> — MuJoCo Playground 0.2.0, [`_src/manipulation/franka_emika_panda/pick.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/manipulation/franka_emika_panda/pick.py) (`PandaPickCubeOrientation`)

#### `playground/panda-pick-cube-cartesian`

Playground's `PandaPickCubeCartesian`.

<img src="gifs/playground__panda-pick-cube-cartesian.gif" alt="16 rollouts of a policy trained on playground/panda-pick-cube-cartesian" loading="lazy">

> A simple task with demonstrating sim2real transfer for pixels observations.
> Pick up a cube to a fixed location using a cartesian controller.
>
> Environment for training the Franka Panda robot to pick up a cube in
> Cartesian space.
>
> — MuJoCo Playground 0.2.0, [`_src/manipulation/franka_emika_panda/pick_cartesian.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/manipulation/franka_emika_panda/pick_cartesian.py) (`PandaPickCubeCartesian`)

> We implement a simple vision-based pick-cube environment on a Franka arm using the Madrona batch renderer.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), section II-C

#### `playground/panda-open-cabinet`

Playground's `PandaOpenCabinet`.

<img src="gifs/playground__panda-open-cabinet.gif" alt="16 rollouts of a policy trained on playground/panda-open-cabinet" loading="lazy">

> Open a cabinet.
>
> Environment for training the Franka Panda robot to bring an object to a
> target.
>
> — MuJoCo Playground 0.2.0, [`_src/manipulation/franka_emika_panda/open_cabinet.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/manipulation/franka_emika_panda/open_cabinet.py) (`PandaOpenCabinet`)

#### `playground/panda-robotiq-push-cube`

Playground's `PandaRobotiqPushCube`.

<img src="gifs/playground__panda-robotiq-push-cube.gif" alt="16 rollouts of a policy trained on playground/panda-robotiq-push-cube" loading="lazy">

> Panda robotiq push cube environment.
>
> Environment for pushing a cube with a Panda robot and Robotiq gripper.
>
> — MuJoCo Playground 0.2.0, [`_src/manipulation/franka_emika_panda_robotiq/push_cube.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/manipulation/franka_emika_panda_robotiq/push_cube.py) (`PandaRobotiqPushCube`)

> The block is initialized at a random position and orientation subject to workspace bounds, and is then pushed, slid, or tapped to a desired goal pose at the center of the workspace.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), appendix C.51

#### `playground/leap-cube-reorient`

Playground's `LeapCubeReorient`.

<img src="gifs/playground__leap-cube-reorient.gif" alt="16 rollouts of a policy trained on playground/leap-cube-reorient" loading="lazy">

> Reorient task for leap hand.
>
> Reorient a cube to match a goal orientation.
>
> — MuJoCo Playground 0.2.0, [`_src/manipulation/leap_hand/reorient.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/manipulation/leap_hand/reorient.py) (`CubeReorient`)

> The in-hand reorientation environment is designed to sequentially re-orient a cube within the palm of a robotic hand, without dropping the cube. The cube is initialized randomly above the palm of the hand.
>
> — Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844), 2025 (CC BY 4.0), appendix C.41

#### `playground/leap-cube-rotate-z-axis`

Playground's `LeapCubeRotateZAxis`.

<img src="gifs/playground__leap-cube-rotate-z-axis.gif" alt="16 rollouts of a policy trained on playground/leap-cube-rotate-z-axis" loading="lazy">

> Rotate-z with leap hand.
>
> Rotate a cube around the z-axis as fast as possible wihout dropping it.
>
> — MuJoCo Playground 0.2.0, [`_src/manipulation/leap_hand/rotate_z.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/manipulation/leap_hand/rotate_z.py) (`CubeRotateZAxis`)

#### `playground/aero-cube-rotate-z-axis`

Playground's `AeroCubeRotateZAxis`.

<img src="gifs/playground__aero-cube-rotate-z-axis.gif" alt="16 rollouts of a policy trained on playground/aero-cube-rotate-z-axis" loading="lazy">

> Rotate-z with TetherIA Aero Hand Open.
>
> Rotate a cube around the z-axis as fast as possible wihout dropping it.
>
> — MuJoCo Playground 0.2.0, [`_src/manipulation/aero_hand/rotate_z.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/manipulation/aero_hand/rotate_z.py) (`CubeRotateZAxis`)

