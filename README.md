# q-learning-cartpole

Based on [this solution](https://github.com/johnnycode8/gym_solutions/blob/main/cartpole_q.py)

## Content
Project contains a sample solution of [this problem](https://gymnasium.farama.org/environments/classic_control/cart_pole/) written in python.

## Prerequisites
To run this sample you need to have installed:
- gymnasium (problem env)
- numpy (calculations)
- pickle (dump and load model)
- pygame (visualise)
- Box2D (visualise)

## Running
Model is trained by running the 'run' method with 'is_training' set to true. In this case, setting 'render' to false recommended, to speed up the training stage.
Trained model is persisted in cartpole.pkl file, and can be then used in test run ('is_training' set to false)