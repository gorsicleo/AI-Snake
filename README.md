# AI-Snake

## Introduction

This is a simple snake game program that utilizes deterministic algorithm and three layer neural network in order to enable computer to play the game by itself.
Currently there is no option for user to play the game!


## Overwiev

Organization of program was inspired by MVC arhitecture:
* _model_ models learning mechanism and simple neural network built using pytorch framework,
* _view_ is game environment itself (or in other words: board where snake is going to move) built using pygame framework,
* _controller_ plays the game.

Here is an example of **deterministic agent** playing the game:

![Deterministic_gameplay](https://github.com/gorsicleo/AI-Snake/blob/main/examples/example3.gif)

Here are examples of **AI agent** playing the game in the early stage of learning (his first game ever):

![AI_gameplay1](https://github.com/gorsicleo/AI-Snake/blob/main/examples/exampleAI(20).gif)

and example of AI agent after 30 games played (10 moves played by deterministic agent and 20 moves AI agent):

![AI_gameplay2](https://github.com/gorsicleo/AI-Snake/blob/main/examples/exampleAI(30).gif)


## Running the program

* To start deterministic agent: run deterministic_controller.py 
>  Example: `python deterministic_controller.py`

* To start AI agent: run ai_controller.py 
> Example: `python ai_controller.py`


**NOTE**

In order to start this program you must have pygame, numpy, and pytorch installed.
For help with installation of these dependencies please refer to:
1. [Numpy install](https://numpy.org/install/)
2. [Pygame install](https://www.pygame.org/wiki/GettingStarted)
3. [Pytorch install](https://pytorch.org/get-started/locally/)

**bonus note for Windows users:** if you are having troubles with pytorch installation please create and open new python environment and install pytorch from there (and start programs that require pytorch from that environment).

## Altering parameters

To tweak learning a bit there are some parameters that could be changed (all of these parameters are in ai_controller.py file):

* ` learning_rate ` on line 104 [Explanation](https://en.wikipedia.org/wiki/Learning_rate)
* ` number_of_deterministic_games_played ` on line 184 represents how many games will deterministic agent play when game starts before AI agent makes moves, this is used to let neural network "see" some number of whole games played by deterministic agent to increase learning.

## Model and brief explanation

Deterministic algorithm uses shortest path method to navigate itself to food. This is why snake develops typical "diaglonal" move pattern.

Neural network used in AI agent is a simple neural network with 3 layers. First layer has 11 input nodes representing current state of the game, second layer is "hidden layer" with 256 nodes (arbitrary) and third layer has only 3 output nodes representing all possible snake moves (turn right, turn left, continue straight).

This program uses [Adam optimizer](https://optimization.cbe.cornell.edu/index.php?title=Adam) with MSE as optimizer criterion that is fed with move that _would_ be played by deterministic algorithm and neural network output in order to calculate loss.

Illustration of loss calculation:

![Loss_illustration](https://github.com/gorsicleo/AI-Snake/blob/main/examples/illustration.svg)

## Possible future features

There is support for saving neural network state whenever snake reaches new highscore. Idea is to train snake and save neural network state to a file which could be loaded on start of the program to have trained snake out of the box!
Since this feature is not completed and tested it is disabled for the time being.

## Licence and usage

This project falls under MIT licence. Feel free to use and contrinute to this code.








