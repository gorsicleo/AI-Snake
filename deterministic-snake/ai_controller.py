import math
import torch
import random
import numpy as np
from collections import deque
from view import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from view import SnakeGameAI, Direction, Point




class AI_agent:

    def __init__(self):

        self.max_memory = 100000
        self.sample_size = 1000
        self.learning_rate = 0.001
        self.n_games = 0
        self.randomness = 0  
        self.discount_rate = 0.9  
        self.memory = deque(maxlen=self.max_memory) 
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=self.learning_rate, discount_rate=self.discount_rate)

    def current_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def save_progress(self, state, action, reward, next_state, is_finished):
        self.memory.append((state, action, reward, next_state, is_finished))

    def learn_from_one_game(self):
        if len(self.memory) > self.sample_size:
            mini_sample = random.sample(
                self.memory, self.sample_size)  # list of tuples
        else:
            mini_sample = self.memory

        for state, action, reward, next_state, is_finished in mini_sample:
            self.trainer.train_step(state, action, reward, next_state, is_finished)

    def learn_from_one_step(self, state, action, reward, next_state, is_finished):
        self.trainer.train_step(state, action, reward, next_state, is_finished)

    def new_action(self, state):
        self.randomness = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.randomness:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    record = 0
    agent = AI_agent()
    game = SnakeGameAI()
    while True:
        old_state = agent.current_state(game)

        final_move = agent.new_action(old_state)

        reward, is_finished, score = game.play_step(final_move)

        new_state = agent.current_state(game)

        agent.learn_from_one_step(
            old_state, final_move, reward, new_state, is_finished)

        agent.save_progress(old_state, final_move, reward, new_state, is_finished)

        if is_finished:
            game.reset()
            agent.n_games += 1
            agent.learn_from_one_game()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)


if __name__ == '__main__':
    train()
