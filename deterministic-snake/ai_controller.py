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
        self.number_of_random_moves = 69
        self.learning_rate = 0.001
        self.number_of_games = 0
        self.discount_rate = 0.9  
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

    def reduce_number_of_random_moves(self):
        self.number_of_random_moves = self.number_of_random_moves - 1

    def learn_from_step(self, state, action, reward, next_state, is_finished):
        self.trainer.train_step(state, action, reward, next_state, is_finished)

    def next_move(self, state):
        self.reduce_number_of_random_moves()
        final_move_template = [0, 0, 0]
        if random.randint(0, 155) < self.number_of_random_moves:
            move_index = random.randint(0, 2)
            final_move_template[move_index] = 1
        else:
            transformed_state_for_neural_network = torch.tensor(state, dtype=torch.float)
            model_output = self.model(transformed_state_for_neural_network)
            move_index = torch.argmax(model_output).item()
            final_move_template[move_index] = 1

        return final_move_template


def train():
    max_score = 0
    agent = AI_agent()
    game = SnakeGameAI()
    while True:
        old_state = agent.current_state(game)

        final_move = agent.next_move(old_state)

        reward, is_finished, score = game.play_step(final_move)

        new_state = agent.current_state(game)

        agent.learn_from_step(
            old_state, final_move, reward, new_state, is_finished)

        if is_finished:
            game.reset()
            agent.number_of_games += 1

            if score > max_score:
                max_score = score
                agent.model.save()

            print('Game Number -> ', agent.number_of_games, 'Score -> ', score, 'Max score -> ', max_score)


if __name__ == '__main__':
    train()
