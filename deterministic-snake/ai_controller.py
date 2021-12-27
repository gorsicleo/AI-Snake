import math
import torch
import random
import numpy as np
from collections import deque
from view import SnakeGameAI, Direction, Point
from model import Neural_Network, Reinforcment_Learner
from view import SnakeGameAI, Direction, Point

number_of_deterministic_games_played = 20

def calculate_distance(point_a, point_b):
        return math.hypot(point_a.x - point_b.x, point_a.y - point_b.y)

def get_deterministic_state(game):
        return [game.food.x, game.food.y, game.head.x, game.head.y]

def next_deterministic_move(state, game):
        point_head = Point(state[2], state[3])
        point_food = Point(state[0], state[1])

        point_left = Point(point_head.x - 20, point_head.y)
        point_right = Point(point_head.x + 20, point_head.y)
        point_up = Point(point_head.x, point_head.y - 20)
        point_down = Point(point_head.x, point_head.y + 20)

        dir_left = game.direction == Direction.LEFT
        dir_right = game.direction == Direction.RIGHT
        dir_up = game.direction == Direction.UP
        dir_down = game.direction == Direction.DOWN

        distances = [calculate_distance(point_left, point_food),
                     calculate_distance(point_right, point_food),
                     calculate_distance(point_up, point_food),
                     calculate_distance(point_down, point_food)]

        dangers = [  # Danger straight 0
            (dir_right and game.is_collision(point_right)) or
            (dir_left and game.is_collision(point_left)) or
            (dir_up and game.is_collision(point_up)) or
            (dir_down and game.is_collision(point_down)),
            # Danger right 1
            (dir_up and game.is_collision(point_right)) or
            (dir_down and game.is_collision(point_left)) or
            (dir_left and game.is_collision(point_up)) or
            (dir_right and game.is_collision(point_down)),
            # Danger left 2
            (dir_down and game.is_collision(point_right)) or
            (dir_up and game.is_collision(point_left)) or
            (dir_right and game.is_collision(point_up)) or
            (dir_left and game.is_collision(point_down))
        ]

        available = []
        if dangers[0] == False:
            available.append([1, 0, 0])
        if dangers[1] == False:
            available.append([0, 1, 0])
        if dangers[2] == False:
            available.append([0, 0, 1])

        lengths = []

        if len(available) == 0:
            return [1, 0, 0]

        # handle available moves
        for move in available:
            if move == [1, 0, 0]:
                if dir_left:
                    lengths.append(distances[0])
                if dir_right:
                    lengths.append(distances[1])
                if dir_up:
                    lengths.append(distances[2])
                if dir_down:
                    lengths.append(distances[3])
            if move == [0, 1, 0]:
                if dir_left:
                    lengths.append(distances[2])
                if dir_right:
                    lengths.append(distances[3])
                if dir_up:
                    lengths.append(distances[1])
                if dir_down:
                    lengths.append(distances[0])
            if move == [0, 0, 1]:
                if dir_left:
                    lengths.append(distances[3])
                if dir_right:
                    lengths.append(distances[2])
                if dir_up:
                    lengths.append(distances[0])
                if dir_down:
                    lengths.append(distances[1])

        _, idx = min((val, idx) for (idx, val) in enumerate(lengths))
        return available[idx]

class AI_agent:

    def __init__(self):
        self.number_of_random_moves = 80
        self.learning_rate = 0.001
        self.number_of_games = 0
        self.discount_rate = 0.9  
        self.model = Neural_Network(11, 256, 3)
        self.trainer = Reinforcment_Learner(self.model, learning_rate=self.learning_rate, discount_rate=self.discount_rate)
    
    
    

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
        self.number_of_random_moves = 0 - self.number_of_games

    def learn_from_step(self, state, action, reward, next_state, is_finished):
        self.trainer.learn_from_one_step(state, action, reward, next_state, is_finished)

    def next_move(self, state, game):
        move = None
        if number_of_deterministic_games_played>0:
            deterministic_state = get_deterministic_state(game)
            move = next_deterministic_move(deterministic_state, game)
        else:
            transformed_state_for_neural_network = torch.tensor(state, dtype=torch.float)
            model_output = self.model(transformed_state_for_neural_network)
            move_index = torch.argmax(model_output).item()
            move = [0, 0, 0]
            move[move_index] = 1

        return move


def train():
    max_score = 0
    agent = AI_agent()
    game = SnakeGameAI()
    while True:
        old_state = agent.current_state(game)

        final_move = agent.next_move(old_state, game)

        reward, is_finished, score = game.play_step(final_move)

        new_state = agent.current_state(game)

        agent.learn_from_step(
            old_state, final_move, reward, new_state, is_finished)

        if is_finished:
            game.reset()
            agent.number_of_games += 1
            number_of_deterministic_games_played = number_of_deterministic_games_played - 1

            if score > max_score:
                max_score = score
                agent.model.save_state(max_score)
                File_object = open("./model/records.txt","a")
                File_object.writelines(str(max_score))
                File_object.close()
                

            print('Game Number -> ', agent.number_of_games, 'Score -> ', score, 'Max score -> ', max_score)


if __name__ == '__main__':
    train()
