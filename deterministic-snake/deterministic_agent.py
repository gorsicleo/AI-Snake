import math
from game import SnakeGameAI, Direction, Point


class DeterministicAgent:

    def get_state(self, game):
        return [game.food.x, game.food.y, game.head.x, game.head.y]

    def get_action(self, state, game):
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

        distances = [_calculate_distance(point_left, point_food),
                     _calculate_distance(point_right, point_food),
                     _calculate_distance(point_up, point_food),
                     _calculate_distance(point_down, point_food)]

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


def _calculate_distance(point_a, point_b):
    return math.hypot(point_a.x - point_b.x, point_a.y - point_b.y)


def run():
    record = 0
    agent = DeterministicAgent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old, game)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        if done:
            game.reset()


if __name__ == '__main__':
    run()
