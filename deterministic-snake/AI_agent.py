import math
from game import SnakeGameAI, Direction, Point

class AI_agent:

    def get_state(self, game):
        return [game.food.x, game.food.y, game.head.x, game.head.y]

    def get_action(self, state, game):
        #popuniti
        return [0, 0, 1]

     
def run():
    record = 0
    agent = AI_agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old,game)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        if done:
            game.reset()

  
if __name__ == '__main__':
    run()