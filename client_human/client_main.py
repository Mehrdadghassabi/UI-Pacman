from base import BaseAgent, Action


class Agent(BaseAgent):

    def do_turn(self) -> Action:
        ans = input("its your turn Human: ")
        if ans == "w" or ans == "W":
            return Action.UP
        if ans == "d" or ans == "D":
            return Action.RIGHT
        if ans == "a" or ans == "A":
            return Action.LEFT
        if ans == "x" or ans == "X":
            return Action.DOWN
        if ans == "t" or ans == "T":
            return Action.TELEPORT
        if ans == "n" or ans == "N":
            return Action.NOOP


if __name__ == '__main__':
    data = Agent().play()
    print("FINISH : ", data)
