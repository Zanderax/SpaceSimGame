from __future__ import division
#Enables Python 3 division and must be kept at top of file.

import random

import numpy as np
import pandas as pd

DATA_FILE = './agent_data.gz'
TURNS = 20
TRAINING_REPEATS = 200
COLONIZE_PERCENTAGE = .001

# Data Classes
class Planet:
    def __init__(self):
        self.resources = 100000
        self.colonized = False

class Star:
    def __init__(self):
        self.resources = 100000
        self.planets = [Planet() for i in range(8)]

class Ruler:
    def __init__(self):
        self.drones = 1000
        self.SolarArray = 0
        self.mine = 0
        self.factories = 0
        self.resources = 0
        self.colonizePercent = 0
        self.planets = []

stars = [Star() for i in range(20)]
stars[0].planets[0].colonized = True

ruler = Ruler()

def Reset():
    for star in stars:
        star.resources = 100000
        for planet in star.planets:
            planet.resources = 100000

    ruler.drones = 1000
    ruler.resources = 0

def GatherResourcesOnColonizedPlanets(drones):
    initalDrones = drones
    for star in stars:
        for planet in star.planets:
            # if planet.colonized:
            if drones < planet.resources:
                # We have used all drones so we finish
                planet.resources -= drones
                ruler.resources += drones
                return
            # We still have drones that can collect, move onto the next start
            drones -= planet.resources
            ruler.resources += planet.resources
            planet.resources = 0

def GatherResourcesOnStars(drones):
    initalDrones = drones
    for star in stars:
        if drones < star.resources:
            star.resources -= drones
            # We have used all drones so we finish
            ruler.resources += drones
            return
        # We still have drones that can collect, move onto the next start
        drones -= star.resources
        ruler.resources += star.resources
        star.resources = 0

def CreateDrones(drones):
    DronesToMake = min(drones, ruler.resources)
    ruler.resources -= DronesToMake
    ruler.drones += DronesToMake

def ColonizePlanet(drones):
    self.colonizePercent += drones * COLONIZE_PERCENTAGE

def TotalStarResources():
    return sum(s.resources for s in stars)

def TotalPlanetResources():
    return sum(sum(p.resources for p in s.planets) for s in stars)

Actions = {
    0 : GatherResourcesOnColonizedPlanets,
    1 : GatherResourcesOnStars,
    2 : CreateDrones,
}

BotActions = [
    GatherResourcesOnColonizedPlanets,
    GatherResourcesOnStars,
    CreateDrones,
]

# QLearningTable Taken from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.5, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]
            
            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
            
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)
        
        q_predict = self.q_table.ix[s, a]
        q_target = r + self.gamma * self.q_table.ix[s_, :].max()
        
        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class QLearningAgent:
    def __init__(self, lr, rd, eg):
        self.lr = lr
        self.rd = rd
        self.eg = eg
        self.maxScore = 0
        self.totalScore = 0
        self.sessionsCompleted = 0
        self.qtable = QLearningTable(
            actions = list(range(len(BotActions))),
            learning_rate = lr,
            reward_decay = rd,
            e_greedy = eg
        )


def IntTryParse(value):
    try:
        return int(value), True
    except ValueError:
        return value, False

def GetInput():
    Input = input("Choose Action - H for help - Q for Quit: ")
    if Input == "H" or Input == "h" :
        print("""Actions:
        0 - Harvest Sun Resources
        1 - Harvest Planet Resources
        2 - Create Drones""")
        return GetInput()
    if Input == "Q" or Input == "q" :
        return -1
    try:
        return int(Input)
    except ValueError:
        return GetInput() 
    
def ExecuteAction(Input):
    if Input < len(Actions):
        Actions[Input](ruler.drones)

def PrintPlanets(star):
    for i, planet in enumerate(star.planets):
        print("  |-> Planet {i} - Resources - {resources}".format(
            i = i,
            resources = planet.resources
        ))

def PrintStars():
    for i, star in enumerate(stars):
        print("Star {i} - Resources - {resources}".format(
            i = i,
            resources = star.resources
        ))
        PrintPlanets(star)

def DisplayStatus():
    print("Resources = {}".format(ruler.resources))
    print("Drones = {}".format(ruler.drones))
    print("Star Resources = {}".format(TotalStarResources()))
    print("Planet Resources = {}".format(TotalPlanetResources()))
    print("Total Score = {}".format(ruler.drones + ruler.resources))
    # PrintStars()


def RunWithAgents(agents):
    # qlearn = QLearningTable(actions=list(range(len(BotActions))))

    # Load
    # qlearn.q_table = pd.read_pickle(DATA_FILE, 'gzip') 

    # DisplayStatus()


    highscore = 0

    for idx, agent in enumerate(agents):
        # print("Running agent {idx}".format(idx=idx))
        for training_session in range(TRAINING_REPEATS):
            current_state = None
            previous_action = None
            for turn in range(TURNS):
                previous_state = current_state
                current_state = [
                    ruler.drones,
                    ruler.resources,
                    TotalStarResources(),
                    TotalPlanetResources(),
                ]

                reward = ruler.drones + ruler.resources 

                if previous_action is not None:
                    agent.qtable.learn(str(previous_state), previous_action, reward, str(current_state))

                # BOT
                # rl_action = agent.qtable.choose_action(str(current_state))

                # RANDOM
                rl_action = random.randint(0,2)

                # print("Turn {t} Action {a}".format(t=turn, a=rl_action))
                # DisplayStatus()


                ExecuteAction(rl_action)

            score = ruler.drones + ruler.resources
            # print("Training Session - {}".format(training_session + 1))
            if highscore < score:
                highscore = score
                # print("Training Session - {}".format(training_session + 1))
                # print("New Highscore - {}".format(highscore))
            # print("Score (percent of high) - {score}  {poh}".format(score=score, poh=score/highscore))
            # DisplayStatus()
            # print("\n")
            agent.totalScore += score
            agent.sessionsCompleted += 1
            agent.averageScore = agent.totalScore / agent.sessionsCompleted
            if score > agent.maxScore:
                agent.maxScore = score
            print("{averageScore}".format(averageScore=agent.averageScore))
            Reset()

        
        print("""Agent {idx} Finished.
lr = {lr} rd = {rd} eg = {eg}
Max Score - {max} Average Score - {avg}""".format(
            idx=idx,
            lr = agent.lr,
            rd = agent.rd,
            eg = agent.eg,
            max = agent.maxScore,
            avg = agent.averageScore
            ))
        print ("\n")

    # qlearn.q_table.to_pickle(DATA_FILE, 'gzip')

def RunAgentParameterSpread():
    agents = []
    for lr in range(10):
        for rd in range(10):
            for eg in range(10):
                agents.append(QLearningAgent(lr * .1,rd * .1,eg * .1))
    RunWithAgents(agents)

def main():
    # Find best params
    # RunAgentParameterSpread()

    agents = []
    # The best params from experiment
    agents.append(QLearningAgent(0.7, 0.7, 0.4))

    RunWithAgents(agents)



if __name__ == "__main__":
    main()