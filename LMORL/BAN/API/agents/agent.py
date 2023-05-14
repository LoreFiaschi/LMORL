import mo_gymnasium as mo_gym
from LMORL.Environment import Environment

from datetime import datetime

class Agent:
    def __init__(self) -> None:
        pass

    def agent_learning(self, env : Environment, episodes : int, mname : str, replay_frequency : int = 1, dump_period : int = 50, reward_threshold : float = None):
        # we need to know the size of the reward, 
        # that same size will be used for BANs dimension
        reward_dim = env.get_reward_dim()
        
        rewards = []
        avg_rewards = []

        timings = [0]

        solved = False

        i = 1

        while i < episodes and not solved:
            state = env.reset()
            done = False
            t = 0
            totrew = 0
            while not done:
                start_time = datetime.now()
                action = self._act(state)   # TODO: check how to modify the state
                next_state,reward,done=self._ban_step(env)
                # reward is MO, then it is a list
                # totrew+=reward
                totrew = [sum(foo) for foo in zip(totrew, reward)]

                self._add_experience(state,action,reward,next_state,done)
                state=next_state
                t+=1
                if t % replay_frequency==0 :
                    self._experience_replay()
                tmng = ( datetime.now() - start_time ).total_seconds()
                timings.append(tmng)
            
            self._episode_end()

            rewards.append(totrew)

            if i % dump_period == 0:
                self._dump_model_to_file(mname)

            if i >= 100:
                # https://www.geeksforgeeks.org/python-ways-to-sum-list-of-lists-and-return-sum-list/
                avg_reward = [sum(comp[i - 100 : i - 1])/100 for comp in zip(*rewards)]
                #( sum(rewards[i - 100: i - 1]) ) / 100  --> not sium because reward is a NA vector

                # TODO: consider if adding an exit condition that use as threshold the
                # input parameter reward_threshold

            else:
                avg_reward = [sum(comp)/i for comp in zip(*rewards)]

            avg_rewards.append(avg_reward)
            print(f"Episode {i} - reward: {totrew} | 100AvgReward: {avg_reward}")

        return rewards, avg_rewards, timings

    def _act(self, state):
        """
        returns the action to perform and updates the state
        """
        # TODO: check how to modify the state
        pass

    def _ban_step(self, env : Environment):
        """
        returns next_state,reward,done
        """
        pass

    def _add_experience(state,action,reward,next_state,done):
        pass

    def _experience_replay():
        pass

    def _episode_end():
        """
        this method is called at each episode end,
        extend this method to add custom methods call and updates for agent-specific needs
        """
        pass

    def _dump_model_to_file(mname):
        pass