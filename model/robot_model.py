import logging
import numpy
import random
from gym import spaces
import gym

logger = logging.getLogger(__name__)


class RobotEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):
        self.states = [state1 for state1 in range(1,26)]
        # self.x = []
        # self.y = []
        self.terminate_states = dict()
        self.terminate_states[4] = 1
        self.terminate_states[9] = 1
        self.terminate_states.values[11:13] = 1
        self.terminate_states.values[23:26] = 1

        self.actions = ['n', 'e', 's', 'w']

        self.rewards = dict()
        # self.rewards[]

        self.t = dict()
        for i in range(1, 4):
            for j in range(1, 4):
                ind = i * 5 + j + 1
                if ind not in [4, 9, 11, 12, 15, 23, 24, 25]:
                    for direc in self.actions:
                        key_ind = str(ind) + '_' + direc
                        if direc == 'n':
                            self.t[key_ind] = ind - 5
                        elif direc == 'e':
                            self.t[key_ind] = ind + 1
                        elif direc == 's':
                            self.t[key_ind] = ind + 5
                        else:
                            self.t[key_ind] = ind - 1
        self.t['1_e'] = 2
        self.t['1_s'] = 6

        self.t['2_e'] = 3
        self.t['2_s'] = 7
        self.t['2_w'] = 1

        self.t['3_e'] = 4
        self.t['3_s'] = 8
        self.t['3_w'] = 2

        self.t['5_s'] = 10
        self.t['5_w'] = 4

        self.t['6_n'] = 1
        self.t['6_e'] = 7
        self.t['6_s'] = 11

        self.t['10_n'] = 5
        self.t['10_s'] = 15
        self.t['10_w'] = 9

        self.t['16_n'] = 11
        self.t['16_e'] = 17
        self.t['16_s'] = 21

        self.t['20_n'] = 15
        self.t['20_s'] = 25
        self.t['20_w'] = 19

        self.t['21_n'] = 16
        self.t['21_e'] = 22

        self.t['22_n'] = 17
        self.t['22_w'] = 21

        self.gamma = 0.8
        self.viewer = None
        self.state = None

    def getTerminal(self):
        return self.terminate_states

    def getGamma(self):
        return self.gamma

    def getStates(self):
        return self.states

    def getAction(self):
        return self.actions

    def getTerminate_states(self):
        return self.terminate_states

    def setAction(self, s):
        self.state = s


    def _step(self, action):
        state = self.state
        if state in self.terminate_states:
            return state, 0, True, {}
        is_terminal = False
        key = '%d_%s'%(state, action)

        if key in self.t:
            next_state = self.t[key]
        else:
            next_state = state
        self.state = next_state
        if next_state in self.terminate_states:
            is_terminal = True
        if key in self.rewards:
            reward = self.rewards[key]
        else:
            reward = 0.0
        return next_state, reward, is_terminal, {}



    def _reset(self):
        self.state = random.choice(self.states)
        return self.state

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
                return

        screen_width = 700
        screen_height = 700

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.line1 = rendering.Line((100, 100), (500, 100))
            self.line2 = rendering.Line((100, 180), (500, 180))
            self.line3 = rendering.Line((100, 260), (500, 260))
            self.line4 = rendering.Line((100, 340), (500, 340))
            self.line5 = rendering.Line((100, 420), (500, 420))
            self.line6 = rendering.Line((100, 500), (500, 500))

            self.line7 = rendering.Line((100, 100), (100, 500))
            self.line8 = rendering.Line((180, 100), (180, 500))
            self.line9 = rendering.Line((260, 100), (260, 500))
            self.line10= rendering.Line((340, 100), (340, 500))
            self.line11= rendering.Line((420, 100), (420, 500))
            self.line12= rendering.Line((500, 100), (500, 500))


            self.bomb1 = rendering.make_circle(40)
            self.bomb1.add_attr(rendering.Transform((300, 140)))
            self.bomb1.set_color(0, 0, 0)

            self.bomb2 = rendering.make_circle(40)
            self.bomb2.add_attr(rendering.Transform((380, 140)))
            self.bomb2.set_color(0, 0, 0)

            self.bomb3 = rendering.make_circle(40)
            self.bomb3.add_attr(rendering.Transform((460, 140)))
            self.bomb3.set_color(0, 0, 0)

            self.bomb4 = rendering.make_circle(40)
            self.bomb4.add_attr(rendering.Transform((140, 300)))
            self.bomb4.set_color(0, 0, 0)

            self.bomb5 = rendering.make_circle(40)
            self.bomb5.add_attr(rendering.Transform((380, 380)))
            self.bomb5.set_color(0, 0, 0)

            self.bomb6 = rendering.make_circle(40)
            self.bomb6.add_attr(rendering.Transform((380, 460)))
            self.bomb6.set_color(0, 0, 0)

            self.gold = rendering.make_circle(40)
            self.gold.add_attr(rendering.Transform((460, 300)))
            self.gold.set_color(1, 1, 0)

            self.robot = rendering.make_circle(20)
            self.robottrans = rendering.Transform(460, 220)
            self.robot.add_attr(self.robottrans)
            self.robot.set_color(0.85, 0.65, 0.13)

            self.line1.set_color(0, 0, 0)
            self.line2.set_color(0, 0, 0)
            self.line3.set_color(0, 0, 0)
            self.line4.set_color(0, 0, 0)
            self.line5.set_color(0, 0, 0)
            self.line6.set_color(0, 0, 0)
            self.line7.set_color(0, 0, 0)
            self.line8.set_color(0, 0, 0)
            self.line9.set_color(0, 0, 0)
            self.line10.set_color(0, 0, 0)
            self.line11.set_color(0, 0, 0)
            self.line12.set_color(0, 0, 0)

            self.viewer.add_geom(self.line1)
            self.viewer.add_geom(self.line2)
            self.viewer.add_geom(self.line3)
            self.viewer.add_geom(self.line4)
            self.viewer.add_geom(self.line5)
            self.viewer.add_geom(self.line6)
            self.viewer.add_geom(self.line7)
            self.viewer.add_geom(self.line8)
            self.viewer.add_geom(self.line9)
            self.viewer.add_geom(self.line10)
            self.viewer.add_geom(self.line11)
            self.viewer.add_geom(self.line12)
            self.viewer.add_geom(self.bomb1)
            self.viewer.add_geom(self.bomb2)
            self.viewer.add_geom(self.bomb3)
            self.viewer.add_geom(self.bomb4)
            self.viewer.add_geom(self.bomb5)
            self.viewer.add_geom(self.bomb6)
            self.viewer.add_geom(self.gold)
            self.viewer.add_geom(self.robot)
        if self.state is None:
            return None
        # self.robottrans.set_translation(self., self.)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
















