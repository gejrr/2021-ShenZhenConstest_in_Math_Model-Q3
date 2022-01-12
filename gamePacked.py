import os
import sys
import math
from abc import ABC

import pygame
import random
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from math import pi
from math import sqrt
from tensorflow.keras import Model
# import matplotlib.pyplot as plt
from tensorflow.keras.layers import *


# ------ PATCH NOTES ------ #
# Fixed: 预测时输入格式包装有误导致展开后shape = (None, size**2)，不能进行预测
# Fixed: 切换模式会造成一帧的None行动被录入
# Fixed: 手动结束后，开新局立即切换AI，预测时触发ValueError
# TODO bug: 自动游戏中途切换人工并完成当前局，训练时偶然出现EagerTensor type导致无法转化为tensor
#  File "C:\Users\...\.conda\envs\...\lib\site-packages\tensorflow_core\python\framework\constant_op.py",
#  line 96, in convert_to_eager_tensor
#     return ops.EagerTensor(value, ctx.device_name, dtype)
#   ValueError: Failed to convert a NumPy array to a Tensor
#   (Unsupported object type tensorflow.python.framework.ops.EagerTensor).
# TODO: 有时出现显存不足问题：failed to create cublas handle: CUBLAS_STATUS_ALLOC_FAILED
# Fixed: 在切换模式时偶然出现None操作
# Fixed: 羊逃脱后仍判定追上
# TODO Fixed: 当前后两帧做出左右横跳的移动，而狗恰好已经在对位，导致不断重复一个动作
# Add: 加入了训练保存机制
# Add: 用类构建卷积神经网络
# TODO Add: 构建循环神经网络，基于一局游戏的进度循环学习
# Add: 加入一个开关，决定本轮是否进行神经网络的训练
# Add: 可进行全连接网络和卷积网络的交换训练
# Opt: 优化奖励函数计算方式，使之与距离有关
# Opt: 再次优化奖励函数计算方式，使之与夹角、半径有关
# Opt: 设定训练集最大容量，优化训练效率
# Opt: 优化了损失函数的数据结构和求法
# Update: 环境更新到TensorFlow2.5和Keras2.6，以配合numpy版本


# 操作方法：
# 用方向键控制羊的移动，↑键沿直径前进，←键逆时针移动，→键顺时针移动。
# 按空格键在人工模式与AI模式间切换。
# 按ESC退出。

# ---GLOBAL VARIABLES--- #

# AI
AI_PLAY = False             # 可在游戏中用SPACE切换
GAME_TRAINABLE = True       # 在游戏过程中不可切换
# NET_TYPE = 'FullLinkNetwork'        # 训练模型，可选全连接网络或卷积神经网络。
NET_TYPE = 'ConvolutionalNetwork'    # 程序内缺省值为卷积网络

# display
FPS = -1
ORIGIN = 100, 100
RECT_SIZE = 3, 3
FIELD_RADIUS = 100
SCREEN_SIZE = (204, 204)

# colors
BLACK = 0, 0, 0
WHITE = 255, 255, 255
GREEN = 0, 200, 0
RED = 250, 0, 200
DARK_BLUE = 0, 0, 75


class Game:
    def __init__(self, auto=False):
        pygame.init()
        scr_info = pygame.display.Info()
        # size = width, height = scr_Info.current_w, scr_Info.current_h
        size = (width, height) = SCREEN_SIZE

        self.games_count = 0
        self.dog_win = 0
        self.sheep_win = 0
        self.round_end = False
        self.step = 0

        self.auto = auto
        self.reward = 0
        # self.hanging_reward = -0.1
        self.losing_reward = 0
        self.winning_reward = 2

        self.human_action = None

        self.display_changed = False
        self.screen = pygame.display.set_mode(size, )
        pygame.display.set_caption("Sheep & Dog")
        self.frame_image = pygame.surfarray.array3d(self.screen)
        self.clock = pygame.time.Clock()

        self.sheep, self.dog = self.__init_player()

    class Animal(pygame.rect.Rect):
        def __init__(self, screen, x, y, speed, angle, distance):
            pygame.rect.Rect.__init__(self, (x, y), RECT_SIZE)

            self.screen = screen

            self.color = WHITE

            self.x = x
            self.y = y
            self.angle = angle
            self.speed = speed
            self.d = distance

        def clockwise(self):
            self.angle += (self.speed / self.d)

        def counterclockwise(self):
            self.angle -= (self.speed / self.d)

        def draw(self):
            self.x = int(self.d * math.cos(self.angle)) + ORIGIN[0]
            self.y = int(self.d * math.sin(self.angle)) + ORIGIN[1]
            # print(self.x, self.y, self.angle, self.d)
            pygame.draw.rect(self.screen, self.color, self)

    class Sheep(Animal):
        def __init__(self, screen, x, y):
            Game.Animal.__init__(self, screen, x, y, speed=3, angle=0, distance=3)
            # pygame.Rect.__init__(self, (x, y), rect_size)

        def forward(self):
            self.d += self.speed

    class Dog(Animal):
        def __init__(self, screen, x, y):
            Game.Animal.__init__(self, screen, x, y, speed=10, angle=-pi, distance=FIELD_RADIUS)
            self.color = RED
            # pygame.Rect.__init__(self, (x, y), rect_size)

        def chase(self, target):
            # standardize angles into range (-pi, pi)
            tar_ang = target.angle % (2 * pi) if target.angle % (2 * pi) < pi else (target.angle % (2 * pi) - 2 * pi)
            dog_ang = self.angle % (2 * pi) if self.angle % (2 * pi) < pi else (self.angle % (2 * pi) - 2 * pi)
            # print(f'{dog_ang - pi: .3f}, {tar_ang: .3f}, {dog_ang: .3f}, {target.angle: .3f}, {self.angle: .3f}')

            if -pi <= dog_ang - pi < tar_ang < dog_ang:  # 顺时针近
                self.counterclockwise()
            # 顺时针近，但在归一化过程中范围溢出
            elif dog_ang - pi < -pi and (-pi < tar_ang < dog_ang or dog_ang + pi < tar_ang < pi):
                self.counterclockwise()
            # 逆时针近
            else:
                self.clockwise()

    def __init_player(self):
        # print('Initializing Game.')
        sheep = self.Sheep(self.screen, x=ORIGIN[0] - RECT_SIZE[0] / 2, y=ORIGIN[1] - RECT_SIZE[1] / 2)
        dog = self.Dog(self.screen, x=ORIGIN[0] - FIELD_RADIUS, y=ORIGIN[1])
        return sheep, dog

    def __reset_game(self, sheep, dog):
        # print('Resetting Game.')

        sheep.__init__(self.screen, x=ORIGIN[0] - RECT_SIZE[0] / 2, y=ORIGIN[1] - RECT_SIZE[1] / 2)
        dog.__init__(self.screen, x=ORIGIN[0] - FIELD_RADIUS, y=ORIGIN[1])

        self.reward = 0.
        self.display_changed = False
        self.step = 0

        return sheep, dog

    def get_hanging_reward(self):
        """
        Calculates the reward of each step.

        The reward is designed to describe how good the action is as reflected in the loss function,
          which therefore, should annotate by figures that how close, having moved in a given way,
          the sheep is to its destination, or to its perfect.
          
        To this end, aspects of a decent game within its duration are evaluated, which include:
        
         - Angle of the inferior arc between two objects.
         
           Sheep would eventually escape if it were to keep a distance against the dog, and we
           should award the neural network for keeping this angle as large as possible.
           
         - Radius to the origin.
         
           Describes how close the sheep is to is freedom, and we should encourage the neural
           network to go further from the origin.

        :return: reward depending on both animal's relative position, a number ranging from -0.015 to 0. 
        """

        # The first version
        # x_s = int(self.sheep.d * math.cos(self.sheep.angle)) + ORIGIN[0]
        # y_s = int(self.sheep.d * math.sin(self.sheep.angle)) + ORIGIN[1]
        # x_d = int(self.dog.d * math.cos(self.dog.angle)) + ORIGIN[0]
        # y_d = int(self.dog.d * math.sin(self.dog.angle)) + ORIGIN[1]
        # distance = sqrt((x_d - x_s)**2 + (y_d - y_s)**2)
        # distance = distance / (2 * FIELD_RADIUS)

        # 距离越远，奖励越大，并且处于 -0.1~0 的范围
        # hanging_reward = (distance - 1) / 10

        tar_ang = self.sheep.angle % (2 * pi) if self.sheep.angle % (2 * pi) < pi \
            else (self.sheep.angle % (2 * pi) - 2 * pi)
        dog_ang = self.dog.angle % (2 * pi) if self.dog.angle % (2 * pi) < pi \
            else (self.dog.angle % (2 * pi) - 2 * pi)

        # 拉开角度奖励（惩罚）
        inferior_ang = (abs(tar_ang - dog_ang)) if abs(tar_ang - dog_ang) <= pi else (2 * pi - abs(tar_ang - dog_ang))
        hanging_reward_ang = (1 - inferior_ang / pi) / 200

        # 拉开距离奖励（惩罚）
        hanging_reward_dis = (1 - self.sheep.d / FIELD_RADIUS) / 100

        hanging_reward = -hanging_reward_ang - hanging_reward_dis
        # print(hanging_reward)

        return hanging_reward

    def show_stat(self):
        print(f"Total {self.games_count}, \tdog's {self.dog_win}, \tsheep's {self.sheep_win}.\t\t"
              f" Mode is {'Auto.  ' if self.auto == True else 'Manual.'} \t"
              f"Reward = {self.reward: .3f}")

    def __manual_game(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # if self.auto is False:
            if event.type == pygame.KEYDOWN:
                # 逆时针移动
                if event.key == pygame.K_LEFT:
                    self.sheep.counterclockwise()
                    self.dog.chase(self.sheep)
                    self.reward += self.get_hanging_reward()
                    self.display_changed = True
                    self.step += 1
                    self.human_action = 1
                # 顺时针移动
                elif event.key == pygame.K_RIGHT:
                    self.sheep.clockwise()  # if speed[0] > 0 else speed[0] - 1
                    self.dog.chase(self.sheep)
                    self.reward += self.get_hanging_reward()
                    self.display_changed = True
                    self.step += 1
                    self.human_action = 2
                # 前进
                elif event.key == pygame.K_UP:
                    self.sheep.forward()  # if speed[1] > 0 else speed[1] - 1
                    self.dog.chase(self.sheep)
                    self.reward += self.get_hanging_reward()
                    self.display_changed = True
                    self.step += 1
                    self.human_action = 0
                # elif event.key == pygame.K_DOWN:
                # pass
                # 切换模式
                elif event.key == pygame.K_SPACE:
                    self.auto = True
                    self.human_action = None
                    self.show_stat()
                    return
                # 不需保存训练结果并退出
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

    def __auto_game(self, ai_action):
        # AI模式下的人工操作
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                # 切换模式
                if event.key == pygame.K_SPACE:
                    self.auto = False
                    self.show_stat()
                    return
                # 保存训练进度并退出
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        # AI操作
        if ai_action == 1:  # 'LEFT'
            self.sheep.counterclockwise()
            self.dog.chase(self.sheep)
            self.reward += self.get_hanging_reward()
            self.display_changed = True
            self.step += 1
        elif ai_action == 2:  # 'RIGHT'
            self.sheep.clockwise()
            self.dog.chase(self.sheep)
            self.reward += self.get_hanging_reward()
            self.display_changed = True
            self.step += 1
        elif ai_action == 0:  # 'UP'
            self.sheep.forward()
            self.dog.chase(self.sheep)
            self.reward += self.get_hanging_reward()
            self.display_changed = True
            self.step += 1
        # TODO 加入静止不动的AI行动
        # elif event.key == pygame.K_DOWN:
        # pass
        # elif event.key == pygame.K_ESCAPE:
        #     pygame.quit()

    def game_frame(self, ai_action=None):
        if self.round_end is True:
            self.sheep, self.dog = self.__reset_game(self.sheep, self.dog)
            self.round_end = False

        self.human_action = None
        self.display_changed = False

        if self.auto is False:
            # print('Manually Proceeding...')
            self.__manual_game()
        else:
            # print('Automatically Proceeding...')
            self.__auto_game(ai_action)

        # 判断游戏结束条件
        if self.sheep.d >= FIELD_RADIUS:
            self.games_count += 1
            self.sheep_win += 1
            self.reward += self.winning_reward

            print(f"Sheep wins.   \t", end='')
            self.show_stat()
            self.round_end = True

        elif abs(self.sheep.x - self.dog.x) <= self.dog.speed - self.sheep.speed \
                and abs(self.sheep.y - self.dog.y) <= self.dog.speed - self.sheep.speed:
            self.games_count += 1
            self.dog_win += 1
            self.reward += self.losing_reward

            print(f"Dog wins.     \t", end='')
            self.show_stat()
            self.round_end = True

        elif self.step >= 120:
            self.games_count += 1

            print(f"Too long game.\t", end='')
            self.show_stat()
            self.round_end = True

        self.screen.fill(DARK_BLUE)  # 每一次移动之后重新填充。RGB。前面定义过colors了
        # time.sleep(0.02)
        # 绘制场地，使用绿色不影响红色图，
        # 也可以绘制在其他surface上
        pygame.draw.circle(self.screen, GREEN, (FIELD_RADIUS + 2, FIELD_RADIUS + 2), 100, 1)
        self.sheep.draw()
        self.dog.draw()
        self.clock.tick(FPS)
        pygame.display.update()
        # print('update', pygame.PixelArray.surface)
        # self.game_image = pygame.surfarray.array3d(pygame.display.get_surface())
        self.frame_image = pygame.surfarray.array3d(self.screen)

        return self.frame_image


class FullLinkNetwork:
    def __init__(self,
                 input_shape=(SCREEN_SIZE[0], SCREEN_SIZE[1], 1),
                 # hidden_units=(256, 128, 64),
                 hidden_units=(512, 128),
                 output_size=3,
                 learning_rate=0.01):
        self.predicted = [1, 0, 0]
        self.action = 0
        self.input_shape = input_shape
        # hidden_units_1, hidden_units_2, hidden_units_3 = hidden_units
        hidden_units_1, hidden_units_2 = hidden_units
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=hidden_units_1, input_dim=input_shape, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=hidden_units_2, activation=tf.nn.relu),
            # tf.keras.layers.Dense(units=output_size, activation=tf.keras.activations.linear)
            tf.keras.layers.Dense(units=output_size, activation='softmax')
        ])


class ConvolutionalNetwork(Model, ABC):
    """
    A VGG Net, with 13 Convolutional nets and 3 full link networks.
    """
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()
        # Standard VGG Net #
        # self.c1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')
        # self.b1 = BatchNormalization()
        # self.a1 = Activation('relu')
        #
        # self.c2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')
        # self.b2 = BatchNormalization()
        # self.a2 = Activation('relu')
        # self.p2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')
        # self.d2 = Dropout(0.2)
        #
        # self.c3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        # self.b3 = BatchNormalization()
        # self.a3 = Activation('relu')
        #
        # self.c4 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        # self.b4 = BatchNormalization()
        # self.a4 = Activation('relu')
        # self.p4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')
        # self.d4 = Dropout(0.2)
        #
        # self.c5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        # self.b5 = BatchNormalization()
        # self.a5 = Activation('relu')
        #
        # self.c6 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        # self.b6 = BatchNormalization()
        # self.a6 = Activation('relu')
        #
        # self.c7 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        # self.b7 = BatchNormalization()
        # self.a7 = Activation('relu')
        # self.p7 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')
        # self.d7 = Dropout(0.2)
        #
        # self.c8 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        # self.b8 = BatchNormalization()
        # self.a8 = Activation('relu')
        #
        # self.c9 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        # self.b9 = BatchNormalization()
        # self.a9 = Activation('relu')
        #
        # self.c10 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        # self.b10 = BatchNormalization()
        # self.a10 = Activation('relu')
        # self.p10 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')
        # self.d10 = Dropout(0.2)
        #
        # self.c11 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        # self.b11 = BatchNormalization()
        # self.a11 = Activation('relu')
        #
        # self.c12 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        # self.b12 = BatchNormalization()
        # self.a12 = Activation('relu')
        #
        # self.c13 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        # self.b13 = BatchNormalization()
        # self.a13 = Activation('relu')
        # self.p13 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')
        # self.d13 = Dropout(0.2)

        # self.flatten = Flatten()
        # self.f14 = Dense(512, activation='relu')
        # self.d14 = Dropout(0.2)
        # self.f15 = Dense(128, activation='relu')
        # self.d15 = Dropout(0.2)
        # self.f16 = Dense(3, activation='softmax')

        # Customized Alex-like Net #
        self.c1 = Conv2D(filters=16, kernel_size=(5, 5), padding='same')
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filters=16, kernel_size=(5, 5), padding='same')
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')
        self.p2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')
        self.d2 = Dropout(0.2)

        self.c3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')
        self.b3 = BatchNormalization()
        self.a3 = Activation('relu')

        self.c4 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')
        self.b4 = BatchNormalization()
        self.a4 = Activation('relu')
        self.p4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')
        self.d4 = Dropout(0.2)

        self.flatten = Flatten()
        self.f14 = Dense(128, activation='relu')
        self.d14 = Dropout(0.2)
        self.f15 = Dense(128, activation='relu')
        self.d15 = Dropout(0.2)
        self.f16 = Dense(3, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)
        x = self.d2(x)

        x = self.c3(x)
        x = self.b3(x)
        x = self.a3(x)

        x = self.c4(x)
        x = self.b4(x)
        x = self.a4(x)
        x = self.p4(x)
        x = self.d4(x)

        # x = self.c5(x)
        # x = self.b5(x)
        # x = self.a5(x)
        #
        # x = self.c6(x)
        # x = self.b6(x)
        # x = self.a6(x)
        #
        # x = self.c7(x)
        # x = self.b7(x)
        # x = self.a7(x)
        # x = self.p7(x)
        # x = self.d7(x)
        #
        # x = self.c8(x)
        # x = self.b8(x)
        # x = self.a8(x)
        #
        # x = self.c9(x)
        # x = self.b9(x)
        # x = self.a9(x)
        #
        # x = self.c10(x)
        # x = self.b10(x)
        # x = self.a10(x)
        # x = self.p10(x)
        # x = self.d10(x)
        #
        # x = self.c11(x)
        # x = self.b11(x)
        # x = self.a11(x)
        #
        # x = self.c12(x)
        # x = self.b12(x)
        # x = self.a12(x)
        #
        # x = self.c13(x)
        # x = self.b13(x)
        # x = self.a13(x)
        # x = self.p13(x)
        # x = self.d13(x)

        x = self.flatten(x)
        x = self.f14(x)
        x = self.d14(x)
        x = self.f15(x)
        x = self.d15(x)
        y = self.f16(x)

        return y


class NeuralNetwork(eval(NET_TYPE)):
    def __init__(self, game, learning_rate=0.001):
        eval(NET_TYPE).__init__(self)

        self.model = eval(NET_TYPE + '()')

        self.predicted = [1, 0, 0]
        self.action = 0
        self.game = game

        # self.model.add_loss(self.my_loss_fun)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            # loss='mse',
            loss=self.my_loss_fun,
            # metrics=['accuracy'],
            metrics=['sparse_categorical_accuracy'],
        )

        checkpoint_save_path = "./Checkpoint/ckpt.ckpt"
        if os.path.exists(checkpoint_save_path + '.index'):
            print("----------------------------- loading model ------------------------------")
            self.model.load_weights(checkpoint_save_path)
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_save_path,
            save_weights_only=True,
            save_best_only=True,
            monitor='loss'
        )

    def my_loss_fun(self, y_true, y_pred):
        # print("calculating loss")
        # squ = (K.square(y_true - y_pred))
        loss = tf.reduce_mean(((- self.game.reward + self.game.winning_reward) / self.game.winning_reward) * y_pred)
        # pre = (- self.game.reward + self.game.winning_reward) / (2 * self.game.winning_reward)

        return loss

    def nn_predict(self, state, batch_size=1):
        self.predicted = self.model.predict(state, batch_size)
        self.action = tf.argmax(self.predicted, axis=1)
        self.action = tf.cast(self.action.numpy().tolist()[0], tf.float32)

        # structure testing code #
        # action = random.randint(1, 3)
        # index = {0: 'UP', 1: 'LEFT', 2: 'RIGHT'}

        # print('action =', self.action)
        return self.action

    def nn_fit(self, x_train, y_train, batch_size):
        history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size, verbose=1, epochs=1,
            callbacks=[self.cp_callback]
        )


def get_red(img):
    """
    Gets the red channel of the image of the screen
    :param img: Colors displayed in surface 'screen'
    :return: red: Numpy array of shape (1, screen_size_x, screen_size_y, 1)
    """
    red = [[0 for __ in range(len(img[0]))] for _ in range(len(img))]
    for line in range(len(img)):
        for col in range(len(img[1])):
            red[col][line] = img[line][col][0]

    red_np = np.array(red) / 255.0  # shape = (size_x, size_y)

    # return red_np.reshape((1, red_np.shape[0], red_np.shape[1], 1))
    return red_np.reshape((red_np.shape[0], red_np.shape[1], 1))


def record_round(img, action, record):
    """
    Record the features of the current frame and add them into an numpy array.
    These features are:
    red channel of the image of the screen;
    real action took;

    :param record: current record. Reinitialized after the round ends.
    :param img: possession of Game
    :param action: possession of Game, type = float32
    :return: record: updated record , type is python list, with elements
     red: numpy array of shape (steps, screen_size_x, screen_size_y, 1) and type float32;
     action: numpy array of shape (steps, ) and type float32;
    """
    red = get_red(img)  # shape = (size_x, size_y, 1)
    # record[0] = np.append(record[0], red)
    # record[1] = np.append(record[1], action)
    record[0].append(red)
    record[1].append(action)
    # print("Step recorded. ")

    # 数据量过大时自动舍弃
    try:
        record[0] = record[0][-2500:]
        record[1] = record[1][-2500:]
        # record = record[-2000:]
    except:
        pass

    return record


def main():
    game = Game(auto=AI_PLAY)
    img = game.game_frame()
    red = get_red(img)
    record = [[], []]

    nn = NeuralNetwork(game, learning_rate=0.002)

    action_hu = None
    action_ai = None

    while True:
        # print('Going into next frame...')

        # -*- Training, Testing, Predicting -*- #
        # 在任意模式下，当动物发生移动，记录移动过程；
        # 在任意模式下，当一局游戏结束，AI整批读入数据，进行训练；
        # 在自动模式下，AI进行预测；
        # index = {0: 'UP', 1: 'LEFT', 2: 'RIGHT'}
        # 游戏结束，开始训练，各项记录初始化（可选）
        if game.round_end is True and GAME_TRAINABLE is True:
            # print(f"type record = {type(record)}, "           # list
            #       f"type record[0] = {type(record[0])}, "     # list
            #       f"type record[1] = {type(record[1])}")      # list
            x_train = np.asarray(record[0])
            y_train = np.asarray(record[1])
            # print(f'Fitting {len(record[0])} data...')
            nn.nn_fit(x_train, y_train, batch_size=32)

        if game.auto is True:
            red = get_red(img)  # shape = (204, 204, 1)
            action_ai = nn.nn_predict(
                red.reshape((1, red.shape[0], red.shape[1], 1)))  # type = float32
        else:
            try:
                action_hu = tf.cast(game.human_action, tf.float32)  # type = python int -> float32
            except:
                pass

        if game.display_changed and GAME_TRAINABLE is True:
            # 先录入采取action时展现的画面，再更新画面
            if game.round_end is False:
                action = action_ai if game.auto else action_hu
                record = record_round(img, action, record)
            # 一轮游戏结束后是否要舍弃数据
            # else:
            #     record = [[], []]

        img = game.game_frame(action_ai)


if __name__ == '__main__':
    main()
