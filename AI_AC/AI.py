class Environment:
    def run(self):
        while not self.stop_signal:
            self.run_episode()

    def run_episode(self):
    # act and get next state
        s = self.env.reset()
        while True:
            time.sleep(THREAD_DELAY)

            a = self.agent.act(s)
            s_, r, done, info = self.env.step(a)

            if done:
                s_ = None

            self.agent.train(s, a, r, s_)

            s = s_

            if done or self.stop_signal:
                break

class Agent:
    def act(self, s):
    # decide what to act
        if random.random() < epsilon:    # explore a random action
            return random.randint(0, NUM_ACTIONS-1)
        else:
            p = brain.predict_p(s)
            return np.random.choice(NUM_ACTIONS, p=p)

    def train(self, s, a, r, s_):
    # process sample
        a_cats = np.zeros(NUM_ACTIONS)
        a_cats[a] = 1

        self.memory.append((s, a_cats, r, s_))

        def get_sample(memory, n):
            r = 0.
            for i in range(n):
                r += memory[i][2] * (GAMMA ** i)
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n-1]
            return s, a, r, s_

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            brain.train_push(s, a, r, s_)
            self.memory.pop(0)

        if s_ is None:    # terminal state
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                brain.train_push(s, a, r, s_)
                self.memory.pop.(0)






class Brain:
    def predict(self, s):    # get pi(s) and V(s)
    def train_push(sample):
    # add sample to training queue
        with self.lock_queue:    # [s, a, r, s_, terminal]
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if s_ is None:
                self.train_queue[3].append(NONE_STATE)
                self.train_queue[4].append(0.)
            else:
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)

    def optimize(self):
    # gradien descent
        if len(self.train_queue[0]) < MIN_BATCH:
            time.sleep(0)
            return


class Optimizer:
    def run(self):
    # call Brain.optimize() in loop
        while not self.stop_signal:
            brain.optimize


l_input = Input(batch_shape=(None, NUM_STATE))
l_dense = Dense(16, activation='relu')(l_inpot)

out_actions = Dense(NUM_ACTIONS, activation ='softmax')(l_dense)
out_value = Dense(1, activation='linear')

model = Model(inputs=[l_input], outputs=[out_actions, out_value])

# Loss function