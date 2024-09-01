class Agent:
    def __init__(self):
        
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01

        self.model = Linear_QNet(11, 256, 3)
        self.target_model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, self.target_model, lr=LR, gamma=self.gamma)
        
        self.update_target_network()
        self.load_model_and_memory()

    def save_model_and_memory(self):
        self.model.save()
        with open('./model/memory.pkl', 'wb') as f:
            pickle.dump(self.memory, f)
        with open('./model/n_games.pkl', 'wb') as f:
            pickle.dump(self.n_games, f)

    def load_model_and_memory(self):
        if os.path.exists('./model/model.pth'):
            self.model.load('./model/model.pth')
        if os.path.exists('./model/memory.pkl'):
            with open('./model/memory.pkl', 'rb') as f:
                self.memory = pickle.load(f)
        if os.path.exists('./model/n_games.pkl'):
            with open('./model/n_games.pkl', 'rb') as f:
                self.n_games = pickle.load(f)