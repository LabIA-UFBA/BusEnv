if __name__ == "__main__":

    # episodes: número de episódios de treino/teste
    # env: ambiente (instância da classe GraphExplorationEnv)
    # alpha: taxa de aprendizado (quanto o agente aprende a cada iteração)
    # gamma: fator de desconto (quão importante é o futuro em relação ao presente)
    # epsilon: taxa de exploração (probabilidade de explorar vs. explorar o conhecimento atual)
    # is_training: se True, treina o agente; se False, carrega a tabela Q de um arquivo
    def run_q(episodes, env, alpha, gamma, epsilon, is_training=True):
        import time
        import matplotlib.pyplot as plt
        startTime = time.time() # Registra o tempo para garantir que a execução não ultrapasse 2 horas
        max_steps = 5000  # Limite (generico) de passos por episódio, valor pode mudar

       #  rewardsPerStep = []  # Array para guardar a recompensa total de cada passo

        rewardsPerEpisode = np.zeros(episodes)


        try:
            if(is_training): # Se for treinamento, começamos com Q-table vazia
                q = {}
            else: # Se não for treinamento, carregamos a Q-table de um arquivo
                f = open('q.pkl', 'rb')
                q = pickle.load(f)
                f.close()
                
            stepsPerEpisode = np.zeros(episodes) # Array para guardar quantos passos foram dados em cada episódio
        
            for i in range(episodes): # Inicia o laço de episódios de treino/teste
                
                # Reinicia o ambiente (env.reset()) e recebe o estado inicial (posição atual + alvo)
                stepCount = 0 # contador de passos do episódio atual
                currentEpisodeReward = 0 # Variável para acumular a recompensa do episódio atual
                state = env.reset()[0]
                start = state[0]
                target = state[1]
                terminated = False # Terminated marca se o episódio terminou (atingiu o alvo)

                while(not terminated and stepCount < max_steps): # Enquanto o episódio não terminar seguir

                    neighbors = list(env.network.neighbors(state[0]))
                    if not neighbors:  # Se não houver vizinhos, termina o episódio
                        terminated = True
                        continue

                    if state not in q: # Se esse estado (posição atual + alvo) ainda não está na Q-table, criamos entradas para ele
                        q[state] = {}
                        for act in range(len(list(env.network.neighbors(state[0])))):
                            q[state][act] = 0 # Cada entrada é associada a um possível vizinho do nó atual     

                    a = random.random() # Gera número aleatório    

                    if is_training and a< epsilon: # Se a < epsilon → agente explora: escolhe ação aleatória
                        n = list(env.network.neighbors(state[0]))
                        action = random.randint(0, max(0,len(n) -1)) # based on number of neighbors of position node
                    else: # Senão → agente explota: escolhe ação com maior valor Q (com desempate aleatório)           
                        values = [ q[state][act] for act in range(len(list(env.network.neighbors(state[0])))) ]
                        ix_max = [ act for act, value in enumerate(values) if value == max(values) ]
                        action = ( random.choice(ix_max) ) if len(ix_max) > 0 else 0
                    
                    # Faz o passo no ambiente, recebendo o novo estado, recompensa, se o episódio terminou (chegou ao alvo?), e metadados
                    new_state,reward,terminated,_,extra = env.step(action)

                    currentEpisodeReward += reward # Acumula a recompensa do episódio atual

                    # rewardsPerStep.append(reward) # Acumula a recompensa total de cada passo

                    rewardsPerEpisode[i] = currentEpisodeReward
                    
                    # Mesma lógica que antes — se o novo estado ainda não está na Q-table, criamos entradas
                    if new_state not in q:
                        q[new_state] = {}
                        for act in range(len(list(env.network.neighbors(new_state[0])))):
                            q[new_state][act] = 0

                    # Atualização da Q-table (apenas se for treino)
                    # Essa é a equação padrão do Q-learning 
                    if is_training:

                        # A Q-table é atualizada com base na recompensa recebida e no valor máximo da próxima ação
                        qValues = [q[new_state][act] for act in q[new_state]] if q[new_state] else [0]
                        # qValues = [ q[new_state][act] for act in range(len(list(env.network.neighbors(new_state[0])))) ]
                        
                        sample = reward + gamma * (max(qValues) if len(qValues) > 0 else 0)

                        part1 = ( (1 - alpha) * q[state][action] ) if len(q[state]) > 0 else float('-inf')
                        part2 = alpha * sample
                        
                        q[state][action] = part1 + part2

                    # Atualiza o estado atual para o novo estado
                    state = new_state

                    stepCount += 1

                    # Se o episódio terminou (o agente chegou ao alvo), imprime o resultado e salva o número de passos
                    if terminated:
                        print(f"({i}) {start} -> {target} in {stepCount} steps with epsilon {epsilon}")
                        stepsPerEpisode[i] = stepCount
                        stepCount = 0
                
                # rewardsPerStep[i] = currentEpisodeReward
                # Reduz o epsilon ao longo dos episódios para explorar menos e explorar mais com o tempo
                epsilon = max(epsilon - 1/(episodes), 0.01) # Nunca deixa epsilon menor que 0.01 (exploração mínima)
                if time.time() - startTime > 60 * 60 * 2: # 2 hours
                    break

            env.close() # Fecha o ambiente

        finally:
            # Calcula a média móvel dos passos por episódio (últimos 100 episódios)
            # Plota esse gráfico e salva
            print("np.zeros(i): ", np.zeros(i))
            sumSteps = np.zeros(i)
            for t in range(i):
                sumSteps[t] = np.mean(stepsPerEpisode[max(0, t-100):(t+1)]) # Média móvel dos passos por episódio

            plt.plot(sumSteps) # Plota a média móvel dos passos por episódio
            plt.savefig(f'q{env.initial}-{env.target}-{alpha}-{gamma}-{epsilon}.png')

            # Plota a RECOMPENSA total por passo
            #window_size = 100
            #moving_avg = [np.mean(rewardsPerStep[max(0, i-window_size):i+1]) for i in range(len(rewardsPerStep))] # Média móvel da recompensa total por passo
            
            # Plota a RECOMPENSA total por episódio (média móvel)
            window_size = 100
            moving_avg = [np.mean(rewardsPerEpisode[max(0, i-window_size):i+1]) for i in range(len(rewardsPerEpisode))]

            plt.figure(figsize=(12,6))
            plt.plot(moving_avg)
            plt.title("Recompensa por episódio (média móvel)")
            plt.xlabel("Episódios")
            plt.ylabel("Recompensa média")
            plt.grid(True)
            plt.savefig(f'rewards-smooth-{env.initial}-{env.target}-{alpha}-{gamma}-{epsilon}.png')

            # np.save(f'steps_{env.initial}_{env.target}.npy', stepsPerEpisode) # Salva o número de passos por episódio em um arquivo .npy
            # Se for treinamento, salva a Q-table em um arquivo
            if is_training:
                # Save Q Table
                f = open(f'q{env.initial}-{env.target}-{alpha}-{gamma}-{epsilon}.pkl',"wb")
                pickle.dump(q, f)
                f.close()


    with open('./sunt/graph_designer/graph_gtfs.gpickle', 'rb') as f: # Carrega o grafo salvo
        G = pickle.load(f)

    
    env = GraphExplorationEnv(G, 9) # Cria o ambiente com o grafo carregado e 9 ações possíveis (número máximo de vizinhos de um nó)

    run_q(10, env, 0.9, 0.9, 0.2, is_training=True) # Executa run_q(...) com 20000 episódios e hiperparâmetros definidos (alpha=0.9, gamma=0.9, epsilon=0.2).

    #run_q(1, env, alpha=0.9, gamma=0.9, epsilon=0.0, is_training=False) # Teste da política aprendida