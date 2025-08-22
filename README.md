# 🚍 Graph Exploration – Multi-Agent Urban Bus Simulation (Modular)

Este projeto fornece um **ambiente de aprendizado por reforço multiagente** para operações de ônibus urbanos, baseado em dados reais do sistema **Salvador Urban Network Transportation (SUNT)**.  
Ele foi refatorado em um **pacote modular**, com estrutura limpa, separação clara de responsabilidades e uma CLI unificada.

---

## 🚌 Visão Geral

O **Ambiente de Simulação Multiagente de Ônibus Urbanos** é construído sobre dados reais de transporte público de Salvador (Brasil).  
Simula a operação de múltiplos ônibus como agentes independentes navegando em uma rede real, permitindo desenvolver e testar estratégias inteligentes de controle para transporte coletivo.

Aspectos principais:
- Cenários realistas e **baseados em dados**.
- Foco em **otimizar eficiência do serviço** e **experiência do passageiro**.
- Uso de **dados de embarque, desembarque e tempos de viagem** reais.

---

## 🎯 Objetivos

Os agentes (ônibus) são treinados para:
- Reduzir o tempo de espera dos passageiros nos pontos.  
- Manter regularidade nos intervalos entre veículos (headways).  
- Balancear ocupação (evitar lotação ou vazio).  
- Operar de forma eficiente em energia e manutenção.  

O sistema aplica **Aprendizado por Reforço Multiagente (MARL)**, onde cada ônibus age de forma autônoma, mas coopera implicitamente através de uma **função de recompensa compartilhada**.

---

## 📊 Sinais de Observação

Durante o treinamento, o ambiente gera métricas como:

- **avg_travel_time_AB** → Tempo médio de viagem entre pontos de referência.  
- **future_demand_at_B** → Demanda prevista de passageiros em um ponto.  
- **occupancy_rate** → Taxa de ocupação do ônibus.  
- **uptime_normalized** → Disponibilidade normalizada do veículo em operação.  

Esses sinais alimentam a função de recompensa e podem ser usados para monitoramento.

---

## 🎮 Ações

Cada ônibus (agente) pode escolher entre:

- **WAIT** → Aguardar antes de seguir, para evitar agrupamento.  
- **MOVE** → Ir ao próximo ponto.  
- **SERVICE_CENTER** → Desviar para manutenção quando necessário.  

---

## 🎯 Função de Recompensa

A recompensa combina:
- Qualidade do serviço ao passageiro (espera menor, demanda atendida).  
- Eficiência operacional (ocupação equilibrada, viagens regulares).  
- Manutenção/combustível (penalidade por ignorar problemas).  
- Fluxo e coordenação (evitar ônibus emparelhados).  

---

## 🛠 Setup de Treinamento

Integra com:

- **Ray RLlib** → Treinamento distribuído.  
- **PettingZoo** → API multiagente.  
- **SuperSuit** → Wrappers de pré-processamento.  
- **Gymnasium** → API padrão.  

**Configuração:**
- Cada ônibus é um agente.  
- Cenário baseado em dados reais (rotas, pontos, demanda).  
- Cada episódio ≈ um dia simulado de operação.  
- PPO (Proximal Policy Optimization) com política compartilhada.  

**Escalabilidade:**  
Suporta **centenas de agentes em paralelo** com Ray distribuído.

---

## 📂 Estrutura do Projeto

```
src/
├─ envs/                        # ambientes PettingZoo
├─ pipelines/                   # observações, rotas, métricas, treino RLlib
├─ tools/                       # utilitários de análise e manipulação de dados
├─ viz/                         # visualização de grafos
├─ tests/                       # testes automatizados
├─ training_observation/        # observações de treino
├─ output_observation_travel_time_sum_amout/  # outputs experimentais
└─ __pycache__/                 # cache python
```

- **CLI** expõe subcomandos que mapeiam para esses módulos.  
- Alguns scripts ainda usam **caminhos hardcoded** → recomendável migrar para configs ou `.env`.  

---

## ⚡ Instalação & Uso

```bash
# 1. Criar ambiente virtual
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 2. Atualizar ferramentas básicas
pip install --upgrade pip setuptools

# 3. Instalar dependências em modo editável
pip install -e ".[rllib,data,viz,test]"

# 4. (Linux / macOS) Exportar PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

#    (Windows PowerShell)
$env:PYTHONPATH = (Get-Location).Path + ";" + $env:PYTHONPATH

# 5. Rodar testes
pytest -q
```

### CLI (`graphx`)

```bash
# Treinamento com RLlib
graphx train -- --help

# Estatísticas do dataset
graphx stats -- --help

# Tamanho do dataset e contagem de itens
graphx look-amount -- --help

# Médias em arquivos PKL
graphx pkl-medias -- --help

# Explorar arquivos de rotas
graphx see-routes -- --help

# Ver conteúdo de PKL
graphx view-pkl -- --help

# Visualizar grafos
graphx view-graph -- --help

# Executar ambiente SUNT
graphx env-sunt --
```

---

## ✅ Próximos Passos

- Substituir **caminhos hardcoded** por configs/env.  
- Aumentar cobertura de **testes**.  
- Adicionar suporte a **experiment tracking** (MLflow, W&B).  
- Modularizar recompensas e observações para experimentos mais flexíveis.  
