# Multi-Agent Urban Bus Simulation Environment (SUNT-based)

Este ambiente simula uma operação multiagente de ônibus urbanos sobre a rede real de transporte da cidade de Salvador, utilizando dados reais do sistema SUNT (Salvador Urban Network Transportation). Ele foi desenvolvido para treinar agentes (ônibus) em cenários realistas com o objetivo de melhorar a eficiência do serviço prestado à população.

---

## 🎯 Objetivo

Treinar agentes de transporte público (ônibus) capazes de:

- Reduzir o tempo de espera dos passageiros nos pontos de parada;
- Regularizar o intervalo entre ônibus (headway);
- Manter um nível adequado de ocupação dos veículos;
- Operar com eficiência energética e mínima degradação de manutenção.

O objetivo é alcançar uma **melhor cobertura do serviço**, otimizando o atendimento aos passageiros a partir de **dados reais de embarque, desembarque e tempo de viagem**.

---

## 🧠 Aprendizado por Reforço Multiagente

Este ambiente segue o paradigma de **MARL (Multi-Agent Reinforcement Learning)** com múltiplos ônibus que:

- Operam de forma **descentralizada**, mas **cooperam implicitamente**;
- Compartilham uma função de recompensa que considera métricas globais de desempenho;
- Aprendem a se coordenar para manter o fluxo do sistema de transporte.

---