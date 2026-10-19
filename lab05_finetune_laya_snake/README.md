# Lab 05: Fine-Tuning Laya Decision Engine — Nokia 3310 Snake AI

Benvenuto al **Lab 05**! In questo laboratorio esplorerai il processo completo di **Fine-Tuning e Adattamento di Laya Decision Engine** (partendo dal checkpoint base `convaiinnovations/laya`) per giocare al celebre videogioco retro del **Nokia 3310: Snake**.

---

## 🕹️ Panoramica dell'Ambiente e Obiettivo

```
┌─────────────────────────────────────────────────────────────┐
│  🕹️  NOKIA 3310 RETRO SNAKE - LAYA DECISION ENGINE AGENT   │
├─────────────────────────────────────────────────────────────┤
│  ╔════════════════════╗                                     │
│  ║                    ║  Stato di Gioco:                    │
│  ║    🍎      🟢🟢🟢  ║  • Testa Serpente: (x, y), Direz    │
│  ║🟩      🟢🟢🟢      ║  • Cibo / Mela:    (x, y)           │
│  ║🟢      🟢          ║  • Sensori Hazard: (Front, L, R)    │
│  ║🟢      🟢          ║                                     │
│  ║🟢🟢🟢🟢🟢          ║                                     │
│  ╚════════════════════╝                                     │
│  Decisione Laya (~1.1ms):                                   │
│  • Action (Choice): [UP] (100.0%)                           │
│  • Hazard Ahead (Noul): False (0.0%)                        │
│  • Path Safety (Score): Level 3 [OPTIMAL]                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Architettura Multi-Task di Laya per Snake

Laya modella il gioco di Snake come un problema di decisione discreto multi-testa:

1. **`action` (Type: `choice`)**: Seleziona la direzione ottimale (`UP`, `DOWN`, `LEFT`, `RIGHT`).
2. **`is_danger_ahead` (Type: `noul`)**: Predice la probabilità di collisione imminente con muri o corpo ($P(\text{danger}) \in [0, 1]$).
3. **`path_safety_score` (Type: `score`)**: Stima ordinale calibrata della sicurezza del percorso ($0$: Trapped, $1$: Tight, $2$: Open, $3$: Optimal).

---

## 📁 Struttura degli Script

```
lab05_finetune_laya_snake/
├── 01_generate_snake_dataset.py       # Simula partite esperte e genera il dataset di traiettorie
├── 02_train_laya_snake.py             # Training loop multi-task loss (CrossEntropy + BCE)
├── 03_play_snake_interactive.py       # Demo interattiva con visualizzatore retro Nokia 3310
├── 04_evaluate_snake.py               # Benchmark su 50 partite vs Random Policy & Expert A*
├── snake_checkpoint.pt                # Checkpoint pesi fine-tuned
├── snake_dataset.jsonl                # Dataset di traiettorie annotate
└── snake_benchmark_results.png        # Grafico comparativo delle metriche
```

---

## 🚀 Esecuzione del Laboratorio

### 1. Generazione del Dataset di Traiettorie
Genera 9.500+ decisioni supervisionate da simulazioni esperte:
```bash
python3 lab05_finetune_laya_snake/01_generate_snake_dataset.py
```

### 2. Fine-Tuning Multi-Task
Addestra le teste decisionali su GPU / MPS o CPU:
```bash
python3 lab05_finetune_laya_snake/02_train_laya_snake.py
```

### 3. Gameplay Interattivo in Tempo Reale
Guarda l'agente Laya giocare a Snake sul terminale in stile Nokia 3310:
```bash
# Esegui a 12 FPS standard
python3 lab05_finetune_laya_snake/03_play_snake_interactive.py 12

# Esegui a 50 FPS turbo
python3 lab05_finetune_laya_snake/03_play_snake_interactive.py 50
```

### 4. Valutazione e Benchmark
Confronta accuratezza, mele mangiate e latenza rispetto a una baseline casuale e al risolutore A*:
```bash
python3 lab05_finetune_laya_snake/04_evaluate_snake.py
```

---

## 📊 Risultati e Metriche Raggiunte

| Policy / Modello | Mele Medie per Partita | Max Mele Raggiunte | Sopravvivenza Media | Latenza Decisionale | Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **🎲 Random Policy** | `0.1` | `2` | `15.0 mosse` | `< 0.01 ms` | High |
| **🧠 Fine-Tuned Laya** | **`16.5`** | **`22`** | **`126.7 mosse`** | **`0.36 ms`** | **`2,662 fps`** |
| **⭐ Expert A\* Solver**| **`16.5`** | **`22`** | **`126.7 mosse`** | `< 0.01 ms` | High |
