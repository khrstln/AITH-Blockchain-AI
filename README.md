# 🎓 Финальный проект по курсу «AI in BlockChain»

**Авторы:** Юрпалов Сергей, Хрусталев Илия

## 🔎 Основная идея

Создать «Oracle»-инструмент, который поможет выбрать оптимальную gas price и временное окно для сделок в Ethereum сети.

## 📚 Использованные статьи

Мы провели обзор трёх ключевых работ, но воспроизвели результаты только первой:

1. **Blockchain Transaction Fee Forecasting: A Comparison of Machine Learning Methods** (Conall Butler, Martin Crane)
2. Step on the Gas? A Better Approach for Recommending the Ethereum Gas Price (Werner et al.)
3. A Practical and Economical Bayesian Approach to Gas Price Prediction (Chuang & Lee)

## 📊 Метрики

- **MAPE** = 1/n ∑ₜ |Aₜ – Fₜ| / Aₜ
- **MAE** = 1/n ∑ₜ |yₜ – ŷₜ|
- **MAPE@5m** - решили пока не использовать, т.к. метрика выглядит как cherry-picking 🍒 и может исказить оценку качества моделей.

**Результаты на тестовой выборке**:

| Модель       | MAPE              | MAE             |
| ------------------ | ----------------- | --------------- |
| RNN                | 7204.23           | 10.79           |
| LSTM               | 7308.68           | 10.70           |
| **bi-LSTM**  | 6641.29           | **10.24** |
| **CNN-LSTM** | **6197.80** | 10.34           |
| Attention‑LSTM    | 6409.60           | 10.28           |

## 💥Исследованные подходы по работе с выбросами

- **Замена MSE на Huber-loss или Quantile-loss** для снижения влияния выбросов при обучении. Оказалось неэффективно из-за того, что пики слишком экстремальные, а Huber слишком мягкий.
- Добавление в выходной слой смесь обобщённого **парето-распределения (GP)** с порогом для экстремальных событий и **автоэнкодер-LSTM** для фич. Получили модель качеством значительно хуже, чем бейзлайн. [Подход](https://arxiv.org/abs/2310.07435)
- Пытались сначала **отделить аномалии** (Isolation Forest, CUSUM), потом сделать **отдельную регрессию для их амплитуды**. Классификация работает 50/50, получили только больше шума. [Подход](https://s-ai-f.github.io/Time-Series/outlier-detection-in-time-series.html)

## 🤖 Модели и результаты

- **RNN** и **LSTM**: бейзлайны без амбиций на хороший результат.
- **BiLSTM**: двунаправленный LSTM, ловит выбросы (MAE ≈10.24).
- **CNN-LSTM**: Conv1D+LSTM, лучшая MAPE ≈6198.
- **Attention-LSTM**: LSTM с вниманием, сбалансированный (MAPE ≈6410).

## 🚀 Планы дальнейшего развития

- Расширить датасет
- Улучшить работу с выбросами и аномалиями
- Сделать UI на Streamlit

## 📱UI

![img](https://github.com/khrstln/AITH-Blockchain-AI/blob/main/docs/UI.png?raw=true)

---

## Как воспроизвести исследование?

> 1. Установить UV package manager.
> 2. `git clone https://github.com/khrstln/AITH-Blockchain-AI`
> 3. `cd AITH-Blockchain-AI && uv sync`
>    *Рекомендуется GPU для ускоренного инференса на PyTorch.*
