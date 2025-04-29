# Импорт необходимых библиотек
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"  # Использование GPU, если доступен

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Импорты из библиотеки simglucose
from simglucose.simulation.env import T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario import CustomScenario

# Импорт алгоритма DQN из stable_baselines3
from stable_baselines3 import DQN
from collections import namedtuple

# Пороговые значения для оценки уровня глюкозы (в мг/дл)
GLUCOSE_IDEAL_LOW = 90
GLUCOSE_IDEAL_HIGH = 140
GLUCOSE_TARGET_LOW = 70
GLUCOSE_TARGET_HIGH = 180
GLUCOSE_CRITICAL_LOW = 30
GLUCOSE_CRITICAL_HIGH = 300

# Базальная скорость подачи инсулина и доза глюкагона (если включён)
BASAL_RATE = 0.5
GLUCAGON_DOSE = 0.3

# Кортеж для хранения действия агента (инсулин/глюкагон/углеводы)
Action = namedtuple('Action', ['basal', 'bolus', 'carbs'])

# Функция генерации сценария с вариативными приёмами пищи
def generate_variable_meal_scenario(start_time, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Базовые точки питания с углеводной нагрузкой
    base_meals = [
        ("01:00", 70),
        ("03:00", 70),
        ("07:00", 70),
        ("10:00", 30),
        ("14:00", 110),
        ("21:00", 90)
    ]

    scenario = []
    for time, base_carb in base_meals:
        # Вносим вариации в количество углеводов (нормальное распределение, CV=10%)
        carb_amount = np.random.normal(loc=base_carb, scale=0.1 * base_carb)
        carb_amount = max(0, round(carb_amount))  # Ограничение снизу: углеводы не могут быть < 0
        scenario.append((str_to_timedelta(time), carb_amount))

    return CustomScenario(start_time=start_time, scenario=scenario)

# Функция преобразования времени из строки в timedelta
def str_to_timedelta(time_str):
    time_obj = datetime.strptime(time_str, '%H:%M')
    return timedelta(hours=time_obj.hour, minutes=time_obj.minute)

# Кастомная среда для обучения агента
class DiabetesRLEnv(gym.Env):
    def __init__(self, patient_name='adolescent#001', dual_hormone=False):
        # Загрузка параметров пациента, сенсора и насоса
        patient = T1DPatient.withName(patient_name)
        sensor = CGMSensor.withName('Dexcom')
        pump = InsulinPump.withName('Insulet')
        start_time = str_to_timedelta('06:00')

        # Вариации метаболических параметров пациента
        param_variability = {
            'Vm0': np.random.normal(1.0, 0.1),
            'Vmx': np.random.normal(1.0, 0.1),
            'EGPb': np.random.normal(1.0, 0.1),
            'CL': np.random.normal(1.0, 0.1),
            'HEb': np.random.normal(1.0, 0.1),
        }
        for key, scale in param_variability.items():
            if key in patient._params:
                patient._params[key] *= scale

        # Генерация сценария питания
        scenario = generate_variable_meal_scenario(start_time)

        # Создание среды симуляции
        self.realistic_env = T1DSimEnv(patient, sensor, pump, scenario)
        self.dual_hormone = dual_hormone
        self.weight = 70  # Вес пациента, влияет на дозы

        # Дискретное пространство действий: 5 (инсулин) + 1 (глюкагон, если включён)
        self.action_space = spaces.Discrete(6 if dual_hormone else 5)

        # Наблюдение: [глюкоза, тренд, время с последнего действия, инсулин, глюкагон]
        self.observation_space = spaces.Box(
            low=np.array([0, -5, 0, 0, 0], dtype=np.float32),
            high=np.array([500, 5, 120, 10, 10], dtype=np.float32),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        self.realistic_env.reset()
        self.current_glucose = 100.0
        self.prev_glucose = 100.0
        self.time = 0
        self.last_insulin = 0.0
        self.last_action_time = -30
        self.trend = 0
        self.total_glucagon = 0
        self.last_glucagon = 0.0
        self.glucose_history = []
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.current_glucose,
            self.trend,
            self.time - self.last_action_time,
            self.last_insulin,
            self.last_glucagon
        ], dtype=np.float32)

    def step(self, action):
        self.glucose_history.append(self.current_glucose)
        if len(self.glucose_history) > 4:
            self.glucose_history = self.glucose_history[-4:]

        # Расчёт тренда глюкозы по полиному 2-й степени
        if len(self.glucose_history) == 4:
            x = np.array([0, 5, 10, 15])
            y = np.array(self.glucose_history)
            coeffs = np.polyfit(x, y, deg=2)
            trend = 2 * coeffs[0] * 15 + coeffs[1]
            trend_ready = 1
        else:
            trend = 0
            trend_ready = 0

        self.trend = trend

        # Обработка действия: выбор между инсулином и глюкагоном
        if self.dual_hormone and action == 5:
            glucagon = GLUCAGON_DOSE * self.weight
            sim_action = Action(basal=0, bolus=0, carbs=30.0)
            self.total_glucagon += glucagon
            self.last_glucagon = glucagon
            self.last_insulin = 0.0
        else:
            basal_rate = BASAL_RATE * [0.0, 0.5, 1.0, 1.5, 2.0][action]
            sim_action = Action(basal=basal_rate, bolus=0, carbs=0)
            self.last_insulin = basal_rate / 12
            self.last_glucagon = 0.0

        # Обновление среды и расчёт награды
        step_result = self.realistic_env.step(sim_action)
        obs = step_result.observation
        self.prev_glucose = self.current_glucose
        self.current_glucose = float(obs.CGM)
        self.time += 5
        reward = self._calculate_reward(action, trend, trend_ready)
        done = self._check_done()

        return self._get_obs(), reward, done, False, {}

    # Расчёт награды за шаг
    def _calculate_reward(self, action, trend, tre):
        glucose = self.current_glucose
        reward = 0.0

        # Награды/штрафы за отклонения от целевых зон
        if GLUCOSE_IDEAL_LOW <= glucose <= GLUCOSE_IDEAL_HIGH:
            reward += 1.0
        elif GLUCOSE_TARGET_LOW <= glucose < GLUCOSE_IDEAL_LOW or GLUCOSE_IDEAL_HIGH < glucose <= GLUCOSE_TARGET_HIGH:
            reward += 0.1
        elif glucose > GLUCOSE_TARGET_HIGH:
            reward -= 0.4 + (glucose - GLUCOSE_TARGET_HIGH) / 200
            if glucose > GLUCOSE_CRITICAL_HIGH:
                reward -= 1.0
        elif glucose < GLUCOSE_TARGET_LOW:
            reward -= 0.6 + (GLUCOSE_TARGET_LOW - glucose) / 100
            if glucose < GLUCOSE_CRITICAL_LOW:
                reward -= 1.0

        # Штраф за ненужные действия в идеальной зоне
        if GLUCOSE_IDEAL_LOW <= glucose <= GLUCOSE_IDEAL_HIGH and action != 0:
            reward -= 3.0

        # Дополнительные условия при наличии тренда
        if tre:
            if glucose < GLUCOSE_TARGET_LOW and action in [1,2,3,4]:
                reward = -5
            elif glucose > GLUCOSE_TARGET_HIGH and trend > 0:
                reward += 1.3 if action in [3,4] else -5

        return reward

    # Проверка на завершение эпизода
    def _check_done(self):
        return (
            self.current_glucose < GLUCOSE_CRITICAL_LOW
            or self.current_glucose > GLUCOSE_CRITICAL_HIGH
            or self.time >= 1440  # 24 часа (по 5 минут на шаг)
            or (self.dual_hormone and self.total_glucagon > 1000)
        )

    def render(self):
        print(f"Time: {self.time} | Glucose: {self.current_glucose:.1f} | Trend: {self.trend:.1f}")

    # ======= Обучение и тестирование =======

    dual_hormone = False  # Можно переключить в True для двухгормональной модели
    env = DiabetesRLEnv(dual_hormone=dual_hormone)

    # Настройка и обучение агента DQN
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=20000,
        batch_size=64,
        gamma=0.95,
        exploration_fraction=0.2,
        tensorboard_log="./logs/",
        device=device
    )

    print("Training agent...")
    model.learn(total_timesteps=300000)
    model.save("dqn_diabetes_model")

    # ======= Тестирование агента =======
    obs, _ = env.reset()
    glucose_levels, actions, basal_rates, glucagon_doses, meal_times = [], [], [], [], []

    for step in range(1440):  # 24 часа = 1440 шагов по 5 минут
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)

        glucose_levels.append(obs[0])
        actions.append(action)

        # Запоминаем моменты приёма пищи
        current_time = env.realistic_env.time
        current_action = env.realistic_env.scenario.get_action(current_time)
        if current_action.meal > 0:
            meal_times.append(step)

        # Сохраняем данные о подаче инсулина и глюкагона
        if dual_hormone and action == 5:
            basal_rates.append(0)
            glucagon_doses.append(GLUCAGON_DOSE * env.weight)
        else:
            basal_rates.append(BASAL_RATE * [0, 0.5, 1.0, 1.5, 2.0][action])
            glucagon_doses.append(0)

        if done:
            break

    # ======= Визуализация результатов =======
    fig, axs = plt.subplots(2, 1, figsize=(12, 14), sharex=True)

    # График уровня глюкозы
    axs[0].plot(glucose_levels, label="Глюкоза", color="blue")
    axs[0].axhline(y=GLUCOSE_IDEAL_LOW, color='green', linestyle='--', label="Идеальный диапазон")
    axs[0].axhline(y=GLUCOSE_IDEAL_HIGH, color='green', linestyle='--')
    axs[0].axhline(y=GLUCOSE_TARGET_LOW, color='red', linestyle=':', label="Границы безопасности")
    axs[0].axhline(y=GLUCOSE_TARGET_HIGH, color='orange', linestyle=':')
    axs[0].set_ylabel("Глюкоза (мг/дл)")
    axs[0].legend()
    axs[0].set_title("Динамика глюкозы под управлением DQN-агента")

    # График поданных доз инсулина
    axs[1].plot(basal_rates, label="Базальный инсулин", color="red")
    if dual_hormone:
        axs[1].bar(range(len(glucagon_doses)), glucagon_doses, width=1.0, label="Глюкагон", color="purple", alpha=0.5)
    for mt in meal_times:
        axs[1].axvline(x=mt, color='brown', linestyle='--', alpha=0.5,
                       label="Приём пищи" if mt == meal_times[0] else None)
    axs[1].set_ylabel("Доза")
    axs[1].legend()
    axs[1].set_xlabel("Время (шаги по 5 мин)")

    plt.tight_layout()
    plt.show()