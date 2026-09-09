#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 generate_loo43_predictions.py 中提取出的 BP 与 GA-BP 对比程序。

保留源文件中的模型逻辑和参数：
1. BP：7-4-1 单隐含层，ReLU，Adam，最大迭代 1000 次。
2. GA-BP：7-4-1 单隐含层，种群 80，进化 150 代，随后进行 1000 次 BP 训练。
3. GA-BP 默认使用 tau=0.2；可通过环境变量 GABP_TAU 设置敏感性分析值。
4. 保留全部 44 个样本，执行 44 折留一交叉验证。
5. 两种模型的最终造价预测均施加严格正值约束。

数据改为 bp1-data.csv：第二行是实际列名，x1~x7 为 7 个输入特征，y 为成本。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(r"C:\python code\BP\bp1-data.csv")
OUTPUT_PATH = Path(r"C:\python code\BP\BP VS GABP_predictions.xlsx")

FEATURE_COLUMNS = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
TARGET_COLUMN = "y"
ID_COLUMN = "index"

RANDOM_STATE = 53
POSITIVE_PREDICTION_FLOOR = 1e-6
DEFAULT_TAU = 0.2
GABP_TAU = float(os.environ.get("GABP_TAU", DEFAULT_TAU))

if not 0.0 < GABP_TAU < 1.0:
    raise ValueError(f"GABP_TAU 必须位于 (0, 1)，当前值为 {GABP_TAU}")


def constrain_positive_prediction(prediction: np.ndarray) -> np.ndarray:
    """将恢复到原始造价尺度后的预测值限制为严格正值。"""
    prediction = np.asarray(prediction, dtype=float)
    return np.maximum(prediction, POSITIVE_PREDICTION_FLOOR)


def load_data(path: Path) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """读取 7 输入特征的 bp1-data.csv。"""
    # CSV 第一行是特征说明，第二行 x1~x7、y 才是实际列名。
    dataset = pd.read_csv(path, header=1)

    required_columns = [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in dataset.columns]
    if missing_columns:
        raise ValueError(
            f"数据文件缺少列：{missing_columns}；实际列为：{list(dataset.columns)}"
        )

    X = dataset.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(dataset[TARGET_COLUMN], errors="coerce")
    project_id = dataset[ID_COLUMN].copy()

    valid_rows = X.notna().all(axis=1) & y.notna()
    if not valid_rows.all():
        removed_count = int((~valid_rows).sum())
        print(f"[提示] 删除含有非数值或缺失数据的样本：{removed_count} 个")

    X = X.loc[valid_rows].reset_index(drop=True)
    y = y.loc[valid_rows].reset_index(drop=True)
    project_id = project_id.loc[valid_rows].reset_index(drop=True)

    if X.shape[1] != 7:
        raise ValueError(f"输入特征数应为 7，实际为 {X.shape[1]}")

    return X, y, project_id


def underestimation_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred < y_true))


def mape_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.abs(y_true) > 1e-12
    if not np.any(valid):
        return float("nan")
    return float(
        np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])) * 100
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    absolute_error = np.abs(y_true - y_pred)

    return {
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MedAE": float(np.median(absolute_error)),
        "R2": float(r2_score(y_true, y_pred)),
        "EVS": float(explained_variance_score(y_true, y_pred)),
        "MAPE_%": mape_safe(y_true, y_pred),
        "UR_%": underestimation_rate(y_true, y_pred) * 100,
    }


def progressive_r2_removal(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_remove: int = 7,
) -> pd.DataFrame:
    """依次删除绝对残差最大的样本并重新计算指标。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    removal_order = np.argsort(np.abs(y_true - y_pred))[::-1]

    rows = []
    sample_count = len(y_true)
    for removed_count in range(0, min(max_remove, sample_count - 2) + 1):
        keep = np.ones(sample_count, dtype=bool)
        if removed_count > 0:
            keep[removal_order[:removed_count]] = False

        rows.append(
            {
                "removed_top_k_residuals": removed_count,
                "remaining_n": int(keep.sum()),
                "R2": float(r2_score(y_true[keep], y_pred[keep])),
                "RMSE": float(
                    math.sqrt(mean_squared_error(y_true[keep], y_pred[keep]))
                ),
                "MAE": float(mean_absolute_error(y_true[keep], y_pred[keep])),
                "MedAE": float(np.median(np.abs(y_true[keep] - y_pred[keep]))),
            }
        )

    return pd.DataFrame(rows)


def relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


@dataclass
class NetworkShape:
    """GA-BP 网络形状；输入维度由数据自动取得，此处将得到 7。"""

    n_features: int
    hidden: int = 4

    @property
    def sizes(self) -> List[Tuple[int, int]]:
        return [
            (self.n_features, self.hidden),
            (self.hidden, 1),
        ]

    @property
    def n_params(self) -> int:
        return sum(input_size * output_size + output_size for input_size, output_size in self.sizes)


def unpack_weights(
    vector: np.ndarray,
    shape: NetworkShape,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """将 GA 的一维染色体还原成各层权重和偏置。"""
    coefs = []
    intercepts = []
    position = 0

    for input_size, output_size in shape.sizes:
        weight_count = input_size * output_size
        weight = vector[position:position + weight_count].reshape(
            input_size,
            output_size,
        )
        position += weight_count
        bias = vector[position:position + output_size]
        position += output_size
        coefs.append(weight)
        intercepts.append(bias)

    return coefs, intercepts


def forward_pass(X: np.ndarray, vector: np.ndarray, shape: NetworkShape) -> np.ndarray:
    """按照 7-4-1 单隐含层结构执行前向传播。"""
    coefs, intercepts = unpack_weights(vector, shape)
    hidden = relu(X @ coefs[0] + intercepts[0])
    output = hidden @ coefs[1] + intercepts[1]
    return output.ravel()


class GABPRegressor(BaseEstimator, RegressorMixin):
    """GA 优化初始权重，然后由 MLPRegressor 进行 BP 训练。"""

    def __init__(
        self,
        hidden_layer_sizes=(4,),
        tau=DEFAULT_TAU,
        population_size=80,
        generations=150,
        crossover_prob=0.8,
        mutation_prob=0.15,
        gene_mutation_prob=0.1,
        mutation_sigma=0.05,
        bp_max_iter=1000,
        random_state=42,
        verbose=False,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.tau = tau
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.gene_mutation_prob = gene_mutation_prob
        self.mutation_sigma = mutation_sigma
        self.bp_max_iter = bp_max_iter
        self.random_state = random_state
        self.verbose = verbose

    def _loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        error = y_pred - y_true
        return float(
            np.mean(
                self.tau * np.maximum(error, 0.0)
                + (1.0 - self.tau) * np.maximum(-error, 0.0)
            )
        )

    @staticmethod
    def _tournament_select(
        population: np.ndarray,
        losses: np.ndarray,
        rng: np.random.Generator,
        k: int = 3,
    ) -> np.ndarray:
        indexes = rng.choice(len(population), size=k, replace=False)
        best_index = indexes[np.argmin(losses[indexes])]
        return population[best_index].copy()

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        shape = NetworkShape(
            n_features=X.shape[1],
            hidden=int(self.hidden_layer_sizes[0]),
        )
        parameter_count = shape.n_params

        # 保留源文件的 Xavier-like 初始化尺度。
        population = rng.normal(
            0,
            0.2,
            size=(self.population_size, parameter_count),
        )

        def evaluate_population(population_array: np.ndarray) -> np.ndarray:
            losses = np.empty(population_array.shape[0])
            for index in range(population_array.shape[0]):
                prediction = forward_pass(X, population_array[index], shape)
                losses[index] = self._loss(y, prediction)
            return losses

        losses = evaluate_population(population)
        best_index = int(np.argmin(losses))
        best = population[best_index].copy()
        best_loss = float(losses[best_index])

        for _ in range(self.generations):
            new_population = [best.copy()]

            while len(new_population) < self.population_size:
                parent1 = self._tournament_select(population, losses, rng)
                parent2 = self._tournament_select(population, losses, rng)

                if rng.random() < self.crossover_prob:
                    point1, point2 = sorted(
                        rng.choice(parameter_count, size=2, replace=False)
                    )
                    child1, child2 = parent1.copy(), parent2.copy()
                    child1[point1:point2], child2[point1:point2] = (
                        parent2[point1:point2],
                        parent1[point1:point2],
                    )
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                for child in (child1, child2):
                    if rng.random() < self.mutation_prob:
                        mutation_mask = rng.random(parameter_count) < self.gene_mutation_prob
                        child[mutation_mask] += rng.normal(
                            0,
                            self.mutation_sigma,
                            size=mutation_mask.sum(),
                        )
                    new_population.append(child)
                    if len(new_population) >= self.population_size:
                        break

            population = np.asarray(new_population)
            losses = evaluate_population(population)
            generation_best_index = int(np.argmin(losses))

            if losses[generation_best_index] < best_loss:
                best_loss = float(losses[generation_best_index])
                best = population[generation_best_index].copy()

        # 保留源文件逻辑：先建立 MLP，再用 GA 最优染色体替换初始权重，最后 BP 训练。
        self.mlp_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=1,
            random_state=self.random_state,
            warm_start=True,
        )
        self.mlp_.fit(X, y)

        coefs, intercepts = unpack_weights(best, shape)
        self.mlp_.coefs_ = [coef.copy() for coef in coefs]
        self.mlp_.intercepts_ = [intercept.copy() for intercept in intercepts]
        self.mlp_.max_iter = self.bp_max_iter
        self.mlp_.fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.mlp_.predict(np.asarray(X, dtype=float))


def make_models(
    random_state: int = 42,
    tau: float = GABP_TAU,
) -> Dict[str, BaseEstimator]:
    """保留源文件中的 BP 和非对称 GA-BP。"""
    models: Dict[str, BaseEstimator] = {}

    models["BP_NN"] = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                TransformedTargetRegressor(
                    regressor=MLPRegressor(
                        hidden_layer_sizes=(4,),
                        activation="relu",
                        solver="adam",
                        learning_rate_init=0.001,
                        max_iter=1000,
                        random_state=random_state,
                    ),
                    transformer=StandardScaler(),
                ),
            ),
        ]
    )

    models[f"GA_BP_tau{tau:g}"] = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                TransformedTargetRegressor(
                    regressor=GABPRegressor(
                        hidden_layer_sizes=(4,),
                        tau=tau,
                        population_size=80,
                        generations=150,
                        crossover_prob=0.8,
                        mutation_prob=0.15,
                        gene_mutation_prob=0.1,
                        mutation_sigma=0.05,
                        bp_max_iter=1000,
                        random_state=random_state,
                    ),
                    transformer=StandardScaler(),
                ),
            ),
        ]
    )

    return models


def run_loo44(
    X: pd.DataFrame,
    y: pd.Series,
    project_id: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """保留全部 44 个样本，执行 44 折 LOOCV。"""
    X44 = X.reset_index(drop=True)
    y44 = y.reset_index(drop=True)
    id44 = project_id.reset_index(drop=True)

    print(f"[信息] LOOCV样本数：{len(y44)}，输入特征数：{X44.shape[1]}")

    models = make_models(random_state=RANDOM_STATE)
    prediction_table = pd.DataFrame(
        {
            "project_id": id44.values,
            "original_row_number": np.arange(1, len(y44) + 1),
            "y_true": y44.values.astype(float),
        }
    )

    loo = LeaveOneOut()
    for model_name, model in models.items():
        print(f"\n[LOO-CV] 正在运行模型：{model_name}")
        predictions = np.zeros(len(y44), dtype=float)
        constrained_count = 0

        for fold, (train_index, test_index) in enumerate(loo.split(X44), start=1):
            X_train = X44.iloc[train_index]
            X_test = X44.iloc[test_index]
            y_train = y44.iloc[train_index].values.astype(float)

            estimator = clone(model)
            estimator.fit(X_train, y_train)
            raw_prediction = np.asarray(estimator.predict(X_test), dtype=float)
            constrained_prediction = constrain_positive_prediction(raw_prediction)
            constrained_count += int(np.count_nonzero(raw_prediction <= 0.0))
            predictions[test_index[0]] = float(np.ravel(constrained_prediction)[0])

            print(
                f"  fold {fold:02d}/{len(y44)}: "
                f"test_id={id44.iloc[test_index[0]]}, "
                f"y={y44.iloc[test_index[0]]:.4f}, "
                f"pred={predictions[test_index[0]]:.4f}"
            )

        if np.any(predictions <= 0.0):
            raise RuntimeError(f"{model_name} 仍包含非正预测值")
        print(
            f"[正值约束] {model_name}: "
            f"截断 {constrained_count} 个非正预测值，"
            f"下限={POSITIVE_PREDICTION_FLOOR:g}"
        )

        true_values = prediction_table["y_true"].values
        prediction_table[f"{model_name}_pred"] = predictions
        prediction_table[f"{model_name}_error"] = predictions - true_values
        prediction_table[f"{model_name}_abs_error"] = np.abs(
            prediction_table[f"{model_name}_error"]
        )
        prediction_table[f"{model_name}_APE_%"] = (
            prediction_table[f"{model_name}_abs_error"]
            / np.maximum(np.abs(true_values), 1e-12)
            * 100
        )
        prediction_table[f"{model_name}_underestimated"] = predictions < true_values

    metric_rows = []
    progressive_rows = []
    true_values = prediction_table["y_true"].values

    for model_name in models:
        predictions = prediction_table[f"{model_name}_pred"].values
        model_metrics = compute_metrics(true_values, predictions)
        model_metrics["Model"] = model_name
        metric_rows.append(model_metrics)

        progressive_metrics = progressive_r2_removal(
            true_values,
            predictions,
            max_remove=7,
        )
        progressive_metrics.insert(0, "Model", model_name)
        progressive_rows.append(progressive_metrics)

    metrics_table = pd.DataFrame(metric_rows)[
        ["Model", "RMSE", "MAE", "MedAE", "R2", "EVS", "MAPE_%", "UR_%"]
    ]
    progressive_table = pd.concat(progressive_rows, ignore_index=True)

    return prediction_table, metrics_table, progressive_table


def main() -> None:
    X, y, project_id = load_data(DATA_PATH)
    print(f"Dataset: {len(y)} samples, {X.shape[1]} features")
    print(f"Features: {FEATURE_COLUMNS}")

    prediction_table, metrics_table, progressive_table = run_loo44(
        X,
        y,
        project_id,
    )

    settings_table = pd.DataFrame(
        {
            "key": [
                "input",
                "target_col",
                "n_original",
                "n_for_loocv",
                "feature_count",
                "feature_cols",
                "models",
                "random_state",
                "gabp_tau",
                "prediction_constraint",
                "positive_prediction_floor",
            ],
            "value": [
                str(DATA_PATH),
                TARGET_COLUMN,
                len(y),
                len(prediction_table),
                len(FEATURE_COLUMNS),
                ", ".join(FEATURE_COLUMNS),
                ", ".join(metrics_table["Model"].tolist()),
                RANDOM_STATE,
                GABP_TAU,
                "max(prediction, positive_prediction_floor)",
                POSITIVE_PREDICTION_FLOOR,
            ],
        }
    )

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        prediction_table.to_excel(writer, index=False, sheet_name="LOO44_predictions")
        metrics_table.to_excel(writer, index=False, sheet_name="Metrics")
        progressive_table.to_excel(
            writer,
            index=False,
            sheet_name="R2_progressive_removal",
        )
        pd.DataFrame({"feature_cols": FEATURE_COLUMNS}).to_excel(
            writer,
            index=False,
            sheet_name="Feature_columns",
        )
        settings_table.to_excel(writer, index=False, sheet_name="Settings")

    print(f"\n完成。输出文件：{OUTPUT_PATH}")
    print("\nMetrics:")
    print(metrics_table.to_string(index=False))


if __name__ == "__main__":
    main()
