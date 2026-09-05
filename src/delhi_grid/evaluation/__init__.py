"""Chronological backtesting and forecasting metrics."""

from delhi_grid.evaluation.backtest import (
    DEFAULT_METRICS_PATH,
    DEFAULT_PREDICTIONS_PATH,
    run_development_baselines,
    write_baseline_results,
)
from delhi_grid.evaluation.failure_analysis import (
    DEFAULT_SLICE_METRICS_PATH,
    DEFAULT_WORST_ERRORS_PATH,
    FailureAnalysisResult,
    build_development_ramps,
    build_error_analysis_frame,
    run_failure_analysis,
    summarize_comparison,
    write_failure_analysis,
)
from delhi_grid.evaluation.metrics import RegressionMetrics, compute_metrics
from delhi_grid.evaluation.regime_analysis import (
    DEFAULT_REGIME_METRICS_PATH,
    RegimeAnalysisResult,
    attach_training_regime,
    build_fold_regime_summary,
    build_regime_error_metrics,
    run_regime_analysis,
    write_regime_analysis,
)
from delhi_grid.evaluation.splits import (
    BacktestConfig,
    MonthlyFold,
    load_backtest_config,
    make_development_folds,
    select_fold_frames,
)
from delhi_grid.evaluation.xgb_backtest import (
    DEFAULT_XGBOOST_METRICS_PATH,
    DEFAULT_XGBOOST_PREDICTIONS_PATH,
    XGBoostDevelopmentResult,
    run_development_xgboost,
    write_xgboost_results,
)

__all__ = [
    "DEFAULT_METRICS_PATH",
    "DEFAULT_PREDICTIONS_PATH",
    "DEFAULT_REGIME_METRICS_PATH",
    "DEFAULT_SLICE_METRICS_PATH",
    "DEFAULT_WORST_ERRORS_PATH",
    "DEFAULT_XGBOOST_METRICS_PATH",
    "DEFAULT_XGBOOST_PREDICTIONS_PATH",
    "BacktestConfig",
    "FailureAnalysisResult",
    "MonthlyFold",
    "RegressionMetrics",
    "RegimeAnalysisResult",
    "XGBoostDevelopmentResult",
    "compute_metrics",
    "attach_training_regime",
    "build_development_ramps",
    "build_error_analysis_frame",
    "build_fold_regime_summary",
    "build_regime_error_metrics",
    "load_backtest_config",
    "make_development_folds",
    "run_development_baselines",
    "run_development_xgboost",
    "run_failure_analysis",
    "run_regime_analysis",
    "select_fold_frames",
    "summarize_comparison",
    "write_baseline_results",
    "write_failure_analysis",
    "write_regime_analysis",
    "write_xgboost_results",
]
