from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk
from datetime import datetime
import pandas as pd

from nge_trader.config.env_utils import read_env, write_env
from nge_trader.services.app_service import AppService
from nge_trader.services.backtester import SimpleBuyHoldBacktester, SignalBacktester, ExecutedSignalBacktester
from nge_trader.services.metrics import compute_sharpe, compute_sortino, compute_max_drawdown, compute_win_rate, compute_calmar
from nge_trader.domain.strategies.moving_average import MovingAverageCrossStrategy
from nge_trader.domain.strategies.rsi import RSIStrategy
from nge_trader.domain.strategies.macd import MACDStrategy
from nge_trader.domain.strategies.bollinger import BollingerBandsStrategy
from nge_trader.domain.strategies.stochastic import StochasticOscillatorStrategy
from nge_trader.domain.strategies.donchian import DonchianBreakoutStrategy
from nge_trader.domain.strategies.momentum import MomentumStrategy
from nge_trader.domain.strategies.cci import CCIStrategy
from nge_trader.domain.strategies.adx import ADXStrategy
from nge_trader.domain.strategies.keltner import KeltnerChannelsStrategy
from nge_trader.domain.strategies.williamsr import WilliamsRStrategy
from nge_trader.domain.strategies.psar import ParabolicSARStrategy
from nge_trader.domain.strategies.ichimoku import IchimokuStrategy
from nge_trader.domain.strategies.supertrend import SupertrendStrategy
from nge_trader.domain.strategies.obv import OBVStrategy
from nge_trader.domain.strategies.ema_crossover import EMACrossoverStrategy
from nge_trader.domain.strategies.pivot import PivotReversalStrategy
from nge_trader.domain.strategies.heikin_ashi import HeikinAshiTrendStrategy
from nge_trader.domain.strategies.trix import TRIXStrategy
from nge_trader.domain.strategies.roc import ROCStrategy
from nge_trader.domain.strategies.zscore import ZScoreReversionStrategy
from nge_trader.domain.strategies.cmf import ChaikinMoneyFlowStrategy
from nge_trader.domain.strategies.vwap import VWAPDeviationStrategy
from nge_trader.domain.strategies.pairs import PairsTradingStrategy
from nge_trader.services.meta_strategy import MetaStrategy, EnsembleConfig
from nge_trader.repository.db import Database
from nge_trader.services.strategy_store import StrategyStore, StrategyConfig
import psutil
from nge_trader.services.analytics import (
    export_equity_csv,
    export_trades_csv,
    export_report_html,
    make_report_dir,
    export_report_zip,
    export_tearsheet,
)
from nge_trader.services.data_agg import resample_ohlc
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from nge_trader.ai.policy import AgentPolicy, PolicyConfig


class DesktopApp:
    """Aplicación de escritorio simple para configuración y backtests."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("NGEtrader - Escritorio")
        self.root.geometry("960x680")

        # Campos de configuración
        self.var_data_provider = tk.StringVar()
        self.var_broker = tk.StringVar()
        self.var_profile = tk.StringVar()
        self.var_alpha = tk.StringVar()
        self.var_alpaca_key = tk.StringVar()
        self.var_alpaca_secret = tk.StringVar()
        self.var_symbol = tk.StringVar(value="AAPL")
        self.var_benchmark = tk.StringVar(value="SPY")

        self._build_ui()
        self._load_env()
        self.db = Database()
        self._start_refresh()

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 6}
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Tabs
        tab_dashboard = ttk.Frame(notebook)
        tab_portfolio = ttk.Frame(notebook)
        tab_analytics = ttk.Frame(notebook)
        tab_journal = ttk.Frame(notebook)
        tab_backtest = ttk.Frame(notebook)
        tab_strategies = ttk.Frame(notebook)
        tab_logs = ttk.Frame(notebook)
        tab_settings = ttk.Frame(notebook)
        tab_altdata = ttk.Frame(notebook)
        tab_allocation = ttk.Frame(notebook)
        tab_oms = ttk.Frame(notebook)

        notebook.add(tab_dashboard, text="Dashboard")
        notebook.add(tab_portfolio, text="Portafolio")
        notebook.add(tab_analytics, text="Analíticas")
        notebook.add(tab_journal, text="Journal")
        notebook.add(tab_backtest, text="Backtesting")
        notebook.add(tab_strategies, text="Estrategias")
        notebook.add(tab_logs, text="Logs")
        notebook.add(tab_settings, text="Configuración")
        notebook.add(tab_altdata, text="Datos Alternativos")
        notebook.add(tab_allocation, text="Asset Allocation")
        notebook.add(tab_oms, text="OMS")

        # Dashboard
        lbl_state = tk.Label(tab_dashboard, text="Estado del Bot: DETENIDO", font=("Segoe UI", 14, "bold"))
        lbl_state.pack(anchor="w", **pad)
        self.lbl_connectivity = ttk.Label(tab_dashboard, text="Conexiones: Alpaca: -- | Alpha Vantage: -- | Binance: --")
        self.lbl_connectivity.pack(anchor="w", padx=8)

        frm_metrics = ttk.LabelFrame(tab_dashboard, text="Resumen Financiero y Métricas")
        frm_metrics.pack(fill=tk.X, **pad)
        self.lbl_balance = ttk.Label(frm_metrics, text="Balance Total: --")
        self.lbl_balance.pack(anchor="w", padx=8, pady=2)
        self.lbl_pl_today = ttk.Label(frm_metrics, text="P&L Hoy: --")
        self.lbl_pl_today.pack(anchor="w", padx=8, pady=2)
        self.lbl_pl_open = ttk.Label(frm_metrics, text="P&L Abierto: --")
        self.lbl_pl_open.pack(anchor="w", padx=8, pady=2)
        ttk.Label(frm_metrics, text="Métricas (última curva guardada):").pack(anchor="w", padx=8, pady=2)
        self.lbl_metrics = ttk.Label(frm_metrics, text="Sharpe: -- | Sortino: -- | Max DD: -- | Win Rate: -- | Slip(avg bps): -- | Skew(ms): --")
        self.lbl_metrics.pack(anchor="w", padx=8, pady=2)
        # Kill switch controls
        ks_frame = ttk.Frame(frm_metrics)
        ks_frame.pack(anchor="e", padx=8, pady=2)
        ttk.Button(ks_frame, text="Armar Kill-Switch", command=self._arm_kill_switch).pack(side=tk.LEFT, padx=4)
        ttk.Button(ks_frame, text="Desarmar Kill-Switch", command=self._disarm_kill_switch).pack(side=tk.LEFT, padx=4)
        self.lbl_connectivity = ttk.Label(frm_metrics, text="Conexiones: Alpaca: -- | Alpha Vantage: -- | Binance: --")
        self.lbl_connectivity.pack(anchor="w", padx=8, pady=2)
        self.lbl_sentiment = ttk.Label(frm_metrics, text="Sentimiento agregado: --")
        self.lbl_sentiment.pack(anchor="w", padx=8, pady=2)

        # Barras rápidas de salud
        health_bar = ttk.Frame(frm_metrics)
        health_bar.pack(fill=tk.X, padx=8, pady=2)
        self.lbl_ws_health = ttk.Label(health_bar, text="WS: -- | reconnects: -- | hb(ms): --")
        self.lbl_ws_health.pack(side=tk.LEFT)
        self.lbl_ntp_skew = ttk.Label(health_bar, text="NTP skew(ms): --")
        self.lbl_ntp_skew.pack(side=tk.LEFT, padx=12)

        # Semáforo Edge-vs-Coste
        frm_semaforo = ttk.LabelFrame(tab_dashboard, text="Semáforo Edge-vs-Coste (global)")
        frm_semaforo.pack(fill=tk.X, padx=8, pady=6)
        self.lbl_semaforo = ttk.Label(frm_semaforo, text="--", width=12, anchor="center")
        self.lbl_semaforo.pack(side=tk.LEFT, padx=8, pady=6)
        self.lbl_semaforo_causa = ttk.Label(frm_semaforo, text="Causa: --")
        self.lbl_semaforo_causa.pack(side=tk.LEFT, padx=8)

        # Riesgo y Budgets
        frm_risk = ttk.LabelFrame(tab_dashboard, text="Riesgo y Budgets (intradiario)")
        frm_risk.pack(fill=tk.X, padx=8, pady=6)
        self.lbl_risk = ttk.Label(frm_risk, text="DD actual: -- | MaxDD: -- | R/Trade: -- | R usados: -- | R pendientes: --")
        self.lbl_risk.pack(anchor="w", padx=8, pady=4)

        # TCA en vivo
        frm_tca = ttk.LabelFrame(tab_dashboard, text="TCA en vivo")
        frm_tca.pack(fill=tk.X, padx=8, pady=6)
        self.lbl_tca = ttk.Label(frm_tca, text="slip(1d): -- | slip(7d): -- | fill_ratio: -- | maker_ratio: -- | p95(ms): -- | SLO: --")
        self.lbl_tca.pack(anchor="w", padx=8, pady=4)

        # Estado del modelo
        frm_model = ttk.LabelFrame(tab_dashboard, text="Estado del modelo")
        frm_model.pack(fill=tk.X, padx=8, pady=6)
        self.lbl_model = ttk.Label(frm_model, text="version: -- | canary%: -- | PSI: -- | acciones: buy/sell/none --/--/--")
        self.lbl_model.pack(anchor="w", padx=8, pady=4)

        ttk.Label(tab_dashboard, text="Curva de Capital").pack(anchor="w", **pad)
        self.canvas_equity = tk.Canvas(tab_dashboard, height=180, bg="#0b1020")
        self.canvas_equity.pack(fill=tk.X, padx=8)

        # Últimas operaciones
        last_frame = ttk.LabelFrame(tab_dashboard, text="Últimas 5 Operaciones")
        last_frame.pack(fill=tk.BOTH, expand=False, padx=8, pady=8)
        self.last_ops = ttk.Treeview(last_frame, columns=("symbol","side","qty","price","time"), show="headings", height=5)
        for key, title in [("symbol","Símbolo"),("side","Lado"),("qty","Cantidad"),("price","Precio"),("time","Tiempo")]:
            self.last_ops.heading(key, text=title)
            self.last_ops.column(key, width=110, anchor="center")
        self.last_ops.pack(fill=tk.X, expand=False)

        # Portafolio
        columns = ("symbol", "quantity", "entry", "price", "value", "upl", "rpl", "stop", "take", "risk_R", "r_total", "exposure", "exposure_pct", "strategy", "model", "traffic", "semaforo")
        self.tree_positions = ttk.Treeview(tab_portfolio, columns=columns, show="headings", height=12)
        headers = [
            ("symbol", "Símbolo"),
            ("quantity", "Cantidad"),
            ("entry", "Precio medio"),
            ("price", "Actual"),
            ("value", "Valor"),
            ("upl", "P&L Unreal"),
            ("rpl", "P&L Real"),
            ("stop", "Stop"),
            ("take", "TP"),
            ("risk_R", "R en riesgo"),
            ("r_total", "R ±"),
            ("exposure", "Exposición €"),
            ("exposure_pct", "Exposición %"),
            ("strategy", "Estrategia"),
            ("model", "Modelo"),
            ("traffic", "%Canary"),
            ("semaforo", "Semáforo"),
        ]
        for key, title in headers:
            self.tree_positions.heading(key, text=title)
            self.tree_positions.column(key, width=110, anchor="center")
        self.tree_positions.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        btns = tk.Frame(tab_portfolio)
        btns.pack(anchor="e", padx=8, pady=4)
        ttk.Button(btns, text="Close", command=self._close_selected_position).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Panic (Close All)", command=self._close_all_positions).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Iniciar Live (demo)", command=self._start_live_demo).pack(side=tk.LEFT, padx=4)
        ttk.Label(btns, text="(usa riesgo 1% y stop=ATR)").pack(side=tk.LEFT, padx=4)
        # Acciones inline SL/TP
        ttk.Label(btns, text="SL").pack(side=tk.LEFT)
        self.var_new_sl = tk.StringVar()
        ttk.Entry(btns, textvariable=self.var_new_sl, width=10).pack(side=tk.LEFT, padx=4)
        ttk.Label(btns, text="TP").pack(side=tk.LEFT)
        self.var_new_tp = tk.StringVar()
        ttk.Entry(btns, textvariable=self.var_new_tp, width=10).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Set SL", command=self._set_selected_sl).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Set TP", command=self._set_selected_tp).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Break-even (+1R)", command=self._move_sl_to_breakeven).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Trailing", command=self._move_sl_trailing).pack(side=tk.LEFT, padx=4)

        # Órdenes recientes
        orders_frame = ttk.LabelFrame(tab_portfolio, text="Órdenes recientes")
        orders_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.orders_tree = ttk.Treeview(orders_frame, columns=("ts","symbol","side","qty","price","status","order_id","parent_id","cid","lat_ms","slip_bps","liq"), show="headings", height=8)
        for key, title in [("ts","Hora"),("symbol","Símbolo"),("side","Lado"),("qty","Cantidad"),("price","Precio"),("status","Estado"),("order_id","Orden"),("parent_id","ParentID"),("cid","CorrID"),("lat_ms","Lat(ms)"),("slip_bps","Slip(bps)"),("liq","Liq")]:
            self.orders_tree.heading(key, text=title)
            self.orders_tree.column(key, width=110, anchor="center")
        self.orders_tree.pack(fill=tk.BOTH, expand=True)
        order_btns = tk.Frame(orders_frame)
        order_btns.pack(anchor="e", padx=8, pady=4)
        from nge_trader.config.settings import Settings as _S
        _role = _S().ui_role
        can_operate = (_role.lower() == "operator")
        btn_cancel = ttk.Button(order_btns, text="Cancelar orden", command=self._cancel_selected_order)
        btn_cancel.pack(side=tk.LEFT, padx=4)
        ttk.Label(order_btns, text="Símbolo").pack(side=tk.LEFT)
        self.var_cancel_sym = tk.StringVar()
        ttk.Entry(order_btns, textvariable=self.var_cancel_sym, width=10).pack(side=tk.LEFT, padx=4)
        ttk.Button(order_btns, text="Cancel-All Símbolo", command=self._cancel_all_by_symbol_inline).pack(side=tk.LEFT, padx=4)
        ttk.Label(order_btns, text="Nuevo SL").pack(side=tk.LEFT)
        self.var_new_sl = tk.StringVar()
        ttk.Entry(order_btns, textvariable=self.var_new_sl, width=10).pack(side=tk.LEFT, padx=4)
        btn_mod = ttk.Button(order_btns, text="Modificar SL", command=self._replace_selected_order_sl)
        btn_mod.pack(side=tk.LEFT, padx=4)
        btn_all = ttk.Button(order_btns, text="Cancelar TODAS", command=self._cancel_all_orders)
        btn_all.pack(side=tk.LEFT, padx=8)
        if not can_operate:
            btn_cancel.state(["disabled"])
            btn_mod.state(["disabled"])
            btn_all.state(["disabled"])

        fills_frame = ttk.LabelFrame(tab_portfolio, text="Ejecuciones recientes (Fills)")
        fills_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.fills_tree = ttk.Treeview(fills_frame, columns=("ts","symbol","side","qty","price","fees","slip_bps","liq","order_id"), show="headings", height=6)
        for key, title in [("ts","Hora"),("symbol","Símbolo"),("side","Lado"),("qty","Cantidad"),("price","Precio"),("fees","Fees"),("slip_bps","Slip(bps)"),("liq","Liq"),("order_id","Orden")]:
            self.fills_tree.heading(key, text=title)
            self.fills_tree.column(key, width=110, anchor="center")
        self.fills_tree.pack(fill=tk.BOTH, expand=True)
        self.lbl_fills_totals = ttk.Label(fills_frame, text="Totales (hoy): PnL Real: -- | Fees: -- | Turnover: --")
        self.lbl_fills_totals.pack(anchor="e", padx=8, pady=4)

        # Analíticas
        analytics_ctrl = ttk.LabelFrame(tab_analytics, text="Controles de Analítica")
        analytics_ctrl.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(analytics_ctrl, text="Benchmark").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(analytics_ctrl, textvariable=self.var_benchmark, width=12).grid(row=0, column=1, sticky="w", **pad)
        ttk.Button(analytics_ctrl, text="Actualizar", command=self._refresh_analytics).grid(row=0, column=2, sticky="w", **pad)
        analytics_ctrl.columnconfigure(3, weight=1)

        self.analytics_fig1_frame = ttk.LabelFrame(tab_analytics, text="Equity vs Benchmark")
        self.analytics_fig1_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.analytics_fig2_frame = ttk.LabelFrame(tab_analytics, text="Drawdown")
        self.analytics_fig2_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.analytics_fig3_frame = ttk.LabelFrame(tab_analytics, text="Distribución de Retornos")
        self.analytics_fig3_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        # Panel de métricas live
        live_metrics = ttk.LabelFrame(tab_analytics, text="Métricas Live (rolling)")
        live_metrics.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.live_metrics_fig = ttk.Frame(live_metrics)
        self.live_metrics_fig.pack(fill=tk.BOTH, expand=True)
        # Reporte diario rápido
        rep_btns = ttk.Frame(live_metrics)
        rep_btns.pack(anchor="e", padx=8, pady=4)
        ttk.Button(rep_btns, text="Generar Reporte ZIP", command=self._export_daily_report).pack(side=tk.LEFT, padx=4)
        # Correlaciones
        corr_ctrl = ttk.LabelFrame(tab_analytics, text="Correlaciones")
        corr_ctrl.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(corr_ctrl, text="Símbolos (coma)").grid(row=0, column=0, sticky="w", **pad)
        self.corr_symbols = tk.StringVar(value="AAPL,MSFT,GOOG,AMZN")
        ttk.Entry(corr_ctrl, textvariable=self.corr_symbols).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(corr_ctrl, text="Calcular", command=self._refresh_correlations).grid(row=0, column=2, sticky="w", **pad)
        corr_ctrl.columnconfigure(1, weight=1)
        self.analytics_corr_frame = ttk.LabelFrame(tab_analytics, text="Matriz de Correlación de Retornos")
        self.analytics_corr_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._analytics_canvases: dict[str, FigureCanvasTkAgg] = {}

        # Journal
        journal_cols = ("symbol", "in_time", "in_price", "out_time", "out_price", "qty", "fees", "realized")
        self.journal_tree = ttk.Treeview(tab_journal, columns=journal_cols, show="headings", height=14)
        journal_headers = [
            ("symbol", "Símbolo"),
            ("in_time", "Entrada"),
            ("in_price", "Precio Entrada"),
            ("out_time", "Salida"),
            ("out_price", "Precio Salida"),
            ("qty", "Cantidad"),
            ("fees", "Comisiones"),
            ("realized", "P&L Realizado"),
        ]
        for key, title in journal_headers:
            self.journal_tree.heading(key, text=title)
            self.journal_tree.column(key, width=110, anchor="center")
        self.journal_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Backtesting
        backtest_frame = ttk.LabelFrame(tab_backtest, text="Panel de Backtesting")
        backtest_frame.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(backtest_frame, text="Estrategia").grid(row=0, column=0, sticky="w", **pad)
        self.strategy_var = tk.StringVar(value="buyhold")
        strategy_combo = ttk.Combobox(
            backtest_frame,
            textvariable=self.strategy_var,
            values=[
                "buyhold",
                "ma_cross",
                "rsi",
                "macd",
                "bollinger",
                "stochastic",
                "donchian",
                "momentum",
                "cci",
                "adx",
                "keltner",
                "williamsr",
                "psar",
                "ichimoku",
                "supertrend",
                "obv",
                "ema_x",
                "pivot",
                "heikin",
                "trix",
                "roc",
                "zscore",
                "cmf",
                "vwap_dev",
                "pairs",
                "meta",
                "agent",
            ],
            state="readonly",
        )
        strategy_combo.grid(row=0, column=1, sticky="ew", **pad)
        ttk.Label(backtest_frame, text="Símbolo").grid(row=0, column=2, sticky="w", **pad)
        ttk.Entry(backtest_frame, textvariable=self.var_symbol).grid(row=0, column=3, sticky="ew", **pad)
        ttk.Label(backtest_frame, text="Exec Algo").grid(row=0, column=4, sticky="w", **pad)
        self.exec_algo = tk.StringVar(value="")
        ttk.Combobox(backtest_frame, textvariable=self.exec_algo, values=["", "twap", "vwap", "pov"], width=8, state="readonly").grid(row=0, column=5, sticky="w", **pad)
        ttk.Label(backtest_frame, text="TF").grid(row=0, column=6, sticky="w", **pad)
        self.bt_tf = tk.StringVar(value="D")
        ttk.Combobox(backtest_frame, textvariable=self.bt_tf, values=["D","W","M"], width=6, state="readonly").grid(row=0, column=7, sticky="w", **pad)
        ttk.Label(backtest_frame, text="Rango Fechas").grid(row=1, column=0, sticky="w", **pad)
        self.date_start = tk.StringVar()
        self.date_end = tk.StringVar()
        ttk.Entry(backtest_frame, textvariable=self.date_start).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Entry(backtest_frame, textvariable=self.date_end).grid(row=1, column=2, sticky="ew", **pad)
        btn_frame = tk.Frame(backtest_frame)
        btn_frame.grid(row=1, column=7, sticky="e", **pad)
        ttk.Button(btn_frame, text="Ejecutar", command=self._run_backtest).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Exportar Reporte", command=self._export_report).pack(side=tk.LEFT, padx=4)
        backtest_frame.columnconfigure(1, weight=1)
        backtest_frame.columnconfigure(3, weight=1)
        backtest_frame.columnconfigure(7, weight=1)

        self.txt = tk.Text(tab_backtest, height=18)
        self.txt.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        # Asset Allocation
        alloc = ttk.LabelFrame(tab_allocation, text="Asignación de Carteras")
        alloc.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(alloc, text="Símbolos (coma)").grid(row=0, column=0, sticky="w", **pad)
        self.alloc_symbols = tk.StringVar(value="AAPL,MSFT,GOOG,AMZN")
        ttk.Entry(alloc, textvariable=self.alloc_symbols).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Label(alloc, text="Ventana retornos").grid(row=0, column=2, sticky="w", **pad)
        self.alloc_window = tk.StringVar(value="60")
        ttk.Entry(alloc, textvariable=self.alloc_window, width=8).grid(row=0, column=3, sticky="w", **pad)
        ttk.Button(alloc, text="Risk Parity", command=self._alloc_risk_parity).grid(row=0, column=4, sticky="w", **pad)
        ttk.Button(alloc, text="Min Var", command=self._alloc_min_var).grid(row=0, column=5, sticky="w", **pad)
        alloc.columnconfigure(1, weight=1)
        self.alloc_out = tk.Text(tab_allocation, height=16)
        self.alloc_out.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # OMS Tab
        oms = ttk.LabelFrame(tab_oms, text="Envío de Órdenes (OCO/OSO)")
        oms.pack(fill=tk.X, padx=8, pady=8)
        self.oms_symbol = tk.StringVar(value="AAPL")
        ttk.Label(oms, text="Símbolo").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(oms, textvariable=self.oms_symbol, width=10).grid(row=0, column=1, sticky="w", **pad)
        self.oms_side = tk.StringVar(value="buy")
        ttk.Combobox(oms, textvariable=self.oms_side, values=["buy","sell"], width=8, state="readonly").grid(row=0, column=2, sticky="w", **pad)
        self.oms_qty = tk.StringVar(value="10")
        ttk.Label(oms, text="Qty").grid(row=0, column=3, sticky="w", **pad)
        ttk.Entry(oms, textvariable=self.oms_qty, width=10).grid(row=0, column=4, sticky="w", **pad)
        self.oms_tp = tk.StringVar(value="")
        self.oms_sl = tk.StringVar(value="")
        ttk.Label(oms, text="TP").grid(row=0, column=5, sticky="w", **pad)
        ttk.Entry(oms, textvariable=self.oms_tp, width=10).grid(row=0, column=6, sticky="w", **pad)
        ttk.Label(oms, text="SL").grid(row=0, column=7, sticky="w", **pad)
        ttk.Entry(oms, textvariable=self.oms_sl, width=10).grid(row=0, column=8, sticky="w", **pad)
        ttk.Label(oms, text="TIF").grid(row=0, column=9, sticky="w", **pad)
        self.oms_tif = tk.StringVar(value="day")
        ttk.Combobox(oms, textvariable=self.oms_tif, values=["day","ioc","fok","gtc"], width=8, state="readonly").grid(row=0, column=10, sticky="w", **pad)
        ttk.Button(oms, text="Enviar", command=self._oms_send).grid(row=0, column=11, sticky="w", **pad)
        oms.columnconfigure(11, weight=1)
        # Acciones masivas
        mass = ttk.LabelFrame(tab_oms, text="Acciones masivas")
        mass.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(mass, text="Cancelar TODAS", command=self._cancel_all_orders).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Label(mass, text="Nuevo SL").pack(side=tk.LEFT)
        self.oms_new_sl_mass = tk.StringVar()
        ttk.Entry(mass, textvariable=self.oms_new_sl_mass, width=10).pack(side=tk.LEFT, padx=4)
        ttk.Button(mass, text="Reemplazar SL (masivo)", command=self._mass_replace_sl).pack(side=tk.LEFT, padx=4)
        self.oms_out = tk.Text(tab_oms, height=16)
        self.oms_out.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Sweep
        sweep_frame = ttk.LabelFrame(tab_backtest, text="Barrido de parámetros / Multi-activo")
        sweep_frame.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(sweep_frame, text="Símbolos (coma)").grid(row=0, column=0, sticky="w", **pad)
        self.sweep_symbols = tk.StringVar(value="AAPL,MSFT,GOOG")
        ttk.Entry(sweep_frame, textvariable=self.sweep_symbols).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Label(sweep_frame, text="Grid params (JSON)").grid(row=1, column=0, sticky="w", **pad)
        self.sweep_params = tk.Text(sweep_frame, height=4)
        self.sweep_params.insert("1.0", "{\n  \"fast_window\": [5,10],\n  \"slow_window\": [20,50]\n}")
        self.sweep_params.grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(sweep_frame, text="Ejecutar Sweep", command=self._run_sweep).grid(row=0, column=2, rowspan=2, sticky="nsew", **pad)
        sweep_frame.columnconfigure(1, weight=1)

        # Backtest de Cartera
        portf = ttk.LabelFrame(tab_backtest, text="Backtest de Cartera")
        portf.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(portf, text="Símbolos (coma)").grid(row=0, column=0, sticky="w", **pad)
        self.btpf_symbols = tk.StringVar(value="AAPL,MSFT,GOOG,AMZN")
        ttk.Entry(portf, textvariable=self.btpf_symbols).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Label(portf, text="Método").grid(row=0, column=2, sticky="w", **pad)
        self.btpf_method = tk.StringVar(value="risk_parity")
        ttk.Combobox(portf, textvariable=self.btpf_method, values=["risk_parity","min_variance","black_litterman"], width=16, state="readonly").grid(row=0, column=3, sticky="w", **pad)
        ttk.Label(portf, text="Ventana").grid(row=0, column=4, sticky="w", **pad)
        self.btpf_window = tk.StringVar(value="60")
        ttk.Entry(portf, textvariable=self.btpf_window, width=8).grid(row=0, column=5, sticky="w", **pad)
        ttk.Label(portf, text="Rebalance (días)").grid(row=0, column=6, sticky="w", **pad)
        self.btpf_reb = tk.StringVar(value="21")
        ttk.Entry(portf, textvariable=self.btpf_reb, width=8).grid(row=0, column=7, sticky="w", **pad)
        ttk.Button(portf, text="Ejecutar", command=self._run_portfolio_bt).grid(row=0, column=8, sticky="e", **pad)
        portf.columnconfigure(1, weight=1)

        # Estrategias
        strategies_frame = ttk.LabelFrame(tab_strategies, text="Gestión de Estrategias")
        strategies_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.store = StrategyStore()
        self.strat_list = tk.Listbox(strategies_frame, height=6)
        self.strat_list.pack(fill=tk.X, padx=8, pady=4)
        frm_edit = tk.Frame(strategies_frame)
        frm_edit.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(frm_edit, text="Strategy Key").grid(row=0, column=0, sticky="w")
        self.edit_key = tk.StringVar(value="ma_cross")
        ttk.Entry(frm_edit, textvariable=self.edit_key).grid(row=0, column=1, sticky="ew")
        ttk.Label(frm_edit, text="Params (JSON)").grid(row=1, column=0, sticky="w")
        self.edit_params = tk.Text(frm_edit, height=4)
        self.edit_params.grid(row=1, column=1, sticky="ew")
        frm_edit.columnconfigure(1, weight=1)
        frm_btn = tk.Frame(strategies_frame)
        frm_btn.pack(anchor="e", padx=8, pady=4)
        ttk.Button(frm_btn, text="Añadir/Actualizar", command=self._save_strategy).pack(side=tk.LEFT, padx=4)
        ttk.Button(frm_btn, text="Eliminar", command=self._delete_strategy).pack(side=tk.LEFT, padx=4)
        self._reload_strategies()

        assign_frame = ttk.LabelFrame(strategies_frame, text="Asignación de estrategias por símbolo")
        assign_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.assign_list = tk.Listbox(assign_frame, height=6)
        self.assign_list.pack(fill=tk.X, padx=8, pady=4)
        frm_assign = tk.Frame(assign_frame)
        frm_assign.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(frm_assign, text="Símbolo").grid(row=0, column=0, sticky="w")
        self.assign_symbol = tk.StringVar(value="AAPL")
        ttk.Entry(frm_assign, textvariable=self.assign_symbol).grid(row=0, column=1, sticky="ew")
        ttk.Label(frm_assign, text="Strategy Key").grid(row=0, column=2, sticky="w")
        self.assign_key = tk.StringVar(value="ma_cross")
        ttk.Entry(frm_assign, textvariable=self.assign_key).grid(row=0, column=3, sticky="ew")
        frm_assign.columnconfigure(1, weight=1)
        frm_assign.columnconfigure(3, weight=1)
        frm_assign_btns = tk.Frame(assign_frame)
        frm_assign_btns.pack(anchor="e", padx=8, pady=4)
        ttk.Button(frm_assign_btns, text="Asignar/Actualizar", command=self._save_assignment).pack(side=tk.LEFT, padx=4)
        ttk.Button(frm_assign_btns, text="Eliminar", command=self._delete_assignment).pack(side=tk.LEFT, padx=4)
        self._reload_assignments()

        # Entrenamiento del agente IA
        train_frame = ttk.LabelFrame(strategies_frame, text="Autoaprendizaje (Agente IA)")
        train_frame.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(train_frame, text="Símbolo").grid(row=0, column=0, sticky="w")
        self.train_symbol = tk.StringVar(value="AAPL")
        ttk.Entry(train_frame, textvariable=self.train_symbol, width=16).grid(row=0, column=1, sticky="w")
        ttk.Button(train_frame, text="Entrenar", command=self._train_agent_ui).grid(row=0, column=2, sticky="w", padx=8)
        ttk.Button(train_frame, text="Walk-Forward", command=self._wf_agent_ui).grid(row=0, column=3, sticky="w")

        # Logs
        logs_frame = ttk.LabelFrame(tab_logs, text="Logs en tiempo real")
        logs_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        filter_frame = tk.Frame(logs_frame)
        filter_frame.pack(anchor="w", padx=4, pady=4)
        ttk.Label(filter_frame, text="Nivel:").pack(side=tk.LEFT)
        self.log_level = tk.StringVar(value="ALL")
        ttk.Combobox(filter_frame, textvariable=self.log_level, values=["ALL", "INFO", "WARNING", "ERROR"], width=10, state="readonly").pack(side=tk.LEFT, padx=6)
        ttk.Button(filter_frame, text="Refrescar", command=self._refresh_logs).pack(side=tk.LEFT)
        self.txt_logs = tk.Text(logs_frame, height=18)
        self.txt_logs.pack(fill=tk.BOTH, expand=True)
        self.lbl_sys = ttk.Label(logs_frame, text="CPU: --% | Mem: --%")
        self.lbl_sys.pack(anchor="e", padx=8, pady=4)

        # Audits
        audits_frame = ttk.LabelFrame(tab_logs, text="Audit Trail")
        audits_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.audit_text = tk.Text(audits_frame, height=10)
        self.audit_text.pack(fill=tk.BOTH, expand=True)

        # Configuración
        cfg = ttk.LabelFrame(tab_settings, text="Configuración General")
        cfg.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(cfg, text="Proveedor de datos").grid(row=0, column=0, sticky="w", **pad)
        ttk.Combobox(cfg, textvariable=self.var_data_provider, values=["alpha_vantage","binance","tiingo"], state="readonly").grid(row=0, column=1, sticky="ew", **pad)
        ttk.Label(cfg, text="Broker").grid(row=1, column=0, sticky="w", **pad)
        ttk.Combobox(cfg, textvariable=self.var_broker, values=["paper","alpaca","binance","coinbase","ibkr"], state="readonly").grid(row=1, column=1, sticky="ew", **pad)
        ttk.Label(cfg, text="Perfil").grid(row=2, column=0, sticky="w", **pad)
        ttk.Combobox(cfg, textvariable=self.var_profile, values=["dev","paper","live"], state="readonly").grid(row=2, column=1, sticky="ew", **pad)
        ttk.Label(cfg, text="Alpha Vantage API Key").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(cfg, textvariable=self.var_alpha, show="*").grid(row=3, column=1, sticky="ew", **pad)
        ttk.Label(cfg, text="Alpaca API Key").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(cfg, textvariable=self.var_alpaca_key, show="*").grid(row=4, column=1, sticky="ew", **pad)
        ttk.Label(cfg, text="Alpaca API Secret").grid(row=5, column=0, sticky="w", **pad)
        ttk.Entry(cfg, textvariable=self.var_alpaca_secret, show="*").grid(row=5, column=1, sticky="ew", **pad)
        # Binance/Coinbase/IBKR credenciales
        self.var_binance_key = tk.StringVar()
        self.var_binance_secret = tk.StringVar()
        ttk.Label(cfg, text="Binance API Key").grid(row=6, column=0, sticky="w", **pad)
        ttk.Entry(cfg, textvariable=self.var_binance_key, show="*").grid(row=6, column=1, sticky="ew", **pad)
        ttk.Label(cfg, text="Binance API Secret").grid(row=7, column=0, sticky="w", **pad)
        ttk.Entry(cfg, textvariable=self.var_binance_secret, show="*").grid(row=7, column=1, sticky="ew", **pad)
        self.var_cb_key = tk.StringVar()
        self.var_cb_secret = tk.StringVar()
        self.var_cb_pass = tk.StringVar()
        ttk.Label(cfg, text="Coinbase API Key").grid(row=8, column=0, sticky="w", **pad)
        ttk.Entry(cfg, textvariable=self.var_cb_key, show="*").grid(row=8, column=1, sticky="ew", **pad)
        ttk.Label(cfg, text="Coinbase API Secret").grid(row=9, column=0, sticky="w", **pad)
        ttk.Entry(cfg, textvariable=self.var_cb_secret, show="*").grid(row=9, column=1, sticky="ew", **pad)
        ttk.Label(cfg, text="Coinbase Passphrase").grid(row=10, column=0, sticky="w", **pad)
        ttk.Entry(cfg, textvariable=self.var_cb_pass, show="*").grid(row=10, column=1, sticky="ew", **pad)
        self.var_ib_host = tk.StringVar()
        self.var_ib_port = tk.StringVar()
        self.var_ib_cid = tk.StringVar()
        ttk.Label(cfg, text="IBKR Host").grid(row=11, column=0, sticky="w", **pad)
        ttk.Entry(cfg, textvariable=self.var_ib_host).grid(row=11, column=1, sticky="ew", **pad)
        ttk.Label(cfg, text="IBKR Port").grid(row=12, column=0, sticky="w", **pad)
        ttk.Entry(cfg, textvariable=self.var_ib_port).grid(row=12, column=1, sticky="ew", **pad)
        ttk.Label(cfg, text="IBKR Client ID").grid(row=13, column=0, sticky="w", **pad)
        ttk.Entry(cfg, textvariable=self.var_ib_cid).grid(row=13, column=1, sticky="ew", **pad)
        ttk.Label(cfg, text="Telegram Bot Token").grid(row=14, column=0, sticky="w", **pad)
        self.var_tg_token = tk.StringVar()
        ttk.Entry(cfg, textvariable=self.var_tg_token, show="*").grid(row=14, column=1, sticky="ew", **pad)
        ttk.Label(cfg, text="Telegram Chat ID").grid(row=15, column=0, sticky="w", **pad)
        self.var_tg_chat = tk.StringVar()
        ttk.Entry(cfg, textvariable=self.var_tg_chat).grid(row=15, column=1, sticky="ew", **pad)
        # Hugging Face LLMs
        llm = ttk.LabelFrame(tab_settings, text="Hugging Face (Inference API)")
        llm.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(llm, text="API Token").grid(row=0, column=0, sticky="w", **pad)
        self.var_hf_token = tk.StringVar()
        ttk.Entry(llm, textvariable=self.var_hf_token, show="*").grid(row=0, column=1, sticky="ew", **pad)
        ttk.Label(llm, text="Modelo Sentimiento").grid(row=1, column=0, sticky="w", **pad)
        self.var_hf_sent = tk.StringVar()
        ttk.Entry(llm, textvariable=self.var_hf_sent).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Label(llm, text="Modelo Embeddings").grid(row=2, column=0, sticky="w", **pad)
        self.var_hf_emb = tk.StringVar()
        ttk.Entry(llm, textvariable=self.var_hf_emb).grid(row=2, column=1, sticky="ew", **pad)
        ttk.Label(llm, text="Modelo Resumen").grid(row=3, column=0, sticky="w", **pad)
        self.var_hf_sum = tk.StringVar()
        ttk.Entry(llm, textvariable=self.var_hf_sum).grid(row=3, column=1, sticky="ew", **pad)
        ttk.Button(llm, text="Guardar", command=self._save_env).grid(row=4, column=1, sticky="e", **pad)
        llm.columnconfigure(1, weight=1)
        # Riesgo/Exposición y caché
        riskf = ttk.LabelFrame(tab_settings, text="Riesgo / Exposición / Caché")
        riskf.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(riskf, text="MAX_OPEN_POSITIONS").grid(row=0, column=0, sticky="w", **pad)
        self.var_max_pos = tk.StringVar()
        ttk.Entry(riskf, textvariable=self.var_max_pos, width=12).grid(row=0, column=1, sticky="w", **pad)
        ttk.Label(riskf, text="MAX_EXPOSURE_PCT").grid(row=1, column=0, sticky="w", **pad)
        self.var_max_exp = tk.StringVar()
        ttk.Entry(riskf, textvariable=self.var_max_exp, width=12).grid(row=1, column=1, sticky="w", **pad)
        ttk.Label(riskf, text="MAX_SYMBOL_EXPOSURE_PCT").grid(row=2, column=0, sticky="w", **pad)
        self.var_max_symexp = tk.StringVar()
        ttk.Entry(riskf, textvariable=self.var_max_symexp, width=12).grid(row=2, column=1, sticky="w", **pad)
        ttk.Label(riskf, text="CACHE_TTL_SECONDS").grid(row=3, column=0, sticky="w", **pad)
        self.var_cache_ttl = tk.StringVar()
        ttk.Entry(riskf, textvariable=self.var_cache_ttl, width=12).grid(row=3, column=1, sticky="w", **pad)
        # Online Policy
        polf = ttk.LabelFrame(tab_settings, text="Política Online (Aprendizaje)")
        polf.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(polf, text="ONLINE_LOOKBACK").grid(row=0, column=0, sticky="w", **pad)
        self.var_on_lb = tk.StringVar()
        ttk.Entry(polf, textvariable=self.var_on_lb, width=12).grid(row=0, column=1, sticky="w", **pad)
        ttk.Label(polf, text="ONLINE_LEARNING_RATE").grid(row=0, column=2, sticky="w", **pad)
        self.var_on_lr = tk.StringVar()
        ttk.Entry(polf, textvariable=self.var_on_lr, width=12).grid(row=0, column=3, sticky="w", **pad)
        ttk.Label(polf, text="ONLINE_L2_REG").grid(row=0, column=4, sticky="w", **pad)
        self.var_on_l2 = tk.StringVar()
        ttk.Entry(polf, textvariable=self.var_on_l2, width=12).grid(row=0, column=5, sticky="w", **pad)

        # Parámetros de ejecución (slicing)
        execf = ttk.LabelFrame(tab_settings, text="Ejecución (Slicing)")
        execf.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(execf, text="SLICING_TRANCHES").grid(row=0, column=0, sticky="w", **pad)
        self.var_tranches = tk.StringVar()
        ttk.Entry(execf, textvariable=self.var_tranches, width=10).grid(row=0, column=1, sticky="w", **pad)
        ttk.Label(execf, text="SLICING_BPS").grid(row=0, column=2, sticky="w", **pad)
        self.var_bps = tk.StringVar()
        ttk.Entry(execf, textvariable=self.var_bps, width=10).grid(row=0, column=3, sticky="w", **pad)
        ttk.Label(execf, text="VWAP_WINDOW_SEC").grid(row=0, column=4, sticky="w", **pad)
        self.var_vwap_sec = tk.StringVar()
        ttk.Entry(execf, textvariable=self.var_vwap_sec, width=10).grid(row=0, column=5, sticky="w", **pad)
        ttk.Label(execf, text="POV_RATIO").grid(row=0, column=6, sticky="w", **pad)
        self.var_pov = tk.StringVar()
        ttk.Entry(execf, textvariable=self.var_pov, width=10).grid(row=0, column=7, sticky="w", **pad)
        ttk.Label(execf, text="PAUSED_SYMBOLS").grid(row=2, column=0, sticky="w", **pad)
        self.var_paused = tk.StringVar()
        ttk.Entry(execf, textvariable=self.var_paused).grid(row=2, column=1, columnspan=3, sticky="ew", **pad)
        ttk.Label(execf, text="SLICING_PROFILE_MAP").grid(row=1, column=0, sticky="w", **pad)
        self.var_slice_map = tk.StringVar()
        ttk.Entry(execf, textvariable=self.var_slice_map).grid(row=1, column=1, columnspan=3, sticky="ew", **pad)
        ttk.Label(execf, text="RISK_PROFILE_MAP").grid(row=1, column=4, sticky="w", **pad)
        self.var_risk_map = tk.StringVar()
        ttk.Entry(execf, textvariable=self.var_risk_map).grid(row=1, column=5, columnspan=3, sticky="ew", **pad)
        # Sesgo de sentimiento
        sentf = ttk.LabelFrame(tab_settings, text="Sesgo de Sentimiento")
        sentf.pack(fill=tk.X, padx=8, pady=8)
        self.var_use_sent = tk.BooleanVar()
        ttk.Checkbutton(sentf, text="USE_SENTIMENT_BIAS", variable=self.var_use_sent).grid(row=0, column=0, sticky="w", **pad)
        ttk.Label(sentf, text="SENTIMENT_WINDOW_MINUTES").grid(row=0, column=1, sticky="w", **pad)
        self.var_sent_win = tk.StringVar()
        ttk.Entry(sentf, textvariable=self.var_sent_win, width=12).grid(row=0, column=2, sticky="w", **pad)
        ttk.Button(sentf, text="Guardar Todo", command=self._save_env).grid(row=0, column=3, sticky="e", **pad)
        btns_cfg = tk.Frame(cfg)
        btns_cfg.grid(row=16, column=1, sticky="e", **pad)
        ttk.Button(btns_cfg, text="Probar Conexión", command=self._test_connectivity).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns_cfg, text="Guardar", command=self._save_env).pack(side=tk.LEFT, padx=4)
        # Secret store
        secf = ttk.LabelFrame(tab_settings, text="Secretos (cifrado en disco)")
        secf.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(secf, text="Guardar Secretos", command=self._save_secrets).pack(side=tk.LEFT, padx=4)
        ttk.Button(secf, text="Restaurar Secretos", command=self._load_secrets).pack(side=tk.LEFT, padx=4)
        cfg.columnconfigure(1, weight=1)

        # Pestaña Datos Alternativos (Hugging Face)
        alt = ttk.LabelFrame(tab_altdata, text="Señales de NLP (Hugging Face)")
        alt.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        ttk.Label(alt, text="Texto").grid(row=0, column=0, sticky="w", **pad)
        self.alt_text = tk.Text(alt, height=6)
        self.alt_text.grid(row=0, column=1, sticky="nsew", **pad)
        ttk.Button(alt, text="Analizar Sentimiento", command=self._alt_sentiment).grid(row=0, column=2, sticky="n", **pad)
        ttk.Button(alt, text="Resumir", command=self._alt_summarize).grid(row=0, column=3, sticky="n", **pad)
        alt.rowconfigure(0, weight=1)
        alt.columnconfigure(1, weight=1)
        self.alt_out = tk.Text(alt, height=12)
        self.alt_out.grid(row=1, column=0, columnspan=4, sticky="nsew", **pad)
        rss = ttk.LabelFrame(tab_altdata, text="RSS a Señales")
        rss.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        ttk.Label(rss, text="URL RSS").grid(row=0, column=0, sticky="w", **pad)
        self.rss_url = tk.StringVar(value="https://www.investing.com/rss/news.rss")
        ttk.Entry(rss, textvariable=self.rss_url).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Label(rss, text="Símbolo (opcional)").grid(row=0, column=2, sticky="w", **pad)
        self.rss_symbol = tk.StringVar(value="")
        ttk.Entry(rss, textvariable=self.rss_symbol, width=12).grid(row=0, column=3, sticky="w", **pad)
        ttk.Button(rss, text="Ingerir y Analizar", command=self._rss_ingest).grid(row=0, column=4, sticky="w", **pad)
        ttk.Label(rss, text="Intervalo (min)").grid(row=0, column=5, sticky="w", **pad)
        self.rss_interval = tk.StringVar(value="30")
        ttk.Entry(rss, textvariable=self.rss_interval, width=6).grid(row=0, column=6, sticky="w", **pad)
        ttk.Button(rss, text="Auto ON", command=self._rss_auto_on).grid(row=0, column=7, sticky="w", **pad)
        ttk.Button(rss, text="Auto OFF", command=self._rss_auto_off).grid(row=0, column=8, sticky="w", **pad)
        rss.columnconfigure(1, weight=1)
        self.rss_out = tk.Text(rss, height=10)
        self.rss_out.grid(row=1, column=0, columnspan=5, sticky="nsew", **pad)
        # EDGAR Filings
        edgar = ttk.LabelFrame(tab_altdata, text="SEC EDGAR (últimos filings)")
        edgar.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        ttk.Button(edgar, text="Traer EDGAR", command=self._edgar_fetch).grid(row=0, column=0, sticky="w", **pad)
        self.edgar_tree = ttk.Treeview(edgar, columns=("updated","title","link"), show="headings", height=6)
        for key, title in [("updated","Fecha"),("title","Título"),("link","Enlace")]:
            self.edgar_tree.heading(key, text=title)
            self.edgar_tree.column(key, width=200 if key!="title" else 360, anchor="w")
        self.edgar_tree.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=8, pady=4)
        edgar.columnconfigure(0, weight=1)

        # FRED Series
        fred = ttk.LabelFrame(tab_altdata, text="FRED (Series económicas)")
        fred.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        ttk.Label(fred, text="Series ID").grid(row=0, column=0, sticky="w", **pad)
        self.fred_series = tk.StringVar(value="CPIAUCSL")
        ttk.Entry(fred, textvariable=self.fred_series, width=20).grid(row=0, column=1, sticky="w", **pad)
        ttk.Button(fred, text="Traer FRED", command=self._fred_fetch).grid(row=0, column=2, sticky="w", **pad)
        self.fred_tree = ttk.Treeview(fred, columns=("date","value"), show="headings", height=6)
        for key, title in [("date","Fecha"),("value","Valor")]:
            self.fred_tree.heading(key, text=title)
            self.fred_tree.column(key, width=160 if key=="date" else 120, anchor="w")
        self.fred_tree.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=8, pady=4)
        fred.columnconfigure(1, weight=1)

        # Twitter (X) búsqueda reciente
        tw = ttk.LabelFrame(tab_altdata, text="X/Twitter (búsqueda reciente)")
        tw.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        ttk.Label(tw, text="Query").grid(row=0, column=0, sticky="w", **pad)
        self.tw_query = tk.StringVar(value="(AAPL OR MSFT) lang:en -is:retweet")
        ttk.Entry(tw, textvariable=self.tw_query).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Label(tw, text="Max").grid(row=0, column=2, sticky="w", **pad)
        self.tw_max = tk.StringVar(value="20")
        ttk.Entry(tw, textvariable=self.tw_max, width=6).grid(row=0, column=3, sticky="w", **pad)
        ttk.Button(tw, text="Buscar", command=self._tw_search).grid(row=0, column=4, sticky="w", **pad)
        tw.columnconfigure(1, weight=1)
        self.tw_tree = ttk.Treeview(tw, columns=("created","text","tickers","sent","score"), show="headings", height=6)
        for key, title in [("created","Fecha"),("text","Texto"),("tickers","Tickers"),("sent","Sent."),("score","Score")]:
            self.tw_tree.heading(key, text=title)
            self.tw_tree.column(key, width=120 if key!="text" else 420, anchor="w")
        self.tw_tree.grid(row=1, column=0, columnspan=5, sticky="nsew", padx=8, pady=4)
        # Señales recientes persistidas
        sigf = ttk.LabelFrame(tab_altdata, text="Señales NLP recientes")
        sigf.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.sig_tree = ttk.Treeview(sigf, columns=("ts","source","title","sent","score"), show="headings", height=6)
        for key, title in [("ts","Hora"),("source","Fuente"),("title","Título"),("sent","Sent."),("score","Score")]:
            self.sig_tree.heading(key, text=title)
            self.sig_tree.column(key, width=140 if key=="title" else 100, anchor="w")
        self.sig_tree.pack(fill=tk.BOTH, expand=True)

    def _load_env(self) -> None:
        env = read_env()
        self.var_data_provider.set(env.get("DATA_PROVIDER", "alpha_vantage"))
        self.var_broker.set(env.get("BROKER", "paper"))
        if hasattr(self, 'var_profile'):
            self.var_profile.set(env.get("PROFILE", "dev"))
        self.var_alpha.set(env.get("ALPHA_VANTAGE_API_KEY", ""))
        self.var_alpaca_key.set(env.get("ALPACA_API_KEY", ""))
        self.var_alpaca_secret.set(env.get("ALPACA_API_SECRET", ""))
        # Nuevos brokers
        if hasattr(self, 'var_binance_key'):
            self.var_binance_key.set(env.get("BINANCE_API_KEY", ""))
        if hasattr(self, 'var_binance_secret'):
            self.var_binance_secret.set(env.get("BINANCE_API_SECRET", ""))
        if hasattr(self, 'var_cb_key'):
            self.var_cb_key.set(env.get("COINBASE_API_KEY", ""))
        if hasattr(self, 'var_cb_secret'):
            self.var_cb_secret.set(env.get("COINBASE_API_SECRET", ""))
        if hasattr(self, 'var_cb_pass'):
            self.var_cb_pass.set(env.get("COINBASE_PASSPHRASE", ""))
        if hasattr(self, 'var_ib_host'):
            self.var_ib_host.set(env.get("IBKR_HOST", "127.0.0.1"))
        if hasattr(self, 'var_ib_port'):
            self.var_ib_port.set(str(env.get("IBKR_PORT", "7497")))
        if hasattr(self, 'var_ib_cid'):
            self.var_ib_cid.set(str(env.get("IBKR_CLIENT_ID", "1")))
        if hasattr(self, 'var_tg_token'):
            self.var_tg_token.set(env.get("TELEGRAM_BOT_TOKEN", ""))
        if hasattr(self, 'var_tg_chat'):
            self.var_tg_chat.set(env.get("TELEGRAM_CHAT_ID", ""))
        if hasattr(self, 'var_hf_token'):
            self.var_hf_token.set(env.get("HUGGINGFACE_API_TOKEN", ""))
        if hasattr(self, 'var_hf_sent'):
            self.var_hf_sent.set(env.get("HUGGINGFACE_SENTIMENT_MODEL", "ProsusAI/finbert"))
        if hasattr(self, 'var_hf_emb'):
            self.var_hf_emb.set(env.get("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
        if hasattr(self, 'var_hf_sum'):
            self.var_hf_sum.set(env.get("HUGGINGFACE_SUMMARIZATION_MODEL", "facebook/bart-large-cnn"))
        if hasattr(self, 'var_max_pos'):
            self.var_max_pos.set(env.get("MAX_OPEN_POSITIONS", "10"))
        if hasattr(self, 'var_max_exp'):
            self.var_max_exp.set(env.get("MAX_EXPOSURE_PCT", "1.0"))
        if hasattr(self, 'var_max_symexp'):
            self.var_max_symexp.set(env.get("MAX_SYMBOL_EXPOSURE_PCT", "0.2"))
        if hasattr(self, 'var_cache_ttl'):
            self.var_cache_ttl.set(env.get("CACHE_TTL_SECONDS", "43200"))
        if hasattr(self, 'var_on_lb'):
            self.var_on_lb.set(env.get("ONLINE_LOOKBACK", "30"))
        if hasattr(self, 'var_on_lr'):
            self.var_on_lr.set(env.get("ONLINE_LEARNING_RATE", "0.01"))
        if hasattr(self, 'var_on_l2'):
            self.var_on_l2.set(env.get("ONLINE_L2_REG", "0.0001"))
        if hasattr(self, 'var_tranches'):
            self.var_tranches.set(env.get("SLICING_TRANCHES", "4"))
        if hasattr(self, 'var_bps'):
            self.var_bps.set(env.get("SLICING_BPS", "5"))
        if hasattr(self, 'var_vwap_sec'):
            self.var_vwap_sec.set(env.get("VWAP_WINDOW_SEC", "60"))
        if hasattr(self, 'var_pov'):
            self.var_pov.set(env.get("POV_RATIO", "0.10"))
        if hasattr(self, 'var_slice_map'):
            self.var_slice_map.set(env.get("SLICING_PROFILE_MAP", ""))
        if hasattr(self, 'var_risk_map'):
            self.var_risk_map.set(env.get("RISK_PROFILE_MAP", ""))
        if hasattr(self, 'var_paused'):
            self.var_paused.set(env.get("PAUSED_SYMBOLS", ""))
        if hasattr(self, 'var_use_sent'):
            self.var_use_sent.set((env.get("USE_SENTIMENT_BIAS", "True").lower() in ["1","true","yes"]))
        if hasattr(self, 'var_sent_win'):
            self.var_sent_win.set(env.get("SENTIMENT_WINDOW_MINUTES", "180"))

    def _save_env(self) -> None:
        try:
            # Validaciones de JSON en perfiles
            from nge_trader.services.schema import validate_json_profile, SLICING_PROFILE_SCHEMA, RISK_PROFILE_SCHEMA
            if hasattr(self, 'var_slice_map') and (self.var_slice_map.get() or '').strip():
                validate_json_profile(self.var_slice_map.get(), SLICING_PROFILE_SCHEMA)
            if hasattr(self, 'var_risk_map') and (self.var_risk_map.get() or '').strip():
                validate_json_profile(self.var_risk_map.get(), RISK_PROFILE_SCHEMA)
            write_env(
                {
                    "DATA_PROVIDER": self.var_data_provider.get(),
                    "BROKER": self.var_broker.get(),
                    "PROFILE": getattr(self, 'var_profile', tk.StringVar(value="dev")).get(),
                    "ALPHA_VANTAGE_API_KEY": self.var_alpha.get(),
                    "ALPACA_API_KEY": self.var_alpaca_key.get(),
                    "ALPACA_API_SECRET": self.var_alpaca_secret.get(),
                    # Nuevos brokers
                    "BINANCE_API_KEY": getattr(self, 'var_binance_key', tk.StringVar(value="")).get(),
                    "BINANCE_API_SECRET": getattr(self, 'var_binance_secret', tk.StringVar(value="")).get(),
                    "COINBASE_API_KEY": getattr(self, 'var_cb_key', tk.StringVar(value="")).get(),
                    "COINBASE_API_SECRET": getattr(self, 'var_cb_secret', tk.StringVar(value="")).get(),
                    "COINBASE_PASSPHRASE": getattr(self, 'var_cb_pass', tk.StringVar(value="")).get(),
                    "IBKR_HOST": getattr(self, 'var_ib_host', tk.StringVar(value="127.0.0.1")).get(),
                    "IBKR_PORT": getattr(self, 'var_ib_port', tk.StringVar(value="7497")).get(),
                    "IBKR_CLIENT_ID": getattr(self, 'var_ib_cid', tk.StringVar(value="1")).get(),
                    "TELEGRAM_BOT_TOKEN": getattr(self, 'var_tg_token', tk.StringVar(value="")).get(),
                    "TELEGRAM_CHAT_ID": getattr(self, 'var_tg_chat', tk.StringVar(value="")).get(),
                    "HUGGINGFACE_API_TOKEN": getattr(self, 'var_hf_token', tk.StringVar(value="")).get(),
                    "HUGGINGFACE_SENTIMENT_MODEL": getattr(self, 'var_hf_sent', tk.StringVar(value="")).get(),
                    "HUGGINGFACE_EMBEDDING_MODEL": getattr(self, 'var_hf_emb', tk.StringVar(value="")).get(),
                    "HUGGINGFACE_SUMMARIZATION_MODEL": getattr(self, 'var_hf_sum', tk.StringVar(value="")).get(),
                    "MAX_OPEN_POSITIONS": getattr(self, 'var_max_pos', tk.StringVar(value="")).get(),
                    "MAX_EXPOSURE_PCT": getattr(self, 'var_max_exp', tk.StringVar(value="")).get(),
                    "MAX_SYMBOL_EXPOSURE_PCT": getattr(self, 'var_max_symexp', tk.StringVar(value="")).get(),
                    "CACHE_TTL_SECONDS": getattr(self, 'var_cache_ttl', tk.StringVar(value="")).get(),
                    "ONLINE_LOOKBACK": getattr(self, 'var_on_lb', tk.StringVar(value="")).get(),
                    "ONLINE_LEARNING_RATE": getattr(self, 'var_on_lr', tk.StringVar(value="")).get(),
                    "ONLINE_L2_REG": getattr(self, 'var_on_l2', tk.StringVar(value="")).get(),
                    "SLICING_TRANCHES": getattr(self, 'var_tranches', tk.StringVar(value="")).get(),
                    "SLICING_BPS": getattr(self, 'var_bps', tk.StringVar(value="")).get(),
                    "VWAP_WINDOW_SEC": getattr(self, 'var_vwap_sec', tk.StringVar(value="")).get(),
                    "POV_RATIO": getattr(self, 'var_pov', tk.StringVar(value="")).get(),
                    "SLICING_PROFILE_MAP": getattr(self, 'var_slice_map', tk.StringVar(value="")).get(),
                    "RISK_PROFILE_MAP": getattr(self, 'var_risk_map', tk.StringVar(value="")).get(),
                    "PAUSED_SYMBOLS": getattr(self, 'var_paused', tk.StringVar(value="")).get(),
                    "USE_SENTIMENT_BIAS": "True" if getattr(self, 'var_use_sent', tk.BooleanVar(value=True)).get() else "False",
                    "SENTIMENT_WINDOW_MINUTES": getattr(self, 'var_sent_win', tk.StringVar(value="")).get(),
                }
            )
            messagebox.showinfo("Guardado", "Configuración guardada en .env")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", f"No se pudo guardar: {exc}")

    def _save_secrets(self) -> None:
        try:
            from nge_trader.services.secret_store import SecretStore
            secrets = {
                "ALPACA_API_KEY": self.var_alpaca_key.get(),
                "ALPACA_API_SECRET": self.var_alpaca_secret.get(),
                "BINANCE_API_KEY": getattr(self, 'var_binance_key', tk.StringVar(value="")).get(),
                "BINANCE_API_SECRET": getattr(self, 'var_binance_secret', tk.StringVar(value="")).get(),
                "COINBASE_API_KEY": getattr(self, 'var_cb_key', tk.StringVar(value="")).get(),
                "COINBASE_API_SECRET": getattr(self, 'var_cb_secret', tk.StringVar(value="")).get(),
                "COINBASE_PASSPHRASE": getattr(self, 'var_cb_pass', tk.StringVar(value="")).get(),
                "TELEGRAM_BOT_TOKEN": getattr(self, 'var_tg_token', tk.StringVar(value="")).get(),
            }
            SecretStore().save(secrets)
            messagebox.showinfo("Secretos", "Secretos cifrados guardados en data/secrets.enc")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _load_secrets(self) -> None:
        try:
            from nge_trader.services.secret_store import SecretStore
            sec = SecretStore().load()
            for k, v in sec.items():
                if k == "ALPACA_API_KEY":
                    self.var_alpaca_key.set(v)
                elif k == "ALPACA_API_SECRET":
                    self.var_alpaca_secret.set(v)
                elif k == "BINANCE_API_KEY":
                    getattr(self, 'var_binance_key', tk.StringVar()).set(v)
                elif k == "BINANCE_API_SECRET":
                    getattr(self, 'var_binance_secret', tk.StringVar()).set(v)
                elif k == "COINBASE_API_KEY":
                    getattr(self, 'var_cb_key', tk.StringVar()).set(v)
                elif k == "COINBASE_API_SECRET":
                    getattr(self, 'var_cb_secret', tk.StringVar()).set(v)
                elif k == "COINBASE_PASSPHRASE":
                    getattr(self, 'var_cb_pass', tk.StringVar()).set(v)
                elif k == "TELEGRAM_BOT_TOKEN":
                    getattr(self, 'var_tg_token', tk.StringVar()).set(v)
            messagebox.showinfo("Secretos", "Secretos cargados")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))
    
    def _test_connectivity(self) -> None:
        try:
            service = AppService()
            st = service.get_connectivity_status()
            messagebox.showinfo("Conexión", f"Alpha Vantage: {st.get('alpha_vantage')} | Alpaca: {st.get('alpaca')} | Binance: {st.get('binance')} | Coinbase: {st.get('coinbase')} | IBKR: {st.get('ibkr')}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))
    def _start_live_demo(self) -> None:
        try:
            from nge_trader.services.live_engine import LiveConfig, LiveEngine
            symbols = [self.var_symbol.get()]
            strategy = self.strategy_var.get() if hasattr(self, "strategy_var") else "ma_cross"
            cfg = LiveConfig(symbols=symbols, strategy=strategy, capital_per_trade=1000.0, poll_seconds=30)
            engine = LiveEngine(cfg)
            threading.Thread(target=engine.run, daemon=True).start()
            messagebox.showinfo("Live", "Live Trading (demo) iniciado en background")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _run_backtest(self) -> None:
        def worker() -> None:
            try:
                service = AppService()
                df = service.data_provider.get_daily_adjusted(self.var_symbol.get())
                tf = self.bt_tf.get() if hasattr(self, "bt_tf") else "D"
                if tf in ("W", "M"):
                    rule = "W" if tf == "W" else "M"
                    df = resample_ohlc(df, rule)
                date_range = None
                if self.date_start.get() and self.date_end.get():
                    try:
                        s = pd.to_datetime(self.date_start.get())
                        e = pd.to_datetime(self.date_end.get())
                        date_range = (s, e)
                    except Exception:
                        date_range = None
                if self.strategy_var.get() == "ma_cross":
                    strat = MovingAverageCrossStrategy(10, 20)
                    signal = strat.generate_signals(df)
                    backtester = SignalBacktester()
                    result = backtester.run(df, signal, date_range=date_range)
                elif self.strategy_var.get() == "rsi":
                    strat = RSIStrategy(14, 30.0, 70.0)
                    signal = strat.generate_signals(df)
                    # Modo ejecución con slippage/comisiones
                    exec_bt = ExecutedSignalBacktester()
                    result = exec_bt.run(df, signal, exec_algo=self.exec_algo.get() or None, date_range=date_range)
                elif self.strategy_var.get() == "macd":
                    strat = MACDStrategy()
                    signal = strat.generate_signals(df)
                    exec_bt = ExecutedSignalBacktester()
                    result = exec_bt.run(df, signal, exec_algo=self.exec_algo.get() or None, date_range=date_range)
                elif self.strategy_var.get() == "bollinger":
                    strat = BollingerBandsStrategy()
                    signal = strat.generate_signals(df)
                    exec_bt = ExecutedSignalBacktester()
                    result = exec_bt.run(df, signal, exec_algo=self.exec_algo.get() or None, date_range=date_range)
                elif self.strategy_var.get() == "stochastic":
                    strat = StochasticOscillatorStrategy()
                    signal = strat.generate_signals(df)
                    exec_bt = ExecutedSignalBacktester()
                    result = exec_bt.run(df, signal, exec_algo=self.exec_algo.get() or None, date_range=date_range)
                elif self.strategy_var.get() == "donchian":
                    strat = DonchianBreakoutStrategy()
                    signal = strat.generate_signals(df)
                    exec_bt = ExecutedSignalBacktester()
                    result = exec_bt.run(df, signal, exec_algo=self.exec_algo.get() or None, date_range=date_range)
                elif self.strategy_var.get() == "momentum":
                    strat = MomentumStrategy()
                    signal = strat.generate_signals(df)
                    exec_bt = ExecutedSignalBacktester()
                    result = exec_bt.run(df, signal, exec_algo=self.exec_algo.get() or None, date_range=date_range)
                elif self.strategy_var.get() == "cci":
                    strat = CCIStrategy()
                    signal = strat.generate_signals(df)
                    exec_bt = ExecutedSignalBacktester()
                    result = exec_bt.run(df, signal, exec_algo=self.exec_algo.get() or None, date_range=date_range)
                elif self.strategy_var.get() == "adx":
                    strat = ADXStrategy()
                    signal = strat.generate_signals(df)
                    exec_bt = ExecutedSignalBacktester()
                    result = exec_bt.run(df, signal, exec_algo=self.exec_algo.get() or None, date_range=date_range)
                elif self.strategy_var.get() == "keltner":
                    strat = KeltnerChannelsStrategy()
                    signal = strat.generate_signals(df)
                    exec_bt = ExecutedSignalBacktester()
                    result = exec_bt.run(df, signal, date_range=date_range)
                elif self.strategy_var.get() == "williamsr":
                    strat = WilliamsRStrategy()
                    signal = strat.generate_signals(df)
                    exec_bt = ExecutedSignalBacktester()
                    result = exec_bt.run(df, signal, date_range=date_range)
                elif self.strategy_var.get() == "psar":
                    strat = ParabolicSARStrategy()
                    signal = strat.generate_signals(df)
                    exec_bt = ExecutedSignalBacktester()
                    result = exec_bt.run(df, signal, date_range=date_range)
                elif self.strategy_var.get() == "ichimoku":
                    strat = IchimokuStrategy()
                    signal = strat.generate_signals(df)
                    exec_bt = ExecutedSignalBacktester()
                    result = exec_bt.run(df, signal, date_range=date_range)
                elif self.strategy_var.get() == "supertrend":
                    strat = SupertrendStrategy()
                    signal = strat.generate_signals(df)
                    exec_bt = ExecutedSignalBacktester()
                    result = exec_bt.run(df, signal, date_range=date_range)
                elif self.strategy_var.get() == "obv":
                    strat = OBVStrategy()
                    signal = strat.generate_signals(df)
                    exec_bt = ExecutedSignalBacktester()
                    result = exec_bt.run(df, signal, date_range=date_range)
                elif self.strategy_var.get() == "ema_x":
                    strat = EMACrossoverStrategy()
                    signal = strat.generate_signals(df)
                    exec_bt = ExecutedSignalBacktester()
                    result = exec_bt.run(df, signal, date_range=date_range)
                elif self.strategy_var.get() == "pivot":
                    strat = PivotReversalStrategy()
                    signal = strat.generate_signals(df)
                    exec_bt = ExecutedSignalBacktester()
                    result = exec_bt.run(df, signal, date_range=date_range)
                elif self.strategy_var.get() == "heikin":
                    strat = HeikinAshiTrendStrategy()
                    signal = strat.generate_signals(df)
                    exec_bt = ExecutedSignalBacktester()
                    result = exec_bt.run(df, signal, date_range=date_range)
                elif self.strategy_var.get() == "trix":
                    strat = TRIXStrategy()
                    signal = strat.generate_signals(df)
                    exec_bt = ExecutedSignalBacktester()
                    result = exec_bt.run(df, signal, date_range=date_range)
                elif self.strategy_var.get() == "roc":
                    strat = ROCStrategy()
                    signal = strat.generate_signals(df)
                    exec_bt = ExecutedSignalBacktester()
                    result = exec_bt.run(df, signal, date_range=date_range)
                elif self.strategy_var.get() == "zscore":
                    strat = ZScoreReversionStrategy()
                    signal = strat.generate_signals(df)
                    exec_bt = ExecutedSignalBacktester()
                    result = exec_bt.run(df, signal, date_range=date_range)
                elif self.strategy_var.get() == "cmf":
                    strat = ChaikinMoneyFlowStrategy()
                    signal = strat.generate_signals(df)
                    exec_bt = ExecutedSignalBacktester()
                    result = exec_bt.run(df, signal, date_range=date_range)
                elif self.strategy_var.get() == "vwap_dev":
                    strat = VWAPDeviationStrategy()
                    signal = strat.generate_signals(df)
                    exec_bt = ExecutedSignalBacktester()
                    result = exec_bt.run(df, signal, date_range=date_range)
                elif self.strategy_var.get() == "pairs":
                    # Símbolo como AAA/BBB
                    symtxt = self.var_symbol.get().strip().upper()
                    if "/" not in symtxt:
                        raise ValueError("Para 'pairs' usa formato AAA/BBB")
                    left, right = [s.strip() for s in symtxt.split("/", 1)]
                    svc = AppService()
                    ldf = svc.data_provider.get_daily_adjusted(left)
                    rdf = svc.data_provider.get_daily_adjusted(right)
                    strat = PairsTradingStrategy()
                    psignal = strat.generate_pair_signals(ldf, rdf)
                    # construir precio sintético como ratio A/B para marcar-to-market
                    left_close = ldf["close"].astype(float).reset_index(drop=True)
                    right_close = rdf["close"].astype(float).reset_index(drop=True)
                    n = min(len(left_close), len(right_close))
                    synth = (left_close.iloc[-n:] / right_close.iloc[-n:]).reset_index(drop=True)
                    sdf = pd.DataFrame({"close": synth})
                    exec_bt = ExecutedSignalBacktester()
                    result = exec_bt.run(sdf, psignal.reset_index(drop=True), exec_algo=self.exec_algo.get() or None, date_range=None)
                elif self.strategy_var.get() == "meta":
                    # Ensemble de señales base
                    sigs = {}
                    sigs["ma_cross"] = MovingAverageCrossStrategy(10, 20).generate_signals(df)
                    sigs["rsi"] = RSIStrategy(14, 30.0, 70.0).generate_signals(df)
                    sigs["macd"] = MACDStrategy().generate_signals(df)
                    meta = MetaStrategy(EnsembleConfig())
                    msignal = meta.combine(df, sigs)
                    exec_bt = ExecutedSignalBacktester()
                    result = exec_bt.run(df, msignal, exec_algo=self.exec_algo.get() or None, date_range=date_range)
                elif self.strategy_var.get() == "agent":
                    policy = AgentPolicy(PolicyConfig())
                    try:
                        policy.load()
                    except Exception:
                        # Si no hay modelo, entrenar rápido con el mismo dataset
                        policy.fit(df)
                        policy.save()
                    signal = policy.generate_signals_series(df)
                    backtester = SignalBacktester()
                    result = backtester.run(df, signal, date_range=date_range)
                else:
                    backtester = SimpleBuyHoldBacktester()
                    result = backtester.run(df, date_range=date_range)
                head = df.head().to_string()
                out = (
                    f"Filas: {len(df)}\n"
                    f"Equity final: {result.equity_curve.iloc[-1]:.2f}\n"
                    f"Retorno total: {result.stats.get('total_return', float('nan')):.2%}\n\n"
                    f"Preview de datos:\n{head}"
                )
                # Persistir equity y trade de ejemplo (usar fechas si están disponibles)
                series_to_save = result.equity_curve.copy()
                if "date" in df.columns:
                    try:
                        series_to_save.index = pd.to_datetime(df["date"]).values
                        series_to_save.index.name = "date"
                    except Exception:
                        pass
                self.db.save_equity_curve(series_to_save)
                for t in result.trades:
                    self.db.record_trade(t)
                self.db.append_log("INFO", f"Backtest {self.var_symbol.get()} completado", datetime.now(datetime.UTC).isoformat())
            except Exception as exc:  # noqa: BLE001
                out = f"Error: {exc}"
            self._set_output(out)

        self._set_output("Ejecutando backtest...")
        threading.Thread(target=worker, daemon=True).start()

    def _set_output(self, text: str) -> None:
        self.txt.configure(state=tk.NORMAL)
        self.txt.delete("1.0", tk.END)
        self.txt.insert(tk.END, text)
        self.txt.configure(state=tk.DISABLED)

    def _alloc_fetch_prices(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        service = AppService()
        out: dict[str, pd.DataFrame] = {}
        for s in symbols:
            try:
                out[s] = service.data_provider.get_daily_adjusted(s)
            except Exception:
                out[s] = pd.DataFrame()
        return out

    def _alloc_risk_parity(self) -> None:
        try:
            from nge_trader.services.portfolio import PortfolioEngine
            syms = [s.strip().upper() for s in self.alloc_symbols.get().split(',') if s.strip()]
            wnd = int(self.alloc_window.get()) if self.alloc_window.get() else 60
            prices = self._alloc_fetch_prices(syms)
            # construir matriz de retornos alineada
            rets = {}
            for k, df in prices.items():
                if df.empty:
                    continue
                p = df["close"].astype(float).reset_index(drop=True)
                rets[k] = p.pct_change().dropna()
            if not rets:
                self.alloc_out.insert(tk.END, "Sin datos")
                return
            ret_df = pd.DataFrame(rets).dropna()
            ret_df = ret_df.iloc[-wnd:]
            eng = PortfolioEngine()
            res = eng.risk_parity(ret_df)
            self.alloc_out.configure(state=tk.NORMAL)
            self.alloc_out.delete("1.0", tk.END)
            self.alloc_out.insert(tk.END, f"Pesos (risk parity):\n{res.weights}\nRiesgo: {res.risk:.4f}")
            self.alloc_out.configure(state=tk.DISABLED)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _alloc_min_var(self) -> None:
        try:
            from nge_trader.services.portfolio import PortfolioEngine
            syms = [s.strip().upper() for s in self.alloc_symbols.get().split(',') if s.strip()]
            wnd = int(self.alloc_window.get()) if self.alloc_window.get() else 60
            prices = self._alloc_fetch_prices(syms)
            rets = {}
            for k, df in prices.items():
                if df.empty:
                    continue
                p = df["close"].astype(float).reset_index(drop=True)
                rets[k] = p.pct_change().dropna()
            if not rets:
                self.alloc_out.insert(tk.END, "Sin datos")
                return
            ret_df = pd.DataFrame(rets).dropna()
            ret_df = ret_df.iloc[-wnd:]
            eng = PortfolioEngine()
            res = eng.min_variance(ret_df)
            self.alloc_out.configure(state=tk.NORMAL)
            self.alloc_out.delete("1.0", tk.END)
            self.alloc_out.insert(tk.END, f"Pesos (min var):\n{res.weights}\nRiesgo: {res.risk:.4f}")
            self.alloc_out.configure(state=tk.DISABLED)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _oms_send(self) -> None:
        try:
            symbol = self.oms_symbol.get().strip().upper()
            side = self.oms_side.get()
            qty = float(self.oms_qty.get())
            tp_txt = self.oms_tp.get().strip()
            sl_txt = self.oms_sl.get().strip()
            tif = (self.oms_tif.get() if hasattr(self, "oms_tif") else "day") or "day"
            service = AppService()
            if tp_txt or sl_txt:
                tp = float(tp_txt) if tp_txt else None
                sl = float(sl_txt) if sl_txt else None
                method = getattr(service.broker, "place_bracket_order", None)
                if method:
                    resp = method(symbol, side, qty, take_profit_price=tp, stop_loss_price=sl, tif=tif)
                else:
                    raise ValueError("El broker no soporta brackets")
            else:
                resp = service.broker.place_order(symbol, side, qty, tif=tif)
            self.oms_out.configure(state=tk.NORMAL)
            self.oms_out.insert(tk.END, f"Orden enviada: {resp}\n")
            self.oms_out.configure(state=tk.DISABLED)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _mass_replace_sl(self) -> None:
        try:
            new_sl = float(self.oms_new_sl_mass.get())
        except Exception:
            messagebox.showwarning("Atención", "Introduce un SL válido")
            return
        try:
            service = AppService()
            if not hasattr(service.broker, "list_orders") or not hasattr(service.broker, "replace_order"):
                messagebox.showwarning("No soportado", "El broker no soporta replace masivo")
                return
            open_orders = service.broker.list_orders(status="open", limit=50)
            count = 0
            for o in open_orders:
                try:
                    oid = o.get("id")
                    if not oid:
                        continue
                    service.broker.replace_order(oid, {"stop_price": round(new_sl, 4)})
                    count += 1
                except Exception:
                    continue
            messagebox.showinfo("Masivo", f"Reemplazados SL en {count} órdenes")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _refresh_dashboard(self) -> None:
        # Redibuja la curva de capital y refresca widgets básicos
        series = self.db.load_equity_curve()
        canvas = self.canvas_equity
        canvas.delete("all")
        if series.empty:
            return
        width = int(canvas.winfo_width() or 800)
        height = int(canvas.winfo_height() or 180)
        vals = series.values
        min_v, max_v = float(vals.min()), float(vals.max())
        rng = max(max_v - min_v, 1e-9)
        points = []
        for i, v in enumerate(vals):
            x = int(i * (width - 20) / max(len(vals) - 1, 1)) + 10
            y = int(height - 10 - (v - min_v) * (height - 20) / rng)
            points.append((x, y))
        for i in range(1, len(points)):
            canvas.create_line(points[i - 1][0], points[i - 1][1], points[i][0], points[i][1], fill="#22d3ee", width=2)
        # Conectividad
        try:
            service = AppService()
            st = service.get_connectivity_status()
            txt = f"Conexiones: AV: {st.get('alpha_vantage')} | Alpaca: {st.get('alpaca')} | Binance: {st.get('binance')} | Coinbase: {st.get('coinbase')} | IBKR: {st.get('ibkr')}"
            self.lbl_connectivity.configure(text=txt)
        except Exception:
            pass
        # Métricas
        try:
            vals = series.pct_change().fillna(0.0)
            sharpe = compute_sharpe(vals)
            sortino = compute_sortino(vals)
            mdd = compute_max_drawdown(series.values)
            wr = compute_win_rate(vals)
            calmar = compute_calmar(series.values)
            # slippage/skew desde logs recientes
            try:
                logs = self.db.tail_logs(200, level=None)
                import json as _json
                slip = [float(_json.loads(r['message']).get('value')) for r in logs if 'slippage_bps' in r['message']]
                skew = [float(_json.loads(r['message']).get('value')) for r in logs if 'market_ws_skew_ms' in r['message']]
                slip_avg = (sum(slip)/len(slip)) if slip else 0.0
                skew_avg = (sum(skew)/len(skew)) if skew else 0.0
            except Exception:
                slip_avg = 0.0
                skew_avg = 0.0
            self.lbl_metrics.configure(text=f"Sharpe: {sharpe:.2f} | Sortino: {sortino:.2f} | Calmar: {calmar:.2f} | Max DD: {mdd:.2%} | Win Rate: {wr:.2%} | Slip(avg bps): {slip_avg:.1f} | Skew(ms): {skew_avg:.0f}")
        except Exception:
            pass
        # Resumen de cuenta Alpaca
        try:
            service = AppService()
            summary = service.get_account_summary()
            if summary["balance"] is not None:
                self.lbl_balance.configure(text=f"Balance Total: {summary['balance']:.2f}")
                self.lbl_pl_today.configure(text=f"P&L Hoy: {summary['pl_today']:.2f}")
                self.lbl_pl_open.configure(text=f"P&L Abierto: {summary['pl_open']:.2f}")
        except Exception:
            pass
        # Sentimiento agregado
        try:
            from nge_trader.repository.db import Database
            sent = Database().aggregated_sentiment(symbol=self.var_symbol.get() or None, minutes=180)
            self.lbl_sentiment.configure(text=f"Sentimiento agregado: {sent:+.2f}")
        except Exception:
            pass
        # Posiciones
        try:
            service = AppService()
            positions = service.get_portfolio_positions()
            self._reload_positions_tree(positions)
            # Semáforo por símbolo (edge neto)
            try:
                from nge_trader.services import oms as _OMS
                for iid in self.tree_positions.get_children():
                    vals = self.tree_positions.item(iid)["values"]
                    if not vals:
                        continue
                    sym = vals[0]
                    try:
                        res = _OMS.compute_executability(sym, "buy", None)
                        net = float(res.get("edge_bps", 0.0)) - float(res.get("threshold_bps", 0.0))
                        color = "GREEN" if net >= 0 else ("YELLOW" if net >= -2 else "RED")
                    except Exception:
                        color = "--"
                    vals_list = list(vals)
                    # Asegurar longitud columnas
                    if len(vals_list) < len(self.tree_positions["columns"]):
                        vals_list += [""] * (len(self.tree_positions["columns"]) - len(vals_list))
                    vals_list[-1] = color
                    self.tree_positions.item(iid, values=vals_list)
            except Exception:
                pass
        except Exception:
            pass
        # Últimas operaciones
        try:
            for iid in self.last_ops.get_children():
                self.last_ops.delete(iid)
            for t in self.db.recent_trades(5):
                self.last_ops.insert("", tk.END, values=(t.get("symbol"), t.get("side"), t.get("qty"), t.get("price"), t.get("in_time")))
        except Exception:
            pass
        # Órdenes recientes
        try:
            for iid in self.orders_tree.get_children():
                self.orders_tree.delete(iid)
            for o in self.db.recent_orders(50):
                self.orders_tree.insert("", tk.END, values=(o.get("ts"), o.get("symbol"), o.get("side"), o.get("qty"), o.get("price"), o.get("status"), o.get("order_id")))
        except Exception:
            pass
        # Fills recientes
        try:
            for iid in self.fills_tree.get_children():
                self.fills_tree.delete(iid)
            for f in self.db.recent_fills(50):
                self.fills_tree.insert("", tk.END, values=(f.get("ts"), f.get("symbol"), f.get("side"), f.get("qty"), f.get("price"), f.get("order_id")))
        except Exception:
            pass

    def _reload_positions_tree(self, positions: list[dict]) -> None:
        tree = self.tree_positions
        for iid in tree.get_children():
            tree.delete(iid)
        from nge_trader.config.settings import Settings as _S
        cshare = float(getattr(_S(), "canary_traffic_pct", 0.0) or 0.0)
        for p in positions:
            symbol = p.get("symbol") or p.get("Symbol") or "?"
            qty = p.get("qty") or p.get("Qty") or p.get("quantity") or "?"
            entry = p.get("avg_entry_price") or p.get("entry") or "?"
            price = p.get("current_price") or p.get("price") or "?"
            market_value = p.get("market_value") or p.get("value") or "?"
            upl = p.get("unrealized_pl") or p.get("upl") or "?"
            rpl = p.get("realized_pnl") or "?"
            stop = p.get("stop_price") or "--"
            take = p.get("take_profit") or "--"
            risk_R = p.get("risk_in_R") or "--"
            r_total = p.get("r_total") or "--"
            exposure = market_value or "--"
            exposure_pct = "--"
            strategy = p.get("strategy_key") or p.get("strategy") or "?"
            model = p.get("model_version") or "--"
            traffic = f"{cshare:.0%}"
            sem = "--"
            tree.insert("", tk.END, values=(symbol, qty, entry, price, market_value, upl, rpl, stop, take, risk_R, r_total, exposure, exposure_pct, strategy, model, traffic, sem))

    def _close_selected_position(self) -> None:
        sel = self.tree_positions.selection()
        if not sel:
            messagebox.showwarning("Atención", "Selecciona una posición para cerrar.")
            return
        item = self.tree_positions.item(sel[0])
        symbol = item["values"][0]
        if messagebox.askyesno("Confirmar", f"¿Cerrar posición en {symbol}?"):
            try:
                service = AppService()
                resp = service.close_position(symbol)
                messagebox.showinfo("Resultado", str(resp))
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Error", str(exc))

    def _close_all_positions(self) -> None:
        if not messagebox.askyesno("Confirmar", "¿Seguro que deseas cerrar TODAS las posiciones?"):
            return
        try:
            service = AppService()
            resp = service.close_all_positions()
            messagebox.showinfo("Resultado", str(resp))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _set_selected_sl(self) -> None:
        sel = self.tree_positions.selection()
        if not sel:
            messagebox.showwarning("Atención", "Selecciona una posición")
            return
        item = self.tree_positions.item(sel[0])
        symbol = item["values"][0]
        try:
            new_sl = float(self.var_new_sl.get())
        except Exception:
            messagebox.showwarning("Atención", "Introduce un SL válido")
            return
        try:
            from nge_trader.services.oms import quantize_quantity_price
            q_qty, q_px = quantize_quantity_price(symbol, 1.0, new_sl)
            from nge_trader.services.app_service import AppService
            service = AppService()
            if hasattr(service.broker, "replace_order"):
                resp = service.broker.replace_order(None, {"stop_price": q_px})
                messagebox.showinfo("OK", str(resp))
            else:
                messagebox.showwarning("No soportado", "El broker actual no soporta replace")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _set_selected_tp(self) -> None:
        sel = self.tree_positions.selection()
        if not sel:
            messagebox.showwarning("Atención", "Selecciona una posición")
            return
        item = self.tree_positions.item(sel[0])
        symbol = item["values"][0]
        try:
            new_tp = float(self.var_new_tp.get())
        except Exception:
            messagebox.showwarning("Atención", "Introduce un TP válido")
            return
        try:
            from nge_trader.services.oms import quantize_quantity_price
            q_qty, q_px = quantize_quantity_price(symbol, 1.0, new_tp)
            from nge_trader.services.app_service import AppService
            service = AppService()
            if hasattr(service.broker, "replace_order"):
                resp = service.broker.replace_order(None, {"take_profit": q_px})
                messagebox.showinfo("OK", str(resp))
            else:
                messagebox.showwarning("No soportado", "El broker actual no soporta replace")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _move_sl_to_breakeven(self) -> None:
        sel = self.tree_positions.selection()
        if not sel:
            messagebox.showwarning("Atención", "Selecciona una posición")
            return
        item = self.tree_positions.item(sel[0])
        symbol = item["values"][0]
        entry = float(item["values"][2] or 0.0)
        try:
            from nge_trader.services.oms import quantize_quantity_price
            _, q_px = quantize_quantity_price(symbol, 1.0, entry)
            from nge_trader.services.app_service import AppService
            service = AppService()
            if hasattr(service.broker, "replace_order"):
                resp = service.broker.replace_order(None, {"stop_price": q_px})
                messagebox.showinfo("OK", str(resp))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _move_sl_trailing(self) -> None:
        # Placeholder simple: mover SL al 98% del precio actual
        sel = self.tree_positions.selection()
        if not sel:
            messagebox.showwarning("Atención", "Selecciona una posición")
            return
        item = self.tree_positions.item(sel[0])
        symbol = item["values"][0]
        price = float(item["values"][3] or 0.0)
        try:
            new_sl = price * 0.98 if price > 0 else price
            from nge_trader.services.oms import quantize_quantity_price
            _, q_px = quantize_quantity_price(symbol, 1.0, new_sl)
            from nge_trader.services.app_service import AppService
            service = AppService()
            if hasattr(service.broker, "replace_order"):
                resp = service.broker.replace_order(None, {"stop_price": q_px})
                messagebox.showinfo("OK", str(resp))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _refresh_logs(self) -> None:
        level = self.log_level.get() if hasattr(self, "log_level") else None
        rows = self.db.tail_logs(200, level=level)
        text = "\n".join(f"[{r['ts']}] {r['level']}: {r['message']}" for r in rows)
        self.txt_logs.configure(state=tk.NORMAL)
        self.txt_logs.delete("1.0", tk.END)
        self.txt_logs.insert(tk.END, text)
        self.txt_logs.configure(state=tk.DISABLED)

    def _alt_sentiment(self) -> None:
        try:
            text = self.alt_text.get("1.0", tk.END).strip()
            if not text:
                return
            from nge_trader.services.nlp_service import NLPService

            nlp = NLPService()
            out = nlp.analyze_sentiment([text])
            self.alt_out.configure(state=tk.NORMAL)
            self.alt_out.delete("1.0", tk.END)
            self.alt_out.insert(tk.END, str(out))
            self.alt_out.configure(state=tk.DISABLED)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _alt_summarize(self) -> None:
        try:
            text = self.alt_text.get("1.0", tk.END).strip()
            if not text:
                return
            from nge_trader.services.nlp_service import NLPService

            nlp = NLPService()
            out = nlp.summarize(text)
            self.alt_out.configure(state=tk.NORMAL)
            self.alt_out.delete("1.0", tk.END)
            self.alt_out.insert(tk.END, out)
            self.alt_out.configure(state=tk.DISABLED)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _rss_ingest(self) -> None:
        try:
            from nge_trader.services.altdata_ingest import AltDataIngest
            from nge_trader.repository.db import Database
            svc = AltDataIngest()
            arts = svc.fetch_rss(self.rss_url.get())
            df = svc.analyze_articles(arts)
            self.rss_out.configure(state=tk.NORMAL)
            self.rss_out.delete("1.0", tk.END)
            head = df.head(10).to_string(index=False)
            self.rss_out.insert(tk.END, head)
            self.rss_out.configure(state=tk.DISABLED)
            # Persistir
            rows = []
            now = datetime.now(datetime.UTC).isoformat()
            sym_val = (self.rss_symbol.get() or None)
            for _, row in df.iterrows():
                rows.append({
                    "ts": now,
                    "source": "rss",
                    "symbol": sym_val,
                    "title": str(row.get("title", ""))[:256],
                    "sentiment_label": row.get("sent_label"),
                    "sentiment_score": row.get("sent_score"),
                    "summary": str(row.get("gen_summary", ""))[:512],
                    "link": row.get("link"),
                })
            Database().save_alt_signals(rows)
            # Recargar árbol
            self._reload_alt_signals()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _reload_alt_signals(self) -> None:
        try:
            from nge_trader.repository.db import Database
            rows = Database().recent_alt_signals(50)
            for iid in self.sig_tree.get_children():
                self.sig_tree.delete(iid)
            for r in rows:
                self.sig_tree.insert("", tk.END, values=(r.get("ts"), r.get("source"), r.get("title"), r.get("sentiment_label"), r.get("sentiment_score")))
        except Exception:
            pass

    def _rss_loop(self) -> None:
        import time
        while getattr(self, "_rss_stop", True) is False:
            try:
                self._rss_ingest()
            except Exception:
                pass
            try:
                mins = int(self.rss_interval.get()) if hasattr(self, "rss_interval") else 30
            except Exception:
                mins = 30
            for _ in range(max(mins, 1)):
                if getattr(self, "_rss_stop", True):
                    return
                time.sleep(60)

    def _rss_auto_on(self) -> None:
        if getattr(self, "_rss_thread", None) and self._rss_thread.is_alive():
            messagebox.showinfo("Auto RSS", "Ya está activo")
            return
        self._rss_stop = False
        self._rss_thread = threading.Thread(target=self._rss_loop, daemon=True)
        self._rss_thread.start()
        messagebox.showinfo("Auto RSS", "Auto-ingesta iniciada")

    def _rss_auto_off(self) -> None:
        self._rss_stop = True
        messagebox.showinfo("Auto RSS", "Auto-ingesta detenida")

    def _edgar_fetch(self) -> None:
        try:
            from nge_trader.services.edgar import fetch_recent_filings
            rows = fetch_recent_filings()
            # rellenar árbol
            for iid in getattr(self, 'edgar_tree', ttk.Treeview()).get_children():
                self.edgar_tree.delete(iid)
            for r in rows:
                self.edgar_tree.insert("", tk.END, values=(r.get("updated"), r.get("title"), r.get("link")))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _fred_fetch(self) -> None:
        try:
            from nge_trader.services.fred import FREDService
            series_id = self.fred_series.get().strip()
            svc = FREDService()
            rows = svc.fetch_series_observations(series_id, limit=120)
            for iid in getattr(self, 'fred_tree', ttk.Treeview()).get_children():
                self.fred_tree.delete(iid)
            for r in rows[-60:]:
                self.fred_tree.insert("", tk.END, values=(r.get("date"), r.get("value")))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _tw_search(self) -> None:
        try:
            from nge_trader.services.twitter import TwitterService
            from nge_trader.services.ticker_ner import extract_tickers
            from nge_trader.services.nlp_service import NLPService
            from nge_trader.repository.db import Database
            q = self.tw_query.get().strip()
            mx = int(self.tw_max.get() or 20)
            svc = TwitterService()
            nlp = NLPService()
            tweets = svc.search_recent(q, max_results=mx)
            # analizar
            for iid in getattr(self, 'tw_tree', ttk.Treeview()).get_children():
                self.tw_tree.delete(iid)
            texts = [t.get("text", "") for t in tweets]
            sents = nlp.analyze_sentiment(texts)
            rows = []
            for t, s in zip(tweets, sents):
                ticks = ",".join(extract_tickers(t.get("text", "")))
                self.tw_tree.insert("", tk.END, values=(t.get("created_at"), t.get("text"), ticks, s.get("label"), s.get("score")))
                rows.append({
                    "ts": t.get("created_at") or datetime.now(datetime.UTC).isoformat(),
                    "source": "twitter",
                    "symbol": None,
                    "title": t.get("text", "")[:256],
                    "sentiment_label": s.get("label"),
                    "sentiment_score": float(s.get("score") or 0.0),
                    "summary": "",
                    "link": None,
                })
            if rows:
                Database().save_alt_signals(rows)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _start_refresh(self) -> None:
        def tick() -> None:
            try:
                self._refresh_dashboard()
                self._refresh_logs()
                self._refresh_system()
                self._refresh_audit()
            finally:
                self.root.after(2000, tick)
        self.root.after(1000, tick)

    def _refresh_system(self) -> None:
        try:
            cpu = psutil.cpu_percent(interval=0)
            mem = psutil.virtual_memory().percent
            self.lbl_sys.configure(text=f"CPU: {cpu:.0f}% | Mem: {mem:.0f}%")
        except Exception:
            pass

    def _arm_kill_switch(self) -> None:
        if not messagebox.askyesno("Confirmar", "¿Activar KILL-SWITCH y bloquear nuevas órdenes?" ):
            return
        try:
            from nge_trader.config.env_utils import write_env
            write_env({"KILL_SWITCH_ARMED": "True"})
            messagebox.showinfo("Kill-Switch", "Activado. Reinicia live para aplicar si es necesario.")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _disarm_kill_switch(self) -> None:
        if not messagebox.askyesno("Confirmar", "¿Desactivar KILL-SWITCH y permitir nuevas órdenes?" ):
            return
        try:
            from nge_trader.config.env_utils import write_env
            write_env({"KILL_SWITCH_ARMED": "False"})
            messagebox.showinfo("Kill-Switch", "Desactivado.")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _refresh_audit(self) -> None:
        try:
            rows = self.db.recent_audits(200)
            text = "\n".join(f"[{r['ts']}] {r['event']} | {r.get('symbol') or ''} | {r.get('order_id') or ''} | {r.get('details') or ''}" for r in rows)
            self.audit_text.configure(state=tk.NORMAL)
            self.audit_text.delete("1.0", tk.END)
            self.audit_text.insert(tk.END, text)
            self.audit_text.configure(state=tk.DISABLED)
        except Exception:
            pass

    def _reload_strategies(self) -> None:
        self.strat_list.delete(0, tk.END)
        for cfg in self.store.load_all():
            self.strat_list.insert(tk.END, f"{cfg.key} {cfg.params}")

    def _reload_assignments(self) -> None:
        self.assign_list.delete(0, tk.END)
        for cfg in self.store.load_assignments():
            self.assign_list.insert(tk.END, f"{cfg.key} -> {cfg.params}")

    def _save_strategy(self) -> None:
        try:
            key = self.edit_key.get().strip()
            params_text = self.edit_params.get("1.0", tk.END).strip() or "{}"
            import json

            params = json.loads(params_text)
            items = self.store.load_all()
            # upsert
            found = False
            for i, it in enumerate(items):
                if it.key == key:
                    items[i] = StrategyConfig(key=key, params=params)
                    found = True
                    break
            if not found:
                items.append(StrategyConfig(key=key, params=params))
            self.store.save_all(items)
            self._reload_strategies()
            messagebox.showinfo("OK", "Estrategia guardada")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _delete_strategy(self) -> None:
        sel = self.strat_list.curselection()
        if not sel:
            messagebox.showwarning("Atención", "Selecciona una estrategia")
            return
        idx = sel[0]
        items = self.store.load_all()
        if idx >= len(items):
            return
        items.pop(idx)
        self.store.save_all(items)
        self._reload_strategies()

    def _save_assignment(self) -> None:
        try:
            symbol = self.assign_symbol.get().strip().upper()
            key = self.assign_key.get().strip()
            if not symbol or not key:
                messagebox.showwarning("Atención", "Símbolo y Strategy Key requeridos")
                return
            items = self.store.load_assignments()
            found = False
            for i, it in enumerate(items):
                if it.key == symbol:
                    items[i] = StrategyConfig(key=symbol, params={"strategy": key})
                    found = True
                    break
            if not found:
                items.append(StrategyConfig(key=symbol, params={"strategy": key}))
            self.store.save_assignments(items)
            self._reload_assignments()
            messagebox.showinfo("OK", "Asignación guardada")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _delete_assignment(self) -> None:
        sel = self.assign_list.curselection()
        if not sel:
            messagebox.showwarning("Atención", "Selecciona una asignación")
            return
        idx = sel[0]
        items = self.store.load_assignments()
        if idx >= len(items):
            return
        items.pop(idx)
        self.store.save_assignments(items)
        self._reload_assignments()

    def _cancel_selected_order(self) -> None:
        sel = self.orders_tree.selection()
        if not sel:
            messagebox.showwarning("Atención", "Selecciona una orden")
            return
        item = self.orders_tree.item(sel[0])
        order_id = item["values"][6]
        if not order_id:
            messagebox.showwarning("Atención", "Orden sin ID")
            return
        if not messagebox.askyesno("Confirmar", f"¿Cancelar orden {order_id}?"):
            return
        try:
            from nge_trader.services.app_service import AppService
            service = AppService()
            if hasattr(service.broker, "cancel_order"):
                resp = service.broker.cancel_order(order_id)
                messagebox.showinfo("OK", str(resp))
            else:
                messagebox.showwarning("No soportado", "El broker actual no soporta cancelación")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _replace_selected_order_sl(self) -> None:
        sel = self.orders_tree.selection()
        if not sel:
            messagebox.showwarning("Atención", "Selecciona una orden")
            return
        item = self.orders_tree.item(sel[0])
        order_id = item["values"][6]
        if not order_id:
            messagebox.showwarning("Atención", "Orden sin ID")
            return
        try:
            new_sl = float(self.var_new_sl.get())
        except Exception:
            messagebox.showwarning("Atención", "Introduce un precio SL válido")
            return
        if not messagebox.askyesno("Confirmar", f"¿Modificar SL a {new_sl} para orden {order_id}?"):
            return
        try:
            from nge_trader.services.app_service import AppService
            service = AppService()
            if hasattr(service.broker, "replace_order"):
                resp = service.broker.replace_order(order_id, {"stop_price": round(new_sl, 4)})
                messagebox.showinfo("OK", str(resp))
            else:
                messagebox.showwarning("No soportado", "El broker actual no soporta replace")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _cancel_all_orders(self) -> None:
        if not messagebox.askyesno("Confirmar", "¿Cancelar TODAS las órdenes abiertas?"):
            return
        try:
            from nge_trader.services.app_service import AppService
            service = AppService()
            if hasattr(service.broker, "cancel_all_orders"):
                resp = service.broker.cancel_all_orders()
                messagebox.showinfo("OK", str(resp))
            else:
                messagebox.showwarning("No soportado", "El broker actual no soporta cancelación masiva")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _cancel_all_by_symbol_inline(self) -> None:
        sym = (self.var_cancel_sym.get() or "").strip()
        if not sym:
            messagebox.showwarning("Atención", "Introduce símbolo")
            return
        if not messagebox.askyesno("Confirmar", f"¿Cancelar TODAS las órdenes de {sym}?"):
            return
        try:
            from nge_trader.entrypoints.api import api_cancel_all
            res = api_cancel_all(sym)
            messagebox.showinfo("OK", str(res))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _embed_figure(self, key: str, frame: ttk.LabelFrame, fig: Figure) -> None:
        old = self._analytics_canvases.get(key)
        if old:
            try:
                old.get_tk_widget().destroy()
            except Exception:
                pass
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._analytics_canvases[key] = canvas

    def _refresh_analytics(self) -> None:
        try:
            # Equity desde DB
            eq = self.db.load_equity_curve()
            if eq.empty:
                messagebox.showwarning("Atención", "No hay curva de capital guardada.")
                return
            # Normalizar equity
            eq = eq.sort_index()
            eq_norm = eq / float(eq.iloc[0])
            # Benchmark desde proveedor
            service = AppService()
            bench_df = service.data_provider.get_daily_adjusted(self.var_benchmark.get())
            b = bench_df[["date", "close"]].copy()
            b["date"] = pd.to_datetime(b["date"])  # type: ignore[assignment]
            b = b.set_index("date").sort_index()
            bench_norm = (b["close"].astype(float) / float(b["close"].iloc[0]))
            # Alinear
            joined = pd.concat([eq_norm.rename("equity"), bench_norm.rename("benchmark")], axis=1).dropna()

            # Figura 1: Equity vs Benchmark
            fig1 = Figure(figsize=(6, 2.4), dpi=100)
            ax1 = fig1.add_subplot(111)
            joined["equity"].plot(ax=ax1, label="Equity")
            joined["benchmark"].plot(ax=ax1, label=self.var_benchmark.get())
            ax1.legend()
            ax1.set_title("Equity vs Benchmark (normalizado)")
            self._embed_figure("fig1", self.analytics_fig1_frame, fig1)

            # Figura 2: Drawdown
            from nge_trader.services.metrics import compute_drawdown_series

            dd = compute_drawdown_series(eq.values)
            dd.index = eq.index
            fig2 = Figure(figsize=(6, 2.0), dpi=100)
            ax2 = fig2.add_subplot(111)
            dd.plot(ax=ax2, color="tomato")
            ax2.set_title("Drawdown")
            self._embed_figure("fig2", self.analytics_fig2_frame, fig2)

            # Figura 3: Distribución de Retornos
            ret = eq.pct_change().dropna()
            fig3 = Figure(figsize=(6, 2.0), dpi=100)
            ax3 = fig3.add_subplot(111)
            ax3.hist(ret.values, bins=50, color="#22d3ee")
            ax3.set_title("Distribución de Retornos Diarios")
            self._embed_figure("fig3", self.analytics_fig3_frame, fig3)

            # Live: reward rolling y slippage/skew
            try:
                from nge_trader.repository.db import Database
                db = Database()
                rewards = db.recent_rewards(200)
                slips = db.recent_metric_values("slippage_bps", 200)
                skews = db.recent_metric_values("market_ws_skew_ms", 200)
                sharpe_live = db.recent_metric_series("sharpe_live", 200)
                hit_rate = db.recent_metric_series("hit_rate", 200)
                import numpy as _np
                def _series(vals):
                    if not vals:
                        return _np.array([]), _np.array([])
                    xs = _np.arange(len(vals))
                    ys = _np.array([v for _, v in vals], dtype=float)
                    return xs, ys
                xr, yr = _series(rewards)
                xs, ys = _series(slips)
                xk, yk = _series(skews)
                xh, yh = _series(hit_rate)
                xs2, ys2 = _series(sharpe_live)
                fig4 = Figure(figsize=(6, 2.0), dpi=100)
                ax4 = fig4.add_subplot(111)
                if len(yr):
                    ax4.plot(xr, yr, label="reward", color="#22c55e")
                if len(ys):
                    ax4.plot(xs, ys, label="slip bps", color="#3b82f6")
                if len(yk):
                    ax4.plot(xk, yk, label="skew ms", color="#f59e0b")
                if len(yh):
                    ax4.plot(xh, yh, label="hit rate", color="#ec4899")
                if len(ys2):
                    ax4.plot(xs2, ys2, label="sharpe live", color="#10b981")
                ax4.legend()
                ax4.set_title("Reward/Slippage/Skew (últimos 200)")
                # incrustar
                self._embed_figure("fig_live", self.live_metrics_fig, fig4)
            except Exception:
                pass
        except Exception:
            pass
        
        except Exception:
            pass

    def _export_daily_report(self) -> None:
        try:
            import subprocess
            import sys
            cmd = [sys.executable, "scripts/export_daily_report.py"]
            out = subprocess.check_output(cmd, timeout=60)
            messagebox.showinfo("Reporte", out.decode("utf-8").strip())
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _refresh_correlations(self) -> None:
        try:
            syms = [s.strip().upper() for s in self.corr_symbols.get().split(',') if s.strip()]
            if not syms:
                return
            service = AppService()
            closes = {}
            for s in syms:
                df = service.data_provider.get_daily_adjusted(s)
                if df.empty:
                    continue
                c = df[["date", "close"]].copy()
                c["date"] = pd.to_datetime(c["date"])  # type: ignore[assignment]
                c = c.set_index("date").sort_index()
                closes[s] = c["close"].astype(float)
            if not closes:
                return
            prices = pd.DataFrame(closes).dropna()
            rets = prices.pct_change().dropna()
            corr = rets.corr()
            # plot heatmap
            fig = Figure(figsize=(5.6, 2.8), dpi=100)
            ax = fig.add_subplot(111)
            im = ax.imshow(corr.values, cmap="viridis", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(corr.index)))
            ax.set_yticklabels(corr.index)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title("Correlación de Retornos")
            self._embed_figure("corr", self.analytics_corr_frame, fig)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _export_report(self) -> None:
        try:
            eq = self.db.load_equity_curve()
            trades = self.db.recent_trades(100000)
            if eq.empty:
                messagebox.showwarning("Atención", "No hay curva de capital para exportar.")
                return
            vals = eq.pct_change().fillna(0.0)
            sharpe = compute_sharpe(vals)
            sortino = compute_sortino(vals)
            mdd = compute_max_drawdown(eq.values)
            wr = compute_win_rate(vals)
            total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
            metrics = {
                "sharpe": float(sharpe),
                "sortino": float(sortino),
                "max_drawdown": float(mdd),
                "win_rate": float(wr),
                "total_return": total_return,
            }
            report_dir = make_report_dir()
            export_equity_csv(report_dir, eq.values)
            export_trades_csv(report_dir, trades)
            export_tearsheet(report_dir, eq)
            html_path = export_report_html(report_dir, metrics)
            zip_path = export_report_zip(report_dir)
            messagebox.showinfo("Exportado", f"Reporte en: {html_path}\nZIP: {zip_path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def run(self) -> None:
        self.root.mainloop()

    def _train_agent_ui(self) -> None:
        try:
            sym = self.train_symbol.get().strip().upper()
            if not sym:
                messagebox.showwarning("Atención", "Introduce un símbolo")
                return
            # Entrena política con datos del proveedor
            from nge_trader.ai.policy import AgentPolicy, PolicyConfig
            service = AppService()
            df = service.data_provider.get_daily_adjusted(sym)
            policy = AgentPolicy(PolicyConfig())
            policy.fit(df)
            policy.save()
            messagebox.showinfo("OK", "Política del agente entrenada y guardada")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _wf_agent_ui(self) -> None:
        try:
            sym = self.train_symbol.get().strip().upper()
            if not sym:
                messagebox.showwarning("Atención", "Introduce un símbolo")
                return
            from nge_trader.ai.policy import AgentPolicy, PolicyConfig
            service = AppService()
            df = service.data_provider.get_daily_adjusted(sym)
            policy = AgentPolicy(PolicyConfig())
            res = policy.walk_forward(df, folds=5)
            policy.save()
            messagebox.showinfo("Walk-Forward", str(res))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _run_sweep(self) -> None:
        try:
            from nge_trader.services.sweeper import run_param_sweep
            # Fecha
            date_range = None
            if self.date_start.get() and self.date_end.get():
                try:
                    s = pd.to_datetime(self.date_start.get())
                    e = pd.to_datetime(self.date_end.get())
                    date_range = (s, e)
                except Exception:
                    date_range = None
            # Símbolos y estrategia actual
            symbols = [s.strip().upper() for s in self.sweep_symbols.get().split(',') if s.strip()]
            strat = self.strategy_var.get()
            # Grid
            import json
            grid_text = self.sweep_params.get("1.0", tk.END).strip()
            param_grid = json.loads(grid_text) if grid_text else {}
            res = run_param_sweep(symbols, strat, param_grid, date_range=date_range)
            # Mostrar top 5
            head = res.rows.head().to_string(index=False)
            self._set_output(f"Sweep completado. Directorio: {res.out_dir}\nTop 5:\n{head}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def _run_portfolio_bt(self) -> None:
        def worker() -> None:
            try:
                from nge_trader.services.backtester_portfolio import run_portfolio_backtest
                syms = [s.strip().upper() for s in self.btpf_symbols.get().split(',') if s.strip()]
                method = self.btpf_method.get()
                window = int(self.btpf_window.get() or 60)
                reb = int(self.btpf_reb.get() or 21)
                svc = AppService()
                data = {s: svc.data_provider.get_daily_adjusted(s) for s in syms}
                date_range = None
                if self.date_start.get() and self.date_end.get():
                    try:
                        s = pd.to_datetime(self.date_start.get())
                        e = pd.to_datetime(self.date_end.get())
                        date_range = (s, e)
                    except Exception:
                        date_range = None
                res = run_portfolio_backtest(data, method=method, window=window, rebalance_days=reb, date_range=date_range)
                final_eq = res.equity_curve.iloc[-1] if not res.equity_curve.empty else float('nan')
                tr = float(final_eq / res.equity_curve.iloc[0] - 1.0) if not res.equity_curve.empty else float('nan')
                out = f"Cartera: {method} sobre {len(syms)} símbolos\nRetorno total: {tr:.2%}\nPesos recientes: {res.weights_history[-1][1] if res.weights_history else {}}"
            except Exception as exc:  # noqa: BLE001
                out = f"Error: {exc}"
            self._set_output(out)
        self._set_output("Ejecutando portfolio backtest...")
        threading.Thread(target=worker, daemon=True).start()


def main() -> None:
    app = DesktopApp()
    app.run()


if __name__ == "__main__":
    main()


