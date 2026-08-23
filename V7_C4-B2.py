"""
MOTOR KMT C4_B2 -- VARIANTE de C4_B con SL/RR reoptimizado (estructura
1v, solo largo, timeframe 60s)
Databento live -> velas 60s + ticks -> Senal 1v (delta extremo dinamico,
SIN Volume Profile) -> sizing -> PickMyTrade -> Tradovate/Apex

======================================================================
ORIGEN DE ESTA VARIANTE (ago 2026)
======================================================================
Este motor es IDENTICO a C4_B (V7_C4-E.py) en TODO excepto dos
parametros: sl_atr y tp_atr. Nace de un analisis posterior a C4_B ya
en produccion, donde se observo un trade real con SL=0.35 ATR tocado
por una mecha de ruido segundos antes de que el precio continuara
hacia donde habria estado el TP -- lo cual motivo re-correr un grid
SL x RR sobre la MISMA señal ya validada de C4_B (estructura "1v",
60s, k=1.5, atr_regimen_max=1.3, solo-largo), usando el mismo dataset
de 397 dias y el mismo motor Numba (backtest_scalping_universo.py).

Grid corrido: SL en {0.35, 0.40, 0.45, 0.50, 0.55, 0.60} ATR x RR en
{1, 2, 3, 4, 5} (30 combinaciones), TP = SL * RR, señal de entrada SIN
TOCAR (mismo k, mismo atr_regimen_max, mismo filtro de solo-largo).

Hallazgo: el baseline de C4_B (SL=0.35, RR=5:1) resulto ser la
combinacion MENOS estable del grid completo -- la que mas degrado
entre train y validacion (WR 82.18% -> 76.68%, -5.5 puntos) y no la de
mayor PnL ni WR. La mejor combinacion encontrada fue:

    SL=0.60 ATR, TP=1.80 ATR (RR=3:1)
    Grid completo (1149 vs 1119 trades, ambos in-sample sobre los
    mismos 397 dias):
      C4_B  (sl=0.35, rr=5): WR=80.68%, pnl=$714,362
      Este  (sl=0.60, rr=3): WR=85.08%, pnl=$749,584  (mejor en ambos)
    Validacion train/val 70/30 (split cronologico, mismo estandar que
    el resto del sistema):
      C4_B:  TRAIN WR=82.18% -> VAL WR=76.68%  (degrada 5.5 puntos)
      Este:  TRAIN WR=85.19% -> VAL WR=84.77%  (degrada 0.42 puntos --
             la combinacion MAS estable de las 30 evaluadas)

Analisis de mecanismo (no solo correlacion): de los 184 trades que
C4_B cierra en SL con su configuracion actual, 99 (53.8%) habrian
llegado al TP con este SL/RR nuevo, caminando el precio real vela por
vela desde la entrada -- confirma que el TP demasiado lejano (RR=5:1)
es el cuello de botella real, no solo el SL angosto en si.

IMPORTANTE -- esto sigue siendo el MISMO backtest/dataset con el que
ya se valido C4_B, reinterpretado con otro grid, NO evidencia
independiente. Por eso este motor arranca en SHADOW, replicando
exactamente el mismo criterio que se aplico a C0 (V7.2PY.py): no se
reemplaza nada en vivo hasta acumular una muestra de shadow
comparable (misma meta de 20-30 señales) y confirmar que el
comportamiento con datos NUNCA vistos por ningun backtest sostiene la
mejora encontrada aqui.

Corre en su propio nodo Railway, su propio Postgres (DATABASE_URL
propio), su propio bot de Telegram -- NO comparte tabla ni chat con
C4_B, para poder comparar ambos en paralelo sin contaminar metricas.

======================================================================
TODO LO DEMAS ES IDENTICO A C4_B (V7_C4-E.py) -- ver ese archivo para
el historial completo de bugs encontrados/corregidos en la
infraestructura compartida (dollar_sl/dollar_tp anclados a fill real,
trailing sincronizado con broker, policy engine RiskAval, filtro de
salto anomalo, kill switch, watchdog, etc.)
======================================================================

Variables de entorno: DATABASE_URL, DATABENTO_API_KEY, PICKMYTRADE_WEBHOOK,
                      PICKMYTRADE_TOKEN, PICKMYTRADE_ACCOUNT, MODO_SHADOW,
                      TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, FRED_API_KEY
"""
import os, sys, json, time, logging, threading, queue
from datetime import datetime, timedelta, time as dtime
from dataclasses import dataclass
from typing import Optional, List, Dict
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import psycopg2

from policy_engine import PolicyEngine
from audit_schema import ActionType, MarketContext, RiskState
from authority_calibrator import AuthorityCalibrator

# ============================================================
# CONFIG -- variante de C4_B: 60s, k=1.5, atr_regimen_max=1.3,
# UNICO CAMBIO: sl=0.60 ATR / tp=1.80 ATR (RR=3:1, antes 0.35/1.75
# RR=5:1). Ver docstring arriba para el analisis completo que
# respalda este cambio.
# ============================================================
CFG = {
    "symbol_db": "MNQ.c.0",
    "symbol_es": "ES.c.0",
    "symbol_exec": "MNQ1!",
    "dataset": "GLBX.MDP3",
    "tf_seg": 60,  # timeframe en SEGUNDOS (no minutos) -- identico a C4_B

    # Señal 1v: IDENTICA a C4_B, sin tocar -- este motor prueba SOLO el
    # efecto de SL/TP, no de la señal de entrada.
    "delta_ventana": 20,
    "delta_std": 1.5,  # k, identico a C4_B

    "atr_regimen_max": 1.3,  # identico a C4_B

    "max_salto_atr_entrada": 3.0,  # identico a C4_B

    # NO hay Señal F / Volume Profile -- mismo motivo que C4_B (Etapa 2
    # demostro que ningun filtro de confirmacion mejora la señal "1v").

    "hora_ini": dtime(0, 0),
    "hora_fin_entradas": dtime(23, 50),
    "hora_cierre": dtime(23, 55),
    "mantenimiento_ini": dtime(17, 0),
    "mantenimiento_fin": dtime(18, 0),

    # >>> UNICO CAMBIO respecto a C4_B <<<
    # Antes (C4_B): sl_atr=0.35, tp_atr=1.75 (RR=5:1)
    # Ahora (esta variante): sl_atr=0.60, tp_atr=1.80 (RR=3:1)
    "sl_atr": 0.60,
    "tp_atr": 1.80,
    "atr_per": 20,

    # Cuenta y riesgo -- IDENTICOS a C4_B/C1/C2.
    "capital": 50_000,
    "riesgo_trade": 400,
    "riesgo_minimo_dolares": 400,
    "usd_punto": 2.0,
    "max_contratos": 2,
    "perdida_max_dia": 800,
    "perdida_max_sem": 2000,

    "kelly_fraccion": 0.50,
    "kelly_min_trades": 30,
    "kelly_ventana": 50,
    "kelly_techo_riesgo_pct": 0.02,
    "riesgo_sonda_fraccion": 0.25,

    "usar_corr_es": False,
    "corr_ventana": 20,
    "corr_min": 0.70,

    # Trailing stop dinamico -- EXACTO a C4_B (no se reoptimizo, solo
    # se probo SL inicial/TP; el trailing sigue siendo el mismo del
    # motor original).
    "breakeven_a_atr": 1.0,
    "bloqueo_50pct_a_atr": 1.5,
    "bloqueo_75pct_a_atr": 3.0,

    "reversion_reduccion_30pct": 0.35,
    "reversion_reduccion_60pct": 0.50,
    "reversion_cierre_total": 0.65,
    "rechazo_volumen_ventana": 60,
    "rechazo_volumen_umbral": 1.5,
    "rechazo_volumen_rango_atr": 0.7,

    "comision_por_contrato_rt": 1.25,
    "spread_puntos_ny": 1.5,
    "spread_puntos_overnight": 3.5,
    "hora_ny_inicio": dtime(9, 30),
    "hora_ny_fin": dtime(16, 0),

    "regimen_vol_ventana": 400,
    "vol_ratio_alto": 1.3,
    "vol_ratio_factor_reduccion": 0.7,

    "news_blackout_min": 20,
    "fred_series": ["DFF", "UNRATE", "CPIAUCSL"],

    # RiskAval: ARCHIVO NUEVO, DEBE CREARSE antes de desplegar --
    # copia de policy_config_c4_b.yaml con los MISMOS limites
    # (max_daily_loss=-800, max_weekly_loss=-2000, max_contracts=2),
    # guardado con nombre propio para no compartir estado de policy
    # engine con C4_B. Si falta o no coincide, Ejecutor.__init__ DEBE
    # fallar al arrancar (mismo criterio que C4_B/C2).
    "policy_config_path": "policy_config_c4_b2.yaml",
}
NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger("motor_kmt_c4_b2")


class Telegram:
    """Alertas a Telegram -- nunca debe tumbar el motor si falla."""
    def __init__(self, token: Optional[str], chat_id: Optional[str]):
        self.token = token
        self.chat_id = chat_id
        self.activo = bool(token and chat_id)
        if not self.activo:
            log.warning("Telegram no configurado (faltan TELEGRAM_TOKEN/TELEGRAM_CHAT_ID) -- alertas desactivadas")

    def enviar(self, texto: str):
        if not self.activo:
            return
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            requests.post(url, json={"chat_id": self.chat_id, "text": texto}, timeout=5)
        except Exception as e:
            log.error(f"Telegram error (no critico, se ignora): {e}")


@dataclass
class Pos:
    precio_entrada: float
    contratos: int
    sl: float
    tp: float
    atr: float
    ts_entrada: datetime
    delta: float
    decision_id: int
    etapa_stop: str = "PROTECCION"
    contratos_originales: int = 0

    def __post_init__(self):
        if self.contratos_originales == 0:
            self.contratos_originales = self.contratos


class Estado:
    def __init__(self):
        self.pos: Optional[Pos] = None
        self.pnl_dia = 0.0
        self.pnl_sem = 0.0
        self.dia = None
        self.sem = None
        self.velas = pd.DataFrame()
        self.velas_es = pd.DataFrame()
        self.trades: List[float] = []
        self.senal_pendiente: Optional[Dict] = None
        self.pausado = False


class DB:
    """Conexion persistente a Postgres, con reconexion automatica. Thread-safe."""
    def __init__(self, url):
        self.url = url
        self.conn = None
        self._lock = threading.Lock()
        self._connect()
        self._crear_tablas()

    def _connect(self):
        self.conn = psycopg2.connect(self.url)
        self.conn.autocommit = False
        log.info("Conexion a Postgres establecida")

    def _ensure(self):
        if self.conn is None or self.conn.closed:
            log.warning("Conexion a DB caida, reconectando...")
            self._connect()

    def _crear_tablas(self):
        self._ensure()
        with self.conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS decisiones (
                id SERIAL PRIMARY KEY, ts TIMESTAMPTZ DEFAULT NOW(),
                operar BOOLEAN, razon TEXT, contratos INT, precio NUMERIC,
                sl NUMERIC, tp NUMERIC, delta NUMERIC, atr NUMERIC,
                corr_es NUMERIC, shadow BOOLEAN);
            CREATE TABLE IF NOT EXISTS ordenes (
                id SERIAL PRIMARY KEY, decision_id INT, ts TIMESTAMPTZ DEFAULT NOW(),
                payload JSONB, respuesta TEXT, exito BOOLEAN);
            CREATE TABLE IF NOT EXISTS trades_live (
                id SERIAL PRIMARY KEY, decision_id INT,
                ts_entrada TIMESTAMPTZ, ts_salida TIMESTAMPTZ,
                precio_entrada NUMERIC, precio_salida NUMERIC, contratos INT,
                pnl NUMERIC, razon TEXT, shadow BOOLEAN);
            CREATE TABLE IF NOT EXISTS control_motor (
                id INT PRIMARY KEY DEFAULT 1, pausado BOOLEAN DEFAULT FALSE,
                ts_actualizado TIMESTAMPTZ DEFAULT NOW(),
                CONSTRAINT single_row CHECK (id = 1));
            INSERT INTO control_motor (id, pausado) VALUES (1, FALSE)
                ON CONFLICT (id) DO NOTHING;
            CREATE TABLE IF NOT EXISTS policy_audit (
                id SERIAL PRIMARY KEY, decision_id INT,
                ts TIMESTAMPTZ DEFAULT NOW(),
                entry_id TEXT, decision TEXT, rule_triggered TEXT, reason TEXT);
            """)
        self.conn.commit()
        log.info("DB lista")

    def get_pausado(self) -> bool:
        with self._lock:
            self._ensure()
            try:
                with self.conn.cursor() as cur:
                    cur.execute("SELECT pausado FROM control_motor WHERE id = 1")
                    row = cur.fetchone()
                    return bool(row[0]) if row else False
            except Exception as e:
                log.error(f"DB error (get_pausado): {e}")
                return False

    def set_pausado(self, valor: bool):
        with self._lock:
            self._ensure()
            try:
                with self.conn.cursor() as cur:
                    cur.execute(
                        "UPDATE control_motor SET pausado = %s, ts_actualizado = NOW() WHERE id = 1",
                        (valor,))
                self.conn.commit()
            except Exception as e:
                log.error(f"DB error (set_pausado): {e}")
                try:
                    self.conn.rollback()
                except Exception:
                    pass

    def decision(self, operar, razon, contratos=None, precio=None, sl=None, tp=None,
                 delta=None, atr=None, corr=None, shadow=True):
        with self._lock:
            self._ensure()
            try:
                with self.conn.cursor() as cur:
                    cur.execute("""INSERT INTO decisiones
                        (operar,razon,contratos,precio,sl,tp,delta,atr,corr_es,shadow)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                        (operar, razon, contratos, precio, sl, tp, delta, atr, corr, shadow))
                    i = cur.fetchone()[0]
                self.conn.commit()
                return i
            except Exception as e:
                log.error(f"DB error (decision): {e}")
                try:
                    self.conn.rollback()
                except Exception:
                    pass
                return None

    def orden(self, did, payload, resp, ok):
        with self._lock:
            self._ensure()
            try:
                with self.conn.cursor() as cur:
                    cur.execute("INSERT INTO ordenes (decision_id,payload,respuesta,exito) VALUES (%s,%s,%s,%s)",
                                (did, json.dumps(payload), str(resp)[:2000], ok))
                self.conn.commit()
            except Exception as e:
                log.error(f"DB error (orden): {e}")
                try:
                    self.conn.rollback()
                except Exception:
                    pass

    def policy_audit(self, did, entry_id, decision_str, rule_triggered, reason):
        with self._lock:
            self._ensure()
            try:
                with self.conn.cursor() as cur:
                    cur.execute("""INSERT INTO policy_audit
                        (decision_id, entry_id, decision, rule_triggered, reason)
                        VALUES (%s,%s,%s,%s,%s)""",
                        (did, entry_id, decision_str, rule_triggered, reason))
                self.conn.commit()
            except Exception as e:
                log.error(f"DB error (policy_audit): {e}")
                try:
                    self.conn.rollback()
                except Exception:
                    pass

    def trade(self, did, p, salida, ts_out, pnl, razon, shadow):
        with self._lock:
            self._ensure()
            try:
                with self.conn.cursor() as cur:
                    cur.execute("""INSERT INTO trades_live
                        (decision_id,ts_entrada,ts_salida,precio_entrada,precio_salida,contratos,pnl,razon,shadow)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (did, p.ts_entrada, ts_out, p.precio_entrada, salida, p.contratos, pnl, razon, shadow))
                self.conn.commit()
            except Exception as e:
                log.error(f"DB error (trade): {e}")
                try:
                    self.conn.rollback()
                except Exception:
                    pass


class CalendarioNoticias:
    """Bloquea entradas cerca de eventos USD de alto impacto (ForexFactory, gratis)."""
    def __init__(self):
        self.eventos = []
        self.ultima_carga = None

    def actualizar(self):
        hoy = datetime.now(UTC).date()
        if self.ultima_carga == hoy:
            return
        try:
            r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=10)
            data = r.json()
            self.eventos = [e for e in data if e.get("country") == "USD" and e.get("impact") == "High"]
            self.ultima_carga = hoy
            log.info(f"Calendario actualizado: {len(self.eventos)} eventos USD alto impacto")
        except Exception as e:
            log.error(f"No se pudo actualizar calendario: {e}")

    def hay_noticia_cerca(self, ts, minutos_ventana):
        self.actualizar()
        for e in self.eventos:
            try:
                ts_evento = datetime.fromisoformat(e["date"]).astimezone(UTC)
            except Exception:
                continue
            if abs((ts - ts_evento).total_seconds()) / 60 <= minutos_ventana:
                return e.get("title", "evento")
        return None

    def minutos_a_proxima_noticia(self, ts):
        self.actualizar()
        mejor = None
        for e in self.eventos:
            try:
                ts_evento = datetime.fromisoformat(e["date"]).astimezone(UTC)
            except Exception:
                continue
            minutos = (ts_evento - ts).total_seconds() / 60
            if minutos >= 0 and (mejor is None or minutos < mejor):
                mejor = minutos
        return mejor


class FRED:
    """Contexto macro gratuito -- no bloquea trading, solo referencia en logs."""
    def __init__(self, api_key):
        self.api_key = api_key
        self.cache = {}
        self.ultima_carga = None

    def actualizar(self):
        if not self.api_key:
            return
        hoy = datetime.now(UTC).date()
        if self.ultima_carga == hoy:
            return
        for serie in CFG["fred_series"]:
            try:
                r = requests.get(
                    "https://api.stlouisfed.org/fred/series/observations",
                    params={"series_id": serie, "api_key": self.api_key,
                            "file_type": "json", "sort_order": "desc", "limit": 1},
                    timeout=10
                )
                obs = r.json().get("observations", [])
                if obs:
                    self.cache[serie] = obs[0]["value"]
            except Exception as e:
                log.error(f"FRED error {serie}: {e}")
        self.ultima_carga = hoy
        log.info(f"FRED actualizado: {self.cache}")


class Ejecutor:
    def __init__(self, url, token, account_id, shadow, db, tg: Optional[Telegram] = None):
        self.url = url
        self.token = token
        self.account = account_id
        self.shadow = shadow
        self.db = db
        self.tg = tg or Telegram(None, None)
        self.authority_calibrator = AuthorityCalibrator(
            base_authority=CFG["riesgo_trade"],
            floor=CFG["riesgo_minimo_dolares"],
            capacity=CFG["capital"],
            kelly_fraction=CFG["kelly_fraccion"],
            min_sample_size=CFG["kelly_min_trades"],
            window=CFG["kelly_ventana"],
            ceiling_fraction=CFG["kelly_techo_riesgo_pct"],
            probe_fraction=CFG["riesgo_sonda_fraccion"],
        )
        self.policy_engine = PolicyEngine(
            CFG["policy_config_path"],
            authority_calibrator=self.authority_calibrator,
        )

    def _base(self, action, qty, price):
        return {
            "symbol": CFG["symbol_exec"],
            "strategy_name": "MOTOR_KMT_C4_B2_1v_60s_k1.5_SLRR3",
            "date": datetime.now(UTC).isoformat(),
            "data": action,
            "quantity": str(qty),
            "risk_percentage": 0,
            "price": str(round(price, 2)),
            "tp": 0, "percentage_tp": 0, "dollar_tp": 0,
            "sl": 0, "dollar_sl": 0, "percentage_sl": 0,
            "trail": 0, "trail_stop": 0, "trail_trigger": 0, "trail_freq": 0,
            "update_tp": False, "update_sl": False,
            "breakeven": 0, "breakeven_offset": 0,
            "token": self.token,
            "pyramid": False,
            "same_direction_ignore": True,
            "reverse_order_close": False,
            "multiple_accounts": [{
                "token": self.token, "account_id": self.account,
                "risk_percentage": 0, "quantity_multiplier": 1
            }]
        }

    def _post(self, did, payload):
        if self.shadow:
            log.info(f"[SHADOW] {payload['data']} {payload['quantity']}c @ {payload['price']} "
                      f"tp={payload['tp']} sl={payload['sl']} "
                      f"dollar_tp={payload['dollar_tp']} dollar_sl={payload['dollar_sl']}")
            self.db.orden(did, payload, "SHADOW", True)
            return True
        try:
            r = requests.post(self.url, json=payload, timeout=10)
            ok = 200 <= r.status_code < 300
            log.info(f"[LIVE] {r.status_code} {r.text[:200]}")
            self.db.orden(did, payload, r.text, ok)
            if not ok:
                self.tg.enviar(f"WARNING ORDEN RECHAZADA por PickMyTrade\nHTTP {r.status_code}\n{r.text[:200]}")
            return ok
        except Exception as e:
            log.error(f"[LIVE] fallo: {e}")
            self.db.orden(did, payload, str(e), False)
            self.tg.enviar(f"WARNING ORDEN FALLIDA (error de red hacia PickMyTrade)\n{e}")
            return False

    def entrar(self, did, qty, precio, sl, tp, atr, daily_pnl, weekly_pnl,
               open_contracts, minutes_to_next_news=None, outcome_history=None):
        audit_entry = self.policy_engine.evaluate(
            action_type=ActionType.OPEN_POSITION,
            proposed_params={"contracts": qty},
            market_context=MarketContext(
                timestamp=datetime.now(UTC), instrument=CFG["symbol_exec"],
                price=precio, atr=atr,
                minutes_to_next_news_event=minutes_to_next_news,
            ),
            risk_state=RiskState(
                daily_pnl=daily_pnl, weekly_pnl=weekly_pnl,
                open_contracts=open_contracts, account_equity=CFG["capital"],
            ),
            agent_id="MOTOR_KMT_C4_B2", account_id=self.account,
            outcome_history=outcome_history,
        )

        self.db.policy_audit(
            did, audit_entry.entry_id, audit_entry.decision.value,
            audit_entry.rule_triggered, audit_entry.reason,
        )

        if audit_entry.decision.value == "blocked":
            log.warning(f"[POLICY][{audit_entry.entry_id}] Orden BLOQUEADA: {audit_entry.reason}")
            self.tg.enviar(f"Orden bloqueada por policy engine: {audit_entry.reason}")
            return False

        if audit_entry.decision.value == "logged":
            log.info(f"[POLICY][{audit_entry.entry_id}] Aprobada con marca de revision: {audit_entry.reason}")

        sl_distancia_dolares = round(atr * CFG["sl_atr"] * CFG["usd_punto"], 2)
        tp_distancia_dolares = round(atr * CFG["tp_atr"] * CFG["usd_punto"], 2)

        p = self._base("buy", qty, precio)
        p["sl"] = 0
        p["dollar_sl"] = sl_distancia_dolares
        p["tp"] = 0
        p["dollar_tp"] = tp_distancia_dolares
        return self._post(did, p)

    def salir(self, did, qty, precio):
        return self._post(did, self._base("flat", qty, precio))

    def actualizar_stop(self, did, qty, precio_ref, nuevo_sl):
        """IMPORTANTE: se asume que 'data=buy' + 'update_sl' es el
        mecanismo para modificar el stop sin reabrir posicion -- esto
        DEBE confirmarse contra la documentacion real de PickMyTrade
        antes de confiar en produccion (mismo aviso pendiente que C4_B/C2)."""
        p = self._base("buy", qty, precio_ref)
        p["update_sl"] = True
        p["sl"] = round(nuevo_sl, 2)
        return self._post(did, p)


class Cerebro:
    """
    Misma señal 1v de C4_B (delta extremo dinamico, k=1.5) + cierre en
    la direccion de la señal -> SOLO CAMBIA: SL 0.60 ATR / TP 1.80 ATR
    (antes 0.35/1.75) -> Kelly con floor. SOLO LARGO, igual a C4_B.
    """
    def __init__(self, e, db, ex, tg: Optional[Telegram] = None):
        self.e, self.db, self.ex = e, db, ex
        self.calendario = CalendarioNoticias()
        self.tg = tg or Telegram(None, None)
        self._breaker_avisado_dia = None
        self._breaker_avisado_sem = None

    def _atr(self, df):
        n = CFG["atr_per"]
        if len(df) < n + 1:
            return np.nan
        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift()).abs()
        lc = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return float(tr.rolling(n).mean().iloc[-1])

    def _vol_ratio(self, df, atr_actual):
        n = min(CFG["regimen_vol_ventana"], len(df))
        if n < 50 or pd.isna(atr_actual):
            return 1.0
        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift()).abs()
        lc = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr_referencia = tr.rolling(CFG["atr_per"]).mean().tail(n).mean()
        if pd.isna(atr_referencia) or atr_referencia <= 0:
            return 1.0
        return float(atr_actual / atr_referencia)

    def _spread_por_sesion(self, hora_ny):
        if CFG["hora_ny_inicio"] <= hora_ny <= CFG["hora_ny_fin"]:
            return CFG["spread_puntos_ny"]
        return CFG["spread_puntos_overnight"]

    def _rechazo_volumen(self, df, atr_actual):
        n = CFG["rechazo_volumen_ventana"]
        if len(df) < n + 1 or pd.isna(atr_actual) or atr_actual <= 0:
            return False
        vol_media = df["volume"].tail(n).mean()
        if pd.isna(vol_media) or vol_media <= 0:
            return False
        v = df.iloc[-1]
        vol_relativo = v["volume"] / vol_media
        rango_relativo = (v["high"] - v["low"]) / atr_actual
        return bool(vol_relativo > CFG["rechazo_volumen_umbral"]
                    and rango_relativo < CFG["rechazo_volumen_rango_atr"])

    def _umbral_delta_reversion(self, df):
        n = CFG["delta_ventana"]
        if len(df) < n + 1:
            return np.nan
        m = df["delta"].rolling(n).mean().iloc[-1]
        s = df["delta"].rolling(n).std().iloc[-1]
        if pd.isna(m) or pd.isna(s):
            return np.nan
        return abs(m) + s

    def _probabilidad_reversion(self, df, atr_actual, direccion="largo"):
        p_reversion = 0.0
        v = df.iloc[-1]
        umbral = self._umbral_delta_reversion(df)
        delta = float(v["delta"])

        if pd.notna(umbral) and umbral > 0:
            if direccion == "largo" and delta < -umbral:
                p_reversion += 0.35
                if v["close"] < v["open"]:
                    p_reversion += 0.15

        if self._rechazo_volumen(df, atr_actual):
            p_reversion += 0.15

        return min(p_reversion, 1.0)

    def _senal_1v(self, df):
        """
        Estructura 1v -- IDENTICA a C4_B, sin tocar (esta variante solo
        prueba SL/TP, no la señal de entrada).
        """
        n = CFG["delta_ventana"]
        if len(df) < n + 2:
            return False
        m = df["delta"].rolling(n).mean().iloc[-1]
        s = df["delta"].rolling(n).std().iloc[-1]
        if pd.isna(m) or pd.isna(s):
            return False
        umbral = -(m + CFG["delta_std"] * s)
        v = df.iloc[-1]
        return bool(v["delta"] < umbral and v["close"] > v["open"])

    def _umbral_1v(self, df):
        n = CFG["delta_ventana"]
        if len(df) < n + 2:
            return float("nan")
        m = df["delta"].rolling(n).mean().iloc[-1]
        s = df["delta"].rolling(n).std().iloc[-1]
        if pd.isna(m) or pd.isna(s):
            return float("nan")
        return float(-(m + CFG["delta_std"] * s))

    def _en_mantenimiento(self, ts_ny_time):
        return CFG["mantenimiento_ini"] <= ts_ny_time < CFG["mantenimiento_fin"]

    def _corr(self):
        if not CFG["usar_corr_es"]:
            return None
        n = CFG["corr_ventana"]
        if len(self.e.velas) < n + 1 or len(self.e.velas_es) < n + 1:
            return None
        a = self.e.velas["close"].pct_change().tail(n).reset_index(drop=True)
        b = self.e.velas_es["close"].pct_change().tail(n).reset_index(drop=True)
        c = a.corr(b)
        return None if pd.isna(c) else float(c)

    def _actualizar_trailing(self, pos: Pos, precio_actual: float, atr_entrada: float):
        ganancia_puntos = precio_actual - pos.precio_entrada
        ganancia_en_atr = ganancia_puntos / atr_entrada if atr_entrada > 0 else 0

        if ganancia_en_atr >= CFG["bloqueo_75pct_a_atr"]:
            nuevo_stop = pos.precio_entrada + ganancia_puntos * 0.75
            nueva_etapa = "GANANCIA_FUERTE_75"
        elif ganancia_en_atr >= CFG["bloqueo_50pct_a_atr"]:
            nuevo_stop = pos.precio_entrada + ganancia_puntos * 0.50
            nueva_etapa = "GANANCIA_PARCIAL_50"
        elif ganancia_en_atr >= CFG["breakeven_a_atr"]:
            nuevo_stop = pos.precio_entrada
            nueva_etapa = "BREAKEVEN"
        else:
            return pos

        if nuevo_stop > pos.sl:
            ok = self.ex.actualizar_stop(pos.decision_id, pos.contratos, precio_actual, nuevo_stop)
            if ok:
                pos.sl = nuevo_stop
                pos.etapa_stop = nueva_etapa
            else:
                log.error(f"FALLO actualizar stop en broker a {nuevo_stop:.2f} -- "
                          f"se mantiene SL interno en {pos.sl:.2f}")
                self.tg.enviar(
                    f"No se pudo mover el stop en el broker a {nuevo_stop:.2f}.\n"
                    f"El SL real sigue en {pos.sl:.2f}. Revisar Tradovate manualmente."
                )
        return pos

    def _riesgo_kelly(self):
        riesgo = CFG["riesgo_trade"]
        piso = CFG["riesgo_minimo_dolares"]
        if len(self.e.trades) >= CFG["kelly_min_trades"]:
            u = self.e.trades[-CFG["kelly_ventana"]:]
            w = [t for t in u if t > 0]
            l = [t for t in u if t <= 0]
            if w and l:
                p = len(w) / len(u)
                b = np.mean(w) / abs(np.mean(l))
                if b > 0:
                    kelly = (p * b - (1 - p)) / b
                    kelly_ajustado = kelly * CFG["kelly_fraccion"]
                    if kelly_ajustado <= 0:
                        riesgo = max(CFG["riesgo_trade"] * CFG["riesgo_sonda_fraccion"], piso)
                    else:
                        riesgo = CFG["capital"] * min(kelly_ajustado, CFG["kelly_techo_riesgo_pct"])
                        riesgo = max(riesgo, piso)
        return max(riesgo, piso)

    def _contratos(self, stop_pts, riesgo, vol_ratio):
        if vol_ratio > CFG["vol_ratio_alto"]:
            riesgo = riesgo * CFG["vol_ratio_factor_reduccion"]
        denom = stop_pts * CFG["usd_punto"]
        if denom <= 0:
            return 0
        return max(0, min(int(riesgo / denom), CFG["max_contratos"]))

    def _breakers(self, ts):
        ny = ts.astimezone(NY)
        d, w = ny.date(), ny.isocalendar()[1]
        if self.e.dia != d:
            self.e.dia, self.e.pnl_dia = d, 0.0
        if self.e.sem != w:
            self.e.sem, self.e.pnl_sem = w, 0.0
        if self.e.pnl_dia <= -CFG["perdida_max_dia"]:
            if self._breaker_avisado_dia != d:
                self._breaker_avisado_dia = d
                self.tg.enviar(f"BREAKER DIARIO activado\nPnL del dia: ${self.e.pnl_dia:.2f}\nEntradas nuevas bloqueadas hasta manana.")
            return "BREAKER_DIARIO"
        if self.e.pnl_sem <= -CFG["perdida_max_sem"]:
            if self._breaker_avisado_sem != w:
                self._breaker_avisado_sem = w
                self.tg.enviar(f"BREAKER SEMANAL activado\nPnL de la semana: ${self.e.pnl_sem:.2f}\nEntradas nuevas bloqueadas hasta la proxima semana.")
            return "BREAKER_SEMANAL"
        return None

    def _reducir_posicion(self, p: Pos, fraccion: float, precio_actual: float, ts):
        contratos_a_cerrar = max(1, int(p.contratos * fraccion))
        contratos_a_cerrar = min(contratos_a_cerrar, p.contratos - 1)
        if contratos_a_cerrar <= 0:
            return
        hora_salida = ts.astimezone(NY).time()
        spread_aplicable = self._spread_por_sesion(hora_salida)
        costo_spread = spread_aplicable * CFG["usd_punto"] * contratos_a_cerrar * 0.5
        costo_comision = CFG["comision_por_contrato_rt"] * contratos_a_cerrar
        pnl_bruto = (precio_actual - p.precio_entrada) * CFG["usd_punto"] * contratos_a_cerrar
        pnl_parcial = pnl_bruto - costo_spread - costo_comision

        ok = self.ex.salir(p.decision_id, contratos_a_cerrar, precio_actual)
        if not ok:
            log.error(f"FALLO reduccion parcial en broker ({contratos_a_cerrar}c)")
            self.tg.enviar(
                f"REDUCCION PARCIAL RECHAZADA por broker ({contratos_a_cerrar}c). "
                f"La posicion sigue completa. Revisar manualmente."
            )
            return

        self.db.trade(p.decision_id, p, precio_actual, ts, pnl_parcial, "REVERSION_PARCIAL", self.ex.shadow)
        p.contratos -= contratos_a_cerrar
        self.e.pnl_dia += pnl_parcial
        self.e.pnl_sem += pnl_parcial
        self.e.trades.append(pnl_parcial)
        log.info(f"REDUCCION PARCIAL {contratos_a_cerrar}c pnl={pnl_parcial:.2f}")
        self.tg.enviar(f"REDUCCION PARCIAL {contratos_a_cerrar}c\nPnL parcial: ${pnl_parcial:.2f}")

    def _cerrar_total(self, p: Pos, precio_salida: float, ts, razon: str):
        ok = self.ex.salir(p.decision_id, p.contratos, precio_salida)
        if not ok:
            log.error(f"FALLO el cierre en broker (razon={razon}) -- posicion se mantiene ABIERTA internamente")
            self.tg.enviar(
                f"CIERRE RECHAZADO por broker ({razon}). "
                f"El motor sigue tratando la posicion como ABIERTA. Revisar manualmente si persiste."
            )
            return

        hora_salida = ts.astimezone(NY).time()
        spread_aplicable = self._spread_por_sesion(hora_salida)
        costo_spread = spread_aplicable * CFG["usd_punto"] * p.contratos * 0.5
        costo_comision = CFG["comision_por_contrato_rt"] * p.contratos
        pnl_bruto = (precio_salida - p.precio_entrada) * CFG["usd_punto"] * p.contratos
        pnl = pnl_bruto - costo_spread - costo_comision
        self.db.trade(p.decision_id, p, precio_salida, ts, pnl, razon, self.ex.shadow)
        self.e.pnl_dia += pnl
        self.e.pnl_sem += pnl
        self.e.trades.append(pnl)
        log.info(f"CIERRE {razon} pnl={pnl:.2f} dia={self.e.pnl_dia:.2f}")
        emoji = "OK" if pnl > 0 else ("=" if pnl == 0 else "X")
        modo = "SHADOW" if self.ex.shadow else "LIVE"
        self.tg.enviar(
            f"[{emoji}] CIERRE [{modo}] {razon}\n"
            f"PnL: ${pnl:.2f}\n"
            f"PnL del dia: ${self.e.pnl_dia:.2f}"
        )
        self.e.pos = None

    def _gestionar(self, v, ts):
        p = self.e.pos
        h = ts.astimezone(NY).time()

        if h >= CFG["hora_cierre"]:
            self._cerrar_total(p, float(v["close"]), ts, "EOD_SALVAGUARDA_OPERATIVA")
            return

        atr_actual = self._atr(self.e.velas)
        p = self._actualizar_trailing(p, float(v["close"]), p.atr)

        sl_tocado = v["low"] <= p.sl
        tp_tocado = v["high"] >= p.tp

        if sl_tocado:
            self._cerrar_total(p, p.sl, ts, "SL")
            return
        if tp_tocado:
            self._cerrar_total(p, p.tp, ts, "TP")
            return

        p_reversion = self._probabilidad_reversion(self.e.velas, atr_actual, "largo")

        if p_reversion >= CFG["reversion_cierre_total"]:
            self._cerrar_total(p, float(v["close"]), ts, "REVERSION_TOTAL")
            return

        if p.etapa_stop != "PROTECCION" and p.contratos > 1:
            if p_reversion >= CFG["reversion_reduccion_60pct"]:
                self._reducir_posicion(p, 0.60, float(v["close"]), ts)
            elif p_reversion >= CFG["reversion_reduccion_30pct"]:
                self._reducir_posicion(p, 0.30, float(v["close"]), ts)

    def evaluar(self, ts):
        if len(self.e.velas) < CFG["delta_ventana"] + 2:
            return
        v = self.e.velas.iloc[-1]

        if self.e.senal_pendiente is not None and self.e.pos is None:
            if self.e.pausado:
                log.info("Señal pendiente retenida: motor pausado (kill switch)")
                return

            ts_det = self.e.senal_pendiente.get("ts_detectada")
            if ts_det is not None and (ts - ts_det) > timedelta(seconds=CFG["tf_seg"] * 2):
                antiguedad_seg = (ts - ts_det).total_seconds()
                log.warning(f"Señal pendiente EXPIRADA ({antiguedad_seg:.0f} seg de antiguedad), descartada")
                self.db.decision(False, f"SENAL_EXPIRADA({antiguedad_seg:.0f}seg)", shadow=self.ex.shadow)
                self.tg.enviar(f"Señal pendiente expirada ({antiguedad_seg:.0f} seg sin ejecutarse) -- descartada por seguridad")
                self.e.senal_pendiente = None
                return

            sp = self.e.senal_pendiente
            self.e.senal_pendiente = None
            hora_ny_actual = ts.astimezone(NY).time()
            medio_spread = self._spread_por_sesion(hora_ny_actual) * 0.5

            if len(self.e.velas) >= 2 and sp["atr"] and sp["atr"] > 0:
                close_anterior = float(self.e.velas.iloc[-2]["close"])
                salto = abs(float(v["open"]) - close_anterior)
                salto_en_atr = salto / sp["atr"]
                if salto_en_atr > CFG["max_salto_atr_entrada"]:
                    log.warning(f"ENTRADA DESCARTADA: salto de {salto_en_atr:.1f}x ATR")
                    self.db.decision(False, f"SALTO_ANOMALO({salto_en_atr:.1f}xATR)",
                                     precio=float(v["open"]), atr=sp["atr"],
                                     delta=sp["delta"], shadow=self.ex.shadow)
                    self.tg.enviar(
                        f"Entrada DESCARTADA por salto de precio anomalo: "
                        f"{salto_en_atr:.1f}x ATR. Posible tick erroneo o gap extremo."
                    )
                    return

            precio_entrada = float(v["open"]) + medio_spread
            sl = precio_entrada - sp["atr"] * CFG["sl_atr"]
            tp = precio_entrada + sp["atr"] * CFG["tp_atr"]
            did = self.db.decision(True, "SENAL_1V", sp["qty"], precio_entrada, sl, tp,
                                    sp["delta"], sp["atr"], sp["corr"], self.ex.shadow)

            open_contratos_actual = self.e.pos.contratos if self.e.pos else 0
            minutos_noticia = self.calendario.minutos_a_proxima_noticia(ts)

            if self.ex.entrar(did, sp["qty"], precio_entrada, sl, tp,
                               atr=sp["atr"], daily_pnl=self.e.pnl_dia,
                               weekly_pnl=self.e.pnl_sem,
                               open_contracts=open_contratos_actual,
                               minutes_to_next_news=minutos_noticia,
                               outcome_history=self.e.trades):
                self.e.pos = Pos(precio_entrada, sp["qty"], sl, tp, sp["atr"], ts, sp["delta"], did)
                sl_dist_usd = round(sp["atr"] * CFG["sl_atr"] * CFG["usd_punto"], 2)
                tp_dist_usd = round(sp["atr"] * CFG["tp_atr"] * CFG["usd_punto"], 2)
                log.info(f"ENTRADA {sp['qty']}c @ {precio_entrada:.2f} SL={sl:.2f} TP={tp:.2f} "
                          f"(dollar_sl=${sl_dist_usd} dollar_tp=${tp_dist_usd} desde fill real) "
                          f"delta={sp['delta']:.0f} atr={sp['atr']:.2f} riesgo=${sp['riesgo']:.0f}")
                modo = "SHADOW" if self.ex.shadow else "LIVE"
                self.tg.enviar(
                    f"ENTRADA [{modo}] [C4_B2]\n"
                    f"{sp['qty']} contrato(s) @ {precio_entrada:.2f}\n"
                    f"SL ref={sl:.2f}  TP ref={tp:.2f}\n"
                    f"Enviado al broker: dollar_sl=${sl_dist_usd}  dollar_tp=${tp_dist_usd} (desde fill real)\n"
                    f"delta={sp['delta']:.0f}  atr={sp['atr']:.2f}  riesgo=${sp['riesgo']:.0f}"
                )
            return

        if self.e.pos:
            self._gestionar(v, ts)
            return

        if self.e.pausado:
            return

        br = self._breakers(ts)
        if br:
            self.db.decision(False, br, shadow=self.ex.shadow)
            return

        ny = ts.astimezone(NY)
        if self._en_mantenimiento(ny.time()):
            return
        if not (CFG["hora_ini"] <= ny.time() <= CFG["hora_fin_entradas"]):
            return

        titulo_noticia = self.calendario.hay_noticia_cerca(ts, CFG["news_blackout_min"])
        if titulo_noticia:
            self.db.decision(False, f"NOTICIA_ALTA({titulo_noticia})", shadow=self.ex.shadow)
            return

        if not self._senal_1v(self.e.velas):
            self.db.decision(False, "SIN_SENAL_1V", precio=float(v["close"]),
                             delta=float(v["delta"]),
                             corr=self._umbral_1v(self.e.velas),
                             shadow=self.ex.shadow)
            return

        atr = self._atr(self.e.velas)
        if pd.isna(atr) or atr <= 0:
            self.db.decision(False, "ATR_INVALIDO", precio=float(v["close"]),
                             delta=float(v["delta"]), shadow=self.ex.shadow)
            return

        vol_ratio_senal = self._vol_ratio(self.e.velas, atr)
        if vol_ratio_senal > CFG["atr_regimen_max"]:
            self.db.decision(False, f"REGIMEN_CALIENTE(vol_ratio={vol_ratio_senal:.2f})",
                             precio=float(v["close"]), atr=atr,
                             delta=float(v["delta"]), shadow=self.ex.shadow)
            return

        corr = self._corr()
        if CFG["usar_corr_es"] and corr is not None and corr < CFG["corr_min"]:
            self.db.decision(False, f"CORR_BAJA({corr:.2f})", atr=atr, corr=corr, shadow=self.ex.shadow)
            return

        delta = float(v["delta"])
        stop_pts = atr * CFG["sl_atr"]
        riesgo = self._riesgo_kelly()
        vol_ratio = self._vol_ratio(self.e.velas, atr)
        qty = self._contratos(stop_pts, riesgo, vol_ratio)
        if qty <= 0:
            self.db.decision(False, "SIZING_CERO", precio=float(v["close"]), delta=delta, atr=atr,
                              shadow=self.ex.shadow)
            return

        self.e.senal_pendiente = {"atr": atr, "delta": delta, "qty": qty, "riesgo": riesgo,
                                   "corr": corr, "ts_detectada": ts}
        log.info(f"SEÑAL 1V DETECTADA (pendiente de ejecutar en la proxima vela) | "
                  f"delta={delta:.0f} atr={atr:.2f} qty={qty}")


class Feed:
    """
    Igual arquitectura que C4_B (productor-consumidor, resolucion de
    simbolo por clase de mensaje), bucketing de velas en SEGUNDOS
    (tf_seg=60) -- ver tick()/_cerrar().
    """
    def __init__(self, key, e, cerebro, tg: Optional["Telegram"] = None):
        self.key, self.e, self.cerebro = key, e, cerebro
        self.tg = tg or Telegram(None, None)
        self.buf = {"mnq": [], "es": []}
        self.vela_ini = None
        self.map_instrumento = {}
        self._cola = queue.Queue(maxsize=50000)
        self._hilo_consumidor = None

    def _asegurar_consumidor(self):
        if self._hilo_consumidor is None or not self._hilo_consumidor.is_alive():
            self._hilo_consumidor = threading.Thread(target=self._consumidor, daemon=True)
            self._hilo_consumidor.start()
            log.info("Hilo consumidor iniciado")

    def _cerrar(self, ts):
        for k, b in self.buf.items():
            if not b:
                continue
            pr = [t["p"] for t in b]
            compras = sum(t["s"] for t in b if t["a"])
            ventas = sum(t["s"] for t in b if not t["a"])
            fila = {"ts": self.vela_ini, "open": pr[0], "high": max(pr), "low": min(pr),
                    "close": pr[-1], "volume": sum(t["s"] for t in b), "delta": compras - ventas}
            df = self.e.velas if k == "mnq" else self.e.velas_es
            nuevo = pd.concat([df, pd.DataFrame([fila])], ignore_index=True).tail(300).reset_index(drop=True)
            if k == "mnq":
                self.e.velas = nuevo
            else:
                self.e.velas_es = nuevo
            self.buf[k] = []
        if len(self.e.velas):
            self.cerebro.evaluar(ts)

    def _perder_buffer_parcial(self, motivo: str):
        perdidos = {k: len(b) for k, b in self.buf.items() if b}
        if perdidos:
            log.warning(f"Buffer parcial descartado por {motivo}: {perdidos}")
            self.tg.enviar(f"Reconexion del feed: se perdieron ticks parciales de vela en curso: {perdidos}")
        self.buf = {"mnq": [], "es": []}
        self.vela_ini = None

    def tick(self, k, ts, price, size, is_buy):
        tf_seg = CFG["tf_seg"]
        epoch_seg = int(ts.timestamp())
        ini_epoch = epoch_seg - (epoch_seg % tf_seg)
        ini = datetime.fromtimestamp(ini_epoch, tz=UTC)
        if self.vela_ini is None:
            self.vela_ini = ini
        elif ini > self.vela_ini:
            self._cerrar(ini)
            self.vela_ini = ini
        self.buf[k].append({"p": price, "s": size, "a": is_buy})

    def _instrument_id(self, r):
        hd = getattr(r, "hd", None)
        if hd is not None and hasattr(hd, "instrument_id"):
            return hd.instrument_id
        return getattr(r, "instrument_id", None)

    def _consumidor(self):
        avisado_cola_llena = False
        while True:
            try:
                r = self._cola.get(timeout=1)
            except queue.Empty:
                continue
            try:
                tipo = type(r).__name__

                if tipo == "SymbolMappingMsg":
                    out_sym = getattr(r, "stype_out_symbol", None)
                    iid = self._instrument_id(r)
                    if out_sym and iid is not None:
                        su = out_sym.upper()
                        if "MNQ" in su:
                            self.map_instrumento[iid] = "mnq"
                            log.info(f"Mapeo confirmado: instrument_id={iid} -> MNQ ({out_sym})")
                        elif "ES" in su:
                            self.map_instrumento[iid] = "es"
                            log.info(f"Mapeo confirmado: instrument_id={iid} -> ES ({out_sym})")
                    continue

                if tipo != "TradeMsg":
                    continue

                iid = self._instrument_id(r)
                if iid is None:
                    continue

                k = self.map_instrumento.get(iid)
                if k is None:
                    log.warning(f"Tick sin simbolo resoluble aun (instrument_id={iid}), descartado")
                    continue

                ts = datetime.fromtimestamp(r.ts_event / 1e9, tz=UTC)
                is_buy = getattr(r, "side", "A") == "B"
                self.tick(k, ts, r.price / 1e9, r.size, is_buy)
            except Exception as ex:
                log.error(f"consumidor err: {ex}")

            qsize = self._cola.qsize()
            if qsize > 5000 and not avisado_cola_llena:
                avisado_cola_llena = True
                log.warning(f"Cola de procesamiento acumulando: {qsize} pendientes")
                self.tg.enviar(f"Cola de ticks acumulando ({qsize} pendientes) -- el consumidor va atrasado")
            elif qsize < 500:
                avisado_cola_llena = False

    def correr(self):
        import databento as dbn
        self._perder_buffer_parcial("reconexion o arranque")
        self._asegurar_consumidor()

        log.info("Conectando Databento live...")
        cli = dbn.Live(key=self.key)
        cli.subscribe(dataset=CFG["dataset"], schema="trades",
                      symbols=[CFG["symbol_db"], CFG["symbol_es"]], stype_in="continuous")
        log.info("Feed OK")
        for r in cli:
            try:
                self._cola.put_nowait(r)
            except queue.Full:
                try:
                    self._cola.get_nowait()
                except queue.Empty:
                    pass
                self._cola.put_nowait(r)


def kill_switch_listener(e: Estado, db: DB, tg: Telegram):
    if not tg.activo:
        return
    offset = None
    while True:
        try:
            url = f"https://api.telegram.org/bot{tg.token}/getUpdates"
            params = {"timeout": 25}
            if offset is not None:
                params["offset"] = offset
            r = requests.get(url, params=params, timeout=30)
            data = r.json()
            for update in data.get("result", []):
                offset = update["update_id"] + 1
                msg = update.get("message", {})
                texto = (msg.get("text") or "").strip().lower()
                chat_id_msg = str(msg.get("chat", {}).get("id", ""))
                if chat_id_msg != str(tg.chat_id):
                    continue

                if texto == "/pausar":
                    e.pausado = True
                    db.set_pausado(True)
                    tg.enviar("Motor PAUSADO. No se abriran entradas nuevas.\n"
                              "Una posicion abierta (si la hay) se sigue gestionando normalmente.\n"
                              "Envia /reanudar para reactivar.")
                    log.info("Kill switch: PAUSADO por comando de Telegram")
                elif texto == "/reanudar":
                    e.pausado = False
                    db.set_pausado(False)
                    tg.enviar("Motor REANUDADO. Volviendo a evaluar señales nuevas.")
                    log.info("Kill switch: REANUDADO por comando de Telegram")
                elif texto == "/estado":
                    estado_txt = "PAUSADO" if e.pausado else "ACTIVO"
                    pos_txt = "Sin posicion abierta" if e.pos is None else f"Posicion abierta: {e.pos.contratos}c @ {e.pos.precio_entrada:.2f}"
                    tg.enviar(f"[C4_B2] {estado_txt}\n{pos_txt}\nPnL del dia: ${e.pnl_dia:.2f}")
        except Exception as ex:
            log.error(f"kill_switch_listener error (no critico): {ex}")
            time.sleep(5)


def watchdog(e: Estado, tg: Telegram, umbral_min: int = 15):
    avisado = False
    ultimo_ts_vela = None
    ultima_vez_cambio = time.time()

    while True:
        time.sleep(300)
        try:
            ahora = datetime.now(UTC)
            ny = ahora.astimezone(NY)
            if CFG["mantenimiento_ini"] <= ny.time() < CFG["mantenimiento_fin"]:
                avisado = False
                continue

            ts_actual = None
            if len(e.velas):
                ts_actual = e.velas.iloc[-1]["ts"]

            if ts_actual is not None and ts_actual != ultimo_ts_vela:
                ultimo_ts_vela = ts_actual
                ultima_vez_cambio = time.time()
                avisado = False
                continue

            minutos_silencio = (time.time() - ultima_vez_cambio) / 60
            if minutos_silencio >= umbral_min and not avisado:
                avisado = True
                tg.enviar(
                    f"WATCHDOG [C4_B2]: sin velas nuevas hace ~{int(minutos_silencio)} min\n"
                    f"El proceso sigue vivo pero el feed puede haberse quedado mudo. Revisar Railway."
                )
                log.warning(f"WATCHDOG: sin velas nuevas hace {int(minutos_silencio)} min")
        except Exception as ex:
            log.error(f"watchdog error: {ex}")


def main():
    db_url = os.environ["DATABASE_URL"]
    shadow = os.environ.get("MODO_SHADOW", "true").lower() != "false"
    log.info("=" * 60)
    log.info(f"MOTOR KMT -- NODO C4_B2 (1v solo-largo, k=1.5, 60s, SL/RR reoptimizado) | "
             f"{'SHADOW' if shadow else 'LIVE'} | capital={CFG['capital']}")
    log.info(f"Sin Volume Profile (mismo criterio que C4_B)")
    log.info(f"delta_std={CFG['delta_std']} | atr_regimen_max={CFG['atr_regimen_max']}")
    log.info(f"SL {CFG['sl_atr']}ATR / TP {CFG['tp_atr']}ATR (RR={CFG['tp_atr']/CFG['sl_atr']:.1f}:1) "
             f"-- antes en C4_B: SL 0.35ATR / TP 1.75ATR (RR=5.0:1)")
    log.info(f"Ventana: {CFG['hora_ini']}-{CFG['hora_fin_entradas']} NY (abierta, sin restriccion horaria)")
    log.info(f"riesgo_trade=${CFG['riesgo_trade']} riesgo_minimo=${CFG['riesgo_minimo_dolares']} "
             f"max_contratos={CFG['max_contratos']}")
    log.info(f"perdida_max_dia=${CFG['perdida_max_dia']} perdida_max_sem=${CFG['perdida_max_sem']}")
    log.info(f"Policy engine (RiskAval): {CFG['policy_config_path']}")
    log.info("=" * 60)

    tg = Telegram(os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID"))

    db = DB(db_url)
    e = Estado()
    e.pausado = db.get_pausado()
    ex = Ejecutor(os.environ.get("PICKMYTRADE_WEBHOOK", ""),
                  os.environ.get("PICKMYTRADE_TOKEN", ""),
                  os.environ.get("PICKMYTRADE_ACCOUNT", ""),
                  shadow, db, tg)
    cerebro = Cerebro(e, db, ex, tg)

    fred = FRED(os.environ.get("FRED_API_KEY"))
    fred.actualizar()

    hilo_watchdog = threading.Thread(target=watchdog, args=(e, tg), daemon=True)
    hilo_watchdog.start()

    hilo_kill_switch = threading.Thread(target=kill_switch_listener, args=(e, db, tg), daemon=True)
    hilo_kill_switch.start()

    estado_inicial = "PAUSADO (recuperado)" if e.pausado else "ACTIVO"
    tg.enviar(f"Motor KMT [C4_B2, 1v solo-largo, k=1.5, 60s, SL=0.60/TP=1.80 RR3:1] iniciado | "
              f"{'SHADOW' if shadow else 'LIVE'} | capital=${CFG['capital']} | {estado_inicial}\n"
              f"Comandos: /pausar /reanudar /estado")

    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        log.warning("Sin DATABENTO_API_KEY - standby")
        while True:
            time.sleep(60)

    feed = Feed(key, e, cerebro, tg)
    while True:
        try:
            feed.correr()
        except Exception as ex2:
            log.error(f"Feed caido: {ex2}. Retry 30s")
            tg.enviar(f"Feed de Databento caido: {str(ex2)[:200]}\nReintentando en 30s...")
            time.sleep(30)


if __name__ == "__main__":
    main()