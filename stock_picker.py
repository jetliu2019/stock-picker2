import os
import re
import time
import random
import numpy as np
import pandas as pd
import requests
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')


# ================================================================
#                    PushPlus 推送模块
# ================================================================

PUSHPLUS_TOKEN ='70a87015756f483ab09f70a5ebe5d6ff'


def send_pushplus(title, content, template='html'):

    url = 'https://www.pushplus.plus/send'
    data = {
        'token': PUSHPLUS_TOKEN,
        'title': title,
        'content': content,
        'template': template,
    }
    try:
        resp = requests.post(url, json=data, timeout=15)
        result = resp.json()
        if result.get('code') == 200:
            print(f"✅ PushPlus 推送成功: {title}")
            return True
        else:
            print(f"❌ PushPlus 推送失败: {result.get('msg', result)}")
            return False
    except Exception as e:
        print(f"❌ PushPlus 推送异常: {e}")
        return False


# ================================================================
#                   第一部分: 基础工具函数
# ================================================================

def TDX_SMA(series, n, m):
    arr = np.zeros(len(series))
    arr[0] = series.iloc[0] if pd.notna(series.iloc[0]) else 0
    for i in range(1, len(series)):
        xi = series.iloc[i] if pd.notna(series.iloc[i]) else 0
        arr[i] = (xi * m + arr[i - 1] * (n - m)) / n
    return pd.Series(arr, index=series.index)


def EMA(series, n):
    return series.ewm(span=n, adjust=False).mean()


def MA(series, n):
    return series.rolling(window=n, min_periods=1).mean()


def LLV(series, n):
    return series.rolling(window=n, min_periods=1).min()


def HHV(series, n):
    return series.rolling(window=n, min_periods=1).max()


def REF(series, n):
    return series.shift(n)


def AVEDEV(series, n):
    return series.rolling(window=n, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )


def CROSS(a, b):
    if isinstance(b, (int, float)):
        b = pd.Series(b, index=a.index)
    if isinstance(a, (int, float)):
        a = pd.Series(a, index=b.index)
    return ((a > b) & (a.shift(1) <= b.shift(1))).fillna(False)


def COUNT(cond, n):
    return cond.astype(int).rolling(window=n, min_periods=1).sum()


def FILTER(cond, n):
    result = pd.Series(False, index=cond.index)
    last_pos = -n - 1
    for i in range(len(cond)):
        if bool(cond.iloc[i]) and (i - last_pos) > n:
            result.iloc[i] = True
            last_pos = i
    return result


def BARSCOUNT(series):
    return pd.Series(range(len(series)), index=series.index, dtype=float)


def SAFE_DIV(a, b):
    b_safe = b.replace(0, np.nan)
    return a / b_safe


def WINNER_APPROX(close, vol, lookback=250):
    result = pd.Series(0.5, index=close.index)
    c_vals = close.values
    v_vals = vol.values
    for i in range(len(close)):
        start = max(0, i - lookback)
        hc = c_vals[start:i + 1]
        hv = v_vals[start:i + 1]
        total = hv.sum()
        if total > 0:
            result.iloc[i] = (hv[hc <= c_vals[i]]).sum() / total
    return result


# ================================================================
#                第二部分: ZIG / PEAKBARS / TROUGHBARS
# ================================================================

def ZIG(price, pct):
    n = len(price)
    if n < 3:
        return price.copy()
    vals = price.values.astype(float)
    pivots = {0: vals[0]}
    trend = 0
    hi, hi_i = vals[0], 0
    lo, lo_i = vals[0], 0
    for i in range(1, n):
        if trend == 0:
            if vals[i] > hi:
                hi, hi_i = vals[i], i
            if vals[i] < lo:
                lo, lo_i = vals[i], i
            if lo > 0 and (hi - lo) / lo * 100 >= pct:
                if hi_i > lo_i:
                    pivots[lo_i] = lo
                    trend = 1
                    hi, hi_i = vals[i], i
                else:
                    pivots[hi_i] = hi
                    trend = -1
                    lo, lo_i = vals[i], i
        elif trend == 1:
            if vals[i] > hi:
                hi, hi_i = vals[i], i
            if hi > 0 and (hi - vals[i]) / hi * 100 >= pct:
                pivots[hi_i] = hi
                trend = -1
                lo, lo_i = vals[i], i
        elif trend == -1:
            if vals[i] < lo:
                lo, lo_i = vals[i], i
            if lo > 0 and (vals[i] - lo) / lo * 100 >= pct:
                pivots[lo_i] = lo
                trend = 1
                hi, hi_i = vals[i], i
    if trend == 1:
        pivots[hi_i] = hi
    elif trend == -1:
        pivots[lo_i] = lo
    pivots[n - 1] = vals[n - 1]
    result = pd.Series(np.nan, index=price.index)
    for idx, val in pivots.items():
        result.iloc[idx] = val
    result = result.interpolate()
    return result


def PEAKBARS(close, k, n_pct, nth):
    z = ZIG(close, n_pct)
    zv = z.values
    n = len(zv)
    peaks = []
    for i in range(1, n - 1):
        if zv[i] >= zv[i - 1] and zv[i] > zv[i + 1]:
            peaks.append(i)
    result = pd.Series(9999.0, index=close.index)
    for i in range(n):
        past = [p for p in peaks if p <= i]
        if len(past) >= nth:
            result.iloc[i] = i - past[-nth]
    return result


def TROUGHBARS(close, k, n_pct, nth):
    z = ZIG(close, n_pct)
    zv = z.values
    n = len(zv)
    troughs = []
    for i in range(1, n - 1):
        if zv[i] <= zv[i - 1] and zv[i] < zv[i + 1]:
            troughs.append(i)
    result = pd.Series(9999.0, index=close.index)
    for i in range(n):
        past = [t for t in troughs if t <= i]
        if len(past) >= nth:
            result.iloc[i] = i - past[-nth]
    return result


# ================================================================
#                 第三部分: 核心信号计算
# ================================================================

def calculate_signals(df, index_df, capital):
    C = df['close'].astype(float)
    O = df['open'].astype(float)
    H = df['high'].astype(float)
    L = df['low'].astype(float)
    VOL = df['volume'].astype(float)

    INDEXC = index_df['close'].astype(float)
    INDEXH = index_df['high'].astype(float)
    INDEXL = index_df['low'].astype(float)

    signals = {}

    winner_c = WINNER_APPROX(C, VOL)
    signals['大机构仓位'] = 100 * (1 - winner_c)

    var5 = LLV(L, 120)
    var6 = HHV(H, 120)
    var7 = (var6 - var5) / 100.0
    var8 = TDX_SMA(SAFE_DIV(C - var5, var7), 20, 1)
    var9 = TDX_SMA(SAFE_DIV(O - var5, var7), 20, 1)
    varA = 3 * var8 - 2 * TDX_SMA(var8, 10, 1)
    varB = 3 * var9 - 2 * TDX_SMA(var9, 10, 1)
    signals['基金私募仓位'] = 100 - varA

    vare1 = REF(L, 1) * 0.9
    varf1 = L * 0.9
    var101 = (varf1 * VOL + vare1 * (capital - VOL)) / capital
    var111 = EMA(var101, 30)
    var121 = C - REF(C, 1)
    var131 = var121.clip(lower=0)
    var141 = var121.abs()
    var151 = SAFE_DIV(TDX_SMA(var131, 7, 1), TDX_SMA(var141, 7, 1)) * 100
    var161 = SAFE_DIV(TDX_SMA(var131, 13, 1), TDX_SMA(var141, 13, 1)) * 100
    var171 = BARSCOUNT(C)
    var181 = SAFE_DIV(TDX_SMA(var121.clip(lower=0), 6, 1), TDX_SMA(var121.abs(), 6, 1)) * 100
    hhv60 = HHV(H, 60)
    llv60 = LLV(L, 60)
    var191 = (-200) * SAFE_DIV(hhv60 - C, hhv60 - llv60) + 100
    var1a1 = SAFE_DIV(C - LLV(L, 15), HHV(H, 15) - LLV(L, 15)) * 100
    var1b1 = TDX_SMA((TDX_SMA(var1a1, 4, 1) - 50) * 2, 3, 1)
    var1c1 = SAFE_DIV(INDEXC - LLV(INDEXL, 14), HHV(INDEXH, 14) - LLV(INDEXL, 14)) * 100
    var1d = TDX_SMA(var1c1, 4, 1)
    var1e = TDX_SMA(var1d, 3, 1)
    var1f = SAFE_DIV(HHV(H, 30) - C, C) * 100
    var20 = (
        (var181 <= 25) & (var191 < -95) & (var1f > 20) &
        (var1b1 < -30) & (var1e < 30) &
        ((var111 - C) >= -0.25) & (var151 < 22) &
        (var161 < 28) & (var171 > 50)
    )
    signals['超级主力建仓'] = CROSS(var20.astype(float), 0.5) & (COUNT(var20, 10) == 1)

    hhv34 = HHV(H, 34)
    llv34_l = LLV(L, 34)
    llv34_c = LLV(C, 34)
    varE_val = MA(100 * SAFE_DIV(C - llv34_c, hhv34 - llv34_l), 5) - 20
    hhv75 = HHV(H, 75)
    llv75 = LLV(L, 75)
    c_r75 = SAFE_DIV(C - llv75, hhv75 - llv75) * 100
    o_r75 = SAFE_DIV(O - llv75, hhv75 - llv75) * 100
    sma_c75 = TDX_SMA(c_r75, 20, 1)
    varF_val = 100 - 3 * sma_c75 + 2 * TDX_SMA(sma_c75, 15, 1)
    sma_o75 = TDX_SMA(o_r75, 20, 1)
    var10_val = 100 - 3 * sma_o75 + 2 * TDX_SMA(sma_o75, 15, 1)
    var11_sig = (varF_val < REF(var10_val, 1)) & (VOL > REF(VOL, 1)) & (C > REF(C, 1))
    signals['大资金进场'] = var11_sig & (COUNT(var11_sig, 30) == 1)

    V1 = (C * 2 + H + L) / 4 * 10
    V2 = EMA(V1, 13) - EMA(V1, 34)
    V3 = EMA(V2, 5)
    V4 = 2 * (V2 - V3) * 5.5
    hhv_ix8 = HHV(INDEXH, 8)
    llv_ix8 = LLV(INDEXL, 8)
    diff_ix8 = (hhv_ix8 - llv_ix8).replace(0, np.nan)
    V5 = (hhv_ix8 - INDEXC) / diff_ix8 * 8
    V8 = (INDEXC * 2 + INDEXH + INDEXL) / 4
    V91 = EMA(V8, 13) - EMA(V8, 34)
    VA_v = EMA(V91, 3)
    VB_v = (V91 - VA_v) / 2
    hhv55 = HHV(H, 55)
    llv55 = LLV(L, 55)
    c_r55 = SAFE_DIV(C - llv55, hhv55 - llv55) * 100
    sma_c55 = TDX_SMA(c_r55, 5, 1)
    V111 = 3 * sma_c55 - 2 * TDX_SMA(sma_c55, 3, 1)
    ema_v111 = EMA(V111, 3)
    ref_ema_v111 = REF(ema_v111, 1).replace(0, np.nan)
    V121_v = SAFE_DIV(ema_v111 - ref_ema_v111, ref_ema_v111) * 100

    cash_cond = ema_v111 <= 13
    signals['准备现金'] = cash_cond & FILTER(cash_cond, 15)
    buy_cond = (ema_v111 <= 13) & (V121_v > 13)
    signals['主力进'] = buy_cond & FILTER(buy_cond, 10)
    signals['卖临界'] = (ema_v111 > 60) & (ema_v111 > REF(ema_v111, 1))
    cc_cond = (ema_v111 >= 90) & (V121_v != 0)
    signals['主力减仓'] = cc_cond & FILTER(cc_cond, 10)
    dd_cond = (ema_v111 >= 120) & (V121_v != 0)
    signals['主力大幅减仓'] = dd_cond & FILTER(dd_cond, 10)

    pb = PEAKBARS(C, 3, 15, 1)
    head_val = pd.Series(np.where(pb < 10, 100.0, 0.0), index=C.index)
    signals['主力清仓'] = (head_val > REF(head_val, 1)).fillna(False)
    tb = TROUGHBARS(C, 3, 15, 1)
    bottom_val = pd.Series(np.where(tb < 10, 50.0, 0.0), index=C.index)
    signals['主力抄底'] = (bottom_val > REF(bottom_val, 1)).fillna(False)

    AA1 = (O + H + L + C) / 4
    a1 = HHV(AA1, 10)
    a2 = LLV(AA1, 30)
    A4 = EMA(SAFE_DIV(AA1 - a2, a1 - a2), 1) * 100
    b1 = HHV(AA1, 16)
    b2 = LLV(AA1, 90)
    B4 = EMA(SAFE_DIV(AA1 - b2, b1 - b2), 1) * 100
    c1_ = HHV(AA1, 30)
    c2_ = LLV(AA1, 240)
    C4 = EMA(SAFE_DIV(AA1 - c2_, c1_ - c2_), 1) * 100
    signals['中线趋势'] = B4
    signals['拉升'] = (A4 > 90) & (B4 > 70)

    zig_val = ZIG(C, 10)
    ma_zig = MA(zig_val, 2)
    signals['上涨确立'] = CROSS(zig_val, ma_zig)
    signals['下跌确立'] = CROSS(ma_zig, zig_val)

    var1_dk = (2 * C + H + L + O) / 5
    var2_dk = LLV(L, 34)
    var3_dk = HHV(H, 34)
    duo = EMA(SAFE_DIV(var1_dk - var2_dk, var3_dk - var2_dk) * 100, 13)
    kong = EMA(duo, 3)
    signals['多'] = duo
    signals['空'] = kong
    signals['多头趋势'] = duo > kong

    var12_tp = (H + L + C) / 3
    ad14 = AVEDEV(var12_tp, 14).replace(0, np.nan)
    ad70 = AVEDEV(var12_tp, 70).replace(0, np.nan)
    var13_cci = (var12_tp - MA(var12_tp, 14)) / (0.015 * ad14)
    var14_cci = (var12_tp - MA(var12_tp, 70)) / (0.015 * ad70)
    var15_s = pd.Series(np.where(
        (var13_cci >= 150) & (var13_cci < 200) & (var14_cci >= 150) & (var14_cci < 200),
        10, 0), index=C.index, dtype=float)
    var16_s = pd.Series(np.where(
        (var13_cci <= -150) & (var13_cci > -200) & (var14_cci <= -150) & (var14_cci > -200),
        -10, var15_s), index=C.index, dtype=float)
    var17_dev = SAFE_DIV(C - MA(C, 13), MA(C, 13)) * 100
    var18_abs = 100 - var17_dev.abs()
    var19_v = pd.Series(np.where(var18_abs < 90, var18_abs, 100), index=C.index, dtype=float)
    var1a_v = varE_val.clip(lower=0)
    var1b_v = pd.Series(np.where(
        (var14_cci >= 200) & (var13_cci >= 150), 15,
        np.where((var14_cci <= -200) & (var13_cci <= -150), -15, var16_s)
    ), index=C.index, dtype=float) + 60
    var1c_cond = (var1a_v > 48) & (var1b_v > 60) & (var19_v < 100)
    signals['大资金出货'] = var1c_cond & (COUNT(var1c_cond, 30) == 1)

    return signals


# ================================================================
#                 第四部分: 选股筛选函数
# ================================================================

def screen_single_stock(df, index_df, capital):
    signals = calculate_signals(df, index_df, capital)
    i = -1
    last_date = df.index[i]
    last_open = float(df['open'].iloc[i])
    last_close = float(df['close'].iloc[i])

    buy = []
    sell = []

    # ====== 只保留"上涨确立"信号 ======
    if signals['上涨确立'].iloc[i]:
        buy.append('上涨确立')

    return {
        '日期': last_date,
        '开盘价': last_open,
        '收盘价': last_close,
        '买入信号': buy,
        '卖出信号': sell,
        '大机构仓位': round(float(signals['大机构仓位'].iloc[i]), 2),
        '基金私募仓位': round(float(signals['基金私募仓位'].iloc[i]), 2),
        '多线': round(float(signals['多'].iloc[i]), 2),
        '空线': round(float(signals['空'].iloc[i]), 2),
        '多头趋势': bool(signals['多头趋势'].iloc[i]),
        '中线趋势': round(float(signals['中线趋势'].iloc[i]), 2),
    }


def batch_screen(stock_list, fetch_stock_func, fetch_index_func, fetch_capital_func,
                 signal_filter=None, max_workers=10):
    index_df = fetch_index_func()
    results = []
    total = len(stock_list)
    progress_lock = threading.Lock()
    progress_counter = [0]  # 用列表以便在闭包中修改

    def _process_one(code):
        """单只股票的 获取数据 + 分析 逻辑（在子线程中执行）"""
        try:
            df = fetch_stock_func(code)
            if df is None or len(df) < 120:
                return None
            cap = fetch_capital_func(code)
            common = df.index.intersection(index_df.index)
            if len(common) < 120:
                return None
            df_a = df.loc[common].copy()
            idx_a = index_df.loc[common].copy()
            res = screen_single_stock(df_a, idx_a, cap)
            all_signals = res['买入信号'] + res['卖出信号']
            if not all_signals:
                return None
            if signal_filter:
                if not any(s in all_signals for s in signal_filter):
                    return None
            date_str = pd.Timestamp(res['日期']).strftime('%Y-%m-%d')
            return {
                '代码': code,
                '日期': date_str,
                '开盘价': res['开盘价'],
                '收盘价': res['收盘价'],
                '买入信号': ' | '.join(res['买入信号']) if res['买入信号'] else '-',
                '卖出信号': ' | '.join(res['卖出信号']) if res['卖出信号'] else '-',
                '大机构仓位': res['大机构仓位'],
                '基金私募仓位': res['基金私募仓位'],
                '多线': res['多线'],
                '空线': res['空线'],
                '多头趋势': '是' if res['多头趋势'] else '否',
                '中线趋势': res['中线趋势'],
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_code = {executor.submit(_process_one, code): code for code in stock_list}
        for future in as_completed(future_to_code):
            code = future_to_code[future]
            with progress_lock:
                progress_counter[0] += 1
                print(f"\r[{progress_counter[0]}/{total}] 已完成 {code} ...", end='', flush=True)
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception:
                continue

    print(f"\n选股完成, 共发现 {len(results)} 只符合条件的股票")
    return pd.DataFrame(results)


# ================================================================
#      第五部分: 格式化函数（控制台打印 + HTML 生成）
# ================================================================

def print_single_result(code, res):
    date_str = pd.Timestamp(res['日期']).strftime('%Y-%m-%d')
    print("┌" + "─" * 50 + "┐")
    print(f"│ {'股票代码':<10}: {code:<36} │")
    print(f"│ {'日    期':<10}: {date_str:<36} │")
    print(f"│ {'开 盘 价':<10}: {res['开盘价']:<36.2f} │")
    print(f"│ {'收 盘 价':<10}: {res['收盘价']:<36.2f} │")
    print("├" + "─" * 50 + "┤")
    buy_str = ' | '.join(res['买入信号']) if res['买入信号'] else '无'
    sell_str = ' | '.join(res['卖出信号']) if res['卖出信号'] else '无'
    print(f"│ {'买入信号':<10}: {buy_str:<36} │")
    print(f"│ {'卖出信号':<10}: {sell_str:<36} │")
    print("├" + "─" * 50 + "┤")
    print(f"│ {'大机构仓位':<8}: {res['大机构仓位']:<35.2f}% │")
    print(f"│ {'基金私募仓位':<7}: {res['基金私募仓位']:<35.2f}% │")
    print(f"│ {'多    线':<10}: {res['多线']:<36.2f} │")
    print(f"│ {'空    线':<10}: {res['空线']:<36.2f} │")
    trend_str = '是' if res['多头趋势'] else '否'
    print(f"│ {'多头趋势':<10}: {trend_str:<36} │")
    print(f"│ {'中线趋势':<10}: {res['中线趋势']:<36.2f} │")
    print("└" + "─" * 50 + "┘")


def format_single_result_html(code, res):
    date_str = pd.Timestamp(res['日期']).strftime('%Y-%m-%d')
    buy_str = ' | '.join(res['买入信号']) if res['买入信号'] else '无'
    sell_str = ' | '.join(res['卖出信号']) if res['卖出信号'] else '无'
    trend_str = '是' if res['多头趋势'] else '否'
    buy_html = f'<span style="color:#FF4444;font-weight:bold;">{buy_str}</span>' if res[
        '买入信号'] else '无'
    sell_html = f'<span style="color:#00AA00;font-weight:bold;">{sell_str}</span>' if res[
        '卖出信号'] else '无'
    html = f'''
    <div style="border:1px solid #ddd; border-radius:8px; padding:12px; margin:10px 0;
                background:#fafafa; font-family:Arial,sans-serif;">
        <h3 style="margin:0 0 8px 0; color:#333;">📈 {code}</h3>
        <table style="width:100%; border-collapse:collapse; font-size:14px;">
            <tr><td style="padding:4px 8px; color:#666;">日期</td>
                <td style="padding:4px 8px;">{date_str}</td></tr>
            <tr><td style="padding:4px 8px; color:#666;">开盘价</td>
                <td style="padding:4px 8px;">{res['开盘价']:.2f}</td></tr>
            <tr><td style="padding:4px 8px; color:#666;">收盘价</td>
                <td style="padding:4px 8px;font-weight:bold;">{res['收盘价']:.2f}</td></tr>
            <tr><td style="padding:4px 8px; color:#666;">买入信号</td>
                <td style="padding:4px 8px;">{buy_html}</td></tr>
            <tr><td style="padding:4px 8px; color:#666;">卖出信号</td>
                <td style="padding:4px 8px;">{sell_html}</td></tr>
            <tr><td style="padding:4px 8px; color:#666;">大机构仓位</td>
                <td style="padding:4px 8px;">{res['大机构仓位']:.2f}%</td></tr>
            <tr><td style="padding:4px 8px; color:#666;">基金私募仓位</td>
                <td style="padding:4px 8px;">{res['基金私募仓位']:.2f}%</td></tr>
            <tr><td style="padding:4px 8px; color:#666;">多线 / 空线</td>
                <td style="padding:4px 8px;">{res['多线']:.2f} / {res['空线']:.2f}</td></tr>
            <tr><td style="padding:4px 8px; color:#666;">多头趋势</td>
                <td style="padding:4px 8px;">{trend_str}</td></tr>
            <tr><td style="padding:4px 8px; color:#666;">中线趋势</td>
                <td style="padding:4px 8px;">{res['中线趋势']:.2f}</td></tr>
        </table>
    </div>
    '''
    return html


def print_batch_results(result_df):
    if result_df.empty:
        print("=" * 60)
        print("  未找到符合条件的股票")
        print("=" * 60)
        return
    col_configs = [
        ('代码', 'str', 8), ('日期', 'str', 12),
        ('开盘价', 'float', 10), ('收盘价', 'float', 10),
        ('买入信号', 'str', 30), ('卖出信号', 'str', 22),
        ('大机构仓位', 'float', 10), ('基金私募仓位', 'float', 10),
        ('多线', 'float', 8), ('空线', 'float', 8),
        ('多头趋势', 'str', 8), ('中线趋势', 'float', 8),
    ]
    col_widths = {}
    for col_name, col_type, min_w in col_configs:
        if col_name not in result_df.columns:
            continue
        header_w = sum(2 if '\u4e00' <= ch <= '\u9fff' else 1 for ch in col_name)
        max_data_w = 0
        for val in result_df[col_name]:
            s = f"{val:.2f}" if col_type == 'float' else str(val)
            w = sum(2 if '\u4e00' <= ch <= '\u9fff' else 1 for ch in s)
            max_data_w = max(max_data_w, w)
        col_widths[col_name] = max(min_w, header_w, max_data_w) + 2

    def pad_str(text, width):
        text = str(text)
        display_w = sum(2 if '\u4e00' <= ch <= '\u9fff' else 1 for ch in text)
        padding = width - display_w
        return text + ' ' * max(padding, 0)

    active_cols = [(n, t) for n, t, _ in col_configs if n in result_df.columns]
    total_width = sum(col_widths[n] for n, _ in active_cols) + len(active_cols) + 1
    print("\n" + "=" * total_width)
    print(pad_str("  选股结果汇总", total_width))
    print("=" * total_width)
    header_parts = [pad_str(n, col_widths[n]) for n, _ in active_cols]
    print("│" + "│".join(header_parts) + "│")
    sep_parts = ["─" * col_widths[n] for n, _ in active_cols]
    print("├" + "┼".join(sep_parts) + "┤")
    for _, row in result_df.iterrows():
        row_parts = []
        for col_name, col_type in active_cols:
            val = row[col_name]
            text = f"{val:.2f}" if col_type == 'float' else str(val)
            row_parts.append(pad_str(text, col_widths[col_name]))
        print("│" + "│".join(row_parts) + "│")
    bottom_parts = ["─" * col_widths[n] for n, _ in active_cols]
    print("└" + "┴".join(bottom_parts) + "┘")
    print(f"  共 {len(result_df)} 只股票\n")


def format_batch_results_html(result_df):
    if result_df.empty:
        return '<p style="text-align:center;color:#999;">未找到符合条件的股票</p>'
    html = '''
    <style>
        .stock-table {width:100%;border-collapse:collapse;font-size:13px;font-family:Arial,sans-serif;}
        .stock-table th {background:#1a73e8;color:white;padding:8px 6px;text-align:center;font-size:12px;position:sticky;top:0;}
        .stock-table td {padding:6px;border-bottom:1px solid #eee;text-align:center;white-space:nowrap;}
        .stock-table tr:nth-child(even) {background:#f8f9fa;}
        .stock-table tr:hover {background:#e8f0fe;}
        .buy-signal {color:#d32f2f;font-weight:bold;font-size:12px;white-space:normal;}
        .sell-signal {color:#2e7d32;font-weight:bold;font-size:12px;white-space:normal;}
        .summary {background:#f0f4ff;padding:10px;border-radius:6px;margin-bottom:10px;font-size:14px;}
    </style>
    '''
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    html += f'''
    <div class="summary">
        📊 <b>上涨确立选股结果</b><br>
        🕐 时间: {now_str}<br>
        📋 共筛选出 <b style="color:#d32f2f;">{len(result_df)}</b> 只股票
    </div>
    '''
    buy_df = result_df[result_df['买入信号'] != '-'].copy()
    if not buy_df.empty:
        html += '<h3 style="color:#d32f2f;margin:15px 0 5px 0;">🔴 上涨确立信号</h3>'
        html += '<div style="overflow-x:auto;"><table class="stock-table">'
        html += '''<tr><th>代码</th><th>日期</th><th>收盘价</th>
            <th>信号</th><th>机构仓位</th><th>多/空线</th>
            <th>多头</th><th>中线趋势</th></tr>'''
        for _, row in buy_df.iterrows():
            trend_icon = '🔴' if row['多头趋势'] == '是' else '🟢'
            html += f'''<tr>
                <td><b>{row['代码']}</b></td><td>{row['日期']}</td>
                <td>{row['收盘价']:.2f}</td>
                <td class="buy-signal">{row['买入信号']}</td>
                <td>{row['大机构仓位']:.1f}%</td>
                <td>{row['多线']:.1f}/{row['空线']:.1f}</td>
                <td>{trend_icon}</td><td>{row['中线趋势']:.1f}</td></tr>'''
        html += '</table></div>'
    html += '<p style="color:#999;font-size:11px;margin-top:15px;">⚠️ 以上数据仅供参考，不构成投资建议</p>'
    return html


# ================================================================
#   第六部分: 主程序 —— 数据源: 腾讯财经，仅筛选【上涨确立】
# ================================================================

if __name__ == '__main__':

    END_DATE = datetime.now().strftime('%Y%m%d')
    START_DATE = (datetime.now() - timedelta(days=600)).strftime('%Y%m%d')
    _INDEX_CACHE = {}

    print("=" * 60)
    print("  📡 数据源: 腾讯财经 (web.ifzq.gtimg.cn)")
    print(f"  📅 数据区间: {START_DATE} ~ {END_DATE}")
    print("  🎯 筛选目标: 仅【上涨确立】信号")
    print("=" * 60)

    # ---------------------------------------------------------------
    #  腾讯财经 K 线通用请求函数
    # ---------------------------------------------------------------

    def _to_dash(d):
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"

    def _tencent_kline(symbol, start, end, fq="qfq", count=800, retries=3):
        url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        param_str = f"{symbol},day,{start},{end},{count},{fq}"
        headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"),
            "Referer": "https://web.ifzq.gtimg.cn/",
        }
        for attempt in range(1, retries + 1):
            try:
                r = requests.get(url, params={"param": param_str},
                                 headers=headers, timeout=30)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                if attempt < retries:
                    wait = 3 * attempt + random.uniform(0, 2)
                    print(f"\n  ⚠️ 请求 {symbol} 失败({attempt}/{retries}), "
                          f"{wait:.0f}s 后重试: {e}")
                    time.sleep(wait)
                else:
                    raise

        sd = data.get("data", {}).get(symbol, {})
        key_map = {"qfq": "qfqday", "hfq": "hfqday"}
        primary_key = key_map.get(fq, "day")
        klines = sd.get(primary_key) or sd.get("day", [])
        if not klines:
            return None

        rows = []
        for k in klines:
            if len(k) < 6:
                continue
            rows.append({
                "date": k[0], "open": float(k[1]), "close": float(k[2]),
                "high": float(k[3]), "low": float(k[4]), "volume": float(k[5]),
            })
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)
        return df

    # ---------------------------------------------------------------
    #  三个业务数据接口
    # ---------------------------------------------------------------

    def fetch_stock(code):
        if code.startswith(('6', '9', '5')):
            symbol = f"sh{code}"
        else:
            symbol = f"sz{code}"
        start = _to_dash(START_DATE)
        end = _to_dash(END_DATE)
        time.sleep(random.uniform(0.2, 0.6))
        df = _tencent_kline(symbol, start, end, fq="qfq")
        if df is None or df.empty:
            return None
        return df[['open', 'high', 'low', 'close', 'volume']]

    def fetch_index():
        if 'idx' in _INDEX_CACHE:
            return _INDEX_CACHE['idx']
        start = _to_dash(START_DATE)
        end = _to_dash(END_DATE)
        df = _tencent_kline("sh000001", start, end, fq="")
        if df is None or df.empty:
            raise RuntimeError("❌ 获取上证指数数据失败，请检查网络")
        _INDEX_CACHE['idx'] = df[['open', 'high', 'low', 'close']]
        print(f"  ✅ 上证指数已加载 {len(df)} 条日线")
        return _INDEX_CACHE['idx']

    def fetch_capital(code):
        try:
            if code.startswith(('6', '9', '5')):
                symbol = f"sh{code}"
            else:
                symbol = f"sz{code}"
            url = f"https://qt.gtimg.cn/q={symbol}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Referer": "https://finance.qq.com",
            }
            r = requests.get(url, headers=headers, timeout=10)
            m = re.search(r'"(.*?)"', r.text)
            if m:
                fields = m.group(1).split('~')
                price = float(fields[3]) if len(fields) > 3 else 0
                if price > 0 and len(fields) > 50:
                    for idx in (44, 49, 45):
                        try:
                            cap_yi = float(fields[idx])
                            if cap_yi > 1:
                                return cap_yi * 1e8 / price
                        except (ValueError, IndexError):
                            continue
        except Exception:
            pass
        return 1e9

    # ---------------------------------------------------------------
    #  主流程
    # ---------------------------------------------------------------

    all_html_parts = []

    # ==================== 单只股票分析 ====================
    print("\n" + "=" * 60)
    print("  示例1: 分析单只股票 000001(平安银行)")
    print("=" * 60)
    try:
        stock_df = fetch_stock('000001')
        index_df = fetch_index()
        cap = fetch_capital('000001')

        common_dates = stock_df.index.intersection(index_df.index)
        stock_aligned = stock_df.loc[common_dates]
        index_aligned = index_df.loc[common_dates]

        result = screen_single_stock(stock_aligned, index_aligned, cap)

        print_single_result('000001', result)

        all_html_parts.append('<h2>📈 单股分析: 000001 平安银行</h2>')
        all_html_parts.append(format_single_result_html('000001', result))

    except Exception as e:
        print(f"分析出错: {e}")
        all_html_parts.append(f'<p style="color:red;">单股分析出错: {e}</p>')

    # ==================== 批量选股 ====================
    print("\n" + "=" * 60)
    print("  示例2: 批量选股 —— 仅筛选【上涨确立】")
    print("=" * 60)

    test_stocks = [
        '000001', '000002', '000004', '000006', '000007', '000008', '000009', '000010', '000011', '000012', '000014',
        '000016', '000017', '000019', '000020', '000021', '000025', '000026', '000027', '000028', '000029', '000030',
        '000031', '000032', '000034', '000035', '000036', '000037', '000039', '000042', '000045', '000048', '000049',
        '000050', '000055', '000056', '000058', '000059', '000060', '000061', '000062', '000063', '000065', '000066',
        '000068', '000069', '000070', '000078', '000088', '000089', '000090', '000096', '000099', '000100', '000151',
        '000153', '000155', '000156', '000157', '000158', '000159', '000166', '000301', '000333', '000338', '000400',
        '000401', '000402', '000403', '000404', '000407', '000408', '000409', '000410', '000411', '000415', '000417',
        '000419', '000420', '000421', '000422', '000423', '000425', '000426', '000428', '000429', '000430', '000488',
        '000498', '000501', '000503', '000504', '000505', '000506', '000507', '000509', '000510', '000513', '000514',
        '000516', '000517', '000518', '000519', '000520', '000521', '000523', '000524', '000525', '000526', '000528',
        '000529', '000530', '000531', '000532', '000533', '000534', '000536', '000537', '000538', '000539', '000541',
        '000543', '000544', '000545', '000546', '000547', '000548', '000550', '000551', '000552', '000553', '000554',
        '000555', '000557', '000558', '000559', '000560', '000561', '000563', '000564', '000565', '000566', '000567',
        '000568', '000570', '000571', '000572', '000573', '000576', '000581', '000582', '000586', '000589', '000590',
        '000591', '000592', '000593', '000595', '000596', '000597', '000598', '000599', '000600', '000601', '000603',
        '000605', '000607', '000608', '000609', '000610', '000612', '000615', '000617', '000619', '000620', '000623',
        '000625', '000626', '000628', '000629', '000630', '000631', '000632', '000633', '000635', '000636', '000637',
        '000638', '000639', '000650', '000651', '000652', '000655', '000656', '000657', '000659', '000661', '000663',
        '000665', '000668', '000669', '000670', '000672', '000676', '000677', '000678', '000679', '000680', '000681',
        '000682', '000683', '000685', '000686', '000688', '000690', '000691', '000692', '000695', '000697', '000698',
        '000700', '000701', '000702', '000703', '000705', '000707', '000708', '000709', '000710', '000711', '000712',
        '000713', '000715', '000716', '000717', '000718', '000719', '000720', '000721', '000722', '000723', '000725',
        '000726', '000727', '000728', '000729', '000731', '000733', '000735', '000736', '000737', '000738', '000739',
        '000750', '000751', '000752', '000753', '000755', '000756', '000757', '000758', '000759', '000761', '000762',
        '000766', '000767', '000768', '000776', '000777', '000778', '000779', '000782', '000783', '000785', '000786',
        '000788', '000789', '000790', '000791', '000792', '000793', '000795', '000796', '000797', '000798', '000799',
        '000800', '000801', '000802', '000803', '000807', '000809', '000810', '000811', '000812', '000813', '000815',
        '000816', '000818', '000819', '000820', '000821', '000822', '000823', '000825', '000826', '000828', '000829',
        '000830', '000831', '000833', '000837', '000838', '000839', '000848', '000850', '000852', '000856', '000858',
        '000859', '000860', '000862', '000863', '000868', '000869', '000875', '000876', '000877', '000878', '000880',
        '000881', '000882', '000883', '000885', '000886', '000887', '000888', '000889', '000890', '000892', '000893',
        '000895', '000897', '000898', '000899', '000900', '000901', '000902', '000903', '000905', '000906', '000908',
        '000909', '000910', '000911', '000912', '000913', '000915', '000917', '000919', '000920', '000921', '000922',
        '000923', '000925', '000926', '000927', '000928', '000929', '000930', '000931', '000932', '000933', '000935',
        '000936', '000937', '000938', '000948', '000949', '000950', '000951', '000952', '000953', '000955', '000957',
        '000958', '000959', '000960', '000962', '000963', '000965', '000966', '000967', '000968', '000969', '000970',
        '000972', '000973', '000975', '000977', '000978', '000980', '000981', '000983', '000985', '000987', '000988',
        '000989', '000990', '000993', '000995', '000997', '000998', '000999', '001201', '001202', '001203', '001205',
        '001206', '001207', '001208', '001209', '001210', '001211', '001212', '001213', '001215', '001216', '001217',
        '001218', '001219', '001221', '001222', '001223', '001225', '001226', '001227', '001228', '001229', '001230',
        '001231', '001233', '001234', '001236', '001238', '001239', '001255', '001256', '001258', '001259', '001260',
        '001266', '001267', '001268', '001269', '001270', '001277', '001278', '001279', '001282', '001283', '001285',
        '001286', '001287', '001288', '001289', '001296', '001298', '001299', '001300', '001301', '001306', '001308',
        '001309', '001311', '001313', '001314', '001316', '001317', '001318', '001319', '001322', '001323', '001324',
        '001326', '001328', '001330', '001331', '001332', '001333', '001335', '001336', '001337', '001338', '001339',
        '001356', '001358', '001359', '001360', '001366', '001367', '001368', '001373', '001376', '001378', '001379',
        '001380', '001382', '001386', '001387', '001388', '001389', '001390', '001391', '001395', '001400', '001696',
        '001872', '001896', '001914', '001965', '001979', '002001', '002003', '002004', '002005', '002006', '002007',
        '002008', '002009', '002010', '002011', '002012', '002014', '002015', '002016', '002017', '002019', '002020',
        '002021', '002022', '002023', '002024', '002025', '002026', '002027', '002028', '002029', '002030', '002031',
        '002032', '002033', '002034', '002035', '002036', '002037', '002038', '002039', '002040', '002041', '002042',
        '002043', '002044', '002045', '002046', '002047', '002048', '002049', '002050', '002051', '002052', '002053',
        '002054', '002055', '002056', '002057', '002058', '002059', '002060', '002061', '002062', '002063', '002064',
        '002065', '002066', '002067', '002068', '002069', '002072', '002073', '002074', '002075', '002076', '002077',
        '002078', '002079', '002080', '002081', '002082', '002083', '002084', '002085', '002086', '002088', '002090',
        '002091', '002092', '002093', '002094', '002095', '002096', '002097', '002098', '002099', '002100', '002101',
        '002102', '002103', '002104', '002105', '002106', '002107', '002108', '002109', '002110', '002111', '002112',
        '002114', '002115', '002116', '002117', '002119', '002120', '002121', '002122', '002123', '002124', '002125',
        '002126', '002127', '002128', '002129', '002130', '002131', '002132', '002133', '002134', '002135', '002136',
        '002137', '002138', '002139', '002140', '002141', '002142', '002144', '002145', '002146', '002148', '002149',
        '002150', '002151', '002152', '002153', '002154', '002155', '002156', '002157', '002158', '002159', '002160',
        '002161', '002162', '002163', '002164', '002165', '002166', '002167', '002168', '002169', '002170', '002171',
        '002172', '002173', '002174', '002175', '002176', '002177', '002178', '002179', '002180', '002181', '002182',
        '002183', '002184', '002185', '002186', '002187', '002188', '002189', '002190', '002191', '002192', '002193',
        '002194', '002195', '002196', '002197', '002198', '002199', '002200', '002201', '002202', '002203', '002204',
        '002205', '002206', '002207', '002208', '002209', '002210', '002211', '002212', '002213', '002214', '002215',
        '002216', '002217', '002218', '002219', '002221', '002222', '002223', '002224', '002225', '002226', '002227',
        '002228', '002229', '002230', '002231', '002232', '002233', '002234', '002235', '002236', '002237', '002238',
        '002239', '002240', '002241', '002242', '002243', '002244', '002245', '002246', '002247', '002248', '002249',
        '002250', '002251', '002252', '002253', '002254', '002255', '002256', '002258', '002259', '002261', '002262',
        '002263', '002264', '002265', '002266', '002267', '002268', '002269', '002270', '002271', '002272', '002273',
        '002274', '002275', '002276', '002277', '002278', '002279', '002281', '002282', '002283', '002284', '002285',
        '002286', '002287', '002289', '002290', '002291', '002292', '002293', '002294', '002295', '002296', '002297',
        '002298', '002299', '002300', '002301', '002302', '002303', '002304', '002305', '002306', '002307', '002309',
        '002310', '002311', '002312', '002313', '002314', '002315', '002316', '002317', '002318', '002319', '002320',
        '002321', '002322', '002323', '002324', '002326', '002327', '002328', '002329', '002330', '002331', '002332',
        '002333', '002334', '002335', '002337', '002338', '002339', '002340', '002342', '002343', '002344', '002345',
        '002346', '002347', '002348', '002349', '002350', '002351', '002352', '002353', '002354', '002355', '002356',
        '002357', '002358', '002360', '002361', '002362', '002363', '002364', '002365', '002366', '002367', '002368',
        '002369', '002370', '002371', '002372', '002373', '002374', '002375', '002376', '002377', '002378', '002379',
        '002380', '002381', '002382', '002383', '002384', '002385', '002386', '002387', '002388', '002389', '002390',
        '002391', '002392', '002393', '002394', '002395', '002396', '002397', '002398', '002399', '002400', '002401',
        '002402', '002403', '002404', '002405', '002406', '002407', '002408', '002409', '002410', '002412', '002413',
        '002414', '002415', '002416', '002418', '002419', '002420', '002421', '002422', '002423', '002424', '002425',
        '002426', '002427', '002428', '002429', '002430', '002431', '002432', '002434', '002436', '002437', '002438',
        '002439', '002440', '002441', '002442', '002443', '002444', '002445', '002446', '002448', '002449', '002451',
        '002452', '002453', '002454', '002455', '002456', '002457', '002458', '002459', '002460', '002461', '002462',
        '002463', '002465', '002466', '002467', '002468', '002469', '002470', '002471', '002472', '002474', '002475',
        '002476', '002478', '002479', '002480', '002481', '002482', '002483', '002484', '002485', '002486', '002487',
        '002488', '002489', '002490', '002491', '002492', '002493', '002494', '002495', '002496', '002497', '002498',
        '002500', '002501', '002506', '002507', '002508', '002510', '002511', '002512', '002513', '002514', '002515',
        '002516', '002517', '002518', '002519', '002520', '002521', '002522', '002523', '002524', '002526', '002527',
        '002528', '002529', '002530', '002531', '002532', '002533', '002534', '002535', '002536', '002537', '002538',
        '002539', '002540', '002541', '002542', '002543', '002544', '002545', '002546', '002547', '002548', '002549',
        '002550', '002551', '002552', '002553', '002554', '002555', '002556', '002557', '002558', '002559', '002560',
        '002561', '002562', '002563', '002564', '002565', '002566', '002567', '002568', '002569', '002570', '002571',
        '002572', '002573', '002574', '002575', '002576', '002577', '002578', '002579', '002580', '002581', '002582',
        '002583', '002584', '002585', '002586', '002587', '002588', '002589', '002590', '002591', '002592', '002593',
        '002594', '002595', '002596', '002597', '002598', '002599', '002600', '002601', '002602', '002603', '002605',
        '002606', '002607', '002608', '002609', '002611', '002612', '002613', '002614', '002615', '002616', '002617',
        '002620', '002622', '002623', '002624', '002625', '002626', '002627', '002628', '002629', '002630', '002631',
        '002632', '002633', '002634', '002635', '002636', '002637', '002638', '002639', '002640', '002641', '002642',
        '002643', '002644', '002645', '002646', '002647', '002648', '002649', '002650', '002651', '002652', '002653',
        '002654', '002655', '002656', '002657', '002658', '002659', '002660', '002661', '002662', '002663', '002664',
        '002666', '002667', '002668', '002669', '002670', '002671', '002672', '002673', '002674', '002675', '002676',
        '002677', '002678', '002679', '002681', '002682', '002683', '002685', '002686', '002687', '002688', '002689',
        '002690', '002691', '002692', '002693', '002694', '002695', '002696', '002697', '002698', '002700', '002701',
        '002702', '002703', '002705', '002706', '002707', '002708', '002709', '002712', '002713', '002714', '002715',
        '002716', '002717', '002718', '002719', '002721', '002722', '002723', '002724', '002725', '002726', '002727',
        '002728', '002729', '002730', '002731', '002732', '002733', '002734', '002735', '002736', '002737', '002738',
        '002739', '002741', '002742', '002743', '002745', '002746', '002747', '002748', '002749', '002752', '002753',
        '002755', '002756', '002757', '002758', '002759', '002760', '002761', '002762', '002763', '002765', '002766',
        '002767', '002768', '002769', '002771', '002772', '002773', '002774', '002775', '002777', '002778', '002779',
        '002780', '002782', '002783', '002785', '002786', '002787', '002788', '002789', '002790', '002791', '002792',
        '002793', '002795', '002796', '002797', '002798', '002799', '002800', '002801', '002802', '002803', '002805',
        '002806', '002807', '002808', '002809', '002810', '002811', '002812', '002813', '002815', '002816', '002817',
        '002818', '002819', '002820', '002821', '002822', '002823', '002824', '002825', '002826', '002827', '002828',
        '002829', '002830', '002831', '002832', '002833', '002835', '002836', '002837', '002838', '002839', '002840',
        '002841', '002842', '002843', '002845', '002846', '002847', '002848', '002849', '002850', '002851', '002852',
        '002853', '002855', '002856', '002857', '002858', '002859', '002860', '002861', '002862', '002863', '002864',
        '002865', '002866', '002867', '002868', '002869', '002870', '002871', '002872', '002873', '002875', '002876',
        '002877', '002878', '002879', '002880', '002881', '002882', '002883', '002884', '002885', '002886', '002887',
        '002888', '002889', '002890', '002891', '002892', '002893', '002895', '002896', '002897', '002898', '002899',
        '002900', '002901', '002902', '002903', '002905', '002906', '002907', '002908', '002909', '002910', '002911',
        '002912', '002913', '002915', '002916', '002917', '002918', '002919', '002920', '002921', '002922', '002923',
        '002925', '002926', '002927', '002928', '002929', '002930', '002931', '002932', '002933', '002935', '002936',
        '002937', '002938', '002939', '002940', '002941', '002942', '002943', '002945', '002946', '002947', '002948',
        '002949', '002950', '002951', '002952', '002953', '002955', '002956', '002957', '002958', '002959', '002960',
        '002961', '002962', '002963', '002965', '002966', '002967', '002968', '002969', '002970', '002971', '002972',
        '002973', '002975', '002976', '002977', '002978', '002979', '002980', '002981', '002982', '002983', '002984',
        '002985', '002986', '002987', '002988', '002989', '002990', '002991', '002992', '002993', '002995', '002996',
        '002997', '002998', '002999', '003000', '003001', '003002', '003003', '003004', '003005', '003006', '003007',
        '003008', '003009', '003010', '003011', '003012', '003013', '003015', '003016', '003017', '003018', '003019',
        '003020', '003021', '003022', '003023', '003025', '003026', '003027', '003028', '003029', '003030', '003031',
        '003032', '003033', '003035', '003036', '003037', '003038', '003039', '003040', '003041', '003042', '003043',
        '003816', '300001', '300002', '300003', '300004', '300005', '300006', '300007', '300008', '300009', '300010',
        '300011', '300012', '300013', '300014', '300015', '300016', '300017', '300018', '300019', '300020', '300021',
        '300022', '300024', '300025', '300026', '300027', '300029', '300030', '300031', '300032', '300033', '300034',
        '300035', '300036', '300037', '300039', '300040', '300041', '300042', '300043', '300044', '300045', '300046',
        '300047', '300048', '300049', '300050', '300051', '300052', '300053', '300054', '300055', '300056', '300057',
        '300058', '300059', '300061', '300062', '300063', '300065', '300066', '300067', '300068', '300069', '300070',
        '300071', '300072', '300073', '300074', '300075', '300076', '300077', '300078', '300079', '300080', '300081',
        '300082', '300083', '300084', '300085', '300086', '300087', '300088', '300091', '300092', '300093', '300094',
        '300095', '300096', '300097', '300098', '300099', '300100', '300101', '300102', '300103', '300105', '300106',
        '300107', '300109', '300110', '300111', '300112', '300113', '300115', '300118', '300119', '300120', '300121',
        '300122', '300123', '300124', '300125', '300126', '300127', '300128', '300129', '300130', '300131', '300132',
        '300133', '300134', '300135', '300136', '300137', '300138', '300139', '300140', '300141', '300142', '300143',
        '300144', '300145', '300146', '300147', '300148', '300149', '300150', '300151', '300152', '300153', '300154',
        '300155', '300157', '300158', '300159', '300160', '300161', '300162', '300163', '300164', '300165', '300166',
        '300167', '300168', '300169', '300170', '300171', '300172', '300173', '300174', '300175', '300176', '300177',
        '300179', '300180', '300181', '300182', '300183', '300184', '300185', '300187', '300188', '300189', '300190',
        '300191', '300192', '300193', '300194', '300195', '300196', '300197', '300198', '300199', '300200', '300201',
        '300203', '300204', '300205', '300206', '300207', '300209', '300210', '300211', '300212', '300213', '300214',
        '300215', '300217', '300218', '300219', '300220', '300221', '300222', '300223', '300224', '300225', '300226',
        '300227', '300228', '300229', '300230', '300231', '300232', '300233', '300234', '300235', '300236', '300237',
        '300238', '300239', '300240', '300241', '300242', '300243', '300244', '300245', '300246', '300247', '300248',
        '300249', '300250', '300251', '300252', '300253', '300254', '300255', '300256', '300257', '300258', '300259',
        '300260', '300261', '300263', '300264', '300265', '300266', '300267', '300268', '300269', '300270', '300271',
        '300272', '300274', '300275', '300276', '300277', '300278', '300279', '300281', '300283', '300284', '300285',
        '300286', '300287', '300288', '300289', '300290', '300291', '300292', '300293', '300294', '300295', '300296',
        '300298', '300299', '300300', '300301', '300302', '300303', '300304', '300305', '300306', '300307', '300308',
        '300310', '300311', '300313', '300314', '300315', '300316', '300317', '300318', '300319', '300320', '300321',
        '300322', '300323', '300324', '300326', '300327', '300328', '300329', '300331', '300332', '300333', '300334',
        '300335', '300337', '300338', '300339', '300340', '300341', '300342', '300343', '300344', '300345', '300346',
        '300347', '300348', '300349', '300350', '300351', '300352', '300353', '300354', '300355', '300357', '300358',
        '300359', '300360', '300363', '300364', '300365', '300366', '300368', '300369', '300370', '300371', '300373',
        '300374', '300375', '300376', '300377', '300378', '300379', '300380', '300381', '300382', '300383', '300384',
        '300385', '300386', '300387', '300388', '300389', '300390', '300391', '300393', '300394', '300395', '300396',
        '300397', '300398', '300399', '300400', '300401', '300402', '300403', '300404', '300405', '300406', '300407',
        '300408', '300409', '300410', '300411', '300412', '300413', '300414', '300415', '300416', '300417', '300418',
        '300419', '300420', '300421', '300422', '300423', '300424', '300425', '300426', '300427', '300428', '300429',
        '300430', '300432', '300433', '300434', '300435', '300436', '300437', '300438', '300439', '300440', '300441',
        '300442', '300443', '300444', '300445', '300446', '300447', '300448', '300449', '300450', '300451', '300452',
        '300453', '300454', '300455', '300456', '300457', '300458', '300459', '300460', '300461', '300462', '300463',
        '300464', '300465', '300466', '300467', '300468', '300469', '300470', '300471', '300472', '300473', '300474',
        '300475', '300476', '300477', '300478', '300479', '300480', '300481', '300482', '300483', '300484', '300485',
        '300486', '300487', '300488', '300489', '300490', '300491', '300492', '300493', '300494', '300496', '300497',
        '300498', '300499', '300500', '300501', '300502', '300503', '300504', '300505', '300506', '300507', '300508',
        '300509', '300510', '300511', '300512', '300513', '300514', '300515', '300516', '300517', '300518', '300519',
        '300520', '300521', '300522', '300523', '300525', '300527', '300528', '300529', '300530', '300531', '300532',
        '300533', '300534', '300535', '300536', '300537', '300538', '300539', '300540', '300541', '300542', '300543',
        '300545', '300546', '300547', '300548', '300549', '300550', '300551', '300552', '300553', '300554', '300555',
        '300556', '300557', '300558', '300559', '300560', '300561', '300562', '300563', '300564', '300565', '300566',
        '300567', '300568', '300569', '300570', '300571', '300572', '300573', '300575', '300576', '300577', '300578',
        '300579', '300580', '300581', '300582', '300583', '300584', '300585', '300586', '300587', '300588', '300589',
        '300590', '300591', '300592', '300593', '300594', '300595', '300596', '300597', '300598', '300599', '300600',
        '300601', '300602', '300603', '300604', '300605', '300606', '300607', '300608', '300609', '300610', '300611',
        '300612', '300613', '300614', '300615', '300616', '300617', '300618', '300619', '300620', '300621', '300622',
        '300623', '300624', '300625', '300626', '300627', '300628', '300629', '300631', '300632', '300633', '300634',
        '300635', '300636', '300637', '300638', '300639', '300640', '300641', '300642', '300643', '300644', '300645',
        '300647', '300648', '300649', '300650', '300651', '300652', '300653', '300654', '300655', '300656', '300657',
        '300658', '300659', '300660', '300661', '300662', '300663', '300664', '300665', '300666', '300667', '300668',
        '300669', '300670', '300671', '300672', '300673', '300674', '300675', '300676', '300677', '300678', '300679',
        '300680', '300681', '300682', '300683', '300684', '300685', '300686', '300687', '300688', '300689', '300690',
        '300691', '300692', '300693', '300694', '300695', '300696', '300697', '300698', '300699', '300700', '300701',
        '300702', '300703', '300705', '300706', '300707', '300708', '300709', '300710', '300711', '300712', '300713',
        '300715', '300716', '300717', '300718', '300719', '300720', '300721', '300722', '300723', '300724', '300725',
        '300726', '300727', '300729', '300730', '300731', '300732', '300733', '300735', '300736', '300737', '300738',
        '300739', '300740', '300741', '300743', '300745', '300746', '300747', '300748', '300749', '300750', '300751',
        '300752', '300753', '300755', '300756', '300757', '300758', '300759', '300760', '300761', '300762', '300763',
        '300765', '300766', '300767', '300768', '300769', '300770', '300771', '300772', '300773', '300774', '300775',
        '300776', '300777', '300778', '300779', '300780', '300781', '300782', '300783', '300784', '300785', '300786',
        '300787', '300788', '300789', '300790', '300791', '300792', '300793', '300795', '300796', '300797', '300798',
        '300800', '300801', '300802', '300803', '300804', '300805', '300806', '300807', '300808', '300809', '300810',
        '300811', '300812', '300813', '300814', '300815', '300816', '300817', '300818', '300819', '300820', '300821',
        '300822', '300823', '300824', '300825', '300826', '300827', '300828', '300829', '300830', '300831', '300832',
        '300833', '300834', '300835', '300836', '300837', '300838', '300839', '300840', '300841', '300842', '300843',
        '300844', '300845', '300846', '300847', '300848', '300849', '300850', '300851', '300852', '300853', '300854',
        '300855', '300856', '300857', '300858', '300859', '300860', '300861', '300862', '300863', '300864', '300865',
        '300866', '300867', '300868', '300869', '300870', '300871', '300872', '300873', '300875', '300876', '300877',
        '300878', '300879', '300880', '300881', '300882', '300883', '300884', '300885', '300886', '300887', '300888',
        '300889', '300890', '300891', '300892', '300893', '300894', '300895', '300896', '300897', '300898', '300899',
        '300900', '300901', '300902', '300903', '300904', '300905', '300906', '300907', '300908', '300909', '300910',
        '300911', '300912', '300913', '300915', '300916', '300917', '300918', '300919', '300920', '300921', '300922',
        '300923', '300925', '300926', '300927', '300928', '300929', '300930', '300931', '300932', '300933', '300935',
        '300936', '300937', '300938', '300939', '300940', '300941', '300942', '300943', '300945', '300946', '300947',
        '300948', '300949', '300950', '300951', '300952', '300953', '300955', '300956', '300957', '300958', '300959',
        '300960', '300961', '300962', '300963', '300964', '300965', '300966', '300967', '300968', '300969', '300970',
        '300971', '300972', '300973', '300975', '300976', '300977', '300978', '300979', '300980', '300981', '300982',
        '300983', '300984', '300985', '300986', '300987', '300988', '300989', '300990', '300991', '300992', '300993',
        '300994', '300995', '300996', '300997', '300998', '300999', '301000', '301001', '301002', '301003', '301004',
        '301005', '301006', '301007', '301008', '301009', '301010', '301011', '301012', '301013', '301015', '301016',
        '301017', '301018', '301019', '301020', '301021', '301022', '301023', '301024', '301025', '301026', '301027',
        '301028', '301029', '301030', '301031', '301032', '301033', '301035', '301036', '301037', '301038', '301039',
        '301040', '301041', '301042', '301043', '301045', '301046', '301047', '301048', '301049', '301050', '301051',
        '301052', '301053', '301055', '301056', '301057', '301058', '301059', '301060', '301061', '301062', '301063',
        '301065', '301066', '301067', '301068', '301069', '301070', '301071', '301072', '301073', '301075', '301076',
        '301077', '301078', '301079', '301080', '301081', '301082', '301083', '301085', '301086', '301087', '301088',
        '301089', '301090', '301091', '301092', '301093', '301095', '301096', '301097', '301098', '301099', '301100',
        '301101', '301102', '301103', '301105', '301106', '301107', '301108', '301109', '301110', '301111', '301112',
        '301113', '301115', '301116', '301117', '301118', '301119', '301120', '301121', '301122', '301123', '301125',
        '301126', '301127', '301128', '301129', '301130', '301131', '301132', '301133', '301135', '301136', '301137',
        '301138', '301139', '301141', '301148', '301149', '301150', '301151', '301152', '301153', '301155', '301156',
        '301157', '301158', '301159', '301160', '301161', '301162', '301163', '301165', '301166', '301167', '301168',
        '301169', '301170', '301171', '301172', '301173', '301175', '301176', '301177', '301178', '301179', '301180',
        '301181', '301182', '301183', '301185', '301186', '301187', '301188', '301189', '301190', '301191', '301192',
        '301193', '301195', '301196', '301197', '301198', '301199', '301200', '301201', '301202', '301203', '301205',
        '301206', '301207', '301208', '301209', '301210', '301211', '301212', '301213', '301215', '301216', '301217',
        '301218', '301219', '301220', '301221', '301222', '301223', '301225', '301226', '301227', '301228', '301229',
        '301230', '301231', '301232', '301233', '301234', '301235', '301236', '301237', '301238', '301239', '301246',
        '301248', '301251', '301252', '301255', '301256', '301257', '301258', '301259', '301260', '301261', '301262',
        '301263', '301265', '301266', '301267', '301268', '301269', '301270', '301272', '301273', '301275', '301276',
        '301277', '301278', '301279', '301280', '301281', '301282', '301283', '301285', '301286', '301287', '301288',
        '301289', '301290', '301291', '301292', '301293', '301295', '301296', '301297', '301298', '301299', '301300',
        '301301', '301302', '301303', '301305', '301306', '301307', '301308', '301309', '301310', '301311', '301312',
        '301313', '301314', '301315', '301316', '301317', '301318', '301319', '301320', '301321', '301322', '301323',
        '301325', '301326', '301327', '301328', '301329', '301330', '301331', '301332', '301333', '301335', '301336',
        '301337', '301338', '301339', '301345', '301348', '301349', '301353', '301355', '301356', '301357', '301358',
        '301359', '301360', '301361', '301362', '301363', '301365', '301366', '301367', '301368', '301369', '301370',
        '301371', '301372', '301373', '301376', '301377', '301378', '301379', '301380', '301381', '301382', '301383'

    ]

    result_df = batch_screen(
        stock_list=test_stocks,
        fetch_stock_func=fetch_stock,
        fetch_index_func=fetch_index,
        fetch_capital_func=fetch_capital,
        signal_filter=['上涨确立']
    )

    print_batch_results(result_df)

    all_html_parts.append('<h2>📋 上涨确立选股结果</h2>')
    all_html_parts.append(format_batch_results_html(result_df))

    if len(result_df) > 0:
        filename = f"上涨确立选股结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        result_df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"  结果已保存至: {filename}")

    # ==================== 推送到手机 ====================
    print("\n" + "=" * 60)
    print("  📱 正在通过 PushPlus 推送结果到手机...")
    print("=" * 60)

    title = f"📊 上涨确立选股报告二 {datetime.now().strftime('%m-%d %H:%M')}"
    full_html = '\n'.join(all_html_parts)
    send_pushplus(title, full_html)

