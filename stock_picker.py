import os
import re
import time
import random
import numpy as np
import pandas as pd
import requests
import warnings
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
                 signal_filter=None):
    index_df = fetch_index_func()
    results = []
    total = len(stock_list)

    for idx, code in enumerate(stock_list):
        print(f"\r[{idx + 1}/{total}] 正在分析 {code} ...", end='', flush=True)
        try:
            df = fetch_stock_func(code)
            if df is None or len(df) < 120:
                continue
            cap = fetch_capital_func(code)
            common = df.index.intersection(index_df.index)
            if len(common) < 120:
                continue
            df_a = df.loc[common].copy()
            idx_a = index_df.loc[common].copy()
            res = screen_single_stock(df_a, idx_a, cap)
            all_signals = res['买入信号'] + res['卖出信号']
            if not all_signals:
                continue
            if signal_filter:
                if not any(s in all_signals for s in signal_filter):
                    continue
            date_str = pd.Timestamp(res['日期']).strftime('%Y-%m-%d')
            results.append({
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
            })
        except Exception as e:
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
        '000001', '000002', '000004', '000006', '000007', '000008',
        '000009', '000010', '000011', '000012', '000014', '000016',
        '000017', '000019', '000020', '000021', '000025', '000026',
        '000027', '000028', '000029', '000030', '000031', '000032',
        '000034', '000035', '000036', '000037', '000039', '000042',
        '000045', '000048', '000049', '000050', '000055', '000056',
        '000058', '000059', '000060', '000061', '000062', '000063',
        '000065', '000066', '000068', '000069'
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

    title = f"📊 上涨确立选股报告 {datetime.now().strftime('%m-%d %H:%M')}"
    full_html = '\n'.join(all_html_parts)
    send_pushplus(title, full_html)
