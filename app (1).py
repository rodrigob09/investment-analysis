import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DCF Valuation Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 { font-family: 'DM Serif Display', serif; }

.main { background-color: #FAFAF8; }

.metric-card {
    background: white;
    border: 1px solid #E8E4DC;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.metric-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8A8578;
    margin-bottom: 6px;
    font-weight: 500;
}
.metric-value {
    font-size: 26px;
    font-weight: 600;
    color: #1A1714;
    font-family: 'DM Serif Display', serif;
}
.metric-sub {
    font-size: 12px;
    color: #8A8578;
    margin-top: 4px;
}
.upside { color: #2D7A4F; }
.downside { color: #C0392B; }

.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.35rem;
    color: #1A1714;
    border-bottom: 2px solid #E8E4DC;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}
.assumption-box {
    background: #F7F4EE;
    border-left: 3px solid #C9A84C;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 13px;
    color: #4A4540;
}
.verdict-overvalued {
    background: #FDF0EF;
    border: 1.5px solid #E8A49A;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    color: #7D1F17;
}
.verdict-undervalued {
    background: #EDF7F2;
    border: 1.5px solid #7DC4A0;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    color: #1A5C38;
}
.verdict-fair {
    background: #F7F4EE;
    border: 1.5px solid #C9A84C;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    color: #6B4F1A;
}
.stAlert { border-radius: 10px; }
.sidebar-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #8A8578;
    font-weight: 500;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_financials(ticker: str):
    """Fetch all needed data from Yahoo Finance."""
    tk = yf.Ticker(ticker)
    info = tk.info
    cf = tk.cashflow
    inc = tk.income_stmt
    bs = tk.balance_sheet
    hist = tk.history(period="1y")
    return info, cf, inc, bs, hist


def get_fcf_series(cf: pd.DataFrame) -> pd.Series:
    """Extract Free Cash Flow = Operating CF - CapEx (last 4 years)."""
    try:
        opcf = cf.loc["Operating Cash Flow"] if "Operating Cash Flow" in cf.index else None
        capex = cf.loc["Capital Expenditure"] if "Capital Expenditure" in cf.index else None
        if opcf is None or capex is None:
            return None
        fcf = opcf + capex  # CapEx is negative in Yahoo
        return fcf.dropna()
    except Exception:
        return None


def get_revenue_series(inc: pd.DataFrame) -> pd.Series:
    try:
        rev = inc.loc["Total Revenue"] if "Total Revenue" in inc.index else None
        if rev is None:
            rev = inc.loc["Revenue"] if "Revenue" in inc.index else None
        return rev.dropna() if rev is not None else None
    except Exception:
        return None


def compute_wacc(info: dict, risk_free_rate: float, equity_risk_premium: float) -> float:
    """Estimate WACC from beta, debt/equity, tax rate."""
    beta = info.get("beta", 1.0) or 1.0
    ke = risk_free_rate + beta * equity_risk_premium  # CAPM cost of equity

    total_debt = info.get("totalDebt", 0) or 0
    market_cap = info.get("marketCap", 1) or 1
    tax_rate = 0.21  # US corporate

    # Cost of debt (approximate)
    interest_exp = info.get("interestExpense", None)
    if interest_exp and total_debt > 0:
        kd = abs(interest_exp) / total_debt
        kd = min(kd, 0.20)  # cap
    else:
        kd = 0.05

    total_value = market_cap + total_debt
    we = market_cap / total_value
    wd = total_debt / total_value

    wacc = we * ke + wd * kd * (1 - tax_rate)
    return max(wacc, 0.04)  # floor at 4%


def run_dcf(
    base_fcf: float,
    growth_rate_1: float,
    growth_rate_2: float,
    terminal_growth: float,
    wacc: float,
    stage1_years: int,
    stage2_years: int,
    shares: float,
    net_cash: float,
) -> dict:
    """Two-stage DCF model. Returns detailed yearly cash flows and intrinsic value."""
    cash_flows = []
    pv_cash_flows = []
    years = []

    fcf = base_fcf
    for y in range(1, stage1_years + 1):
        fcf *= (1 + growth_rate_1)
        pv = fcf / (1 + wacc) ** y
        cash_flows.append(fcf)
        pv_cash_flows.append(pv)
        years.append(f"Y{y}")

    for y in range(1, stage2_years + 1):
        fcf *= (1 + growth_rate_2)
        pv = fcf / (1 + wacc) ** (stage1_years + y)
        cash_flows.append(fcf)
        pv_cash_flows.append(pv)
        years.append(f"Y{stage1_years + y}")

    # Terminal value (Gordon Growth)
    terminal_fcf = fcf * (1 + terminal_growth)
    if wacc <= terminal_growth:
        wacc = terminal_growth + 0.01
    tv = terminal_fcf / (wacc - terminal_growth)
    pv_tv = tv / (1 + wacc) ** (stage1_years + stage2_years)

    sum_pv_fcf = sum(pv_cash_flows)
    enterprise_value = sum_pv_fcf + pv_tv + net_cash
    intrinsic_value = enterprise_value / shares if shares > 0 else 0

    return {
        "years": years,
        "cash_flows": cash_flows,
        "pv_cash_flows": pv_cash_flows,
        "terminal_value": tv,
        "pv_terminal_value": pv_tv,
        "sum_pv_fcf": sum_pv_fcf,
        "enterprise_value": enterprise_value,
        "intrinsic_value": intrinsic_value,
        "tv_pct": pv_tv / enterprise_value * 100 if enterprise_value > 0 else 0,
    }


def sensitivity_table(base_fcf, growth_rate_1, growth_rate_2, terminal_growth,
                       stage1_years, stage2_years, shares, net_cash,
                       wacc_center, tgr_center):
    """Build sensitivity table varying WACC and terminal growth rate."""
    wacc_range = [wacc_center - 0.02, wacc_center - 0.01, wacc_center,
                  wacc_center + 0.01, wacc_center + 0.02]
    tgr_range = [tgr_center - 0.01, tgr_center - 0.005, tgr_center,
                 tgr_center + 0.005, tgr_center + 0.01]

    rows = []
    for tgr in tgr_range:
        row = {}
        for w in wacc_range:
            if w <= tgr:
                row[f"{w*100:.1f}%"] = "N/A"
            else:
                res = run_dcf(base_fcf, growth_rate_1, growth_rate_2, tgr,
                              w, stage1_years, stage2_years, shares, net_cash)
                row[f"{w*100:.1f}%"] = f"${res['intrinsic_value']:.2f}"
        rows.append(row)

    df = pd.DataFrame(rows, index=[f"TGR {t*100:.2f}%" for t in tgr_range])
    df.index.name = "Terminal Growth \\ WACC"
    return df


# ── App Layout ────────────────────────────────────────────────────────────────

st.markdown("# 📊 DCF Valuation Tool")
st.markdown(
    "<p style='color:#8A8578; font-size:15px; margin-top:-10px;'>"
    "Discounted Cash Flow intrinsic value estimator using live financial data</p>",
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Inputs")

    ticker_input = st.text_input("Stock Ticker", value="AAPL", max_chars=10,
                                  help="Enter any valid Yahoo Finance ticker symbol").upper().strip()

    st.markdown("---")
    st.markdown("### 📈 Growth Assumptions")

    g1 = st.slider("Stage 1 growth rate (%)", 0.0, 40.0, 12.0, 0.5,
                   help="High-growth phase annual FCF growth") / 100
    stage1_yrs = st.slider("Stage 1 duration (years)", 1, 7, 5)

    g2 = st.slider("Stage 2 growth rate (%)", 0.0, 25.0, 7.0, 0.5,
                   help="Transition phase annual FCF growth") / 100
    stage2_yrs = st.slider("Stage 2 duration (years)", 1, 7, 5)

    tgr = st.slider("Terminal growth rate (%)", 0.5, 5.0, 2.5, 0.25,
                    help="Perpetual growth rate (usually ≈ long-run GDP growth)") / 100

    st.markdown("---")
    st.markdown("### 💰 Discount Rate")

    rfr = st.slider("Risk-free rate (%)", 2.0, 7.0, 4.5, 0.1,
                    help="Typically the 10-year US Treasury yield") / 100
    erp = st.slider("Equity risk premium (%)", 3.0, 8.0, 5.0, 0.25,
                    help="Damodaran's ERP estimate is ~4.5–5.5%") / 100

    st.markdown("---")
    run_btn = st.button("🔍 Run Valuation", use_container_width=True, type="primary")

# ── Main ──────────────────────────────────────────────────────────────────────

if run_btn or ticker_input:
    with st.spinner(f"Fetching data for **{ticker_input}**..."):
        try:
            info, cf, inc, bs, hist = fetch_financials(ticker_input)
        except Exception as e:
            st.error(f"Could not fetch data: {e}")
            st.stop()

    if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
        st.error(f"Ticker **{ticker_input}** not found. Please check the symbol.")
        st.stop()

    # ── Company header
    name = info.get("longName", ticker_input)
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
    market_cap = info.get("marketCap", 0)
    shares_outstanding = info.get("sharesOutstanding", 1) or 1

    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown(f"## {name} &nbsp; <span style='font-size:16px;color:#8A8578;font-family:DM Sans'>({ticker_input})</span>", unsafe_allow_html=True)
        st.markdown(f"<span style='font-size:13px;color:#8A8578'>{sector} · {industry}</span>", unsafe_allow_html=True)
    with col_h2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Current Price</div>
            <div class='metric-value'>${current_price:,.2f}</div>
            <div class='metric-sub'>Market cap: ${market_cap/1e9:.1f}B</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Extract financial data
    fcf_series = get_fcf_series(cf)
    rev_series = get_revenue_series(inc)

    if fcf_series is None or len(fcf_series) == 0:
        st.error("Could not extract Free Cash Flow data for this ticker. Try a different company (e.g. AAPL, MSFT, GOOGL).")
        st.stop()

    base_fcf = float(fcf_series.iloc[0])  # most recent year

    # Net cash
    total_cash = info.get("totalCash", 0) or 0
    total_debt = info.get("totalDebt", 0) or 0
    net_cash = total_cash - total_debt

    # WACC
    wacc = compute_wacc(info, rfr, erp)
    beta = info.get("beta", 1.0) or 1.0

    # Run DCF
    result = run_dcf(
        base_fcf, g1, g2, tgr, wacc,
        stage1_yrs, stage2_yrs, shares_outstanding, net_cash
    )
    iv = result["intrinsic_value"]

    if iv <= 0 or base_fcf <= 0:
        st.warning("⚠️ This company has negative or near-zero Free Cash Flow, making DCF unreliable. Consider using a different valuation method.")

    upside = (iv - current_price) / current_price * 100 if current_price > 0 else 0

    # ── Key metrics row
    st.markdown('<div class="section-header">Valuation Summary</div>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    direction = "upside" if upside > 5 else ("downside" if upside < -5 else "")

    with m1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Intrinsic Value</div>
            <div class='metric-value'>${iv:,.2f}</div>
            <div class='metric-sub'>Per share</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Market Price</div>
            <div class='metric-value'>${current_price:,.2f}</div>
            <div class='metric-sub'>Current</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Upside / Downside</div>
            <div class='metric-value {"upside" if upside > 0 else "downside"}'>{upside:+.1f}%</div>
            <div class='metric-sub'>vs. intrinsic value</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>WACC</div>
            <div class='metric-value'>{wacc*100:.2f}%</div>
            <div class='metric-sub'>β = {beta:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with m5:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Terminal Val. %</div>
            <div class='metric-value'>{result['tv_pct']:.0f}%</div>
            <div class='metric-sub'>of enterprise value</div>
        </div>""", unsafe_allow_html=True)

    # ── Verdict
    margin = abs(upside)
    if upside > 15:
        verdict_cls = "verdict-undervalued"
        verdict_icon = "🟢"
        verdict_text = f"<b>Potentially Undervalued</b> — Intrinsic value (${iv:,.2f}) is <b>{upside:.1f}% above</b> the current market price. Under these assumptions, the stock appears to offer upside."
    elif upside < -15:
        verdict_cls = "verdict-overvalued"
        verdict_icon = "🔴"
        verdict_text = f"<b>Potentially Overvalued</b> — Intrinsic value (${iv:,.2f}) is <b>{abs(upside):.1f}% below</b> the current market price. The market may be pricing in growth beyond these assumptions."
    else:
        verdict_cls = "verdict-fair"
        verdict_icon = "🟡"
        verdict_text = f"<b>Approximately Fairly Valued</b> — Intrinsic value (${iv:,.2f}) is within ±15% of market price. The stock appears roughly in line with these DCF assumptions."

    st.markdown(f"""
    <div class='{verdict_cls}' style='margin: 1rem 0;'>
        {verdict_icon} {verdict_text}<br>
        <span style='font-size:12px; opacity:0.75; margin-top:6px; display:block;'>
        ⚠️ This is a model output, not investment advice. Results are highly sensitive to growth and discount rate assumptions.
        </span>
    </div>""", unsafe_allow_html=True)

    # ── Charts row
    st.markdown('<div class="section-header">Cash Flow Analysis</div>', unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # FCF history bar
        if fcf_series is not None and len(fcf_series) > 0:
            fcf_df = pd.DataFrame({
                "Year": [str(d.year) for d in fcf_series.index],
                "FCF ($B)": [v / 1e9 for v in fcf_series.values]
            })
            colors = ["#2D7A4F" if v >= 0 else "#C0392B" for v in fcf_df["FCF ($B)"]]
            fig_fcf = go.Figure(go.Bar(
                x=fcf_df["Year"], y=fcf_df["FCF ($B)"],
                marker_color=colors, text=[f"${v:.1f}B" for v in fcf_df["FCF ($B)"]],
                textposition="outside"
            ))
            fig_fcf.update_layout(
                title="Historical Free Cash Flow",
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="DM Sans", size=12),
                height=300, margin=dict(t=40, b=20, l=20, r=20),
                yaxis=dict(gridcolor="#F0EDE6", title="FCF ($ Billions)"),
                xaxis=dict(title=""),
            )
            st.plotly_chart(fig_fcf, use_container_width=True)

    with chart_col2:
        # Projected FCF waterfall
        proj_labels = result["years"] + ["Terminal\nValue"]
        proj_pv = result["pv_cash_flows"] + [result["pv_terminal_value"]]
        colors_proj = ["#4A90D9"] * len(result["pv_cash_flows"]) + ["#C9A84C"]

        fig_proj = go.Figure(go.Bar(
            x=proj_labels, y=[v / 1e9 for v in proj_pv],
            marker_color=colors_proj,
            text=[f"${v/1e9:.1f}B" for v in proj_pv],
            textposition="outside"
        ))
        fig_proj.update_layout(
            title="Present Value of Projected Cash Flows",
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="DM Sans", size=12),
            height=300, margin=dict(t=40, b=20, l=20, r=20),
            yaxis=dict(gridcolor="#F0EDE6", title="PV ($ Billions)"),
            xaxis=dict(title=""),
        )
        st.plotly_chart(fig_proj, use_container_width=True)

    # ── EV Breakdown Pie
    st.markdown('<div class="section-header">Enterprise Value Breakdown</div>', unsafe_allow_html=True)

    pie_col, detail_col = st.columns([1, 1])

    with pie_col:
        pv_fcf_total = result["sum_pv_fcf"]
        pv_tv = result["pv_terminal_value"]
        nc = max(net_cash, 0)

        pie_labels = ["Stage 1+2 FCFs", "Terminal Value", "Net Cash"]
        pie_values = [pv_fcf_total / 1e9, pv_tv / 1e9, nc / 1e9]
        pie_colors = ["#4A90D9", "#C9A84C", "#2D7A4F"]

        fig_pie = go.Figure(go.Pie(
            labels=pie_labels, values=pie_values,
            hole=0.45, marker_colors=pie_colors,
            textinfo="label+percent",
            textfont=dict(family="DM Sans", size=13),
        ))
        fig_pie.update_layout(
            title="EV Components",
            font=dict(family="DM Sans"),
            height=300, margin=dict(t=40, b=20, l=20, r=20),
            paper_bgcolor="white",
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with detail_col:
        st.markdown("#### DCF Bridge")
        ev = result["enterprise_value"]
        rows_bridge = {
            "PV of Stage 1+2 FCFs": f"${pv_fcf_total/1e9:.2f}B",
            "PV of Terminal Value": f"${pv_tv/1e9:.2f}B",
            "Net Cash (Cash − Debt)": f"${net_cash/1e9:.2f}B",
            "**Enterprise Value**": f"**${ev/1e9:.2f}B**",
            "Shares Outstanding": f"{shares_outstanding/1e9:.2f}B",
            "**Intrinsic Value / Share**": f"**${iv:,.2f}**",
        }
        for k, v in rows_bridge.items():
            st.markdown(f"<div style='display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #F0EDE6;font-size:14px'><span>{k}</span><span>{v}</span></div>", unsafe_allow_html=True)

    # ── Sensitivity Table
    st.markdown('<div class="section-header">Sensitivity Analysis</div>', unsafe_allow_html=True)
    st.markdown("<p style='font-size:13px;color:#8A8578;'>Intrinsic value per share across different WACC and terminal growth rate combinations.</p>", unsafe_allow_html=True)

    with st.spinner("Computing sensitivity table..."):
        sens_df = sensitivity_table(
            base_fcf, g1, g2, tgr,
            stage1_yrs, stage2_yrs, shares_outstanding, net_cash,
            wacc, tgr
        )

    # Style: highlight cells close to current price
    def highlight_cell(val):
        try:
            v = float(val.replace("$", "").replace(",", ""))
            pct = (v - current_price) / current_price
            if pct > 0.15:
                return "background-color:#EDF7F2; color:#1A5C38; font-weight:600"
            elif pct < -0.15:
                return "background-color:#FDF0EF; color:#7D1F17; font-weight:600"
            else:
                return "background-color:#FFF9ED; color:#6B4F1A; font-weight:600"
        except:
            return ""

    styled = sens_df.style.map(highlight_cell)
    st.dataframe(styled, use_container_width=True)
    st.markdown("<p style='font-size:11px;color:#8A8578'>🟢 Green = >15% upside · 🟡 Yellow = fairly valued · 🔴 Red = >15% overvalued vs. current price</p>", unsafe_allow_html=True)

    # ── Assumptions summary
    st.markdown('<div class="section-header">Model Assumptions Used</div>', unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown(f"""
        <div class='assumption-box'>
        <b>Base FCF (most recent year):</b> ${base_fcf/1e9:.2f}B<br>
        <b>Stage 1:</b> {g1*100:.1f}% for {stage1_yrs} years<br>
        <b>Stage 2:</b> {g2*100:.1f}% for {stage2_yrs} years<br>
        <b>Terminal growth:</b> {tgr*100:.2f}%
        </div>""", unsafe_allow_html=True)
    with a2:
        ke = rfr + beta * erp
        st.markdown(f"""
        <div class='assumption-box'>
        <b>Risk-free rate:</b> {rfr*100:.1f}%<br>
        <b>Equity risk premium:</b> {erp*100:.1f}%<br>
        <b>Beta:</b> {beta:.2f}<br>
        <b>Cost of equity (CAPM):</b> {ke*100:.2f}%<br>
        <b>WACC:</b> {wacc*100:.2f}%
        </div>""", unsafe_allow_html=True)
    with a3:
        st.markdown(f"""
        <div class='assumption-box'>
        <b>Total cash:</b> ${total_cash/1e9:.2f}B<br>
        <b>Total debt:</b> ${total_debt/1e9:.2f}B<br>
        <b>Net cash:</b> ${net_cash/1e9:.2f}B<br>
        <b>Shares outstanding:</b> {shares_outstanding/1e9:.2f}B
        </div>""", unsafe_allow_html=True)

    # ── Price history chart
    st.markdown('<div class="section-header">12-Month Price History</div>', unsafe_allow_html=True)
    if not hist.empty:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=hist.index, y=hist["Close"],
            line=dict(color="#4A90D9", width=2),
            fill="tozeroy", fillcolor="rgba(74,144,217,0.08)",
            name="Close Price"
        ))
        fig_hist.add_hline(
            y=iv, line_dash="dash", line_color="#2D7A4F",
            annotation_text=f"Intrinsic Value ${iv:,.2f}",
            annotation_position="bottom right"
        )
        fig_hist.add_hline(
            y=current_price, line_dash="dot", line_color="#8A8578",
            annotation_text=f"Current ${current_price:,.2f}",
            annotation_position="top right"
        )
        fig_hist.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="DM Sans", size=12),
            height=320, margin=dict(t=20, b=20, l=20, r=80),
            yaxis=dict(gridcolor="#F0EDE6", title="Price ($)"),
            xaxis=dict(title=""),
            showlegend=False,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

else:
    # Landing state
    st.info("👈 Enter a stock ticker in the sidebar and click **Run Valuation** to begin.")
    st.markdown("""
    ### How to use this tool
    1. **Enter a ticker** in the sidebar (e.g. `AAPL`, `MSFT`, `GOOGL`, `AMZN`)
    2. **Adjust growth assumptions** — Stage 1 is the high-growth period; Stage 2 is a slower transition phase
    3. **Set a terminal growth rate** — typically close to long-run nominal GDP growth (~2–2.5%)
    4. **Review the WACC inputs** — the discount rate is auto-computed using CAPM and the company's beta
    5. **Interpret the output** — compare intrinsic value to market price and examine the sensitivity table

    ### What is DCF?
    Discounted Cash Flow (DCF) analysis estimates a company's intrinsic value by projecting future free cash flows and discounting them back to today's dollars using the Weighted Average Cost of Capital (WACC). If the intrinsic value exceeds the market price, the stock may be undervalued — and vice versa.

    ### Limitations
    - DCF is highly sensitive to growth rate and WACC assumptions
    - Companies with negative FCF (early-stage, capital-intensive) are difficult to value with DCF
    - Does not account for qualitative factors, competitive moats, or management quality
    - Always use alongside other valuation methods (P/E, EV/EBITDA, comparable transactions)
    """)
