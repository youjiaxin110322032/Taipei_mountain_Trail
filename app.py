import io
import time

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import urllib3

# 使用 verify=False 時會出現 InsecureRequestWarning，這裡關閉警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="資料查詢與商品分析", page_icon="📊", layout="wide")

# ------------------------------------------------------------------
# 中文字型處理：
# 原本 Colab 的 `!wget` 下載字型 + matplotlib font_manager 的作法，
# 在 Streamlit 不需要。圖表改用 Plotly（由瀏覽器渲染），直接指定
# 使用者電腦上的中文字型即可，不必在伺服器安裝任何字型檔。
# ------------------------------------------------------------------
CJK_FONT = "Microsoft JhengHei, PingFang TC, Noto Sans TC, Noto Sans CJK TC, sans-serif"


def style_fig(fig):
    fig.update_layout(font=dict(family=CJK_FONT, size=14), margin=dict(t=60, l=10, r=10, b=10))
    return fig


# ==================================================================
# 第一部分：臺北市開放資料（合併 code1 + code2）
# ==================================================================
DATASETS = {
    "資料集 1": (
        "https://data.taipei/api/dataset/b5726297-d172-4ba7-b5c4-31de38e184e1"
        "/resource/0d1d7db3-efc1-40d1-ad24-5a1a1f88e06b/download"
    ),
    "資料集 2": (
        "https://data.taipei/api/dataset/a98a3e0e-a36f-43fa-82f8-b09a3011a47a"
        "/resource/ae7e5986-859d-4294-b289-7c1b2e7c23f1/download"
    ),
    "自訂網址": "",
}


@st.cache_data(ttl=3600, show_spinner="資料下載中...")
def load_csv(url: str, verify_ssl: bool) -> pd.DataFrame:
    """下載 CSV 並轉成 DataFrame（快取 1 小時）"""
    response = requests.get(url, verify=verify_ssl, timeout=30)
    response.raise_for_status()
    content = response.content

    # 繁體中文：先試 utf-8-sig，失敗再試 big5 / cp950
    for enc in ("utf-8-sig", "big5", "cp950"):
        try:
            return pd.read_csv(io.BytesIO(content), encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("無法判斷檔案編碼，請確認資料來源。")


def page_open_data():
    st.title("📊 臺北市資料查詢")

    with st.sidebar:
        st.header("⚙️ 資料設定")
        choice = st.selectbox("選擇資料集", list(DATASETS.keys()))
        url = st.text_input("資料網址 (CSV)", value=DATASETS[choice])
        verify_ssl = st.checkbox("驗證 SSL 憑證", value=False)
        drop_na = st.checkbox("移除含空值的列 (dropna)", value=True)
        if st.button("🔄 重新下載資料"):
            st.cache_data.clear()
            st.rerun()

    if not url:
        st.info("請在左側輸入 CSV 網址。")
        return

    try:
        df = load_csv(url, verify_ssl)
    except Exception as e:
        st.error(f"資料載入失敗：{e}")
        return

    raw_rows = len(df)
    if drop_na:
        df = df.dropna()

    c1, c2, c3 = st.columns(3)
    c1.metric("原始筆數", f"{raw_rows:,}")
    c2.metric("目前筆數", f"{len(df):,}")
    c3.metric("欄位數", len(df.columns))

    # 篩選
    with st.sidebar:
        st.header("🔍 篩選")
        cols = st.multiselect("顯示欄位", df.columns.tolist(), default=df.columns.tolist())
        keyword = st.text_input("關鍵字搜尋（所有欄位）")

    view = df[cols] if cols else df
    if keyword:
        mask = (
            view.astype(str)
            .apply(lambda col: col.str.contains(keyword, case=False, na=False))
            .any(axis=1)
        )
        view = view[mask]

    tab1, tab2, tab3 = st.tabs(["資料表", "統計摘要", "圖表"])

    with tab1:
        st.caption(f"共 {len(view):,} 筆")
        st.dataframe(view, width="stretch")
        st.download_button(
            "⬇️ 下載目前結果 (CSV)",
            data=view.to_csv(index=False).encode("utf-8-sig"),
            file_name="taipei_data_filtered.csv",
            mime="text/csv",
        )

    with tab2:
        st.write("數值欄位統計")
        st.dataframe(view.describe(), width="stretch")
        st.write("欄位型態")
        st.dataframe(
            pd.DataFrame({"型態": view.dtypes.astype(str), "空值數": view.isna().sum()}),
            width="stretch",
        )

    with tab3:
        num_cols = view.select_dtypes("number").columns.tolist()
        if num_cols:
            chart_col = st.selectbox("選擇欄位畫圖", num_cols)
            st.bar_chart(view[chart_col])
        else:
            st.info("目前結果沒有數值欄位可繪圖。")


# ==================================================================
# 第二部分：商品爬蟲（PChome 24h 搜尋 API）+ 折線圖 / 圓餅圖 / 旭日圖
# ==================================================================
PCHOME_API = "https://ecshweb.pchome.com.tw/search/v3.3/all/results"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; StreamlitDemo/1.0)"}
SORT_OPTIONS = {
    "銷量高到低": "sale/dc",
    "相關度": "rnk/dc",
    "價格低到高": "prc/ac",
    "價格高到低": "prc/dc",
}


def _to_price(v):
    """price 欄位可能是數字或 dict，統一轉成數字"""
    if isinstance(v, dict):
        v = v.get("P") or v.get("M") or v.get("Low")
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


@st.cache_data(ttl=1800, show_spinner=False)
def scrape_pchome(keyword: str, pages: int, sort: str):
    """爬取 PChome 搜尋結果，回傳 (DataFrame, 錯誤訊息或 None)"""
    rows, error = [], None
    for page in range(1, pages + 1):
        try:
            r = requests.get(
                PCHOME_API,
                params={"q": keyword, "page": page, "sort": sort},
                headers=HEADERS,
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            error = f"第 {page} 頁抓取失敗：{e}"
            break

        prods = data.get("prods") or data.get("Prods") or []
        if not prods:
            break
        for i, p in enumerate(prods, start=1):
            pid = p.get("Id", "")
            rows.append(
                {
                    "頁數": page,
                    "頁內序號": i,
                    "商品名稱": p.get("name", ""),
                    "價格": _to_price(p.get("price")),
                    "原價": _to_price(p.get("originPrice")),
                    "連結": f"https://24h.pchome.com.tw/prod/{pid}" if pid else "",
                }
            )
        if page >= int(data.get("totalPage", page) or page):
            break
        time.sleep(0.5)  # 禮貌性延遲，避免對網站造成負擔

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["價格"]).reset_index(drop=True)
        df.insert(0, "商品序號", df.index + 1)
    return df, error


def add_price_range(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """依價格切成 n 個等寬區間，產生可讀的區間標籤"""
    df = df.copy()
    lo, hi = df["價格"].min(), df["價格"].max()
    if lo == hi:
        df["價格區間"] = f"{int(lo):,}"
        return df
    edges = np.linspace(lo, hi, n_bins + 1)
    labels = [f"{int(edges[i]):,}–{int(edges[i + 1]):,}" for i in range(n_bins)]
    df["價格區間"] = pd.cut(df["價格"], bins=edges, labels=labels, include_lowest=True).astype(str)
    return df


def page_scraper():
    st.title("🛒 商品爬蟲與價格分析")
    st.caption("資料來源：PChome 24h 購物搜尋。請適度使用，避免短時間大量請求。")

    with st.form("scrape_form"):
        c1, c2, c3 = st.columns([3, 1, 2])
        keyword = c1.text_input("產品名稱", value="行動電源")
        pages = c2.number_input("爬取頁數", min_value=1, max_value=10, value=3, step=1)
        sort_label = c3.selectbox("排序方式", list(SORT_OPTIONS.keys()))
        submitted = st.form_submit_button("🚀 開始爬取")

    if submitted:
        if not keyword.strip():
            st.warning("請輸入產品名稱。")
        else:
            with st.spinner(f"正在爬取「{keyword}」前 {pages} 頁..."):
                df, err = scrape_pchome(keyword.strip(), int(pages), SORT_OPTIONS[sort_label])
            st.session_state["scrape"] = (keyword.strip(), df, err)

    if "scrape" not in st.session_state:
        st.info("請輸入產品名稱與頁數後，按下「開始爬取」。")
        return

    kw, df, err = st.session_state["scrape"]
    if err:
        st.warning(err)
    if df.empty:
        st.error(f"找不到「{kw}」的商品資料。")
        return

    df = add_price_range(df)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("商品數", f"{len(df):,}")
    m2.metric("平均價格", f"{df['價格'].mean():,.0f}")
    m3.metric("最低價", f"{df['價格'].min():,.0f}")
    m4.metric("最高價", f"{df['價格'].max():,.0f}")

    t_data, t_line, t_pie, t_sun = st.tabs(["資料表", "折線圖", "圓餅圖", "旭日圖"])

    with t_data:
        st.dataframe(
            df.drop(columns=["價格區間"]),
            width="stretch",
            column_config={"連結": st.column_config.LinkColumn("連結", display_text="開啟")},
        )
        st.download_button(
            "⬇️ 下載爬取結果 (CSV)",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"pchome_{kw}.csv",
            mime="text/csv",
        )

    with t_line:
        fig = px.line(
            df,
            x="商品序號",
            y="價格",
            color=df["頁數"].map(lambda p: f"第 {p} 頁"),
            markers=True,
            hover_name="商品名稱",
            title=f"「{kw}」商品價格走勢（依搜尋排序）",
            labels={"color": "頁數"},
        )
        st.plotly_chart(style_fig(fig), width="stretch")

        page_avg = df.groupby("頁數", as_index=False)["價格"].mean()
        fig2 = px.line(
            page_avg, x="頁數", y="價格", markers=True, title="各頁平均價格"
        )
        fig2.update_xaxes(dtick=1)
        st.plotly_chart(style_fig(fig2), width="stretch")

    with t_pie:
        pie_df = df["價格區間"].value_counts().rename_axis("價格區間").reset_index(name="商品數")
        fig = px.pie(
            pie_df,
            names="價格區間",
            values="商品數",
            hole=0.35,
            title=f"「{kw}」價格區間占比",
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(style_fig(fig), width="stretch")

    with t_sun:
        sun_df = df.copy()
        sun_df["第幾頁"] = sun_df["頁數"].map(lambda p: f"第 {p} 頁")
        sun_df["商品"] = sun_df["商品名稱"].str.slice(0, 18) + "…#" + sun_df["商品序號"].astype(str)
        fig = px.sunburst(
            sun_df,
            path=["第幾頁", "價格區間", "商品"],
            values="價格",
            title=f"「{kw}」：頁數 → 價格區間 → 商品",
        )
        fig.update_layout(height=700)
        st.plotly_chart(style_fig(fig), width="stretch")


# ==================================================================
# 主程式
# ==================================================================
with st.sidebar:
    st.title("功能選單")
    page = st.radio("選擇功能", ["📊 臺北市開放資料", "🛒 商品爬蟲分析"], label_visibility="collapsed")
    st.divider()

if page.startswith("📊"):
    page_open_data()
else:
    page_scraper()
