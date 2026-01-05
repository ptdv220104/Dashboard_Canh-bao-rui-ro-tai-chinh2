import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# 1. CẤU HÌNH TRANG
# =====================================================
st.set_page_config(
    page_title="Early Warning System – Financial Risk",
    layout="wide"
)

# =====================================================
# 2. CSS – GIAO DIỆN & SIDEBAR
# =====================================================
st.markdown("""
<style>
.main { background-color: #f5f7fb; }

.card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

h1, h2, h3 {
    color: #1f2937;
    font-weight: 700;
}

[data-testid="stMetric"] {
    background: white;
    padding: 16px;
    border-radius: 14px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827, #1f2937);
    min-width: 280px !important;
    max-width: 280px !important;
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb;
    font-size: 15px;
}

div[role="radiogroup"] label {
    white-space: nowrap;
}

thead tr th {
    background-color: #e5e7eb !important;
    color: #111827 !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.title("HỆ THỐNG CẢNH BÁO SỚM RỦI RO TÀI CHÍNH DOANH NGHIỆP")

# =====================================================
# 3. LOAD DATA (KHỚP JUPYTER)
# =====================================================
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "Ket_qua_du_bao.csv")
    df = pd.read_csv(file_path)
    df["nam"] = df["nam"].astype(int)
    df = df[(df["nam"] >= 2019) & (df["nam"] <= 2024)]

    return df

df = load_data()

# =====================================================
# 4. MENU
# =====================================================
page = st.sidebar.radio(
    "📌 Điều hướng",
    [
        "📊 Tổng quan hệ thống",
        "🌍 Toàn cảnh thị trường",
        "🏭 Phân tích theo ngành",
        "🏢 Phân tích doanh nghiệp",
        "🚨 Cảnh báo & So sánh",
    ]
)

# =====================================================
# 📊 TRANG 1 – TỔNG QUAN HỆ THỐNG
# =====================================================
if page == "📊 Tổng quan hệ thống":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Tổng quan hệ thống")

    st.markdown("""
    **Hệ thống cảnh báo sớm rủi ro tài chính cho doanh nghiệp phi tài chính niêm yết tại Việt Nam**
    được xây dựng dựa trên dữ liệu báo cáo tài chính và kết quả từ các mô hình Machine Learning.
    Hệ thống tập trung vào theo dõi **xu hướng rủi ro**, **so sánh động theo thời gian**
    và **phân tích đa cấp độ** từ thị trường, ngành đến từng doanh nghiệp.
    """)

    col1, col2, col3, col4 = st.columns(4)
    total_firms = df["ma_ck"].nunique()
    risky_firms = df[df["target"] == 1]["ma_ck"].nunique()

    col1.metric("Số DN phân tích", total_firms)
    col2.metric("DN rủi ro (target = 1)", risky_firms)
    col3.metric("Tỷ lệ DN rủi ro (%)", round(risky_firms / total_firms * 100, 2))
    col4.metric("Năm dữ liệu mới nhất", df["nam"].max())
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Cơ cấu doanh nghiệp theo trạng thái tài chính")

    latest_year = df["nam"].max()

    df_latest = (
        df[df["nam"] == latest_year]
        .sort_values("ma_ck")
        .drop_duplicates(subset="ma_ck", keep="last")
    )


    # PHÂN LOẠI TRẠNG THÁI TÀI CHÍNH THEO RISK SCORE
    def classify_status(score):
        if score < 40:
            return "An toàn"
        elif score < 70:
            return "Cảnh báo"
        else:
            return "Nguy cơ cao"


    df_latest["Trang_thai_tai_chinh"] = df_latest["diem_rui_ro"].apply(classify_status)

    pie_df = (
        df_latest
        .groupby("Trang_thai_tai_chinh")["ma_ck"]
        .nunique()
        .reset_index()
    )

    fig = px.pie(
        pie_df,
        values="ma_ck",
        names="Trang_thai_tai_chinh",
        hole=0.5,
        color="Trang_thai_tai_chinh",
        color_discrete_map={
            "An toàn": "#2ecc71",
            "Cảnh báo": "#f1c40f",
            "Nguy cơ cao": "#e74c3c"
        }
    )

    fig.update_traces(
        textinfo="percent+label",
        hovertemplate="%{label}: %{value} DN (%{percent})"
    )

    fig.update_layout(
        title=f"Cơ cấu doanh nghiệp theo trạng thái tài chính – năm {latest_year}"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# 🌍 TRANG 2 – TOÀN CẢNH THỊ TRƯỜNG
# =====================================================
elif page == "🌍 Toàn cảnh thị trường":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Toàn cảnh rủi ro tài chính thị trường")

    market = df.groupby("nam")["diem_rui_ro"].mean().reset_index()
    market["delta"] = market["diem_rui_ro"].diff()

    st.plotly_chart(
        px.line(market, x="nam", y="diem_rui_ro", markers=True),
        use_container_width=True
    )

    latest = market.iloc[-1]
    st.info(
        f"Năm {int(latest['nam'])}, Risk Score trung bình thị trường "
        f"{'tăng' if latest['delta'] > 0 else 'giảm'} "
        f"{abs(latest['delta']):.2f} điểm so với năm trước."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Heatmap Risk Score trung bình theo Ngành – Năm")

    heat = df.groupby(["nganh", "nam"])["diem_rui_ro"].mean().reset_index()
    heat_pivot = heat.pivot(index="nganh", columns="nam", values="diem_rui_ro")

    st.plotly_chart(
        px.imshow(heat_pivot, aspect="auto", color_continuous_scale="RdYlGn_r"),
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# 🏭 TRANG 3 – PHÂN TÍCH THEO NGÀNH
# =====================================================
elif page == "🏭 Phân tích theo ngành":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("So sánh xu hướng rủi ro theo ngành")

    # =====================================================
    # 1. BỘ ĐIỀU KHIỂN
    # =====================================================
    years = sorted(df["nam"].unique())
    base_year = st.selectbox("Năm gốc", years, index=0)
    compare_year = st.selectbox(
        "Năm so sánh",
        [y for y in years if y > base_year]
    )

    top_n = st.slider("Chọn Top ngành hiển thị", 5, 30, 15)

    # =====================================================
    # 2. TÍNH TOÁN CORE
    # =====================================================
    df_base = df[df["nam"] == base_year]
    df_comp = df[df["nam"] == compare_year]

    industry_cmp = (
        df_base.groupby("nganh")["diem_rui_ro"].mean()
        .to_frame("Năm gốc")
        .join(
            df_comp.groupby("nganh")["diem_rui_ro"].mean().to_frame("Năm so sánh"),
            how="inner"
        )
    )

    industry_cmp["Chênh lệch"] = industry_cmp["Năm so sánh"] - industry_cmp["Năm gốc"]
    industry_cmp = industry_cmp.sort_values("Chênh lệch", ascending=False)

    # =====================================================
    # 3. BIỂU ĐỒ CHÍNH – DIVERGING BAR (HIỆN ĐẠI)
    # =====================================================
    industry_plot = industry_cmp.head(top_n).reset_index()

    fig = px.bar(
        industry_plot,
        x="Chênh lệch",
        y="nganh",
        orientation="h",
        color="Chênh lệch",
        color_continuous_scale="RdYlGn_r",
        title=f"Thay đổi Risk Score theo ngành ({compare_year} so với {base_year})"
    )

    fig.update_layout(
        xaxis_title="Chênh lệch Risk Score",
        yaxis_title="Ngành",
        coloraxis_showscale=False,
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # 4. KPI INSIGHT NHANH
    # =====================================================
    top_worst = industry_cmp.index[0]
    top_best = industry_cmp.index[-1]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Ngành rủi ro tăng mạnh nhất",
        top_worst,
        round(industry_cmp.loc[top_worst, "Chênh lệch"], 2)
    )

    c2.metric(
        "Ngành cải thiện tốt nhất",
        top_best,
        round(industry_cmp.loc[top_best, "Chênh lệch"], 2)
    )

    c3.metric(
        "Chênh lệch Risk Score TB",
        round(industry_cmp["Chênh lệch"].mean(), 2)
    )

    # =====================================================
    # 5. PHÂN PHỐI RỦI RO – BOX PLOT (CHIỀU SÂU)
    # =====================================================
    st.subheader("Phân phối Risk Score theo ngành (năm so sánh)")

    fig_box = px.box(
        df[df["nam"] == compare_year],
        x="nganh",
        y="diem_rui_ro",
        points="outliers"
    )

    fig_box.update_layout(
        xaxis_title="Ngành",
        yaxis_title="Risk Score",
        height=450
    )

    st.plotly_chart(fig_box, use_container_width=True)

    # =====================================================
    # 6. BẢNG TRA CỨU CHI TIẾT
    # =====================================================
    with st.expander("Xem bảng so sánh chi tiết theo ngành"):
        st.dataframe(
            industry_cmp.round(2),
            use_container_width=True
        )

    st.markdown('</div>', unsafe_allow_html=True)


# =====================================================
# 🏢 TRANG 4 – PHÂN TÍCH DOANH NGHIỆP (NÂNG CẤP)
# =====================================================
elif page == "🏢 Phân tích doanh nghiệp":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Phân tích rủi ro theo doanh nghiệp")

    # =====================================================
    # A. PHÂN TÍCH XU HƯỚNG – SO SÁNH (MACRO)
    # =====================================================
    st.subheader("A. Xu hướng Risk Score theo thời gian")

    nganh_list = st.multiselect(
        "Chọn ngành (phân tích xu hướng)",
        sorted(df["nganh"].dropna().unique())
    )

    if nganh_list:
        df_macro = df[df["nganh"].isin(nganh_list)].copy()

        year_min, year_max = int(df_macro["nam"].min()), int(df_macro["nam"].max())
        year_range = st.slider(
            "Chọn khoảng năm",
            min_value=year_min,
            max_value=year_max,
            value=(year_min, year_max),
            key="year_range_macro"
        )

        df_macro = df_macro[
            (df_macro["nam"] >= year_range[0]) &
            (df_macro["nam"] <= year_range[1])
            ]

        ma_list = st.multiselect(
            "Chọn doanh nghiệp để so sánh",
            sorted(df_macro["ma_ck"].unique()),
            key="ma_list_macro"
        )

        if ma_list:
            df_multi = (
                df_macro[df_macro["ma_ck"].isin(ma_list)]
                .sort_values(["ma_ck", "nam"])
            )

            industry_avg = (
                df_macro.groupby("nam", as_index=False)["diem_rui_ro"]
                .mean()
                .sort_values("nam")
            )

            fig_macro = go.Figure()

            for m in ma_list:
                tmp = df_multi[df_multi["ma_ck"] == m]
                fig_macro.add_trace(go.Scatter(
                    x=tmp["nam"],
                    y=tmp["diem_rui_ro"],
                    mode="lines+markers",
                    name=m
                ))

            fig_macro.add_trace(go.Scatter(
                x=industry_avg["nam"],
                y=industry_avg["diem_rui_ro"],
                mode="lines",
                line=dict(dash="dash", width=3),
                name="Trung bình ngành"
            ))

            fig_macro.update_layout(
                xaxis_title="Năm",
                yaxis_title="Risk Score",
                hovermode="x unified"
            )

            st.plotly_chart(fig_macro, use_container_width=True)
        else:
            st.info("Chọn ít nhất một doanh nghiệp để hiển thị biểu đồ.")
    else:
        st.info("Chọn ngành để bắt đầu phân tích.")

    # =====================================================
    # B. PHÂN TÍCH CHI TIẾT DOANH NGHIỆP (MICRO)
    # =====================================================
    st.subheader("B. Phân tích chi tiết theo doanh nghiệp")

    ma_ck_detail = st.selectbox(
        "Chọn mã cổ phiếu",
        sorted(df["ma_ck"].unique()),
        key="ma_ck_detail"
    )

    df_micro = df[df["ma_ck"] == ma_ck_detail].sort_values("nam")

    year = st.selectbox(
        "Chọn năm phân tích",
        sorted(df_micro["nam"].unique()),
        key="year_detail"
    )

    row = df_micro[df_micro["nam"] == year].iloc[0]

    st.markdown(
        f"""
        **Tên công ty:** {row.get("ten_cong_ty", "Không có dữ liệu")}  
        **Ngành:** {row["nganh"]}  
        **Năm phân tích:** {year}
        """
    )

    # =====================================================
    # KPI + GAUGE RISK SCORE (HIỆN ĐẠI)
    # =====================================================
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("ROA", round(row["roa_tre1"], 4))
    c2.metric("ROE", round(row["roe_tre1"], 4))
    c3.metric("Nợ / VCSH", round(row["no_von_chu_so_huu_tre1"], 2))
    c4.metric("Khả năng trả lãi", round(row["kha_nang_tra_lai_tre1"], 2))

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=row["diem_rui_ro"],
        title={"text": "Risk Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 40], "color": "#2ecc71"},
                {"range": [40, 70], "color": "#f1c40f"},
                {"range": [70, 100], "color": "#e74c3c"}
            ],
            "bar": {"color": "black"}
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # =====================================================
    # NHẬN XÉT & CẢNH BÁO
    # =====================================================
    comments = []

    if row["roa_tre1"] < 0:
        comments.append("ROA âm → hiệu quả sinh lời kém.")
    if row["roe_tre1"] < 0:
        comments.append("ROE âm → lợi ích cổ đông suy giảm.")
    if row["no_von_chu_so_huu_tre1"] > 2:
        comments.append("Đòn bẩy tài chính cao (Nợ/VCSH > 2).")
    if row["kha_nang_tra_lai_tre1"] < 1:
        comments.append("Khả năng trả lãi yếu, tiềm ẩn rủi ro thanh khoản.")

    with st.expander("📌 Nhận xét & cảnh báo tự động"):
        if comments:
            for c in comments:
                st.warning(c)
        else:
            st.success("Các chỉ tiêu tài chính tương đối ổn định.")

    # =====================================================
    # RADAR CHART
    # =====================================================
    radar_labels = ["ROA", "ROE", "Nợ/VCSH", "Khả năng trả lãi", "Risk Score"]
    radar_values = [
        row["roa_tre1"],
        row["roe_tre1"],
        row["no_von_chu_so_huu_tre1"],
        row["kha_nang_tra_lai_tre1"],
        row["diem_rui_ro"]
    ]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=radar_values,
        theta=radar_labels,
        fill="toself"
    ))
    fig_radar.update_layout(showlegend=False)
    st.plotly_chart(fig_radar, use_container_width=True)

    # =====================================================
    # SO SÁNH VỚI TRUNG BÌNH NGÀNH
    # =====================================================
    industry_avg_year = (
        df[
            (df["nganh"] == row["nganh"]) &
            (df["nam"] == year)
            ]["diem_rui_ro"].mean()
    )

    compare_df = pd.DataFrame({
        "Đối tượng": ["Doanh nghiệp", "Trung bình ngành"],
        "Risk Score": [row["diem_rui_ro"], industry_avg_year]
    })

    st.plotly_chart(
        px.bar(compare_df, x="Đối tượng", y="Risk Score"),
        use_container_width=True
    )

    st.markdown('</div>', unsafe_allow_html=True)



# =====================================================
# 🚨 TRANG 5 – CẢNH BÁO & SO SÁNH (KHÔNG LẶP BIỂU ĐỒ)
# =====================================================
elif page == "🚨 Cảnh báo & So sánh":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Biến động Risk Score mạnh nhất theo doanh nghiệp")

    # =========================
    # TÍNH TOÁN
    # =========================
    # CHỌN NĂM SO SÁNH
    # =========================
    years = sorted(df["nam"].unique())

    c_year1, c_year2 = st.columns(2)
    with c_year1:
        base_year = st.selectbox("Năm gốc", years, index=0)
    with c_year2:
        compare_year = st.selectbox(
            "Năm so sánh",
            [y for y in years if y > base_year]
        )

    # =========================
    # TÍNH TOÁN BIẾN ĐỘNG
    # =========================
    df_base = df[df["nam"] == base_year]
    df_comp = df[df["nam"] == compare_year]

    tmp = (
        df_base[["ma_ck", "diem_rui_ro"]]
        .merge(
            df_comp[["ma_ck", "diem_rui_ro"]],
            on="ma_ck",
            how="inner",
            suffixes=("_base", "_comp")
        )
    )

    tmp["delta"] = tmp["diem_rui_ro_comp"] - tmp["diem_rui_ro_base"]

    top_up = tmp.sort_values("delta", ascending=False).head(10)
    top_down = tmp.sort_values("delta").head(10)

    # =========================
    # =========================
    # KPI INSIGHT (AN TOÀN)
    # =========================
    c1, c2 = st.columns(2)

    if not top_up.empty:
        c1.metric(
            "DN tăng Risk Score mạnh nhất",
            top_up.iloc[0]["ma_ck"],
            round(top_up.iloc[0]["delta"], 2)
        )
    else:
        c1.metric(
            "DN tăng Risk Score mạnh nhất",
            "Không đủ dữ liệu",
            "-"
        )

    if not top_down.empty:
        c2.metric(
            "DN giảm Risk Score mạnh nhất",
            top_down.iloc[0]["ma_ck"],
            round(top_down.iloc[0]["delta"], 2)
        )
    else:
        c2.metric(
            "DN giảm Risk Score mạnh nhất",
            "Không đủ dữ liệu",
            "-"
        )

    # =========================
    # BIỂU ĐỒ SO SÁNH (SIDE-BY-SIDE)
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top DN rủi ro tăng mạnh")
        fig_up = px.bar(
            top_up,
            x="delta",
            y="ma_ck",
            orientation="h",
            title="Gia tăng Risk Score",
            color="delta",
            color_continuous_scale=["yellow", "red"]
        )
        fig_up.update_layout(
            xaxis_title="Mức tăng Risk Score",
            yaxis_title="Mã cổ phiếu",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_up, use_container_width=True)

    with col2:
        st.subheader("Top DN rủi ro giảm mạnh")
        fig_down = px.bar(
            top_down,
            x="delta",
            y="ma_ck",
            orientation="h",
            title="Suy giảm Risk Score",
            color="delta",
            color_continuous_scale=["green", "yellow"]
        )
        fig_down.update_layout(
            xaxis_title="Mức giảm Risk Score",
            yaxis_title="Mã cổ phiếu",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_down, use_container_width=True)

    # =========================
    # BẢNG TRA CỨU (PHỤ)
    # =========================
    with st.expander("Xem chi tiết dữ liệu"):
        table_df = pd.concat([
            top_up.assign(Xu_hướng="Tăng"),
            top_down.assign(Xu_hướng="Giảm")
        ])

        table_df = table_df[[
            "ma_ck",
            "diem_rui_ro_base",
            "diem_rui_ro_comp",
            "delta",
            "Xu_hướng"
        ]].rename(columns={
            "diem_rui_ro_base": f"Risk Score {base_year}",
            "diem_rui_ro_comp": f"Risk Score {compare_year}",
            "delta": "Chênh lệch Risk Score"
        })

        st.dataframe(table_df.round(2), use_container_width=True)





