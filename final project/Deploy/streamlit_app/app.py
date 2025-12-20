"""
Credit Card Customer Segmentation - Streamlit App
==================================================
Web app for customer segmentation using trained KMeans model.

Features:
- Upload CSV files
- Automatic preprocessing and prediction
- Display persona names
- Download results with cluster assignments
- Marketing strategy recommendations
"""

from __future__ import annotations

from pathlib import Path
import io
import sys

import pandas as pd
import streamlit as st

# Add parent directory to path for imports
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from segmentation_model import SegmentationModel


# Configuration
DEFAULT_ARTIFACT = PROJECT_DIR / "model_artifacts" / "credit_segmentation_k4.joblib"


# Page config
st.set_page_config(
    page_title="Credit Card Customer Segmentation",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #566573;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8F9F9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86C1;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">💳 Credit Card Customer Segmentation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Phân khúc khách hàng thẻ tín dụng với K-Means Clustering</div>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Cấu hình")
    
    artifact_path = st.text_input(
        "Đường dẫn model artifact (.joblib)",
        value=str(DEFAULT_ARTIFACT),
        help="Đường dẫn tới file .joblib chứa model đã train"
    )
    
    st.divider()
    
    st.subheader("Tùy chọn hiển thị")
    show_preview = st.checkbox("Xem preview dữ liệu", value=True)
    show_persona = st.checkbox("Hiển thị tên persona", value=True)
    show_marketing = st.checkbox("Hiển thị chiến lược marketing", value=True)
    
    st.divider()
    
    st.subheader("ℹ️ Thông tin")
    st.caption("Phiên bản: 1.0.0")
    st.caption("Ngày cập nhật: 2025-12-20")


# Load model
@st.cache_resource
def load_model(p: str) -> SegmentationModel:
    """Load model artifact with caching."""
    return SegmentationModel.load(p)


try:
    model = load_model(artifact_path)
    cluster_names = model.get_cluster_names()
except FileNotFoundError:
    st.error(f"❌ Không tìm thấy file artifact tại: `{artifact_path}`")
    st.info("💡 Hãy chạy notebook để export model trước khi sử dụng app này.")
    st.stop()
except Exception as e:
    st.error(f"❌ Không load được artifact: {e}")
    st.stop()


# Display model info
with st.expander("📊 Thông tin Model", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Số cluster (K)", model.k)
    with col2:
        st.metric("Số features", len(model.preprocessor.feature_names_))
    with col3:
        st.metric("Random state", model.random_state)
    
    # Display persona names
    if cluster_names:
        st.subheader("Persona đã học")
        persona_df = pd.DataFrame([
            {"Cluster": k, "Persona Name": v} 
            for k, v in sorted(cluster_names.items())
        ])
        st.dataframe(persona_df, use_container_width=True, hide_index=True)


# Main section
st.divider()
st.header("📤 1. Upload dữ liệu")

uploaded = st.file_uploader(
    "Chọn file CSV chứa dữ liệu khách hàng",
    type=["csv"],
    help="File CSV phải có cấu trúc tương tự file train (CC GENERAL.csv)"
)

if uploaded is None:
    st.info("💡 Hãy upload file CSV để bắt đầu phân khúc khách hàng.")
    st.info("📁 Gợi ý: Sử dụng file `Dataset/CC GENERAL.csv` để test.")
    st.stop()

# Read uploaded file
raw_bytes = uploaded.read()
try:
    df_in = pd.read_csv(io.BytesIO(raw_bytes))
    st.success(f"✅ Đã load {len(df_in):,} khách hàng với {len(df_in.columns)} cột")
except Exception as e:
    st.error(f"❌ Không đọc được CSV: {e}")
    st.stop()

# Preview data
if show_preview:
    with st.expander("👁️ Preview dữ liệu (5 dòng đầu)", expanded=True):
        st.dataframe(df_in.head(), use_container_width=True)


# Predict clusters
st.divider()
st.header("🎯 2. Kết quả phân khúc")

with st.spinner("Đang xử lý và phân khúc khách hàng..."):
    try:
        labels = model.predict(df_in)
    except Exception as e:
        st.error(f"❌ Predict lỗi: {e}")
        st.stop()

# Prepare result dataframe
df_out = df_in.copy()
df_out["Cluster"] = labels.astype(int)

if show_persona:
    df_out["Persona"] = df_out["Cluster"].map(cluster_names)

st.success("✅ Hoàn thành phân khúc!")

# Display results
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Phân phối Cluster")
    
    if show_persona:
        # Count by persona
        persona_counts = df_out.groupby(["Cluster", "Persona"]).size().reset_index(name="Count")
        persona_counts["Label"] = persona_counts.apply(
            lambda row: f"C{row['Cluster']}: {row['Persona'][:35]}", axis=1
        )
        chart_data = persona_counts.set_index("Label")["Count"]
    else:
        chart_data = df_out["Cluster"].value_counts().sort_index()
    
    st.bar_chart(chart_data, height=350)

with col2:
    st.subheader("📋 Kết quả mẫu (10 dòng)")
    
    # Select columns to display
    display_cols = ["Cluster"]
    if show_persona:
        display_cols.append("Persona")
    
    # Add first few columns from input
    if "CUST_ID" in df_in.columns:
        display_cols.append("CUST_ID")
    
    numeric_cols = [c for c in ["BALANCE", "PURCHASES", "CREDIT_LIMIT"] if c in df_in.columns]
    display_cols.extend(numeric_cols[:3])
    
    # Ensure all columns exist
    display_cols = [c for c in display_cols if c in df_out.columns]
    
    st.dataframe(
        df_out[display_cols].head(10),
        use_container_width=True,
        hide_index=True
    )

# Statistics
st.divider()
st.subheader("📈 Thống kê chi tiết")

if show_persona:
    summary = df_out.groupby(["Cluster", "Persona"]).size().reset_index(name="Số KH")
    summary["Tỷ lệ"] = (summary["Số KH"] / len(df_out) * 100).round(1).astype(str) + "%"
else:
    summary = df_out["Cluster"].value_counts().reset_index()
    summary.columns = ["Cluster", "Số KH"]
    summary["Tỷ lệ"] = (summary["Số KH"] / len(df_out) * 100).round(1).astype(str) + "%"

st.dataframe(summary, use_container_width=True, hide_index=True)


# Download section
st.divider()
st.header("💾 3. Download kết quả")

csv_bytes = df_out.to_csv(index=False).encode("utf-8-sig")
filename = "segmented_customers_with_persona.csv" if show_persona else "segmented_customers.csv"

st.download_button(
    label=f"📥 Download CSV {'(kèm Persona)' if show_persona else '(kèm cột Cluster)'}",
    data=csv_bytes,
    file_name=filename,
    mime="text/csv",
    use_container_width=True
)


# Marketing strategy section
if show_marketing:
    st.divider()
    st.header("🎯 4. Chiến lược Marketing theo Persona")
    
    # Mapping persona keywords to marketing strategies
    campaign_map = {
        "Cash-Advance Heavy": {
            "icon": "⚠️",
            "title": "Kiểm soát rủi ro",
            "strategies": [
                "Giảm hạn mức ứng tiền mặt",
                "Tăng phí ứng tiền để khuyến khích chuyển sang trả góp",
                "Cảnh báo sớm về tình trạng tài chính",
                "Cross-sell sản phẩm vay cá nhân với lãi suất thấp hơn"
            ]
        },
        "Low Activity": {
            "icon": "🎯",
            "title": "Kích hoạt khách hàng",
            "strategies": [
                "Welcome back campaign với ưu đãi hấp dẫn",
                "Miễn phí thường niên năm đầu",
                "Cashback 10-20% cho giao dịch đầu tiên",
                "Gamification: tích điểm khi sử dụng thường xuyên"
            ]
        },
        "VIP": {
            "icon": "👑",
            "title": "Chăm sóc VIP",
            "strategies": [
                "Tăng hạn mức tín dụng không cần yêu cầu",
                "Cashback 3-5% không giới hạn",
                "Quyền lợi cao cấp: Lounge, Concierge, Bảo hiểm",
                "Ưu tiên hỗ trợ 24/7"
            ]
        },
        "Installment": {
            "icon": "🛍️",
            "title": "Thúc đẩy trả góp",
            "strategies": [
                "Partnership với BNPL platforms",
                "0% lãi suất cho trả góp 3-6 tháng",
                "Cashback thêm khi chọn trả góp",
                "Merchant offers tại các đối tác lớn"
            ]
        },
        "Revolver": {
            "icon": "💳",
            "title": "Quản lý nợ",
            "strategies": [
                "Balance transfer với lãi suất ưu đãi",
                "Chuyển đổi sang trả góp có lãi suất cố định",
                "Financial education: webinar quản lý chi tiêu",
                "Increase alerts khi sắp đến hạn mức"
            ]
        },
        "Regular": {
            "icon": "🔄",
            "title": "Duy trì và phát triển",
            "strategies": [
                "Chương trình tích điểm ổn định",
                "Cashback 1-2% cho tất cả giao dịch",
                "Cross-sell: Thẻ tín dụng bổ sung, Bảo hiểm",
                "Referral bonus: Giới thiệu bạn bè nhận thưởng"
            ]
        }
    }
    
    for cluster_id in sorted(df_out["Cluster"].unique()):
        persona_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        count = (df_out["Cluster"] == cluster_id).sum()
        pct = count / len(df_out) * 100
        
        # Find matching strategy
        strategy_info = None
        for keyword, info in campaign_map.items():
            if keyword.lower() in persona_name.lower():
                strategy_info = info
                break
        
        # Default strategy if no match
        if strategy_info is None:
            strategy_info = {
                "icon": "📊",
                "title": "Theo dõi và đánh giá",
                "strategies": ["Phân tích hành vi chi tiêu", "Thiết kế chiến dịch phù hợp"]
            }
        
        with st.expander(
            f"{strategy_info['icon']} **Cluster {cluster_id}: {persona_name}** "
            f"({count:,} KH - {pct:.1f}%)",
            expanded=False
        ):
            st.markdown(f"### {strategy_info['title']}")
            
            for strategy in strategy_info['strategies']:
                st.markdown(f"- {strategy}")
            
            # Show profile stats if available
            numeric_cols = df_in.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                st.markdown("---")
                st.markdown("**Profile trung bình (top 5 metrics):**")
                
                cluster_data = df_out[df_out["Cluster"] == cluster_id][numeric_cols[:5]]
                if len(cluster_data) > 0:
                    stats_df = pd.DataFrame({
                        "Metric": cluster_data.columns,
                        "Mean": cluster_data.mean().round(2).values
                    })
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)


# Footer
st.divider()
st.caption("© 2025 Credit Card Customer Segmentation | Powered by Streamlit & scikit-learn")
