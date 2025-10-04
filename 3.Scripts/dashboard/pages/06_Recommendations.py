import streamlit as st

st.set_page_config(page_title="Recommendations", layout="wide")

# Header image - CitiBike station in NYC
# Save your image locally and update this path, or upload to an image hosting service
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.image("https://images.pexels.com/photos/12168006/pexels-photo-12168006.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1")

st.title("Operational Recommendations")

# Executive Summary Card
st.markdown("""
<div style="background-color: #f0f2f6; padding: 25px; border-radius: 10px; 
            border-left: 5px solid #1e88e5; margin-bottom: 30px;">
    <h3 style="margin-top: 0; color: #1e3a5f;">Executive Summary</h3>
    <p style="font-size: 16px;"><strong>Two actionable recommendations from our 2022 data analysis:</strong></p>
    <ul style="font-size: 15px; line-height: 1.8;">
        <li><strong>Fleet Scaling:</strong> Implement month-by-month adjustments rather than broad seasonal categories</li>
        <li><strong>Waterfront Expansion:</strong> Launch targeted 7-station pilot instead of broad expansion</li>
    </ul>
    <p style="margin-bottom: 0; font-size: 14px; color: #666;">
        For complete analysis and methodology, refer to notebooks exploring questions Q1 and Q2.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==================== FLEET SCALING ====================
st.markdown("""
<div style="background: linear-gradient(90deg, #1565c0 0%, #1976d2 100%); 
            padding: 12px 20px; border-radius: 5px; margin: 20px 0;">
    <h2 style="color: white; margin: 0;">1. Fleet Scaling Throughout the Year</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("""
Our monthly demand analysis reveals significant seasonal variation, with peak demand ranging from 37% to 90% of maximum capacity.
""")

# Key metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Winter Low (January)", "37%", "-63% vs peak", delta_color="inverse")
with col2:
    st.metric("Spring Recovery (March)", "70%+", "+33% vs January")
with col3:
    st.metric("Peak Season Stability", "May-October", "Full capacity")

st.markdown("### Seasonal Breakdown")

# Winter period
with st.container():
    st.markdown("""
    **Winter Contraction (December-February):**
    - **December**: Demand drops to 55% → reduce fleet by 30-40%
    - **January**: Demand falls to under 40% → scale back fleet by approximately 50%
    - **February**: Demand recovers to 55% → restore fleet to 70%
    """)

# Spring period
with st.container():
    st.markdown("""
    **Spring Recovery (March-April):**
    - **March**: Demand exceeds 70% → increase fleet to 80-85%
    - **April**: Demand reaches 90% → restore full fleet capacity
    """)

# Peak and Fall
with st.container():
    st.markdown("""
    **Peak Season (May-October):**
    - Maintain full fleet capacity
    
    **Late Fall (November):**
    - Demand remains strong at 88% → no scale-back needed yet
    """)

# Recommendation box
st.markdown("""
<div style="background-color: #bbdefb; padding: 20px; border-radius: 8px; 
            border-left: 4px solid #1565c0; margin-top: 20px;">
    <h4 style="margin-top: 0; color: #0d47a1;">Recommendation</h4>
    <p style="margin-bottom: 0;">
    Implement <strong>month-by-month fleet adjustments</strong> rather than broad seasonal categories. 
    The wide variation in winter demand (37% to 72% excluding transition months) makes targeted 
    monthly scaling more effective than uniform seasonal reductions.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==================== WATERFRONT EXPANSION ====================
st.markdown("""
<div style="background: linear-gradient(90deg, #00897b 0%, #26a69a 100%); 
            padding: 12px 20px; border-radius: 5px; margin: 20px 0;">
    <h2 style="color: white; margin: 0;">2. Waterfront Station Expansion</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("""
Using a 300-meter proximity definition, our spatial analysis reveals a balanced system with localized pressure points.
""")

# Key findings
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Waterfront Endpoints", "14.2%", "of all trip endpoints")
with col2:
    st.metric("Waterfront Stations", "14.8%", "of all stations")
with col3:
    st.metric("Candidate Stations", "7", "for targeted expansion")

st.markdown("### Analysis Findings")

# System balance
with st.container():
    st.markdown("""
    **System-Level Balance:**
    - Waterfront endpoint share: 14.2%
    - Waterfront station share: 14.8%
    - **Conclusion**: No system-wide waterfront capacity shortfall
    """)

# Concentration
with st.container():
    st.markdown("""
    **Demand Concentration:**
    - 40 stations account for ~50% of all waterfront endpoints
    - Peak-hour shares: 10-15% (vs. 5.9% baseline across 17-hour active day)
    - **Span50** (hours to reach 50% of daily trips): Mostly 5-7 hours
    - Half of daily volume concentrates in roughly one-quarter of operating hours
    """)

# Interpretation
with st.container():
    st.markdown("""
    **Interpretation:**
    
    Peaks are real but not extreme. Demand concentration suggests localized pressure points 
    rather than system-wide shortage. The data supports targeted interventions at specific 
    high-stress locations rather than broad infrastructure expansion.
    """)

# Recommendation box
st.markdown("""
<div style="background-color: #b2dfdb; padding: 20px; border-radius: 8px; 
            border-left: 4px solid #00897b; margin-top: 20px;">
    <h4 style="margin-top: 0; color: #004d40;">Recommendation</h4>
    <p style="margin-bottom: 0;">
    Rather than broad waterfront expansion, <strong>target the most time-concentrated hotspots 
    with a pilot program</strong>. We identified 7 stations as candidates based on peak-hour 
    congestion patterns (>12% of daily trips in single hour OR >30% in top-3 hours with tight 
    daily spread). This focused approach allows for testing and learning before committing to 
    larger-scale infrastructure investments.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Footer note
st.markdown("""
<div style="text-align: center; color: #666; font-size: 14px; margin-top: 40px;">
    <p>These recommendations are based on comprehensive analysis of 2022 CitiBike trip data. 
    Refer to the analysis notebooks for detailed methodology and findings.</p>
</div>
""", unsafe_allow_html=True)