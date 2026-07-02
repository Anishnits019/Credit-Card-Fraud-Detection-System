import folium
from folium.plugins import HeatMap
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit.components.v1 as components
import os
from dotenv import load_dotenv
load_dotenv()

# 🛠️ FIX 1: st.set_page_config must be the absolute FIRST Streamlit command called
st.set_page_config(layout="wide") 

@st.cache_data
def get_cleaned_data():
    s3_destination_path="s3://fraud-detection-pipeline-anish-964043552068-ap-south-1-an/processed_data/processed_data.parquet"
    aws_credentials = {
    "key": os.getenv("AWS_ACCESS_KEY_ID"),
    "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "client_kwargs": {
        "region_name": os.getenv("AWS_DEFAULT_REGION")
    }
}
    df=pd.read_parquet(s3_destination_path,
                engine='fastparquet',
                index=False,
                storage_options=aws_credentials)
    return df

df = get_cleaned_data()

st.title("💳 Credit Card Fraud Analytics Dashboard")

page = st.sidebar.radio("Go to:", ["Overview & Demographics", "Geographic Analysis", "Behavioral Analysis"])

available_categorical_features = ['category', 'merchant', 'job', 'city']

# ==============================================================================
# VIEW 1: OVERVIEW & DEMOGRAPHICS (Your Main Landing View)
# ==============================================================================
if page == "Overview & Demographics":
    st.markdown("### 📊 System Overview & Baseline Diagnostics")
    
    with st.expander("📖 Quick Reference Guide: What do these WoE Scores Mean?"):
        st.markdown("""
        ### Weight of Evidence (WoE) Risk Scale
        | WoE Value | What it Means | Risk Profile |
        | :--- | :--- | :--- |
        | **Positive Score ($>0$)** | The category has a higher share of legitimate transactions than fraudulent ones. | 🟢 Safe / Low Risk |
        | **Zero Score ($=0$)** | The category's fraud-to-legitimate ratio perfectly matches the global average. | 🟡 Neutral / Baseline |
        | **Negative Score ($<0$)** | The category has a higher share of fraudulent transactions than legitimate ones. | 🔴 Dangerous / High Risk |
        """)
        
    target_feature = st.selectbox("Choose a Categorical Axis to Compute WoE Profile:", 
        options=available_categorical_features
    )

    st.markdown(f"#### Processing Data Matrix for Feature Node: `{target_feature.upper()}`")

    category_pivot = df.pivot_table(index=target_feature, columns="is_fraud", aggfunc="size", fill_value=0).reset_index()
    category_pivot.columns = [target_feature, 'not_fraud', 'fraud']
    category_pivot.columns.name = None

    total_global_not_fraud = category_pivot['not_fraud'].sum()
    total_global_fraud = category_pivot['fraud'].sum()

    category_pivot['per_not_fraud'] = category_pivot['not_fraud'] / total_global_not_fraud
    category_pivot['per_fraud'] = category_pivot['fraud'] / total_global_fraud

    epsilon = 1e-4
    category_pivot['WoE'] = np.log((category_pivot['per_not_fraud'] + epsilon) / (category_pivot['per_fraud'] + epsilon))

    category_iv = ((category_pivot['per_not_fraud'] - category_pivot['per_fraud']) * category_pivot['WoE']).sum()

    print(f"Category IV: {category_iv:.4f}")
    if target_feature in ["merchant", "job", "city"]:
        category_pivot = category_pivot.nsmallest(15, 'WoE')
        
    fig_woe_dynamic = px.bar(
        category_pivot,
        x=target_feature,
        y="WoE",
        color="WoE",
        color_continuous_scale=px.colors.sequential.RdBu, 
        title=f"Dynamic WoE Spectrum Distribution — {target_feature.upper()}",
        labels={"WoE": "Weight of Evidence (Risk Scale Values)", target_feature: f"System Log: {target_feature.title()}"},
        height=400 + (len(category_pivot) * 15)
    )
    st.plotly_chart(fig_woe_dynamic, use_container_width=True)

    # --- Dynamic Strategic Insights Panel ---
    st.markdown("---")
    st.markdown("### 🔍 Enterprise Risk Intel & Takeaways")

    if target_feature == 'category':
        st.info("""
    **Executive Risk Summary (CATEGORY):**
    An evaluation of behavioral transaction streams using Weight of Evidence (WoE) reveals a clean, structurally bifurcated distribution. The ecosystem exhibits clear boundaries separating low-risk physical transactions from high-vulnerability Card-Not-Present (CNP) digital channels.

    #### 🔴 High-Threat Risk Vectors (Negative WoE Bounds)
    * **`shopping_net` (WoE ~ -1.15) & `misc_net` (WoE ~ -0.92):** These web-based, digital commerce nodes represent the highest concentration of exposure in the network. Because online checkouts are Card-Not-Present environments, automated fraud syndicates target these endpoints to execute scalable credential-stuffing attacks and liquidate balances using stolen card profiles.
    * **`grocery_net` (WoE ~ -0.88):** Unlike physical supermarkets, online grocery delivery portals show an inflation in risk. Fraudsters exploit these platforms for immediate consumer goods delivery or digital gift card flipping, leveraging weaker authentication barriers typical of rapid-delivery applications.

    #### 🟢 Low-Risk Operational Baselines (Positive WoE Bounds)
    * **`grocery_pos` (WoE ~ +1.22), `health_fitness` (WoE ~ +1.22), & `home` (WoE ~ +1.20):** These brick-and-mortar segments are highly secure. Point-of-Sale (**POS**) transactions require a physical card presence, EMV chip cryptograms, or biometric device taps. Because these channels require physical proximity and are heavily covered by merchant CCTV surveillance, they carry a highly negative correlation with fraudulent behavior.
    * **`food_dining` & `entertainment` (WoE > +0.85):** These represent experiential, real-time localized transactions which historically act as a baseline indicator of genuine customer utility.

    ---

    ### 🛠️ Operational Deployment Strategy (For the Risk Engine)
    1.  **Dynamic Authentication Stepping:** Enforce mandatory 3D-Secure (3DS) multi-factor verification challenges exclusively for transactions routed through `shopping_net` and `misc_net` whenever a transaction deviates from the baseline historical spend velocity.
    2.  **Velocity Floor Adjustments:** Automatically reduce the dollar-amount authorization ceiling for transaction processing windows labeled under high-risk `_net` suffixes, while maintaining frictionless, accelerated routing paths for verified physical `_pos` endpoints to optimize cardholder retention.
    """)

    elif target_feature == 'merchant':
        st.info("""
    **Executive Risk Summary (MERCHANT):**
    An evaluation of the `merchant` dimension using Weight of Evidence (WoE) allows the system to isolate point-source payment vulnerabilities. Because merchant tracking involves thousands of individual nodes, this view programmatically utilizes an `nsmallest(15)` filter to isolate the highest-threat vectors operating across the payment network. 

    Consequently, the entire visible distribution sits within negative bounds ($< -1.15$), representing hyper-concentrations of fraud.

    #### 🔴 Primary Threat Actors (Maximum Negative Scale)
    * **`fraud_Kozey-Boehm` (WoE ~ -1.40):** This node represents the absolute highest statistical risk in the database. When an individual merchant endpoint drops to a WoE of -1.40, it mathematically proves that the ratio of fraudulent activity routed through this terminal vastly outpaces legitimate transactions. 
    * **`fraud_Herman, Treutel and Dickens` & `fraud_Terry-Huel` (WoE < -1.25):** These nodes represent critical risk centers flashing on the high-threat heat signature.

    #### 🔍 Operational Diagnosis (Why these patterns exist)
    In enterprise risk infrastructure, an all-negative, high-threat merchant cluster typically flags one of two operational realities:
    1. **Mule store / Shell Fronts:** Fraud syndicates frequently set up dummy online storefronts or compromise weak digital merchant portals specifically to run automated scripts that test batches of stolen card credentials (carding attacks).
    2. **Data Breaches / Point of Compromise:** These specific merchants may have experienced a security breach, meaning a high percentage of cards used here are subsequently flagged as compromised.

    ---

    ### 🛠️ Operational Deployment Strategy (For the Risk Engine)
    1. **Automated Terminal Holds:** Ingress payment traffic matching the top 3 highest-risk merchant IDs (`Kozey-Boehm`, etc.) should bypass standard soft routing rules and be forced into an immediate clearing hold or strict multi-factor 3D-Secure verification.
    2. **Dynamic Risk-Score Overrides:** Integrate these specific WoE values as direct penalty multipliers within the real-time machine learning inference pipeline to suppress card authorization approvals automatically when transaction velocity spikes at these locations.
    """)

    elif target_feature == 'job':
        st.info("""
        **Analytical Takeaway (JOB):** The occupation profile exposes clear social-engineering and demographic targeting patterns.
        Professions like **'Dancers'** or **'Air Traffic Controllers'** plunge down to a WoE near **-3.0**, proving that individuals within these operational profiles are significantly over-represented in fraud incidents. 
        
        *Strategic Action:* These occupational clusters can serve as high-signal risk vectors. Cardholders matching these specific profiles can be pre-emptively monitored with tighter anomaly detection thresholds for out-of-pattern spending.
        """)

    elif target_feature == 'city':
        st.info("""
        **Analytical Takeaway (CITY):** Geographic aggregation via WoE uncovers highly localized fraud rings. 
        Cities displaying significant negative WoE spikes indicate locations where localized identity theft, mail interception, or coordinate-based skimming devices are heavily active.
        
        *Strategic Action:* Deploy region-specific fraud parameters to dynamically flag physical point-of-sale swipes radiating from these high-vulnerability metropolitan centers.
        """)

# ==============================================================================
# VIEW 2: GEOGRAPHIC ANALYSIS
# ==============================================================================
elif page == "Geographic Analysis":
    st.title("🗺️ Spatial & Geographic Analysis")
    st.write("Analyze fraud data at different regional resolutions: National (States), Regional (ZIP Codes), or Local (Cities).")
    
    analysis_level = st.radio(
        "Select Analysis Resolution:",
        options=["National View (State Level)", "State View (ZIP Level)", "City View (Local Level)"]
    )

    us_states_url = "https://raw.githubusercontent.com/python-visualization/folium/main/examples/data/us-states.json"

    if analysis_level == "National View (State Level)":
        state_fraud = df.groupby('state').agg(
            fraud_count=('is_fraud', 'sum')
        ).reset_index()
        state_fraud = state_fraud[state_fraud['fraud_count'] > 0]

        map_center = [37.0902, -95.7129]
        zoom_level = 4
        cat_map = folium.Map(location=map_center, zoom_start=zoom_level, tiles="CartoDB positron")

        if not state_fraud.empty:
            folium.Choropleth(
                geo_data=us_states_url,
                name="US State Fraud Choropleth",
                data=state_fraud,
                columns=["state", "fraud_count"],
                key_on="feature.id",
                fill_color="YlOrRd",
                fill_opacity=0.7,
                line_color="#000000",
                line_weight=3,
                line_opacity=1.0,
                legend_name="Total Fraud Count by State"
            ).add_to(cat_map)
        else:
            st.warning("⚠️ No fraud signatures recorded across the country.")

    elif analysis_level == "State View (ZIP Level)":
        state_options = sorted(df['state'].dropna().unique().tolist())
        target_state = st.selectbox("Select a Target State:", options=state_options)
        
        filtered_df = df[df['state'] == target_state].copy()

        cat_fraud = filtered_df.groupby('zip').agg(
            fraud_count=('is_fraud', 'sum'),
            lat=('lat', 'mean'),
            long=('long', 'mean')
        ).reset_index()
        cat_fraud = cat_fraud[cat_fraud['fraud_count'] > 0]

        if not cat_fraud.empty:
            map_center = [cat_fraud['lat'].mean(), cat_fraud['long'].mean()]
            zoom_level = 5
        else:
            map_center = [37.0902, -95.7129]
            zoom_level = 4

        cat_map = folium.Map(location=map_center, zoom_start=zoom_level, tiles="CartoDB positron")

        if not cat_fraud.empty:
            heat_data = cat_fraud[['lat', 'long', 'fraud_count']].values.tolist()
            folium.GeoJson(
                us_states_url, 
                name="US State Borders", 
                style_function=lambda x: {
                    'fillColor': 'transparent', 
                    'color': '#000000', 
                    'weight': 3,
                    'opacity': 1.0
                }
            ).add_to(cat_map)
            HeatMap(heat_data, radius=15, max_zoom=10).add_to(cat_map)
        else:
            st.warning(f"⚠️ No fraud signatures recorded matching the state of {target_state}.")

    elif analysis_level == "City View (Local Level)":
        city_options = sorted(df['city'].dropna().unique().tolist())
        target_city = st.selectbox("Select a Target City:", options=city_options)
        
        filtered_df = df[df['city'] == target_city].copy()

        cat_fraud = filtered_df.groupby('zip').agg(
            fraud_count=('is_fraud', 'sum'),
            lat=('lat', 'mean'),
            long=('long', 'mean')
        ).reset_index()
        cat_fraud = cat_fraud[cat_fraud['fraud_count'] > 0]

        if not cat_fraud.empty:
            map_center = [cat_fraud['lat'].mean(), cat_fraud['long'].mean()]
            zoom_level = 5
        else:
            map_center = [37.0902, -95.7129]
            zoom_level = 4

        cat_map = folium.Map(location=map_center, zoom_start=zoom_level, tiles="CartoDB positron")

        if not cat_fraud.empty:
            heat_data = cat_fraud[['lat', 'long', 'fraud_count']].values.tolist()
            folium.GeoJson(
                us_states_url, 
                name="US State Borders", 
                style_function=lambda x: {
                    'fillColor': 'transparent', 
                    'color': '#000000', 
                    'weight': 3,
                    'opacity': 1.0
                }
            ).add_to(cat_map)
            HeatMap(heat_data, radius=15, max_zoom=3).add_to(cat_map)
        else:
            st.warning(f"⚠️ No fraud signatures recorded matching the city of {target_city}.")

    import streamlit.components.v1 as components
    cat_map.save("fraud_hotspots_map.html")
    
    with open("fraud_hotspots_map.html", 'r') as f:
        html_map = f.read()
    components.html(html_map, height=500, scrolling=True)
    
    st.info("💡 Switch options above to explore data grouped by state shapes or local transaction coordinates.")

elif page == "Behavioral Analysis":
    st.title("🕵️‍♂️ Fraud Behavioral & Transaction Metrics")
    st.write("Analyzing temporal patterns, spending velocity, and high-risk transaction profiles.")

    col1, col2, col3 = st.columns(3)
    
    total_fraud_cases = int(df['is_fraud'].sum())
    col1.metric("Total Fraud Cases", f"{total_fraud_cases:,}")
    
    avg_fraud_amt = df[df['is_fraud'] == 1]['amt'].mean()
    col2.metric("Avg Fraudulent Amount", f"${avg_fraud_amt:.2f}")
    
    avg_legit_amt = df[df['is_fraud'] == 0]['amt'].mean()
    col3.metric("Avg Legitimate Amount", f"${avg_legit_amt:.2f}")
    
    st.markdown("---")

    if 'trans_date_trans_time' in df.columns:
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['hour'] = df['trans_date_trans_time'].dt.hour
        
        hourly_analysis = df.groupby(['hour', 'is_fraud']).size().unstack(fill_value=0).reset_index()
        hourly_analysis.columns = ['Hour of Day', 'Legitimate', 'Fraudulent']
        
        st.subheader("⏰ Temporal Heat: When Does Fraud Occur?")
        fraud_per_hour=px.bar(hourly_analysis,
                x='Hour of Day',
                y='Fraudulent',
                color="Fraudulent",
                color_continuous_scale=px.colors.sequential.Reds,
                title="Fraudulent Transactions by Hour of Day",
                labels={"Fraudulent": "Number of Fraudulent Transactions", "Hour of Day": "Hour of Day"} 
                )
        st.plotly_chart(fraud_per_hour, use_container_width=True)


    st.markdown("---")

    R = 6371
    lat1 = np.radians(df['lat'])
    lon1 = np.radians(df['long'])
    lat2 = np.radians(df['merch_lat'])
    lon2 = np.radians(df['merch_long'])

    dlat = lat2 - lat1
    dlong = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlong / 2) ** 2
    d = 2 * np.arcsin(np.sqrt(a))
    df['distance'] = 6371 * d

    bins = [0, 5, 10, 20, 50, 100, 150]
    labels = ['0-5 km', '5-10 km', '10-20 km', '20-50 km', '50-100 km', '100-150 km']
    df['distance_bin'] = pd.cut(df['distance'], bins=bins, labels=labels, right=False)
    
    distance_fraud_pivot = df.pivot_table(index="distance_bin", columns="is_fraud", aggfunc="size", fill_value=0).reset_index()
    distance_amt_pivot = df.pivot_table(index="distance_bin", columns="is_fraud", values="amt", aggfunc="mean", fill_value=0).reset_index()
    
    distance_fraud_pivot.columns = ['Distance Range', 'Legitimate Count', 'Fraud Count']
    distance_amt_pivot.columns = ['Distance Range', 'Avg Legit Amt', 'Avg Fraud Amt']

    st.subheader("📊 Volume vs Value Analysis Across Distance Ranges")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚨 Transaction Frequency")
        st.caption("Absolute metric velocity tracking total fraudulent events.")
        
        st.dataframe(
            distance_fraud_pivot,
            column_config={
                "Distance Range": "Distance Bracket",
                "Legitimate Count": st.column_config.NumberColumn("Legit Volume", format="%d"),
                "Fraud Count": st.column_config.NumberColumn("Fraud Volume", format="%d")
            },
            use_container_width=True,
            hide_index=True
        )
        
        import plotly.express as px
        
        distance_freq_chart = px.bar(
            distance_fraud_pivot,
            x="Distance Range",
            y="Fraud Count",
            color="Fraud Count",
            color_continuous_scale=px.colors.sequential.Reds, 
            title="Fraud Event Velocity Spectrum — Distance Range",
            height=400 + (len(distance_fraud_pivot) * 15)
        )
        st.plotly_chart(distance_freq_chart, use_container_width=True)
        
    with col2:
        st.markdown("### 💰 Financial Ticket Exposure")
        st.caption("Average numerical spending value deviations across distances.")
        
        st.dataframe(
            distance_amt_pivot,
            column_config={
                "Distance Range": "Distance Bracket",
                "Avg Legit Amt": st.column_config.NumberColumn("Avg Legit Ticket", format="$%.2f"),
                "Avg Fraud Amt": st.column_config.NumberColumn("Avg Fraud Ticket", format="$%.2f")
            },
            use_container_width=True,
            hide_index=True
        )
        
        distance_val_chart = px.bar(
            distance_amt_pivot,
            x="Distance Range",
            y="Avg Fraud Amt",
            color="Avg Fraud Amt",
            color_continuous_scale=px.colors.sequential.Reds, 
            title="Average Fraud Value Magnitude — Distance Range",
            height=400 + (len(distance_amt_pivot) * 15)
        )
        st.plotly_chart(distance_val_chart, use_container_width=True)


#     st.subheader("💰 Transaction Value Distribution")
    
# max_amt = float(df['amt'].max())
# amt_range = st.slider("Filter Transaction Amount Range ($):", min_value=0.0, max_value=max_amt, value=(0.0, min(2000.0, max_amt)))

# dist_df = df[(df['amt'] >= amt_range[0]) & (df['amt'] <= amt_range[1])].copy()

# chart_data = dist_df.groupby(['category', 'is_fraud']).size().unstack(fill_value=0).reset_index()
# chart_data.columns = ['Category', 'Legitimate', 'Fraudulent']

# if 'Legitimate' not in chart_data.columns:
#     chart_data['Legitimate'] = 0
# if 'Fraudulent' not in chart_data.columns:
#     chart_data['Fraudulent'] = 0

# chart_data = chart_data.sort_values(by='Fraudulent', ascending=False)

# st.dataframe(
#     chart_data,
#     column_config={
#         "Category": "Merchant Category",
#         "Legitimate": st.column_config.ProgressColumn("Legit Count", format="%d", min_value=0, max_value=int(chart_data['Legitimate'].max()) if not chart_data.empty else 1),
#         "Fraudulent": st.column_config.ProgressColumn("Fraud Count", format="%d", min_value=0, max_value=int(chart_data['Fraudulent'].max()) if not chart_data.empty else 1, color="red")
#     },
#     use_container_width=True,
#     hide_index=True
# )
    