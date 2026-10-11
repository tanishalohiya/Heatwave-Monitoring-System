import streamlit as st
import pandas as pd
import plotly.express as px

from services.preprocessing import (
    load_dataset,
    clean_dataset,
    compute_climatology,
    add_normal_temperature,
)
from services.prediction import predict_heatwave, classify_day, detect_heatwave_days
from services.alerts import generate_alert
from services.geo import load_district_geojson, match_district_names
from services.theme import CUSTOM_CSS, PLOTLY_TEMPLATE_NAME, STATUS_COLORS, status_badge


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Climate Intelligence System",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# DATA LOADING + PROCESSING (cached)
# ============================================================

DATA_PATH = "data/Rajasthan_Heatwave_2006_2025.csv"


@st.cache_data
def get_processed_data(path):
    df = load_dataset(path)
    df["DATE"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]])
    df = clean_dataset(df)

    climatology = compute_climatology(df)
    df = add_normal_temperature(df, climatology)
    df = detect_heatwave_days(df)

    df = df.sort_values("DATE")
    return df


try:
    df = get_processed_data(DATA_PATH)
except Exception as e:
    st.error(f"Unable to load dataset: {e}")
    st.stop()


# ============================================================
# DERIVED SUMMARY VALUES
# ============================================================

total_records = len(df)
district_count = df["DISTRICT"].nunique()
heatwave_days = int(df["HEATWAVE_PREDICTED"].sum())
severe_heatwave_days = int(df["SEVERE_HEATWAVE_PREDICTED"].sum())
max_temperature = df["TMAX"].max()
min_temperature = df["TMIN"].min()
average_temperature = df["TEMP2M"].mean()

latest_date = df["DATE"].max()
latest_per_district = (
    df[df["DATE"] == latest_date]
    .sort_values("TMAX", ascending=False)
    .reset_index(drop=True)
)
districts_in_heatwave_today = latest_per_district[
    latest_per_district["HEATWAVE_PREDICTED"] == 1
]
hottest_today = latest_per_district.iloc[0]


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 🌡️ Climate Intelligence")

st.sidebar.caption(
    "Heatwave Monitoring & Early Warning System"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📊 Analytics",
        "🔮 Prediction",
        "🚨 Alerts",
        "📍 District Analysis",
        "🗺️ District Map",
        "📁 Dataset Explorer",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Historical dataset: Rajasthan\n\n"
    "Period: 2006–2025\n\n"
    "Heatwave logic: IMD-style (departure from "
    "normal + 2 consecutive days)"
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">🌡️ Climate Intelligence System</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Heatwave Monitoring, Prediction and Early Warning'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Latest Status (per district, not a single arbitrary row)
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Current Monitoring Status</div>',
        unsafe_allow_html=True
    )

    n_affected = len(districts_in_heatwave_today)

    if n_affected > 0:
        district_names = ", ".join(districts_in_heatwave_today["DISTRICT"].tolist())
        st.error(
            f"⚠️ Heatwave conditions detected in {n_affected} district(s) "
            f"on {latest_date.strftime('%d %B %Y')}: {district_names}"
        )
    else:
        st.success(
            f"✅ Normal conditions across all monitored districts "
            f"on {latest_date.strftime('%d %B %Y')}."
        )

    # --------------------------------------------------------
    # KPI Cards
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🌡️ Maximum Temperature",
            f"{max_temperature:.1f} °C"
        )

    with col2:
        st.metric(
            "🌤️ Average Temperature",
            f"{average_temperature:.1f} °C"
        )

    with col3:
        st.metric(
            "🔥 Heatwave Days",
            f"{heatwave_days:,}"
        )

    with col4:
        st.metric(
            "📍 Districts Monitored",
            district_count
        )

    # --------------------------------------------------------
    # Temperature Trend
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Temperature Overview</div>',
        unsafe_allow_html=True
    )

    daily_temperature = (
        df.groupby("DATE", as_index=False)["TMAX"]
        .mean()
    )

    fig_temperature = px.line(
        daily_temperature,
        x="DATE",
        y="TMAX",
        title="Maximum Temperature Trend",
        labels={
            "DATE": "Date",
            "TMAX": "Maximum Temperature (°C)"
        }
    )

    fig_temperature.update_layout(
        template=PLOTLY_TEMPLATE_NAME,
        height=420
    )

    st.plotly_chart(
        fig_temperature,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Two charts
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        yearly_heatwave = (
            df.groupby("YEAR")["HEATWAVE_PREDICTED"]
            .sum()
            .reset_index()
        )

        fig_heatwave = px.bar(
            yearly_heatwave,
            x="YEAR",
            y="HEATWAVE_PREDICTED",
            title="Heatwave Days by Year",
            labels={
                "YEAR": "Year",
                "HEATWAVE_PREDICTED": "Heatwave Days"
            }
        )

        fig_heatwave.update_layout(
            template=PLOTLY_TEMPLATE_NAME,
            height=350
        )

        st.plotly_chart(
            fig_heatwave,
            use_container_width=True
        )

    with col2:

        district_heatwave = (
            df.groupby("DISTRICT")["HEATWAVE_PREDICTED"]
            .sum()
            .reset_index()
            .sort_values("HEATWAVE_PREDICTED", ascending=True)
        )

        fig_district = px.bar(
            district_heatwave,
            x="HEATWAVE_PREDICTED",
            y="DISTRICT",
            orientation="h",
            title="Heatwave Occurrences by District",
            labels={
                "HEATWAVE_PREDICTED": "Heatwave Days",
                "DISTRICT": "District"
            },
            category_orders={"DISTRICT": district_heatwave["DISTRICT"].tolist()}
        )

        fig_district.update_layout(
            template=PLOTLY_TEMPLATE_NAME,
            height=350
        )

        st.plotly_chart(
            fig_district,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Recent Data
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Recent Observations</div>',
        unsafe_allow_html=True
    )

    recent_columns = [
        "DATE",
        "DISTRICT",
        "TMAX",
        "TMIN",
        "TEMP2M",
        "HEATWAVE_PREDICTED"
    ]

    recent_data = (
        df[recent_columns]
        .sort_values("DATE", ascending=False)
        .head(10)
        .copy()
    )

    recent_data["DATE"] = recent_data["DATE"].dt.strftime(
        "%d-%m-%Y"
    )

    st.dataframe(
        recent_data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.title("📊 Climate Analytics")

    st.write(
        "Explore temperature and heatwave patterns "
        "across the historical dataset."
    )

    col1, col2 = st.columns(2)

    with col1:

        selected_year = st.selectbox(
            "Select Year",
            sorted(df["YEAR"].unique(), reverse=True)
        )

    with col2:

        selected_district = st.selectbox(
            "Select District",
            sorted(df["DISTRICT"].unique())
        )

    filtered = df[
        (df["YEAR"] == selected_year) &
        (df["DISTRICT"] == selected_district)
    ]

    if filtered.empty:

        st.warning("No data available for the selected filters.")

    else:

        st.subheader(
            f"{selected_district} — {selected_year}"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Maximum Temperature",
                f"{filtered['TMAX'].max():.1f} °C"
            )

        with col2:
            st.metric(
                "Average Temperature",
                f"{filtered['TEMP2M'].mean():.1f} °C"
            )

        with col3:
            st.metric(
                "Heatwave Days",
                int(filtered["HEATWAVE_PREDICTED"].sum())
            )

        fig = px.line(
            filtered,
            x="DATE",
            y=["TMAX", "TEMP2M", "TMIN", "NORMAL_TMAX"],
            title="Temperature Variation vs Normal",
            labels={
                "value": "Temperature (°C)",
                "variable": "Parameter"
            }
        )

        fig.update_layout(
            template=PLOTLY_TEMPLATE_NAME,
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# PREDICTION
# ============================================================

elif page == "🔮 Prediction":

    st.title("🔮 Heatwave Prediction")

    prediction_mode = st.radio(
        "Choose prediction method",
        ["Rule-Based (IMD-style)", "DNN Model"],
        horizontal=True
    )

    if prediction_mode == "Rule-Based (IMD-style)":

        st.write(
            "Evaluate a maximum temperature using the existing "
            "IMD-style absolute or normal-temperature departure criteria."
        )

        col1, col2 = st.columns(2)

        with col1:
            temperature = st.number_input(
                "Maximum Temperature (°C)",
                min_value=-10.0,
                max_value=60.0,
                value=40.0,
                step=0.5
            )

        with col2:
            use_normal = st.checkbox("Use normal-temperature departure")
            normal_temperature = None

            if use_normal:
                normal_temperature = st.number_input(
                    "Normal Temperature for this date/district (°C)",
                    min_value=-10.0,
                    max_value=55.0,
                    value=38.0,
                    step=0.5
                )

        if st.button("Run Prediction", type="primary", key="rule_prediction"):

            day_class = classify_day(temperature, normal_temperature)

            st.markdown("---")

            if day_class == 2:
                st.error("🔴 SEVERE HEATWAVE CONDITION")
                st.write(
                    "This single day meets the severe heatwave criteria. "
                    "An official declaration also depends on persistence "
                    "for at least two consecutive days."
                )
            elif day_class == 1:
                st.error("🔥 HEATWAVE CONDITION")
                st.write(
                    "This single day meets the heatwave criteria. "
                    "An official declaration also depends on persistence "
                    "for at least two consecutive days."
                )
            else:
                st.success("✅ NORMAL CONDITION")
                st.write(
                    "The entered temperature is below the configured "
                    "heatwave threshold."
                )

    else:
        st.write(
            "The trained DNN classifies historical weather observations "
            "using 20 weather and location features. It does not currently "
            "forecast future weather."
        )

        try:
            from services.dnn_prediction import predict_dnn

            # Use the already-loaded and temperature-converted dataset.
            dnn_df = df.copy()

            required_features = [
                "MONTH", "DAY", "WIND_U10", "WIND_V10", "MSLP",
                "BLH", "GEOP", "TEMP2M", "TMAX", "TMIN", "DEW2M",
                "CLOUD", "RAIN", "SRAD", "EVAP", "SOILT1",
                "SOILM1", "LAI", "LAT", "LON"
            ]

            required_columns = required_features + ["YEAR", "DISTRICT"]

            missing_columns = [
                column for column in required_columns
                if column not in dnn_df.columns
            ]

            if missing_columns:
                st.error(
                    "The dataset is missing required columns: "
                    + ", ".join(missing_columns)
                )
            else:
                districts = sorted(
                    dnn_df["DISTRICT"].dropna().unique().tolist()
                )
                selected_district = st.selectbox(
                    "Select district",
                    districts,
                    key="dnn_district"
                )

                district_df = dnn_df[
                    dnn_df["DISTRICT"] == selected_district
                ].copy()

                district_df["RECORD_DATE"] = pd.to_datetime(
                    dict(
                        year=district_df["YEAR"],
                        month=district_df["MONTH"],
                        day=district_df["DAY"]
                    ),
                    errors="coerce"
                )

                district_df = district_df.dropna(subset=["RECORD_DATE"])

                dates = sorted(
                    district_df["RECORD_DATE"].unique(),
                    reverse=True
                )

                if not dates:
                    st.warning("No dated records are available for this district.")
                else:
                    selected_date = st.selectbox(
                        "Select historical date",
                        dates,
                        format_func=lambda value: pd.Timestamp(value).strftime(
                            "%d %B %Y"
                        ),
                        key="dnn_date"
                    )

                    selected_row = district_df[
                        district_df["RECORD_DATE"] == selected_date
                    ]

                    if selected_row.empty:
                        st.warning("No record was found for that date.")
                    else:
                        record = selected_row.iloc[0]

                        st.caption(
                            "Selected record: "
                            f"{selected_district}, "
                            f"{pd.Timestamp(selected_date).strftime('%d %B %Y')}"
                        )

                        if st.button(
                            "Classify Historical Record",
                            type="primary",
                            key="dnn_prediction"
                        ):
                            weather_data = record[required_features].to_dict()
                            result = predict_dnn(weather_data)

                            probability = result["probability"]
                            prediction = result["prediction"]

                            st.markdown("---")
                            st.metric(
                                "DNN Heatwave Probability",
                                f"{probability:.2%}"
                            )

                            if prediction == 1:
                                st.error("🔥 DNN CLASSIFICATION: HEATWAVE")
                            else:
                                st.success(
                                    "✅ DNN CLASSIFICATION: NO HEATWAVE"
                                )

                            st.caption(
                                f"Classification threshold: "
                                f"{result['threshold']:.2f}"
                            )

                            if "HEATWAVE" in record.index:
                                st.write(
                                    "Dataset label:",
                                    int(record["HEATWAVE"])
                                )

        except FileNotFoundError as exc:
            st.error(str(exc))
        except ImportError as exc:
            st.error(
                "Unable to import the DNN dependencies. "
                f"Check the environment: {exc}"
            )
        except Exception as exc:
            st.error(f"DNN classification failed: {exc}")


# ============================================================
# ALERTS
# ============================================================

elif page == "🚨 Alerts":

    st.title("🚨 Heatwave Alerts")

    st.write(
        f"Alert status for all districts on the latest available "
        f"date: **{latest_date.strftime('%d %B %Y')}**"
    )

    st.markdown("---")

    if len(districts_in_heatwave_today) == 0:
        st.success("✅ No districts are currently under heatwave alert.")
    else:
        st.error(
            f"⚠️ {len(districts_in_heatwave_today)} district(s) currently "
            "under heatwave alert."
        )

    st.markdown(
        '<div class="section-title">District Status</div>',
        unsafe_allow_html=True
    )

    status_label_map = {0: "Normal", 1: "Heatwave", 2: "Severe Heatwave"}

    for _, row in latest_per_district.iterrows():
        status_label = status_label_map[row["DAY_CLASS"]]

        col1, col2, col3 = st.columns([3, 2, 2])

        with col1:
            st.write(f"**{row['DISTRICT']}**")

        with col2:
            st.write(f"{row['TMAX']:.1f} °C")

        with col3:
            st.markdown(status_badge(status_label), unsafe_allow_html=True)


# ============================================================
# DISTRICT ANALYSIS
# ============================================================

elif page == "📍 District Analysis":

    st.title("📍 District Analysis")

    selected_district = st.selectbox(
        "Select District",
        sorted(df["DISTRICT"].unique())
    )

    district_data = df[
        df["DISTRICT"] == selected_district
    ]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Maximum Recorded Temperature",
            f"{district_data['TMAX'].max():.1f} °C"
        )

    with col2:
        st.metric(
            "Average Temperature",
            f"{district_data['TEMP2M'].mean():.1f} °C"
        )

    with col3:
        st.metric(
            "Total Heatwave Days",
            int(district_data["HEATWAVE_PREDICTED"].sum())
        )

    yearly = (
        district_data
        .groupby("YEAR")["HEATWAVE_PREDICTED"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        yearly,
        x="YEAR",
        y="HEATWAVE_PREDICTED",
        title=f"Heatwave Days — {selected_district}",
        labels={
            "YEAR": "Year",
            "HEATWAVE_PREDICTED": "Heatwave Days"
        }
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE_NAME,
        height=420
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# DISTRICT MAP
# ============================================================

elif page == "🗺️ District Map":

    st.title("🗺️ District Map")

    GEOJSON_PATH = "data/rajasthan_districts.geojson"

    @st.cache_data
    def get_geojson_and_matches(path, district_names):
        geojson = load_district_geojson(path)
        matched, unmatched = match_district_names(district_names, geojson)
        return geojson, matched, unmatched

    try:
        rajasthan_geojson, matched, unmatched = get_geojson_and_matches(
            GEOJSON_PATH, tuple(sorted(df["DISTRICT"].unique()))
        )
    except FileNotFoundError:
        st.error(
            f"District boundary file not found at `{GEOJSON_PATH}`. "
            "Download it and place it there — see the setup note below."
        )
        st.code(
            'curl -sL -o data/rajasthan_districts.geojson '
            '"https://raw.githubusercontent.com/shuklaneerajdev/'
            'IndiaStateTopojsonFiles/master/Rajasthan.geojson"',
            language="bash"
        )
        st.stop()

    if unmatched:
        st.warning(
            f"{len(unmatched)} district name(s) in your dataset didn't "
            f"match the map boundaries and won't be shaded: "
            f"{', '.join(unmatched)}"
        )

    df["GEO_DISTRICT"] = df["DISTRICT"].map(matched)

    map_view = st.radio(
        "Show on map",
        ["Current status (latest date)", "Total heatwave days (all-time)"],
        horizontal=True
    )

    if map_view == "Current status (latest date)":

        map_data = (
            df[df["DATE"] == df["DATE"].max()]
            .dropna(subset=["GEO_DISTRICT"])
            .copy()
        )
        map_data["STATUS"] = map_data["DAY_CLASS"].map(
            {0: "Normal", 1: "Heatwave", 2: "Severe Heatwave"}
        )

        fig = px.choropleth(
            map_data,
            geojson=rajasthan_geojson,
            locations="GEO_DISTRICT",
            featureidkey="properties.Dist_Name",
            color="STATUS",
            color_discrete_map=STATUS_COLORS,
            category_orders={"STATUS": ["Normal", "Heatwave", "Severe Heatwave"]},
            hover_name="DISTRICT",
            hover_data={
                "TMAX": ":.1f",
                "NORMAL_TMAX": ":.1f",
                "GEO_DISTRICT": False,
                "STATUS": False,
            },
            title=f"Heatwave Status — {latest_date.strftime('%d %B %Y')}"
        )

    else:

        map_data = (
            df.dropna(subset=["GEO_DISTRICT"])
            .groupby(["DISTRICT", "GEO_DISTRICT"], as_index=False)["HEATWAVE_PREDICTED"]
            .sum()
        )

        fig = px.choropleth(
            map_data,
            geojson=rajasthan_geojson,
            locations="GEO_DISTRICT",
            featureidkey="properties.Dist_Name",
            color="HEATWAVE_PREDICTED",
            color_continuous_scale="OrRd",
            hover_name="DISTRICT",
            hover_data={"GEO_DISTRICT": False},
            title="Total Heatwave Days (2006–2025)"
        )

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        template=PLOTLY_TEMPLATE_NAME,
        height=600,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# DATASET EXPLORER
# ============================================================

elif page == "📁 Dataset Explorer":

    st.title("📁 Dataset Explorer")

    st.write(
        "Filter and explore the historical Rajasthan "
        "heatwave dataset."
    )

    col1, col2 = st.columns(2)

    with col1:

        year_filter = st.multiselect(
            "Filter by Year",
            sorted(df["YEAR"].unique()),
            default=[]
        )

    with col2:

        district_filter = st.multiselect(
            "Filter by District",
            sorted(df["DISTRICT"].unique()),
            default=[]
        )

    filtered_df = df.copy()

    if year_filter:
        filtered_df = filtered_df[
            filtered_df["YEAR"].isin(year_filter)
        ]

    if district_filter:
        filtered_df = filtered_df[
            filtered_df["DISTRICT"].isin(district_filter)
        ]

    st.info(
        f"Showing {len(filtered_df):,} records"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )

    csv = filtered_df.to_csv(index=False)

    st.download_button(
        label="⬇️ Download Filtered Dataset",
        data=csv,
        file_name="heatwave_filtered_data.csv",
        mime="text/csv"
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About the System")

    st.markdown(
        """
        ### 🌡️ Climate Intelligence System

        The **Heatwave Monitoring System** is a Python and
        Streamlit-based application designed to monitor
        historical weather conditions, analyze heatwave
        patterns, perform temperature-based prediction and
        generate early warning alerts.

        ### Dataset

        **Rajasthan Heatwave Dataset**

        - Period: 2006–2025
        - Frequency: Daily
        - Region: Rajasthan
        - Multiple districts
        - Meteorological and environmental parameters

        ### Heatwave Detection Methodology

        Classification follows IMD-style criteria rather than a
        single flat threshold:

        - **Departure from normal**: a day's TMAX is compared
          against a smoothed day-of-year climatological normal
          computed per district from the historical record.
        - **Severity tiers**: Normal / Heatwave / Severe Heatwave,
          based on departure magnitude and absolute temperature.
        - **Persistence rule**: a heatwave is only confirmed when
          qualifying conditions persist for at least 2 consecutive
          days at a district.

        ### Technology Stack

        - Python
        - Streamlit
        - Pandas
        - NumPy
        - Plotly
        - PyTest

        ### Main Features

        - Historical heatwave monitoring
        - Temperature analytics
        - District-wise analysis
        - Heatwave prediction
        - Alert generation
        - Dataset exploration
        - Interactive visualizations

        ### Objective

        The objective is to provide a centralized
        climate-intelligence dashboard that helps users
        understand temperature patterns and identify
        potential heatwave conditions.
        """
    )

    st.success(
        "Climate Intelligence System is ready for monitoring."
    )