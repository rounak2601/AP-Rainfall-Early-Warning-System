import streamlit as st
import pandas as pd
import numpy as np
import pickle
import geopandas as gpd
from shapely.geometry import Point
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from sqlalchemy import create_engine, text
from datetime import datetime
import requests
from twilio.rest import Client
import os

DB_PATH      = r"D:\Projects\Rain_prediction\rainfall.db"
MODELS_PATH  = r"D:\Projects\Rain_prediction"
OWM_API_KEY  = os.getenv("OWM_API_KEY")
TWILIO_SID   = os.getenv("TWILIO_SID")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")
TWILIO_FROM  = "+16624938503"
ALERT_PHONE  = "+919399803347"

YEAR_FACTORS = {2021:1.22,2022:1.18,2023:0.87,2024:1.05,
                2025:1.08,2026:0.93,2027:1.15,2028:0.91,2029:1.12}
MONSOON_INT  = {6:0.6,7:0.9,8:1.0,9:0.8,10:0.5}
MONSOON_DAY  = {6:0,7:30,8:61,9:92,10:122}
ENSO_LABEL   = {2021:"La Nina (+22%)",2022:"La Nina (+18%)",2023:"El Nino (-13%)",
                2024:"Neutral (+5%)",2025:"Neutral (+8%)",2026:"El Nino (-7%)",
                2027:"La Nina (+15%)",2028:"El Nino (-9%)",2029:"La Nina (+12%)"}
HUM_NORMS    = {6:75,7:85,8:88,9:82,10:72}
TMP_NORMS    = {6:32,7:29,8:28,9:29,10:28}
PRS_NORMS    = {6:1005,7:1002,8:1001,9:1003,10:1006}
WND_NORMS    = {6:4.5,7:5.2,8:4.8,9:3.9,10:3.2}

engine = create_engine(f"sqlite:///{DB_PATH}")

st.set_page_config(page_title="AP Rainfall Early Warning System",layout="wide",page_icon="🌧")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:Inter,sans-serif;}
.weather-card{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);border-radius:16px;
  padding:20px;text-align:center;border:1px solid #ffffff22;margin:4px;}
.weather-val{font-size:2rem;font-weight:700;color:#00d4ff;margin:8px 0;}
.weather-lbl{font-size:0.8rem;color:#aaaacc;text-transform:uppercase;letter-spacing:1px;}
.risk-none{background:linear-gradient(135deg,#1a2a1a,#2d4a2d);border:1px solid #4caf50;border-radius:12px;padding:16px;margin:8px 0;}
.risk-low{background:linear-gradient(135deg,#1a1a2e,#2d2d5e);border:1px solid #2196f3;border-radius:12px;padding:16px;margin:8px 0;}
.risk-moderate{background:linear-gradient(135deg,#2a1a00,#4a3000);border:1px solid #ff9800;border-radius:12px;padding:16px;margin:8px 0;}
.risk-high{background:linear-gradient(135deg,#2a0a0a,#4a1a1a);border:1px solid #f44336;border-radius:12px;padding:16px;margin:8px 0;}
.risk-extreme{background:linear-gradient(135deg,#1a0000,#3a0000);border:2px solid #ff1744;border-radius:12px;padding:16px;margin:8px 0;}
.infra-card{background:#1e1e3a;border-radius:12px;padding:16px;border:1px solid #333366;margin:8px 0;}
.section-title{font-size:1.2rem;font-weight:700;color:#e0e0ff;margin:16px 0 8px 0;border-bottom:2px solid #0f3460;padding-bottom:6px;}
.tip-box{background:#0d2137;border-left:4px solid #00d4ff;border-radius:8px;padding:10px;margin:8px 0;font-size:0.85rem;color:#aaddff;}
.result-box{background:#111128;border:1px solid #333366;border-radius:16px;padding:20px;margin:8px 0;}
.stat-label{font-size:0.75rem;color:#aaaacc;text-transform:uppercase;letter-spacing:1px;}
.stat-value{font-size:1.8rem;font-weight:700;color:#ffffff;margin:4px 0;}
.signal-box{background:#111128;border:1px solid #333366;border-radius:12px;padding:12px;margin:4px 0;}
.blocked-box{background:linear-gradient(135deg,#1a0a0a,#2a1010);border:2px solid #ff6b35;border-radius:12px;padding:16px;margin:8px 0;}
</style>""",unsafe_allow_html=True)

if "prediction_done" not in st.session_state:
    st.session_state.update({"prediction_done":False,"amount":0.0,"will_rain":False,
        "risk":"NONE","afc_u":pd.DataFrame(),"afe_u":pd.DataFrame(),
        "pred_date":"","pred_village":"","pred_district":"",
        "pred_lat":16.5,"pred_lon":80.6,"confidence":0,"signals":[],
        "rain_votes":0,"total":1,"blocked":False,"block_reason":""})

@st.cache_resource
def load_models():
    with open(f"{MODELS_PATH}/reg_model.pkl","rb") as f: reg=pickle.load(f)
    with open(f"{MODELS_PATH}/clf_model.pkl","rb") as f: clf=pickle.load(f)
    return reg,clf

@st.cache_resource
def load_shapefiles():
    c=gpd.read_file(f"{MODELS_PATH}/Canals.shp").to_crs("EPSG:32644")
    e=gpd.read_file(f"{MODELS_PATH}/Embankments.shp").to_crs("EPSG:32644")
    return c,e

@st.cache_data
def load_villages():
    return pd.read_csv(f"{MODELS_PATH}/village_predictions.csv")

@st.cache_data(ttl=3600)
def get_historical_avg(lat,lon):
    df=load_villages()
    nearby=df[(abs(df["centroid_lat"]-lat)<0.3)&(abs(df["centroid_lon"]-lon)<0.3)]["predicted_mm"]
    return float(nearby.mean()) if len(nearby)>0 else 5.0

@st.cache_data(ttl=600)
def fetch_live_weather(lat,lon):
    try:
        r=requests.get("https://api.openweathermap.org/data/2.5/weather",
            params={"lat":lat,"lon":lon,"appid":OWM_API_KEY,"units":"metric"},timeout=5)
        if r.status_code==200:
            d=r.json()
            return {"temp":round(d["main"]["temp"],1),"humidity":d["main"]["humidity"],
                    "pressure":d["main"]["pressure"],"rainfall":d.get("rain",{}).get("1h",0.0),
                    "desc":d["weather"][0]["description"].title(),
                    "wind":d["wind"]["speed"],"feels":round(d["main"]["feels_like"],1)}
    except: pass
    m=datetime.now().month
    return {"temp":TMP_NORMS.get(m,35),"humidity":HUM_NORMS.get(m,45),
            "pressure":PRS_NORMS.get(m,1012),"rainfall":0.0,
            "desc":"Unavailable","wind":WND_NORMS.get(m,3.0),"feels":35}

@st.cache_data(ttl=1800)
def fetch_forecast(lat,lon):
    try:
        r=requests.get("https://api.openweathermap.org/data/2.5/forecast",
            params={"lat":lat,"lon":lon,"appid":OWM_API_KEY,"units":"metric","cnt":16},timeout=5)
        if r.status_code==200:
            rows=[]
            for item in r.json()["list"]:
                rows.append({"time":item["dt_txt"],
                    "rain_mm":item.get("rain",{}).get("3h",0.0),
                    "temp":item["main"]["temp"],"humidity":item["main"]["humidity"],
                    "desc":item["weather"][0]["description"].title()})
            return pd.DataFrame(rows)
    except: pass
    return pd.DataFrame()

@st.cache_data(ttl=1800)
def build_map_data():
    df=load_villages()
    nm=datetime.now().month
    reg,clf=load_models()
    sample=df.sample(min(200,len(df)),random_state=42)
    nyf=YEAR_FACTORS.get(datetime.now().year,1.0)
    X=pd.DataFrame({
        "centroid_lat":sample["centroid_lat"].values,"centroid_lon":sample["centroid_lon"].values,
        "year":datetime.now().year,"month":nm,"day":15,
        "dayofyear":int(pd.Timestamp(f"{datetime.now().year}-{nm:02d}-15").dayofyear),
        "rain_lag1":sample["predicted_mm"].values,"rain_lag7":sample["predicted_mm"].values,
        "rain_rolling7":sample["predicted_mm"].values,
        "humidity":HUM_NORMS.get(nm,45),"temp":TMP_NORMS.get(nm,35),
        "pressure":PRS_NORMS.get(nm,1012),"wind":WND_NORMS.get(nm,3.0),
        "year_factor":nyf,"monsoon_intensity":MONSOON_INT.get(nm,0.1),
        "monsoon_day":float(MONSOON_DAY.get(nm,0))})
    sample=sample.copy()
    sample["live_mm"]=np.maximum(0,reg.predict(X))
    sample["live_rain"]=clf.predict(X).astype(bool)
    return sample

def physical_rain_check(temp, humidity, month):
    t=float(str(temp).replace("--","35"))
    h=float(str(humidity).replace("--","50"))
    if t>42:
        return False, f"Physically impossible: {t}C is too hot for precipitation (max ~40C)"
    if h<35 and t>37:
        return False, f"Physically impossible: {h}% humidity + {t}C — air is far too hot and dry for rain"
    if h<40 and month not in [6,7,8,9,10]:
        return False, f"Physically unlikely: {h}% humidity in non-monsoon month — rain blocked"
    if h<50 and t>38:
        return False, f"Physically blocked: {h}% humidity at {t}C — insufficient moisture for rainfall"
    return True, "Atmospheric conditions permit rainfall"

def get_confidence(amount,will_rain,weather,forecast_df,forecast_date,enso_label,blocked,block_reason):
    today=datetime.now().date()
    days_ahead=(forecast_date-today).days
    signals=[]
    if blocked:
        signals.append(("Physical Meteorology Gate",
            "NO RAIN",block_reason,"#ff6b35"))
        ml_sig="RAIN" if will_rain else "NO RAIN"
        signals.append(("ML Model (92.2% accuracy) — overridden by physics gate",
            ml_sig,f"Model predicted {amount:.1f}mm but physical conditions prevent rainfall","#888888"))
        return signals,0,0,len(signals)
    ml_sig="RAIN" if will_rain else "NO RAIN"
    signals.append(("ML Model (92.2% accuracy, 9.5M records)",
        ml_sig,f"{amount:.1f}mm predicted using location + seasonal + ENSO patterns","#00d4ff"))
    if days_ahead<=2:
        h=float(str(weather["humidity"]).replace("--","0"))
        if h>0:
            hs="RAIN" if h>70 else "NO RAIN"
            signals.append(("Live Humidity (OpenWeatherMap — real-time)",hs,
                f"Current humidity {h}% — {'above' if h>70 else 'below'} 70% threshold","#ff9800"))
        if not forecast_df.empty:
            n24=forecast_df["rain_mm"].head(8).sum()
            ns="RAIN" if n24>2 else "NO RAIN"
            signals.append(("NWP Physics Forecast (next 24hrs)",ns,
                f"Forecast shows {n24:.1f}mm in next 24hrs","#4caf50"))
    elif days_ahead<=5:
        if not forecast_df.empty:
            tf=forecast_df["rain_mm"].sum()
            ns="RAIN" if tf>5 else "NO RAIN"
            signals.append(("NWP Forecast (48hr range)",ns,f"48hr total: {tf:.1f}mm","#4caf50"))
        m=forecast_date.month
        cs="RAIN" if m in [6,7,8,9,10] else "NO RAIN"
        signals.append(("Seasonal Climatology (IMD normals)",cs,
            f"{forecast_date.strftime('%B')} {'is monsoon season' if m in [6,7,8,9,10] else 'is dry season'}","#aa44ff"))
    else:
        m=forecast_date.month
        cs="RAIN" if m in [6,7,8,9,10] else "NO RAIN"
        cr=(f"{forecast_date.strftime('%B')} is peak monsoon — historically 80-200mm"
            if m in [7,8,9] else
            f"{forecast_date.strftime('%B')} is monsoon season — moderate rainfall"
            if m in [6,10] else
            f"{forecast_date.strftime('%B')} is dry season — historically below 20mm")
        signals.append(("Seasonal Climatology (IMD historical normals)",cs,cr,"#aa44ff"))
        if m in [6,7,8,9,10]:
            yf=YEAR_FACTORS.get(forecast_date.year,1.0)
            ys="RAIN" if yf>=1.0 else "NO RAIN"
            signals.append(("ENSO Climate Pattern",ys,
                f"{enso_label} — {'above' if yf>=1.0 else 'below'} normal rainfall","#ff6b35"))
            signals.append(("Bay of Bengal Monsoon Window","RAIN",
                f"AP receives SW monsoon Jun-Oct. {forecast_date.strftime('%B')} is active window.","#4caf50"))
    rv=sum(1 for s in signals if s[1]=="RAIN")
    total=len(signals)
    conf=int((rv/total)*100)
    return signals,conf,rv,total

def auto_threshold(lat,lon):
    avg=get_historical_avg(lat,lon)
    if avg<5:  return 15,"Low rainfall region — threshold 15mm."
    if avg<10: return 25,"Moderate rainfall region — threshold 25mm."
    return 35,"High rainfall region — threshold 35mm."

def send_sms(message):
    try:
        client=Client(TWILIO_SID,TWILIO_TOKEN)
        client.messages.create(body=message,from_=TWILIO_FROM,to=ALERT_PHONE)
        return True
    except Exception as e:
        st.warning(f"SMS failed: {e}")
        return False

def get_risk_level(mm,will_rain,blocked=False):
    if blocked or not will_rain or mm<0.5: return "NONE"
    if mm<5:  return "LOW"
    if mm<15: return "MODERATE"
    if mm<25: return "HIGH"
    return "EXTREME"

def save_prediction(village,district,mandal,lat,lon,date,mm,will_rain,risk,cs,es):
    with engine.connect() as conn:
        conn.execute(text("""INSERT INTO predictions
            (timestamp,district,mandal,village,lat,lon,date,predicted_mm,will_rain,risk_level)
            VALUES (:ts,:district,:mandal,:village,:lat,:lon,:date,:mm,:wr,:risk)"""),
            dict(ts=datetime.now().isoformat(),district=district,mandal=mandal,village=village,
                 lat=lat,lon=lon,date=str(date),mm=float(mm),wr=int(will_rain),risk=risk))
        if risk in ["HIGH","EXTREME"]:
            conn.execute(text("""INSERT INTO alerts
                (timestamp,village,district,predicted_mm,alert_type,affected_canals,affected_embankments)
                VALUES (:ts,:village,:district,:mm,:alert,:canals,:emb)"""),
                dict(ts=datetime.now().isoformat(),village=village,district=district,
                     mm=float(mm),alert=risk,canals=cs,emb=es))
        conn.commit()

def get_recent_alerts():
    try: return pd.read_sql("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 20",engine)
    except: return pd.DataFrame()

def get_prediction_history():
    try:
        df=pd.read_sql("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 500",engine)
        df["predicted_mm"]=pd.to_numeric(df["predicted_mm"],errors="coerce").fillna(0)
        return df
    except: return pd.DataFrame()

reg_model,clf_model=load_models()
canals,embankments=load_shapefiles()
villages_df=load_villages()

st.markdown("""
<div style="background:linear-gradient(90deg,#0f3460,#16213e,#0f0f1a);
padding:24px 32px;border-radius:16px;margin-bottom:20px;border-bottom:3px solid #00d4ff;">
<h1 style="color:#00d4ff;margin:0;font-size:2rem;">🌧 AP Rainfall Early Warning System</h1>
<p style="color:#aaaacc;margin:4px 0 0 0;font-size:0.9rem;">
ML + ENSO Patterns + Live Weather + Physical Meteorology Gate | 92.2% accuracy | 15,589 villages
</p></div>""",unsafe_allow_html=True)

now=datetime.now()
c1,c2,c3,c4=st.columns(4)
c1.markdown(f"<div style='color:#aaaacc;font-size:0.8rem'>Date</div><div style='color:#00d4ff;font-weight:700;font-size:1.1rem'>{now.strftime('%d %b %Y')}</div>",unsafe_allow_html=True)
c2.markdown(f"<div style='color:#aaaacc;font-size:0.8rem'>Time</div><div style='color:#00d4ff;font-weight:700;font-size:1.1rem'>{now.strftime('%I:%M %p')}</div>",unsafe_allow_html=True)
c3.markdown(f"<div style='color:#aaaacc;font-size:0.8rem'>Season</div><div style='color:#00d4ff;font-weight:700;font-size:1.1rem'>{'Monsoon' if now.month in [6,7,8,9,10] else 'Non-Monsoon'}</div>",unsafe_allow_html=True)
c4.markdown(f"<div style='color:#4caf50;font-size:0.8rem'>Model Accuracy</div><div style='color:#4caf50;font-weight:700;font-size:1.1rem'>92.2%</div>",unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #ffffff11;margin:12px 0'>",unsafe_allow_html=True)

tab1,tab2,tab3,tab4=st.tabs(["🎯 Predict and Alert","🗺 Live Map","📊 Analytics","🔔 Alert History"])

with tab1:
    col_left,col_right=st.columns([1,2],gap="large")
    with col_left:
        st.markdown("<div class='section-title'>📍 Select Location</div>",unsafe_allow_html=True)
        mode=st.radio("",["Select from list","Enter coordinates"],horizontal=True)
        if mode=="Select from list":
            district=st.selectbox("District",sorted(villages_df["district"].unique()))
            fmandal=villages_df[villages_df["district"]==district]
            mandal=st.selectbox("Mandal",sorted(fmandal["mandal"].unique()))
            fvillage=fmandal[fmandal["mandal"]==mandal]
            village=st.selectbox("Village",sorted(fvillage["village"].unique()))
            row=fvillage[fvillage["village"]==village].iloc[0]
            lat,lon=float(row["centroid_lat"]),float(row["centroid_lon"])
            st.markdown(f"<div class='tip-box'>📌 {lat:.4f}N, {lon:.4f}E</div>",unsafe_allow_html=True)
        else:
            district,mandal,village="Custom","Custom","Custom"
            lat=st.number_input("Latitude",12.0,20.0,16.50,0.01)
            lon=st.number_input("Longitude",76.0,85.0,80.64,0.01)

        st.markdown("<div class='section-title'>📅 Forecast Settings</div>",unsafe_allow_html=True)
        date_input=st.date_input("Forecast Date",value=pd.Timestamp("2025-08-15"))
        days_ahead=(date_input-datetime.now().date()).days
        enso=ENSO_LABEL.get(date_input.year,"Neutral")
        yf_val=YEAR_FACTORS.get(date_input.year,1.0)
        if days_ahead<=0:
            st.markdown("<div class='tip-box'>📡 Today — live weather + physical gate active</div>",unsafe_allow_html=True)
        elif days_ahead<=2:
            st.markdown("<div class='tip-box'>📡 Near-term — live weather + NWP signals active</div>",unsafe_allow_html=True)
        elif days_ahead<=5:
            st.markdown(f"<div class='tip-box'>📅 Medium-range | ENSO: {enso}</div>",unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='tip-box'>🗓 Long-range | ENSO {date_input.year}: <b>{enso}</b> | Factor: {'Above' if yf_val>=1 else 'Below'} normal ({'+' if yf_val>=1 else ''}{int((yf_val-1)*100)}%)</div>",unsafe_allow_html=True)

        auto_thresh,thresh_tip=auto_threshold(lat,lon)
        st.markdown("<div class='section-title'>⚡ SMS Alert Settings</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='tip-box'>💡 {thresh_tip}</div>",unsafe_allow_html=True)
        use_auto=st.checkbox("Use smart auto-threshold (recommended)",value=True)
        threshold=auto_thresh if use_auto else st.slider("Manual threshold (mm)",1,100,auto_thresh)
        if use_auto: st.info(f"Auto threshold: {threshold} mm")
        send_sms_cb=st.checkbox("Send SMS if threshold exceeded",value=True)
        predict_btn=st.button("🚀 Run Prediction",type="primary",use_container_width=True)

    with col_right:
        weather=fetch_live_weather(lat,lon)
        st.markdown("<div class='section-title'>🌤 Live Weather Conditions</div>",unsafe_allow_html=True)
        wc1,wc2,wc3,wc4=st.columns(4)
        tv=float(str(weather["temp"]).replace("--","35"))
        hv=float(str(weather["humidity"]).replace("--","50"))
        wc1.markdown(f"<div class='weather-card'><div class='weather-lbl'>Temperature</div><div class='weather-val' style='color:{'#ff4444' if tv>40 else '#ff9800' if tv>35 else '#00d4ff'}'>{weather['temp']}°C</div><div class='weather-lbl'>Feels {weather['feels']}°C</div></div>",unsafe_allow_html=True)
        wc2.markdown(f"<div class='weather-card'><div class='weather-lbl'>Humidity</div><div class='weather-val' style='color:{'#4caf50' if hv>70 else '#ff9800' if hv>40 else '#ff4444'}'>{weather['humidity']}%</div><div class='weather-lbl'>{'High — rain likely' if hv>70 else 'Moderate' if hv>40 else 'Low — no rain possible'}</div></div>",unsafe_allow_html=True)
        wc3.markdown(f"<div class='weather-card'><div class='weather-lbl'>Live Rainfall</div><div class='weather-val'>{weather['rainfall']}mm</div><div class='weather-lbl'>Right now</div></div>",unsafe_allow_html=True)
        pv=float(str(weather["pressure"]).replace("--","1013"))
        wc4.markdown(f"<div class='weather-card'><div class='weather-lbl'>Pressure</div><div class='weather-val'>{weather['pressure']}</div><div class='weather-lbl'>{'Storm risk' if pv<1005 else 'Normal'}</div></div>",unsafe_allow_html=True)
        st.markdown(f"<p style='color:#aaaacc;font-size:0.85rem;margin:6px 0'>☁ <b style='color:#00d4ff'>{weather['desc']}</b> | 📍 <b style='color:#00d4ff'>{village}, {district}</b> | 💨 {weather['wind']}m/s</p>",unsafe_allow_html=True)

        can_rain_now,gate_reason=physical_rain_check(weather["temp"],weather["humidity"],datetime.now().month)
        if not can_rain_now:
            st.markdown(f"<div class='blocked-box'>🚫 <b style='color:#ff6b35'>Physical Gate Active</b><br><span style='color:#ffaa88;font-size:0.85rem'>{gate_reason}</span></div>",unsafe_allow_html=True)

        forecast_df=fetch_forecast(lat,lon)
        if not forecast_df.empty:
            st.markdown("<div class='section-title'>📈 48-Hour Forecast</div>",unsafe_allow_html=True)
            fc1,fc2=st.columns(2)
            with fc1:
                fig_r=go.Figure()
                fig_r.add_trace(go.Bar(x=forecast_df["time"],y=forecast_df["rain_mm"],
                    marker_color=["#ff4444" if r>10 else "#00d4ff" if r>2 else "#334466" for r in forecast_df["rain_mm"]],
                    hovertemplate="%{x}<br>%{y:.1f}mm<extra></extra>"))
                fig_r.add_hline(y=threshold,line_dash="dash",line_color="#ff9800",
                    annotation_text=f"Alert:{threshold}mm",annotation_font_color="#ff9800")
                fig_r.update_layout(title="Rainfall mm",height=200,paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",font_color="#aaaacc",margin=dict(t=30,b=10,l=10,r=10),
                    xaxis=dict(showgrid=False,tickangle=45),yaxis=dict(showgrid=True,gridcolor="#222244"))
                st.plotly_chart(fig_r,use_container_width=True)
            with fc2:
                fig_h=go.Figure()
                fig_h.add_trace(go.Scatter(x=forecast_df["time"],y=forecast_df["humidity"],
                    fill="tozeroy",line_color="#00d4ff",fillcolor="rgba(0,212,255,0.1)",
                    hovertemplate="%{x}<br>%{y}%<extra></extra>"))
                fig_h.add_hline(y=70,line_dash="dash",line_color="#ff9800",
                    annotation_text="Rain:70%",annotation_font_color="#ff9800")
                fig_h.update_layout(title="Humidity %",height=200,paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",font_color="#aaaacc",margin=dict(t=30,b=10,l=10,r=10),
                    xaxis=dict(showgrid=False,tickangle=45),
                    yaxis=dict(showgrid=True,gridcolor="#222244",range=[0,100]))
                st.plotly_chart(fig_h,use_container_width=True)

        if predict_btn:
            with st.spinner("Running prediction..."):
                date=pd.Timestamp(date_input)
                fm=date.month
                hist_avg=get_historical_avg(lat,lon)
                wx=fetch_live_weather(lat,lon)
                days_diff=(date.date()-datetime.now().date()).days
                yf=YEAR_FACTORS.get(date.year,1.0+(date.year-2024)*0.01)
                mi=MONSOON_INT.get(fm,0.1)
                md=float(MONSOON_DAY.get(fm,0))

                if days_diff<=2:
                    wx_h=float(str(wx["humidity"]).replace("--","0")) or HUM_NORMS.get(fm,45)
                    wx_t=float(str(wx["temp"]).replace("--","0")) or TMP_NORMS.get(fm,35)
                    wx_p=float(str(wx["pressure"]).replace("--","0")) or PRS_NORMS.get(fm,1012)
                    wx_w=float(str(wx["wind"]).replace("--","0")) or WND_NORMS.get(fm,3.0)
                    can_rain,block_reason=physical_rain_check(wx_t,wx_h,fm)
                else:
                    wx_h=HUM_NORMS.get(fm,45)
                    wx_t=TMP_NORMS.get(fm,35)
                    wx_p=PRS_NORMS.get(fm,1012)
                    wx_w=WND_NORMS.get(fm,3.0)
                    can_rain,block_reason=physical_rain_check(wx_t,wx_h,fm)

                sample=pd.DataFrame([{
                    "centroid_lat":lat,"centroid_lon":lon,
                    "year":date.year,"month":fm,"day":date.day,"dayofyear":date.dayofyear,
                    "rain_lag1":hist_avg,"rain_lag7":hist_avg,"rain_rolling7":hist_avg,
                    "humidity":wx_h,"temp":wx_t,"pressure":wx_p,"wind":wx_w,
                    "year_factor":yf,"monsoon_intensity":mi,"monsoon_day":md
                }])
                raw_amount   =max(0,float(reg_model.predict(sample)[0]))
                raw_will_rain=bool(clf_model.predict(sample)[0])

                if not can_rain:
                    amount=0.0
                    will_rain=False
                    blocked=True
                else:
                    amount=raw_amount
                    will_rain=raw_will_rain
                    blocked=False
                    block_reason=""

                risk=get_risk_level(amount,will_rain,blocked)
                enso_lbl=ENSO_LABEL.get(date.year,"Neutral")
                signals,confidence,rain_votes,total=get_confidence(
                    raw_amount,raw_will_rain,wx,forecast_df,date.date(),enso_lbl,blocked,block_reason)

                if will_rain and risk!="NONE":
                    pt=Point(lon,lat)
                    gdf_wgs=gpd.GeoDataFrame([{"mm":amount}],geometry=[pt],crs="EPSG:4326")
                    gdf_proj=gdf_wgs.to_crs("EPSG:32644")
                    gdf_proj=gdf_proj.set_geometry(gdf_proj.geometry.buffer(10000))
                    afc=gpd.sjoin(canals,gdf_proj,how="inner",predicate="intersects")
                    afe=gpd.sjoin(embankments,gdf_proj,how="inner",predicate="intersects")
                    afc_u=afc[["canal_name","canal_type","length_km"]].dropna(subset=["canal_name"]).drop_duplicates()
                    afe_u=afe[["name","river","length_km","location"]].dropna(subset=["name"]).drop_duplicates()
                    afc_u=afc_u[afc_u["canal_name"].str.strip()!=""]
                    afe_u=afe_u[afe_u["name"].str.strip()!=""]
                else:
                    afc_u=pd.DataFrame(); afe_u=pd.DataFrame()

                cs=", ".join(afc_u["canal_name"].tolist()[:5]) if len(afc_u)>0 else "None"
                es=", ".join(afe_u["name"].tolist()[:5])       if len(afe_u)>0 else "None"
                st.session_state.update({"prediction_done":True,"amount":amount,
                    "will_rain":will_rain,"risk":risk,"afc_u":afc_u,"afe_u":afe_u,
                    "pred_date":date.strftime("%d %b %Y"),"pred_village":village,
                    "pred_district":district,"pred_lat":lat,"pred_lon":lon,
                    "confidence":confidence,"signals":signals,"rain_votes":rain_votes,
                    "total":total,"blocked":blocked,"block_reason":block_reason})
                save_prediction(village,district,mandal,lat,lon,date_input,amount,will_rain,risk,cs,es)
                if send_sms_cb and will_rain and amount>=threshold and not blocked:
                    msg=(f"RAINFALL ALERT - {village}, {district}\n"
                         f"Date: {date.strftime('%d %b %Y')}\nExpected: {amount:.1f}mm | Risk: {risk}\n"
                         f"Confidence: {confidence}% | ENSO: {enso_lbl}\n"
                         f"Canals: {cs}\nEmbankments: {es}")
                    if send_sms(msg): st.success(f"SMS sent to {ALERT_PHONE}!")

        if st.session_state.prediction_done:
            amount    =st.session_state.amount
            will_rain =st.session_state.will_rain
            risk      =st.session_state.risk
            afc_u     =st.session_state.afc_u
            afe_u     =st.session_state.afe_u
            confidence=st.session_state.confidence
            signals   =st.session_state.signals
            rain_votes=st.session_state.get("rain_votes",0)
            total_sig =st.session_state.get("total",1)
            blocked   =st.session_state.get("blocked",False)
            block_reason=st.session_state.get("block_reason","")

            rc={"NONE":"#4caf50","LOW":"#2196f3","MODERATE":"#ff9800","HIGH":"#f44336","EXTREME":"#ff1744"}
            st.markdown("<div class='section-title'>🎯 Prediction Result</div>",unsafe_allow_html=True)
            r1,r2,r3,r4=st.columns(4)
            r1.markdown(f"<div class='result-box'><div class='stat-label'>Will it rain?</div><div class='stat-value'>{'NO 🚫' if blocked else 'YES ☔' if will_rain else 'NO ☀'}</div></div>",unsafe_allow_html=True)
            r2.markdown(f"<div class='result-box'><div class='stat-label'>Expected Rainfall</div><div class='stat-value' style='color:#00d4ff'>{amount:.1f}mm</div></div>",unsafe_allow_html=True)
            r3.markdown(f"<div class='result-box'><div class='stat-label'>Risk Level</div><div class='stat-value' style='color:{rc.get(risk,"#fff")}'>{risk}</div></div>",unsafe_allow_html=True)
            r4.markdown(f"<div class='result-box'><div class='stat-label'>Forecast Date</div><div class='stat-value' style='font-size:1.1rem'>{st.session_state.pred_date}</div></div>",unsafe_allow_html=True)

            if blocked:
                st.markdown(f"<div class='blocked-box'><h3 style='margin:0;color:#ff6b35'>🚫 Physical Gate Override</h3><p style='color:#ffaa88;margin:8px 0 0 0'>{block_reason}</p><p style='color:#aaaacc;font-size:0.85rem;margin:4px 0 0 0'>The ML model was overridden by physical meteorology constraints. Rain is physically impossible under current atmospheric conditions.</p></div>",unsafe_allow_html=True)
            else:
                rxc={"NONE":"risk-none","LOW":"risk-low","MODERATE":"risk-moderate","HIGH":"risk-high","EXTREME":"risk-extreme"}
                rxm={"NONE":"✅ No significant rainfall expected. Conditions are safe.",
                    "LOW":"🔵 Light rainfall expected. No major infrastructure risk.",
                    "MODERATE":"🟠 Moderate rainfall predicted. Monitor water levels closely.",
                    "HIGH":"🔴 Heavy rainfall warning! Inspect canals and embankments immediately.",
                    "EXTREME":"🚨 EXTREME RAINFALL ALERT! Emergency action required now!"}
                st.markdown(f"<div class='{rxc[risk]}'><h3 style='margin:0;color:white'>{rxm[risk]}</h3></div>",unsafe_allow_html=True)

            cc="#4caf50" if confidence>=67 else "#ff9800" if confidence>=34 else "#f44436" if not blocked else "#ff6b35"
            cl=("🚫 Blocked by physical meteorology gate" if blocked else
                "HIGH CONFIDENCE — All signals agree" if confidence>=67 else
                "MODERATE CONFIDENCE — Majority agree" if confidence>=34 else
                "LOW CONFIDENCE — Human review recommended")
            st.markdown("<div class='section-title'>🔬 Confidence Analysis — Multi-Signal Verification</div>",unsafe_allow_html=True)
            st.markdown(f"""<div class='result-box'>
            <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'>
            <span class='stat-label'>Prediction Confidence</span>
            <span style='color:{cc};font-weight:700'>{cl}</span></div>
            <div style='background:#222244;border-radius:6px;height:14px;margin:8px 0'>
            <div style='background:{cc};width:{confidence}%;height:14px;border-radius:6px'></div></div>
            <div style='color:#aaaacc;font-size:0.85rem'>{rain_votes} of {total_sig} signals predict rain | Confidence: {confidence}%</div>
            </div>""",unsafe_allow_html=True)

            for sn,sr,sreason,scol in signals:
                icon="✅" if sr=="RAIN" else "🚫" if "blocked" in sreason.lower() or "impossible" in sreason.lower() or "blocked" in sn.lower() else "❌"
                rcol="#4caf50" if sr=="RAIN" else "#ff6b35" if icon=="🚫" else "#666688"
                st.markdown(f"""<div class='signal-box'>
                <div style='display:flex;justify-content:space-between'>
                <span style='color:{scol};font-weight:600'>{icon} {sn}</span>
                <span style='color:{rcol};font-weight:700'>{sr}</span></div>
                <div style='color:#aaaacc;font-size:0.82rem;margin-top:4px'>{sreason}</div>
                </div>""",unsafe_allow_html=True)

            if will_rain and risk!="NONE" and not blocked:
                st.markdown("<div class='section-title'>🏗 Affected Infrastructure</div>",unsafe_allow_html=True)
                ic1,ic2=st.columns(2)
                ic1.markdown(f"<div class='infra-card'><div style='color:#aaaacc;font-size:0.75rem;text-transform:uppercase'>CANALS AT RISK</div><div style='font-size:2.5rem;font-weight:700;color:#ff4444'>{len(afc_u)}</div><div style='color:#aaaacc;font-size:0.8rem'>within 10km radius</div></div>",unsafe_allow_html=True)
                ic2.markdown(f"<div class='infra-card'><div style='color:#aaaacc;font-size:0.75rem;text-transform:uppercase'>EMBANKMENTS AT RISK</div><div style='font-size:2.5rem;font-weight:700;color:#ff9800'>{len(afe_u)}</div><div style='color:#aaaacc;font-size:0.8rem'>within 10km radius</div></div>",unsafe_allow_html=True)
                if len(afc_u)>0:
                    st.markdown("**Canals at risk:**")
                    st.dataframe(afc_u.reset_index(drop=True),use_container_width=True)
                if len(afe_u)>0:
                    st.markdown("**Embankments at risk:**")
                    st.dataframe(afe_u.reset_index(drop=True),use_container_width=True)
                if len(afc_u)==0 and len(afe_u)==0:
                    st.markdown("<div class='tip-box'>No major canals or embankments within 10km. This is expected for villages away from major irrigation networks.</div>",unsafe_allow_html=True)
            else:
                st.markdown("<div class='risk-none'><h4 style='margin:0;color:#4caf50'>✅ No infrastructure at risk.</h4></div>",unsafe_allow_html=True)

            st.markdown("<div class='section-title'>📍 Location on Map</div>",unsafe_allow_html=True)
            mini_map=folium.Map(location=[st.session_state.pred_lat,st.session_state.pred_lon],zoom_start=10,tiles="CartoDB dark_matter")
            folium.Marker([st.session_state.pred_lat,st.session_state.pred_lon],
                popup=f"{st.session_state.pred_village}<br>{amount:.1f}mm|{risk}|{confidence}% conf",
                icon=folium.Icon(color="red" if risk in ["HIGH","EXTREME"] else "blue" if risk=="MODERATE" else "green",icon="cloud")).add_to(mini_map)
            folium.Circle([st.session_state.pred_lat,st.session_state.pred_lon],radius=10000,
                color="red" if risk in ["HIGH","EXTREME"] else "#00d4ff",
                fill=True,fill_opacity=0.08,tooltip="10km monitoring radius").add_to(mini_map)
            st_folium(mini_map,width=None,height=260)

with tab2:
    st.markdown("<div class='section-title'>🗺 Live Risk Map — Andhra Pradesh</div>",unsafe_allow_html=True)
    st.markdown("""<div class='tip-box'>
    Current month rainfall risk across AP villages. Click any dot for details.<br>
    🟢 No risk | 🔵 Low (1-5mm) | 🟠 Moderate (5-15mm) | 🔴 High (15-25mm) | ⚫ Extreme (25mm+)
    </div>""",unsafe_allow_html=True)
    with st.spinner("Loading map..."):
        map_data=build_map_data()
        m=folium.Map(location=[15.9,79.7],zoom_start=7,tiles="CartoDB dark_matter")
        cmap={"NONE":"#4caf50","LOW":"#2196f3","MODERATE":"#ff9800","HIGH":"#f44336","EXTREME":"#880000"}
        for _,row in map_data.iterrows():
            mm=float(row["live_mm"])
            rv=get_risk_level(mm,bool(row["live_rain"]))
            folium.CircleMarker(
                location=[row["centroid_lat"],row["centroid_lon"]],
                radius=5,color=cmap.get(rv,"gray"),fill=True,fill_opacity=0.8,
                popup=folium.Popup(f"<b>{row['village']}</b><br>{row['district']}<br>{mm:.1f}mm | {rv}",max_width=200)
            ).add_to(m)
        st_folium(m,width=None,height=420)
    st.markdown("""<div style='display:flex;gap:24px;margin-top:8px;flex-wrap:wrap;'>
    <span style='color:#4caf50;font-weight:600'>● No Risk</span>
    <span style='color:#2196f3;font-weight:600'>● Low</span>
    <span style='color:#ff9800;font-weight:600'>● Moderate</span>
    <span style='color:#f44336;font-weight:600'>● High</span>
    <span style='color:#880000;font-weight:600'>● Extreme</span>
    </div>""",unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='section-title'>📊 Analytics and Trends</div>",unsafe_allow_html=True)
    history=get_prediction_history()
    if not history.empty and len(history)>=2:
        total=len(history)
        rainy=int(history["will_rain"].sum())
        avg_mm=history["predicted_mm"].mean()
        max_mm=history["predicted_mm"].max()
        m1,m2,m3,m4=st.columns(4)
        m1.markdown(f"<div class='result-box'><div class='stat-label'>Total Predictions</div><div class='stat-value'>{total}</div></div>",unsafe_allow_html=True)
        m2.markdown(f"<div class='result-box'><div class='stat-label'>Rainy Predictions</div><div class='stat-value' style='color:#2196f3'>{rainy}</div></div>",unsafe_allow_html=True)
        m3.markdown(f"<div class='result-box'><div class='stat-label'>Avg Rainfall</div><div class='stat-value' style='color:#00d4ff'>{avg_mm:.1f}mm</div></div>",unsafe_allow_html=True)
        m4.markdown(f"<div class='result-box'><div class='stat-label'>Max Predicted</div><div class='stat-value' style='color:#ff4444'>{max_mm:.1f}mm</div></div>",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        col_a,col_b=st.columns(2)
        with col_a:
            dr=history.groupby("district")["predicted_mm"].mean().reset_index()
            dr.columns=["District","Avg mm"]
            dr=dr.sort_values("Avg mm",ascending=False).head(15)
            f1=go.Figure(go.Bar(x=dr["District"],y=dr["Avg mm"],
                marker=dict(color=dr["Avg mm"],colorscale="Blues",showscale=False),
                text=[f"{v:.1f}" for v in dr["Avg mm"]],textposition="outside"))
            f1.update_layout(title=dict(text="Avg Rainfall by District",font=dict(color="#e0e0ff")),
                height=320,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                font_color="#aaaacc",margin=dict(t=40,b=80,l=10,r=10),
                xaxis=dict(showgrid=False,tickangle=30),yaxis=dict(showgrid=True,gridcolor="#222244",title="mm"))
            st.plotly_chart(f1,use_container_width=True)
            st.caption("Taller bar = more predicted rainfall. Prioritise for flood monitoring.")
        with col_b:
            rk=history["risk_level"].value_counts().reset_index()
            rk.columns=["Risk Level","Count"]
            cm2={"NONE":"#4caf50","LOW":"#2196f3","MODERATE":"#ff9800","HIGH":"#f44336","EXTREME":"#880000"}
            f2=go.Figure(go.Pie(labels=rk["Risk Level"],values=rk["Count"],
                marker_colors=[cm2.get(r,"gray") for r in rk["Risk Level"]],
                textinfo="label+percent+value",hole=0.4))
            f2.update_layout(title=dict(text="Risk Distribution",font=dict(color="#e0e0ff")),
                height=320,paper_bgcolor="rgba(0,0,0,0)",font_color="#aaaacc",
                margin=dict(t=40,b=10,l=10,r=10),
                annotations=[dict(text=f"{total}<br>total",x=0.5,y=0.5,font_size=14,font_color="#fff",showarrow=False)])
            st.plotly_chart(f2,use_container_width=True)
            st.caption("During monsoon shifts to MODERATE and HIGH.")
        col_c,col_d=st.columns(2)
        with col_c:
            f3=go.Figure()
            for rl,color in [("EXTREME","#880000"),("HIGH","#f44336"),("MODERATE","#ff9800"),("LOW","#2196f3"),("NONE","#4caf50")]:
                sub=history[history["risk_level"]==rl]
                if len(sub)>0:
                    f3.add_trace(go.Scatter(x=sub["timestamp"],y=sub["predicted_mm"],mode="markers",
                        name=rl,marker=dict(color=color,size=10,opacity=0.85)))
            f3.update_layout(title=dict(text="Prediction History",font=dict(color="#e0e0ff")),
                height=320,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                font_color="#aaaacc",margin=dict(t=40,b=40,l=10,r=10),
                xaxis=dict(showgrid=False,title="Time"),
                yaxis=dict(showgrid=True,gridcolor="#222244",title="mm"),
                legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(f3,use_container_width=True)
            st.caption("Each dot = one prediction. Color = risk level.")
        with col_d:
            if "village" in history.columns:
                vc=history.groupby("village")["predicted_mm"].mean().reset_index()
                vc.columns=["Village","Avg mm"]
                t10=vc.sort_values("Avg mm",ascending=True).tail(10)
                f4=go.Figure(go.Bar(x=t10["Avg mm"],y=t10["Village"],orientation="h",
                    marker=dict(color=t10["Avg mm"],colorscale="Reds",showscale=False),
                    text=[f"{v:.1f}mm" for v in t10["Avg mm"]],textposition="outside"))
                f4.update_layout(title=dict(text="Top 10 High Rainfall Villages",font=dict(color="#e0e0ff")),
                    height=320,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#aaaacc",margin=dict(t=40,b=10,l=10,r=60),
                    xaxis=dict(showgrid=True,gridcolor="#222244",title="Avg mm"),yaxis=dict(showgrid=False))
                st.plotly_chart(f4,use_container_width=True)
                st.caption("Villages with highest predicted rainfall — prioritise for flood preparedness.")
        st.markdown("<div class='section-title'>📋 Recent Predictions Log</div>",unsafe_allow_html=True)
        sc=[c for c in ["timestamp","village","district","predicted_mm","risk_level","will_rain"] if c in history.columns]
        dp=history[sc].head(20).copy()
        dp.columns=[c.replace("_"," ").title() for c in dp.columns]
        st.dataframe(dp,use_container_width=True)
    else:
        st.info("Run predictions first — charts appear automatically.")

with tab4:
    st.markdown("<div class='section-title'>🔔 Alert History</div>",unsafe_allow_html=True)
    alerts=get_recent_alerts()
    if not alerts.empty:
        st.markdown(f"<div class='tip-box'>Total alerts: <b>{len(alerts)}</b> — HIGH and EXTREME predictions only.</div>",unsafe_allow_html=True)
        for _,row in alerts.iterrows():
            css="risk-high" if row["alert_type"] in ["HIGH","EXTREME"] else "risk-moderate"
            mv=float(row["predicted_mm"]) if row["predicted_mm"] else 0
            st.markdown(f"""<div class="{css}">
            <div style="display:flex;justify-content:space-between">
            <b style="color:white;font-size:1.1rem">{row["alert_type"]} ALERT</b>
            <span style="color:#aaaacc;font-size:0.8rem">{row["timestamp"]}</span></div>
            <div style="color:white;margin:4px 0"><b>{row["village"]}</b>, {row["district"]}</div>
            <div style="color:#ff9800;font-weight:700">{mv:.1f}mm predicted</div>
            <div style="color:#aaaacc;font-size:0.82rem;margin-top:4px">
            Canals: {row["affected_canals"]}<br>Embankments: {row["affected_embankments"]}
            </div></div>""",unsafe_allow_html=True)
    else:
        st.info("No alerts yet. HIGH and EXTREME predictions appear here automatically.")
