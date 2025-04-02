import streamlit as st
import streamlit_shadcn_ui as ui
import pandas as pd
import random
import altair as alt
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import datetime
import os
import streamlit as st
import calendar
from sklearn.preprocessing import StandardScaler
import base64
import matplotlib.image as mpimg

from config import CONFIG
from fonctions import *

st.set_page_config(page_title="Streamlit Chelsea ⚽️🔵", page_icon="🔵", layout="wide")

@st.cache_data
def get_data():
    df_gps = pd.read_csv(CONFIG["paths"]["gps_data"], encoding='ISO-8859-1')
    df_photo_club = pd.read_excel(CONFIG["paths"]["photo_club"])
    df_indiv_prio = pd.read_csv(CONFIG["paths"]["individual_priority"])
    df_physical = pd.read_csv(CONFIG["paths"]["physical_data"])
    df_recovery = pd.read_csv(CONFIG["paths"]["recovery_data"])
    df_target_gps = pd.read_csv(CONFIG["paths"]["target_data"])
    df_predicted_hr = pd.read_csv(CONFIG["paths"]["predicted_hr_data"])    
    df_rpe_data = pd.read_csv(CONFIG["paths"]["RPE_data"])    
    df_weight = pd.read_csv(CONFIG["paths"]["WEIGHT_data"])    
    df_injury_history = pd.read_csv(CONFIG["paths"]["Injury_History"])
    df_gps['date'] = pd.to_datetime(df_gps['date'], format=CONFIG["date_format"])
    df_predicted_hr['date'] = pd.to_datetime(df_predicted_hr['date'], format=CONFIG["date_format"])
    df_target_gps['date'] = pd.to_datetime(df_target_gps['date'], format=CONFIG["date_format"])
    df_recovery['sessionDate'] = pd.to_datetime(df_recovery['sessionDate'], format=CONFIG["date_format"])
    df_physical['testDate'] = pd.to_datetime(df_physical['testDate'], format=CONFIG["date_format"])
    df_rpe_data['date'] = pd.to_datetime(df_rpe_data['date'])
    df_weight['date'] = pd.to_datetime(df_weight['date'])
    df_injury_history['injuryDate'] = pd.to_datetime(df_injury_history['injuryDate'])
    return df_gps, df_photo_club, df_indiv_prio, df_physical, df_recovery, df_target_gps, df_predicted_hr, df_rpe_data, df_weight, df_injury_history

today_date = pd.to_datetime("14/03/2025", format="%d/%m/%Y")

df_gps, df_photo_club, df_indiv_prio, df_physical, df_recovery, df_target_gps, df_predicted_hr, df_rpe_data, df_weight, df_injury_history = get_data()

df_matchs = df_gps.dropna()
df_matchs = pd.merge(df_matchs,df_photo_club,how='left',left_on='opposition_full',right_on='Team').drop('Team',axis=1)
df_gps = df_gps[df_gps['date'] <= today_date].reset_index(drop=True)

max_speed_ever = df_gps['peak_speed'].max()

df_gps_7_last_days = df_gps[df_gps['date'].isin([today_date - datetime.timedelta(days=i) for i in range(7)])].reset_index(drop=True).fillna("")
df_gps_15_last_days = df_gps[df_gps['date'].isin([today_date - datetime.timedelta(days=i) for i in range(15)])].reset_index(drop=True).fillna("")
df_gps_last_month = df_gps[df_gps['date'].isin([today_date - datetime.timedelta(days=i) for i in range(30)])].reset_index(drop=True).fillna("")
current_season = list(df_gps["season"])[-1]
df_gps_season = df_gps[df_gps['season'] == current_season].reset_index(drop=True).fillna("")

gps_metrics = ['distance','distance_over_21','distance_over_24','distance_over_27','accel_decel_over_2_5','accel_decel_over_3_5',
                                     'accel_decel_over_4_5','day_duration','peak_speed']
ss = StandardScaler()

df_gps_ss = ss.fit_transform(df_gps[df_gps['day_duration'] > 0][gps_metrics])
df_gps_ss = pd.DataFrame(df_gps_ss,columns=gps_metrics,index=df_gps[df_gps['day_duration'] > 0]['date']).reset_index()

##### SIDEBAR ##### 

# st.sidebar.header("Player Informations",)

##### HEADER PAGE ##### 

cols = st.columns([5,2])

with cols[0]:

    cols_ = st.columns([1,8])
    with cols_[0]:
            img_player = "https://www.leballonrond.fr/img/jogadores/30/162530_ori_hernan_crespo.jpg"
            st.markdown(f"""<div style="display: flex; justify-content: center; align-items: center;"><img src="{img_player}" width="90px" style="border-radius: 30%;" /></div>""",unsafe_allow_html=True)
    with cols_[1]:
        st.markdown(f'<h3>Hernan Crespo</h3>', unsafe_allow_html=True)
        st.markdown(f"<h6>Attaquant</h6>",unsafe_allow_html=True)
        st.markdown(f"<h6>26 ans</h6>",unsafe_allow_html=True)

with cols[1]:
    liste_tabs = ['General', 'Training Session', 'GPS Data', 'Month Overview']
    value_tab = ui.tabs(options=liste_tabs, default_value=liste_tabs[0], key="liste_tabs")
    if value_tab == "Training Session":
        date_session = st.date_input("Date:", value=None, max_value=today_date, format="DD/MM/YYYY")

st.divider()

######### GENERAL INFORMATIONS ######### 

if value_tab == "General":

##### CALENDAR ##### 

    if "start_date" not in st.session_state:

        st.session_state.start_date = today_date

    SLIDING_WINDOW_SIZE = 12

    def get_date_range(start_date):
        date_before = [start_date - datetime.timedelta(days=i) for i in range(1,SLIDING_WINDOW_SIZE//2)]
        date_after = [start_date + datetime.timedelta(days=i) for i in range(SLIDING_WINDOW_SIZE//2)]
        date_range = date_before + date_after
        return sorted(date_range)

    def move_dates(direction,days_diff=1):
        if direction == "previous":
            st.session_state.start_date -= datetime.timedelta(days=days_diff)
        elif direction == "next":
            st.session_state.start_date += datetime.timedelta(days=days_diff)
        elif direction == "today":
            st.session_state.start_date = today_date

    # st.markdown("<h6 align=center><u>Calendar:<u></h6>",unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 5, 0.5])

    with col1:
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        if st.button("⬅ Previous"):
            move_dates("previous")
        if st.button("⬅ Previous 10 days"):
            move_dates("previous",10)

    with col2:
        col2_cols = st.columns([4.1,2,4])
        with col2_cols[1]:
            if st.button("📆 Today"):
                move_dates("today")
        
    with col3:
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        if st.button("Next ➡"):
            move_dates("next")
        if st.button("Next 10 days ➡"):
            move_dates("next",10)

    date_range = get_date_range(st.session_state.start_date)

    col_list = col2.columns(12)

    for col, date in zip(col_list, date_range):
        with col.container(border=True,height=150):
            color = "royalblue" if date == today_date else "white"
            st.markdown(f"<div style='text-align: center; font-weight: bold; color:{color}'>{date.strftime('%d %b')}</div>",unsafe_allow_html=True)
            if date in list(df_matchs['date']):
                df_match_date = df_matchs[df_matchs['date'] == date].reset_index(drop=True)
                opposite_team, url_team = df_match_date['opposition_full'][0], df_match_date['Logo_url'][0]
                opposite_team = opposite_team if len(opposite_team.split()) == 1 else opposite_team.split()[0]
                # st.markdown(f"<div style='text-align: center; font-weight: bold; color: white; font-size: 12px; display: block;'>{opposite_team}</div>",unsafe_allow_html=True)
                st.markdown(f"<br>",unsafe_allow_html=True)
                st.markdown(f"""<div style="display: flex; justify-content: center; align-items: center;"><img src="{url_team}" width="40px" style="border-radius: 50%;" /></div>""",unsafe_allow_html=True)
            else:
                if date <= today_date:
                    st.markdown(f"<br>",unsafe_allow_html=True)
                    duration_day = df_gps[df_gps['date'] == date].reset_index(drop=True)['day_duration'][0]
                    PATH_ICON = "icons/"
                    url_icon = "https://cdn.pixabay.com/photo/2012/04/05/01/08/sleep-25528_1280.png" if duration_day == 0 else "https://img.freepik.com/vecteurs-premium/icone-du-logo-vectoriel-chronometre_414847-333.jpg"
                    st.markdown(f"""<div style="display: flex; justify-content: center; align-items: center;"><img src="{url_icon}" width="35px" style="border-radius: 50%;" /></div>""",unsafe_allow_html=True)

    st.divider()

    a, b, c, d = st.columns(4)

    md_plus_value = int(df_gps['md_plus_code'].iloc[-1])
    df_last_match = df_gps[df_gps['date'] == (today_date - datetime.timedelta(days=md_plus_value))]
    df_last_match = pd.merge(df_last_match,df_photo_club,how='left',left_on='opposition_full',right_on='Team').drop('Team',axis=1)
    last_match_opposite_team_url, last_match_duration = df_last_match['Logo_url'].iloc[0], round(df_last_match['day_duration'].iloc[0])

    df_recovery['sessionDate'] = pd.to_datetime(df_recovery['sessionDate'], format="%d/%m/%Y")
    last_recovery_date = df_recovery['sessionDate'].iloc[-1]
    df_last_recovery = df_recovery[df_recovery['sessionDate'] == last_recovery_date].reset_index(drop=True)
    last_recovery_score = df_last_recovery[df_last_recovery['metric'] == "emboss_baseline_score"]['value'].iloc[0]

    df_physical_last = df_physical.drop_duplicates(subset=['expression','movement','quality'],keep='last').reset_index(drop=True)
    physical_overall = df_physical_last.groupby('expression')['benchmarkPct'].mean().to_frame().reset_index()
    isometric_overall, dynamic_overall = physical_overall[physical_overall['expression'] == "isometric"]['benchmarkPct'].iloc[0] * 100, physical_overall[physical_overall['expression'] == "dynamic"]['benchmarkPct'].iloc[0] * 100

    top_priority = df_indiv_prio[df_indiv_prio['Priority'] == 1].reset_index(drop=True)

    HEIGHT_CONTAINER = 300

    with a.container(border=True,height=HEIGHT_CONTAINER):
        st.markdown("<h6 align=center>Last Game</h6>",unsafe_allow_html=True)
        cols_a = st.columns(2)
        cols_a[0].markdown(f"""<div style="display: flex; justify-content: center; align-items: center;"><img src="{last_match_opposite_team_url}" width="60px" style="border-radius: 50%;" /></div>""",unsafe_allow_html=True)
        cols_a[0].markdown(f"""<p style="font-size:12px; text-align:center; margin-top:10px">{md_plus_value} days ago</p>""",unsafe_allow_html=True)
        cols_a[1].markdown(f"""<h1 style="text-align:center; margin-top:-20px">{last_match_duration}'</h1>""", unsafe_allow_html=True)
        cols_a[1].markdown(f"""<h6 style="text-align:center; margin-top:-10px; font-size:20px; color:red">⬊ 30%</h1>""",unsafe_allow_html=True)

        df_matchs_current_season = df_matchs[(df_matchs['season'] == current_season) & (df_matchs['day_duration'] > 0)].reset_index(drop=True)
        
        st.markdown(f"<br>",unsafe_allow_html=True)
        st.markdown(f"<br>",unsafe_allow_html=True)
        st.markdown(f"""<p style="font-size:16px; margin-left:10px; text-align:center">This season :</p>""",unsafe_allow_html=True)
        st.markdown(f"""<p style="font-size:16px; margin-left:10px; margin-top:-10px; text-align:center"><b>{len(df_matchs_current_season)}</b> games played - <b>{round(df_matchs_current_season['day_duration'].sum())}</b> minutes played</p>""",unsafe_allow_html=True)

    with b.container(border=True,height=HEIGHT_CONTAINER):

        st.markdown("""<h6 align=center>Recovery Score</h6>""",unsafe_allow_html=True)
        url_icon = np.select([last_recovery_score <= -1,(last_recovery_score > -1) & (last_recovery_score <= 0),(last_recovery_score > 0) & (last_recovery_score <= 0.5),(last_recovery_score > 0.5) & (last_recovery_score <= 0.75),(last_recovery_score > 0.75)],
            ["icons/very_low_score_blue.png","icons/low_score_blue.png","icons/middle_score_blue.png","icons/good_score_blue.png","icons/excellent_score_blue.png"],default="icons/middle_score_blue.png"
        ).astype(object)

        cols = st.columns([1.1,2,0.2])
        cols[1].image(str(url_icon), width=150)

        fig, ax = plot_init((20,6),spine_color=False)

        df_recovery_7_last_days = df_recovery[df_recovery['sessionDate'].isin(df_gps_7_last_days['date'])].reset_index(drop=True)
        df_recovery_7_last_days = df_recovery_7_last_days[df_recovery_7_last_days['metric'] == 'emboss_baseline_score'].reset_index(drop=True)

        bars = plt.bar(df_recovery_7_last_days['sessionDate'],df_recovery_7_last_days['value'],width=0.7,linewidth=2)

        for i in range(len(bars)):
            date = df_recovery_7_last_days['sessionDate'].iloc[i]
            bar = bars[i]
            value = bar.get_height()

            color = np.select(
                [value <= -1,(value > -1) & (value <= -0.5),(value > -0.5) & (value <= 0.5),(value > 0.5) & (value <= 1),value > 1],
                ["#d6e6f4","#8fc2de","#66abd4","#3080bd","#083e81"],default="#000000" ).item()

            bar.set_color(color)
            bar.set_edgecolor("white")

        plt.axhline(0,color='lightgrey',linestyle=':')

        min_max_value = df_recovery[df_recovery['metric'] == 'emboss_baseline_score']['value']

        plt.ylabel('Recovery Score\n',color='white',fontsize=30)
        plt.yticks([])
        plt.ylim(min_max_value.min(),min_max_value.max());
        plt.xticks(df_recovery_7_last_days['sessionDate'],[date.strftime('%d %b') for date in df_recovery_7_last_days['sessionDate']],fontsize=25);
    
        st.pyplot(fig)

    with c.container(border=True,height=HEIGHT_CONTAINER):

        st.markdown("""<h6 align=center>RPE Sessions</h6>""",unsafe_allow_html=True)

        cols_pills = st.columns([1,2,0.2])
        selection_pills = cols_pills[1].pills("Directions", ["GPS","Physical","Day"], selection_mode="single", label_visibility="collapsed", default="GPS")
    
        st.markdown("<br>",unsafe_allow_html=True)

        scaler = StandardScaler()
        df_rpe_data['RPE_scaled'] = df_rpe_data.groupby('session')['RPE'].transform(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten())

        df_rpe_7_last_days = df_rpe_data[df_rpe_data['date'].isin(df_gps_7_last_days['date'])].reset_index(drop=True)

        df_rpe_7_last_days_mean_day = df_rpe_7_last_days.drop('session',axis=1).groupby('date').mean()[['RPE','RPE_scaled']].reset_index()

        fig, ax = plot_init((20,6),spine_color=False)

        if selection_pills == "GPS":
            
            df_rpe_values = df_rpe_7_last_days[df_rpe_7_last_days['session'] == "GPS"].reset_index(drop=True)

            bars = plt.bar(df_rpe_values['date'],df_rpe_values['RPE_scaled'],width=0.7,linewidth=2)

            for i in range(len(bars)):
                date = df_rpe_values['date'].iloc[i]
                bar = bars[i]
                value = bar.get_height()

                color = np.select(
                    [value <= -1, (value > -1) & (value <= -0.5), (value > -0.5) & (value <= 0.5), (value > 0.5) & (value <= 1), value > 1],
                ["#d6e6f4", "#8fc2de", "#66abd4", "#3080bd", "#083e81"],default="#000000").item()

                bar.set_color(color)
                bar.set_edgecolor("white")

        elif selection_pills == "Physical":

            df_rpe_values = df_rpe_7_last_days[df_rpe_7_last_days['session'] == "Physical Test"].reset_index(drop=True)

            bars = plt.bar(df_rpe_values['date'],df_rpe_values['RPE_scaled'],width=0.7,linewidth=2)

            for i in range(len(bars)):
                date = df_rpe_values['date'].iloc[i]
                bar = bars[i]
                value = bar.get_height()

                color = np.select(
                    [value <= -1, (value > -1) & (value <= -0.5), (value > -0.5) & (value <= 0.5), (value > 0.5) & (value <= 1), value > 1],
                ["#d6e6f4", "#8fc2de", "#66abd4", "#3080bd", "#083e81"],default="#000000").item()

                bar.set_color(color)
                bar.set_edgecolor("white")

        else:

            bars = plt.bar(df_rpe_7_last_days_mean_day['date'],df_rpe_7_last_days_mean_day['RPE_scaled'],width=0.7,linewidth=2)

            for i in range(len(bars)):
                date = df_rpe_7_last_days_mean_day['date'].iloc[i]
                bar = bars[i]
                value = bar.get_height()

                color = np.select(
                    [value <= -1, (value > -1) & (value <= -0.5), (value > -0.5) & (value <= 0.5), (value > 0.5) & (value <= 1), value > 1],
                ["#d6e6f4", "#8fc2de", "#66abd4", "#3080bd", "#083e81"],default="#000000").item()

                bar.set_color(color)
                bar.set_edgecolor("white")

        plt.axhline(0,color='lightgrey',linestyle=':')

        plt.ylabel('Z-ScoreRPE\n',color='white',fontsize=30)
        plt.yticks([])
        plt.xticks(df_recovery_7_last_days['sessionDate'],[date.strftime('%d %b') for date in df_recovery_7_last_days['sessionDate']],fontsize=25);        
        
        st.pyplot(fig)

    with d.container(border=True,height=HEIGHT_CONTAINER):

        st.markdown("""<h6 align=center>Coach's Note</h6>""",unsafe_allow_html=True)

        cols = st.columns([2,4])
        try:
            cols[0].image("images/Enzo_maresca.png", width=150)
        except:
            try:
                cols[0].image("icons/enzo_maresca_2.webp", width=150)
            except:
                pass


        cols[1].markdown(
            """
            <div style="text-align: justify; font-size: 14px; margin-top: 10px;">
                Hey Hernan,<br>
                You <span style="font-weight:bold; color:#8fc2de;">played a lot</span> these last few weeks, that is why 
                <span style="font-weight:bold; color:#8fc2de;">I managed your play time</span> yesterday.<br>
                You had a great impact and you <span style="font-weight:bold; color:#8fc2de;">deserved to score</span> (0.76 personal xG).<br>
                The <span style="font-weight:bold; color:#8fc2de;">next game against Arsenal</span> will be the opportunity for you to 
                <span style="font-weight:bold; color:#8fc2de;">start</span> (and score of course) ⚽️🔵
            </div>""",unsafe_allow_html=True)

    #pass_br(1)

    a, b, c, d = st.columns(4) 
    
    HEIGHT_CONTAINER = 380

    with a.container(border=True, height=HEIGHT_CONTAINER):

        st.markdown("<h6 align=center>Injury History</h6>", unsafe_allow_html=True)

        fig, ax = plot_init((10, 6))

        img = mpimg.imread('icons/human_body_grey.png')
        ax.imshow(img, extent=[0, 0.25, 0, 0.35])  

        ax.scatter([0.085], [0.08], color='#3080bd', s=500, edgecolor='black',linewidth=2)
        ax.scatter([0.045], [0.08], color='#083e81', s=500, edgecolor='black',linewidth=2)
        ax.scatter([0.16], [0.06], color='#8fc2de', s=300, edgecolor='black',linewidth=2)
        ax.scatter([0.086], [0.022], color='#8fc2de', s=300, edgecolor='black',linewidth=2)

        ax.scatter([0.26], [0.1], color='#0e1117', s=100, edgecolor='#0e1117')

        ax.axis('off')

        col1, col2, col3 = st.columns([1, 2, 1]) 
        with col2:

            st.pyplot(fig)

    with b.expander("Player Weight",expanded=True):

        OPTIMAL_WEIGHT = 90.0

        st.markdown("""<h6 align=center>Player Weight</h6>""",unsafe_allow_html=True)

        last_weight_date, last_weight = df_weight['date'].iloc[-1], df_weight['weight'].iloc[-1]

        color = "orange" if last_weight != OPTIMAL_WEIGHT else "green"

        diff_weight = round(last_weight - OPTIMAL_WEIGHT,1)
        diff_weight = "+" + str(diff_weight)if diff_weight > 0 else str(diff_weight)
 
        st.markdown(f"""
        <p style="text-align:center; font-weight:600;">
            <span style="font-size:30px; color:snow;">{last_weight}</span>
            <span style="font-size:12px; color:white;"> kg</span>
            <span style="font-size:14px; color:{color};">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;( {diff_weight} kg)</span>
        </p>""", unsafe_allow_html=True)

        fig, ax = plot_init((20,6),spine_color=False)

        df_weight_7_last_days = df_weight[df_weight['date'].isin(df_gps_7_last_days['date'])].reset_index(drop=True)
        df_weight_7_last_days['color'] = 'white'
        last_index = list(df_weight_7_last_days.index)[-1]
        df_weight_7_last_days['color'][last_index] = "orange" if df_weight_7_last_days["weight"][last_index] > OPTIMAL_WEIGHT else "#3080bd"

        plt.scatter(df_weight_7_last_days['date'], df_weight_7_last_days['weight'],s=400,color=df_weight_7_last_days['color'],zorder=2)
        plt.plot(df_weight_7_last_days['date'], df_weight_7_last_days['weight'],linewidth=3,zorder=1)

        plt.plot([last_weight_date,last_weight_date],[OPTIMAL_WEIGHT,last_weight],linewidth=3)

        plt.ylabel('Weight\n',color='white',fontsize=30)
        plt.xticks(df_recovery_7_last_days['sessionDate'],[date.strftime('%d %b') for date in df_recovery_7_last_days['sessionDate']],fontsize=25);

        plt.axhline(OPTIMAL_WEIGHT,linewidth=3,linestyle='--',color='snow')

        st.pyplot(fig)

        st.markdown(f"""
        <p style="text-align:center; font-size: 14px; margin-top: 10px">Optimal weight: {OPTIMAL_WEIGHT} kg</p>""", unsafe_allow_html=True)

    with c.expander("Physical Capability Score",expanded=True):

        st.markdown("""<h6 align=center>Physical Capability Score</h6>""",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        cols = st.columns(2)
        cols[0].markdown("<p align=center>Isometric</p>",unsafe_allow_html=True)
        cols[0].markdown(f"""<p style="display: flex; justify-content: center;font-size: 25px; margin-top: -10px; color: white">{round(isometric_overall,1)} %</p>""",unsafe_allow_html=True)
        cols[1].markdown("<p align=center>Dynamic</p>",unsafe_allow_html=True)
        cols[1].markdown(f"""<p style="display: flex; justify-content: center;font-size: 25px; margin-top: -10px; color: white">{round(dynamic_overall,1)} %</p>""",unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown(f"""<p style="display: flex; justify-content: center;font-size: 15px;">Most Recent Test: {' '.join(df_physical_last.sort_values('testDate')[['expression','movement']].iloc[0].values)} ({df_physical_last.sort_values('testDate')['testDate'].iloc[0].strftime('%d %b')})</p>""",unsafe_allow_html=True)
        st.markdown(f"""<p style="display: flex; justify-content: center;font-size: 15px;">Most Outdated Test: {' '.join(df_physical_last.sort_values('testDate',ascending=False)[['expression','movement']].iloc[0].values)} ({df_physical_last.sort_values('testDate',ascending=False)['testDate'].iloc[0].strftime('%d %b')})</p>""",unsafe_allow_html=True)

        pass_br(3)

    with d.expander("Top Priority Areas",expanded=True):

        st.markdown("""<h6 align=center>Top Priority Areas</h6>""",unsafe_allow_html=True)
        cols = st.columns([2,4])
        cols[0].image("icons/sleeping.png", width=100)
        cols[1].markdown(f"""<p style="display: flex; justify-content: justify;font-size: 15px; margin-top: 20px">{top_priority['Target'][0]}</p>""",unsafe_allow_html=True)

        df_recovery_7_last_days = df_recovery[df_recovery['sessionDate'].isin(df_gps_7_last_days['date'])].reset_index(drop=True)
        df_sleep_7_last_days = df_recovery_7_last_days[df_recovery_7_last_days['metric'] == 'sleep_baseline_composite'].reset_index(drop=True)

        fig, ax = plot_init((20,6),spine_color=False)

        bars = plt.bar(df_sleep_7_last_days['sessionDate'], [6,0,0,6,7,6])

        for i in range(len(bars)):
            
            date = df_sleep_7_last_days['sessionDate'].iloc[i]
            bar = bars[i]
            value = bar.get_height()

            if value > 0:
                plt.text(date,1,value,color='white',fontsize=24)

            color = np.select(
                [value <= -1, (value > -1) & (value <= -0.5), (value > -0.5) & (value <= 0.5), (value > 0.5) & (value <= 1), value > 1],
                ["#d6e6f4", "#8fc2de", "#66abd4", "#3080bd", "#083e81"],default="#000000").item()

            bar.set_color(color)
            bar.set_edgecolor("white")

        plt.axhline(8,color='lightgrey',linestyle=':',linewidth=3)
        plt.text(plt.xlim()[0], 8.2, "8 hours of sleep", color='white', fontsize=25, ha='left', va='bottom')

        min_max_value = df_recovery[df_recovery['metric'] == 'emboss_baseline_score']['value']

        plt.ylabel('Hours of Sleep\n',color='white',fontsize=30)
        plt.yticks([])
        plt.ylim(0,10);
        plt.xticks(df_recovery_7_last_days['sessionDate'],[date.strftime('%d %b') for date in df_recovery_7_last_days['sessionDate']],fontsize=25);
    
        st.pyplot(fig)

######### GENERAL GPS INFORMATIONS ######### 

if value_tab == "GPS Data":

    col1, col2, col3 = st.columns([1.98, 2, 1])  
    with col2:
        choice_date = st.radio("Date Range",["7 last days", "15 last days", "Last month","Season"],horizontal=True, label_visibility="collapsed")

    a, b, c, d, e = st.columns(5)

    with a.container(border=True,height=150):
        md_plus_value = int(df_gps['md_plus_code'].iloc[-1])
        df_last_match = df_gps[df_gps['date'] == (today_date - datetime.timedelta(days=md_plus_value))]
        df_last_match = pd.merge(df_last_match,df_photo_club,how='left',left_on='opposition_full',right_on='Team').drop('Team',axis=1)
        last_match_opposite_team_url, last_match_duration = df_last_match['Logo_url'].iloc[0], round(df_last_match['day_duration'].iloc[0])
        st.markdown("<h6 align=center>Dernier Match</h6>",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        cols_last_game = st.columns([2,5])
        with cols_last_game[0]:
            st.markdown(f"""<div style="display: flex; justify-content: center; align-items: center;"><img src="{last_match_opposite_team_url}" width="40px" style="border-radius: 50%;" /></div>""",unsafe_allow_html=True)
        with cols_last_game[1]:
            st.markdown(f"<p align=center>{md_plus_value} days ago<br>{last_match_duration} minutes played",unsafe_allow_html=True)
    with b.container(border=True,height=150):
        distance_7_last_days = round(df_gps_7_last_days['distance'].sum() / 1000,1)
        distance_15_last_days = round(df_gps_15_last_days['distance'].sum() / 1000,1)
        distance_last_month = round(df_gps_last_month['distance'].sum() / 1000,1)
        distance_season = round(df_gps_season['distance'].sum() / 1000,1)
        distance_value = distance_7_last_days if choice_date == "7 last days" else distance_15_last_days if choice_date == "15 last days" else\
              distance_last_month if choice_date == "Last month" else distance_season
        cols_container = st.columns([2,1])
        cols_container[0].markdown("""<h6 style="text-align: center;">Distance (km)</h6>""", unsafe_allow_html=True)
        cols_container[1].markdown(f'<p style="font-size: 10px; margin-top: 0px;">({choice_date})</p>',unsafe_allow_html=True)
        cols_metric = st.columns([1,1,1])
        cols_metric[1].metric("Label",distance_value,"13%",label_visibility="collapsed")
    with c.container(border=True,height=150):
        distance_over_21_7_last_days = round(df_gps_7_last_days['distance_over_21'].sum() / 1000,1)
        distance_over_21_15_last_days = round(df_gps_15_last_days['distance_over_21'].sum() / 1000,1)
        distance__over_21_last_month = round(df_gps_last_month['distance_over_21'].sum() / 1000,1)
        distance__over_21_season = round(df_gps_season['distance_over_21'].sum() / 1000,1)
        distance_over_21_value = distance_over_21_7_last_days if choice_date == "7 last days" else distance_over_21_15_last_days if choice_date == "15 last days" else \
            distance__over_21_last_month if choice_date == "Last month" else distance__over_21_season
        cols_container = st.columns([2,1])
        cols_container[0].markdown("""<h6 style="text-align: center;">Distance > 21 (km)</h6>""", unsafe_allow_html=True)
        cols_container[1].markdown(f'<p style="font-size: 10px; margin-top: 0px;">({choice_date})</p>',unsafe_allow_html=True)
        cols_metric = st.columns([1.2,2,0.5])
        cols_metric[1].metric("Label",distance_over_21_value,"13%",label_visibility="collapsed")
    with d.container(border=True,height=150):
        accel_decel_over_3_5_7_last_days = round(df_gps_7_last_days['accel_decel_over_3_5'].sum())
        accel_decel_over_3_5_15_last_days = round(df_gps_15_last_days['accel_decel_over_3_5'].sum())
        accel_decel_over_3_5_last_month = round(df_gps_last_month['accel_decel_over_3_5'].sum())
        accel_decel_over_3_5_season = round(df_gps_season['accel_decel_over_3_5'].sum())
        accel_decel_over_3_5_value = accel_decel_over_3_5_7_last_days if choice_date == "7 last days" else accel_decel_over_3_5_15_last_days if choice_date == "15 last days" else \
            accel_decel_over_3_5_last_month if choice_date == "Last month" else accel_decel_over_3_5_season
        cols_container = st.columns([2,1])
        cols_container[0].markdown("""<h6 style="text-align: center;">Accel-Decel >3.5 ms2</h6>""", unsafe_allow_html=True)
        cols_container[1].markdown(f'<p style="font-size: 10px; margin-top: 0px;">({choice_date})</p>',unsafe_allow_html=True)
        cols_metric = st.columns([1.2,2,0.5])
        cols_metric[1].metric("Label",accel_decel_over_3_5_value,"13%",label_visibility="collapsed")
    with e.container(border=True,height=150):
        max_speed_7_last_days = round(df_gps_7_last_days['peak_speed'].max(),1)
        max_speed_15_last_days = round(df_gps_15_last_days['peak_speed'].max(),1)
        max_speed_last_month = round(df_gps_last_month['peak_speed'].max(),1)
        max_speed_season = round(df_gps_season['peak_speed'].max(),1)
        max_speed_value = max_speed_7_last_days if choice_date == "7 last days" else max_speed_15_last_days if choice_date == "15 last days" else \
            max_speed_last_month if choice_date == "Last month" else max_speed_season
        cols_container = st.columns([2,1])
        cols_container[0].markdown("""<h6 style="text-align: center;">Max Speed (km/h)</h6>""", unsafe_allow_html=True)
        cols_container[1].markdown(f'<p style="font-size: 10px; margin-top: 0px;">({choice_date})</p>',unsafe_allow_html=True)
        cols_metric = st.columns([1.2,2,0.5])
        cols_metric[1].metric("Label",max_speed_value,"13%",label_visibility="collapsed")

    st.markdown("<br>",unsafe_allow_html=True)

    a, b = st.columns(2)

    with a.expander("Distance",expanded=True):

        st.markdown("<h6 align=center>Distance</h6>",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)

        df_values = df_gps_7_last_days.copy() if choice_date == "7 last days" else df_gps_15_last_days.copy() if choice_date == "15 last days"\
            else df_gps_last_month.copy() if choice_date == "Last month" else df_gps_season.copy()
        size = 30 if choice_date == "7 last days" else 20 if choice_date == "15 last days" else 10 if choice_date == "Last month" else 3
        max_distance = df_values["distance"].max() 

        st.vega_lite_chart(
            data=df_values,
            spec={
                "mark": {"type": "bar", "tooltip": True, "size": size, "cornerRadiusEnd": 5}, 
                "encoding": {
                    "x": {"field": "date", "type": "temporal", "axis": {"title": "Date of Activity"}},
                    "y": {"field": "distance", "type": "quantitative", "axis": {"title": "Distance (m)"}},
                    "color": {  
                        "condition": [
                            {"selection": "hover", "value": "#00148b"}, 
                            {"test": "datum.opposition_full == ''", "value": "grey"} 
                        ],
                        "value": "lightblue",
                    },
                    "stroke": { 
                        "condition": {"test": f"datum.distance == {max_distance}", "value": "gold"},
                        "value": "transparent" 
                    },
                    "strokeWidth": {  
                        "condition": {"test": f"datum.distance == {max_distance}", "value": 5},
                        "value": 0
                    },
                    "tooltip": [
                        {"field": "date", "type": "temporal", "title": "Date"},
                        {"field": "distance", "type": "quantitative", "title": "Distance (m)", "format": ".1f"},
                        {"field": "opposition_full", "type": "nominal", "title": "Adversaire"},
                    ],
                },
                "selection": {
                    "hover": {"type": "single", "on": "mouseover", "empty": "none"}
                },
            },
        )

    with b.expander("Distance over 21",expanded=True):

        st.markdown("<h6 align=center>Distance over 21 km/h</h6>",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)

        df_values = df_gps_7_last_days.copy() if choice_date == "7 last days" else df_gps_15_last_days.copy() if choice_date == "15 last days"\
            else df_gps_last_month.copy() if choice_date == "Last month" else df_gps_season.copy()
        size = 30 if choice_date == "7 last days" else 20 if choice_date == "15 last days" else 10 if choice_date == "Last month" else 3
        max_distance_over_21 = df_values["distance_over_21"].max() 

        st.vega_lite_chart(
            data=df_values,
            spec={
                "mark": {"type": "bar", "tooltip": True, "size": size, "cornerRadiusEnd": 5}, 
                "encoding": {
                    "x": {"field": "date", "type": "temporal", "axis": {"title": "Date of Activity"}},
                    "y": {"field": "distance_over_21", "type": "quantitative", "axis": {"title": "Distance over 21 (m)"}},
                    "color": {  
                        "condition": [
                            {"selection": "hover", "value": "#00148b"}, 
                            {"test": "datum.opposition_full == ''", "value": "grey"}  
                        ],
                        "value": "lightblue"  
                    },
                    "stroke": { 
                        "condition": {"test": f"datum.distance_over_21 == {max_distance_over_21}", "value": "gold"},
                        "value": "transparent" 
                    },
                    "strokeWidth": {  
                        "condition": {"test": f"datum.distance_over_21 == {max_distance_over_21}", "value": 5},
                        "value": 0
                    },
                    "tooltip": [
                        {"field": "date", "type": "temporal", "title": "Date"},
                        {"field": "distance_over_21", "type": "quantitative", "title": "Distance over 21 (m)", "format": ".1f"},
                        {"field": "opposition_full", "type": "nominal", "title": "Adversaire"},
                    ],
                },
                "selection": {
                    "hover": {"type": "single", "on": "mouseover", "empty": "none"}
                },
            },
        )

    a, b = st.columns(2)

    with a.expander("Accel-Decel > 3.5 ms2",expanded=False):

        st.markdown("<h6 align=center>Accel-Decel > 3.5 ms2</h6>",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)

        df_values = df_gps_7_last_days.copy() if choice_date == "7 last days" else df_gps_15_last_days.copy() if choice_date == "15 last days"\
            else df_gps_last_month.copy() if choice_date == "Last month" else df_gps_season.copy()
        size = 30 if choice_date == "7 last days" else 20 if choice_date == "15 last days" else 10 if choice_date == "Last month" else 3
        max_accel_decel = df_values["accel_decel_over_3_5"].max() 

        st.vega_lite_chart(
            data=df_values,
            spec={
                "mark": {"type": "bar", "tooltip": True, "size": size, "cornerRadiusEnd": 5}, 
                "encoding": {
                    "x": {"field": "date", "type": "temporal", "axis": {"title": "Date of Activity"}},
                    "y": {"field": "accel_decel_over_3_5", "type": "quantitative", "axis": {"title": "Accel-Decel > 3.5"}},
                    "color": {  
                        "condition": [
                            {"selection": "hover", "value": "#00148b"}, 
                            {"test": "datum.opposition_full == ''", "value": "grey"}  
                        ],
                        "value": "lightblue"  
                    },
                    "stroke": { 
                        "condition": {"test": f"datum.accel_decel_over_3_5 == {max_accel_decel}", "value": "gold"},
                        "value": "transparent" 
                    },
                    "strokeWidth": {  
                        "condition": {"test": f"datum.accel_decel_over_3_5 == {max_accel_decel}", "value": 5},
                        "value": 0
                    },
                    "tooltip": [
                        {"field": "date", "type": "temporal", "title": "Date"},
                        {"field": "accel_decel_over_3_5", "type": "quantitative", "title": "Accel-Decel > 3.5"},
                        {"field": "opposition_full", "type": "nominal", "title": "Adversaire"},
                    ],
                },
                "selection": {
                    "hover": {"type": "single", "on": "mouseover", "empty": "none"}
                },
            },
        )

    with b.expander("Maximal Speed",expanded=False):

        st.markdown("<h6 align=center>Maximal Speed</h6>",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)

        df_values = df_gps_7_last_days.copy() if choice_date == "7 last days" else df_gps_15_last_days.copy() if choice_date == "15 last days"\
            else df_gps_last_month.copy() if choice_date == "Last month" else df_gps_season.copy()
        size = 30 if choice_date == "7 last days" else 20 if choice_date == "15 last days" else 10 if choice_date == "Last month" else 3
        max_speed = df_values["peak_speed"].max() 

        st.vega_lite_chart(
            data=df_values,
            spec={
                "mark": {"type": "bar", "tooltip": True, "size": size, "cornerRadiusEnd": 5}, 
                "encoding": {
                    "x": {"field": "date", "type": "temporal", "axis": {"title": "Date of Activity"}},
                    "y": {"field": "peak_speed", "type": "quantitative", "axis": {"title": "Maximal Speed (km/h)"}},
                    "color": {  
                        "condition": [
                            {"selection": "hover", "value": "#00148b"}, 
                            {"test": "datum.opposition_full == ''", "value": "grey"}  
                        ],
                        "value": "lightblue"  
                    },
                    "stroke": { 
                        "condition": {"test": f"datum.peak_speed == {max_speed}", "value": "gold"},
                        "value": "transparent" 
                    },
                    "strokeWidth": {  
                        "condition": {"test": f"datum.peak_speed == {max_speed}", "value": 5},
                        "value": 0
                    },
                    "tooltip": [
                        {"field": "date", "type": "temporal", "title": "Date"},
                        {"field": "peak_speed", "type": "quantitative", "title": "Maximal Speed (km/h)"},
                        {"field": "opposition_full", "type": "nominal", "title": "Adversaire"},
                    ],
                },
                "selection": {
                    "hover": {"type": "single", "on": "mouseover", "empty": "none"}
                },
            },
        )

######### TRAINING SESSION INFORMATIONS ######### 

if value_tab == "Training Session":

    if date_session is not None:
        
        try:
            date_session = pd.to_datetime(date_session)
            df_gps_date = df_gps[df_gps['date'] == date_session].reset_index(drop=True).fillna("")
            df_gps_date = pd.merge(df_gps_date,df_target_gps,on="date",how='left')    
            type_activity = "Training" if df_gps_date['opposition_code'][0] == "" else f"Match | {df_gps_date['opposition_full'][0]}"

            df_gps_7_last_days_bis = df_gps[df_gps['date'].isin([date_session - datetime.timedelta(days=i) for i in range(7)])].reset_index(drop=True).fillna("")

            df_predicted_hr_date = df_predicted_hr[df_predicted_hr['date'] == date_session].reset_index(drop=True)

            a, b = st.columns([1,3],gap="small")

            with a.container(border=False):
                st.markdown(f"""<div style='display: flex; flex-direction: column; justify-content: center; height: 100px; text-align: center; margin-top: -10px'><p style='font-size: 10px; margin: 0;'>Inspired by Parmia Calcio.</p>
                        <p style='font-size: 18px; margin: 0;'><b>{type_activity} | </b>{date_session.strftime('%A %d %B %Y')}</p></div>""",unsafe_allow_html=True)
                
                count_target = check_targets(df_gps_date)
                color_target = np.select([count_target == 1,count_target == 2,count_target == 3,count_target == 4,count_target == 5],
                                        ["#d6e6f4", "#8fc2de", "#66abd4", "#3080bd", "#083e81"],default="#000000")
                
                comment = df_predicted_hr_date['comment_session'][0]

                if pd.notna(comment):
                    color_comment = "darkred" if comment == "Harder than expected" else "lightblue"
                    st.markdown(f"""
                        <div style='text-align:center;'>
                            <span style='color:{color_target}; font-weight:600; font-size:14px;'>{count_target} targets achieved.</span><br>
                            <span style='color:{color_comment}; font-weight:600; font-size:14px;'>(HR) {comment}</span>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div style='text-align:center;'>
                            <span style='color:white; font-weight:600; font-size:14px;'>{count_target} targets achieved.</span>
                        </div>
                    """, unsafe_allow_html=True)
            
            time_columns = [col for col in df_gps_date.columns if col.startswith("hr_zone")]

            for col in time_columns:
                df_gps_date[col] = round(pd.to_timedelta(df_gps_date[col]).dt.total_seconds() / 60,2)

            df_gps_date["total_hr_time_min"] = df_gps_date[time_columns].sum(axis=1)

            for col in time_columns:
                proportion_col = col + "_prop"
                df_gps_date[proportion_col] = round(df_gps_date[col] / df_gps_date["total_hr_time_min"] * 100,1)

            df_gps_date = df_gps_date.fillna(0)
            dominant_zone = df_gps_date[[col for col in df_gps_date.columns if 'prop' in col]].idxmax(axis=1).values[0]

            with b.container(border=True):
                cols = st.columns([0.2,3])
                
                cols[0].image(f"icons/pulse-{dominant_zone[8]}.png", width=50)

                fig, ax = plt.subplots(figsize=(50,1))

                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#0e1117')

                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')

                for spine in ax.spines.values():
                    spine.set_color('black')

                bar_5 = plt.barh(['Session'],df_gps_date['hr_zone_1_hms_prop'].iloc[0] + df_gps_date['hr_zone_2_hms_prop'].iloc[0] + df_gps_date['hr_zone_3_hms_prop'].iloc[0] + df_gps_date['hr_zone_4_hms_prop'].iloc[0] + + df_gps_date['hr_zone_5_hms_prop'].iloc[0],color='#d93321',edgecolor='white',label='Zone 5')
                bar_4 = plt.barh(['Session'],df_gps_date['hr_zone_1_hms_prop'].iloc[0] + df_gps_date['hr_zone_2_hms_prop'].iloc[0] + df_gps_date['hr_zone_3_hms_prop'].iloc[0] + df_gps_date['hr_zone_4_hms_prop'].iloc[0],color='#d99921',edgecolor='white',label='Zone 4')
                bar_3 = plt.barh(['Session'],df_gps_date['hr_zone_1_hms_prop'].iloc[0] + df_gps_date['hr_zone_2_hms_prop'].iloc[0] + df_gps_date['hr_zone_3_hms_prop'].iloc[0],color='#b9c110',edgecolor='white',label='Zone 3')
                bar_2 = plt.barh(['Session'],df_gps_date['hr_zone_1_hms_prop'].iloc[0] + df_gps_date['hr_zone_2_hms_prop'].iloc[0],color='#10c133',edgecolor='white',label='Zone 2')
                bar_1 = plt.barh(['Session'],df_gps_date['hr_zone_1_hms_prop'].iloc[0],color='#3988e8',edgecolor='white',label='Zone 1')

                plt.ylim(-0.5,0.5)

                liste_bars = [bar_1,bar_2,bar_3,bar_4,bar_5]
                liste_values = [df_gps_date['hr_zone_1_hms_prop'].iloc[0],df_gps_date['hr_zone_2_hms_prop'].iloc[0],df_gps_date['hr_zone_3_hms_prop'].iloc[0],df_gps_date['hr_zone_4_hms_prop'].iloc[0],df_gps_date['hr_zone_5_hms_prop'].iloc[0]]
                liste_x_ticks = []

                for i in range(len(liste_bars)):

                    bar, value = liste_bars[i], liste_values[i]
                    x = bar[0].get_width()
                    x_previous = liste_bars[i-1][0].get_width() if i>= 1 else 0
                    diff_x = x - x_previous
                    liste_x_ticks.append((x_previous + diff_x//2)+ 1)
                    if value > 5:
                        plt.text((x_previous + diff_x//2),-0.1,str(value) + "%",color='white',fontsize=18)

                plt.xticks(liste_x_ticks,[f"Zone {i}" for i in range(1,6)],fontsize=20)
                plt.yticks([])

                cols[1].pyplot(fig)

                with st.expander("Predicted HR"):

                    fig, ax = plt.subplots(figsize=(50,1))

                    fig.patch.set_facecolor('#0e1117')
                    ax.set_facecolor('#0e1117')

                    ax.tick_params(axis='x', colors='white')
                    ax.tick_params(axis='y', colors='white')

                    for spine in ax.spines.values():
                        spine.set_color('black')

                    df_predicted_hr_date_plot = round(df_predicted_hr_date[[col for col in df_predicted_hr_date.columns if "predict" in col]] * 100,1)

                    bar_5 = plt.barh(['Session'],df_predicted_hr_date_plot['predict_prop_hr1'].iloc[0] + df_predicted_hr_date_plot['predict_prop_hr2'].iloc[0] + df_predicted_hr_date_plot['predict_prop_hr3'].iloc[0] + df_predicted_hr_date_plot['predict_prop_hr4'].iloc[0] + + df_predicted_hr_date_plot['predict_prop_hr5'].iloc[0],color='#d93321',edgecolor='white',label='Zone 5')
                    bar_4 = plt.barh(['Session'],df_predicted_hr_date_plot['predict_prop_hr1'].iloc[0] + df_predicted_hr_date_plot['predict_prop_hr2'].iloc[0] + df_predicted_hr_date_plot['predict_prop_hr3'].iloc[0] + df_predicted_hr_date_plot['predict_prop_hr4'].iloc[0],color='#d99921',edgecolor='white',label='Zone 4')
                    bar_3 = plt.barh(['Session'],df_predicted_hr_date_plot['predict_prop_hr1'].iloc[0] + df_predicted_hr_date_plot['predict_prop_hr2'].iloc[0] + df_predicted_hr_date_plot['predict_prop_hr3'].iloc[0],color='#b9c110',edgecolor='white',label='Zone 3')
                    bar_2 = plt.barh(['Session'],df_predicted_hr_date_plot['predict_prop_hr1'].iloc[0] + df_predicted_hr_date_plot['predict_prop_hr2'].iloc[0],color='#10c133',edgecolor='white',label='Zone 2')
                    bar_1 = plt.barh(['Session'],df_predicted_hr_date_plot['predict_prop_hr1'].iloc[0],color='#3988e8',edgecolor='white',label='Zone 1')

                    plt.ylim(-0.5,0.5)

                    liste_bars = [bar_1,bar_2,bar_3,bar_4,bar_5]
                    liste_values = [df_predicted_hr_date_plot['predict_prop_hr1'].iloc[0],df_predicted_hr_date_plot['predict_prop_hr2'].iloc[0],df_predicted_hr_date_plot['predict_prop_hr3'].iloc[0],
                                    df_predicted_hr_date_plot['predict_prop_hr4'].iloc[0],df_predicted_hr_date_plot['predict_prop_hr5'].iloc[0]]
                    liste_x_ticks = []

                    for i in range(len(liste_bars)):

                        bar, value = liste_bars[i], liste_values[i]
                        x = bar[0].get_width()
                        x_previous = liste_bars[i-1][0].get_width() if i>= 1 else 0
                        diff_x = x - x_previous
                        liste_x_ticks.append((x_previous + diff_x//2)+ 1)
                        if value > 5:
                            plt.text((x_previous + diff_x//2),-0.1,str(value) + "%",color='white',fontsize=18)

                    plt.xticks(liste_x_ticks,[f"Zone {i}" for i in range(1,6)],fontsize=20)
                    plt.yticks([])

                    st.pyplot(fig)

            a, b, c, d, e = st.columns(5,gap="small")

            with a.container(border=True):
                    
                st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 16px; font-weight: bold; color:lightgrey">TOTAL DISTANCE</p></div>""", unsafe_allow_html=True)
                st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 40px; color:lightgrey">{round(df_gps_date['distance'][0],1)} m</p></div>""", unsafe_allow_html=True)
                if type_activity == "Training":
                    target = df_gps_date['target_distance'][0]
                    st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 12px; color:lightgrey">Target: {target} m</p></div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 12px; color:lightgrey">Moyenne Match: x m</p></div>""", unsafe_allow_html=True)
                    target = None

                st.markdown("<br>",unsafe_allow_html=True)

                fig, ax = plot_init((30,2))
                
                plot_barh_session(type_activity,target,df_gps_date,'distance',df_matchs,current_season,"#00148b")

                st.pyplot(fig)

                pass_br(2)

                fig, ax = plot_init((20,6),spine_color=False)

                plot_last7days_session(df_gps_7_last_days_bis,'distance',"#00148b")

                st.pyplot(fig)

                st.divider()

                st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 12px; color:#4b4f53; font-weight: bold">LAST 7 DAYS TOTAL</p></div>""", unsafe_allow_html=True)
                st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 30px; color:#4b4f53">{round(df_gps_7_last_days_bis['distance'].sum()):,} m</p></div>""", unsafe_allow_html=True)

            with b.container(border=True):
                    
                st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 16px; font-weight: bold; color:lightgrey">HIGH SPEED DISTANCE (over 21)</p></div>""", unsafe_allow_html=True)
                st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 40px; color:lightgrey">{round(df_gps_date['distance_over_21'][0],1)} m</p></div>""", unsafe_allow_html=True)
                if type_activity == "Training":
                    target = df_gps_date['target_distance_over_21'][0]
                    st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 12px; color:lightgrey">Target: {target} m</p></div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 12px; color:lightgrey">Moyenne Match: x m</p></div>""", unsafe_allow_html=True)
                    target = None

                st.markdown("<br>",unsafe_allow_html=True)

                fig, ax = plot_init((30,2))
                
                plot_barh_session(type_activity,target,df_gps_date,'distance_over_21',df_matchs,current_season,"#00148b")

                st.pyplot(fig)

                pass_br(2)

                fig, ax = plot_init((20,6),spine_color=False)

                plot_last7days_session(df_gps_7_last_days_bis,'distance_over_21',"#00148b")

                st.pyplot(fig)

                st.divider()

                st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 12px; color:#4b4f53; font-weight: bold">LAST 7 DAYS TOTAL</p></div>""", unsafe_allow_html=True)
                st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 30px; color:#4b4f53">{round(df_gps_7_last_days_bis['distance_over_21'].sum()):,} m</p></div>""", unsafe_allow_html=True)

            with c.container(border=True):
                    
                st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 16px; font-weight: bold; color:lightgrey">VERY HIGH SPEED DISTANCE (over 27)</p></div>""", unsafe_allow_html=True)
                st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 40px; color:lightgrey">{round(df_gps_date['distance_over_27'][0],1)} m</p></div>""", unsafe_allow_html=True)
                if type_activity == "Training":
                    target = df_gps_date['target_distance_over_27'][0]
                    st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 12px; color:lightgrey">Target: {target} m</p></div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 12px; color:lightgrey">Moyenne Match: x m</p></div>""", unsafe_allow_html=True)

                st.markdown("<br>",unsafe_allow_html=True)

                fig, ax = plot_init((30,2))
                
                plot_barh_session(type_activity,target,df_gps_date,'distance_over_27',df_matchs,current_season,"#00148b")

                st.pyplot(fig)

                pass_br(2)

                fig, ax = plot_init((20,6),spine_color=False)

                plot_last7days_session(df_gps_7_last_days_bis,'distance_over_27',"#00148b")

                st.pyplot(fig)

                st.divider()

                st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 12px; color:#4b4f53; font-weight: bold">LAST 7 DAYS TOTAL</p></div>""", unsafe_allow_html=True)
                st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 30px; color:#4b4f53">{round(df_gps_7_last_days_bis['distance_over_27'].sum()):,} m</p></div>""", unsafe_allow_html=True)

            with d.container(border=True):
                    
                st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 16px; font-weight: bold; color:lightgrey">ACCEL-DECEL (over 3.5)</p></div>""", unsafe_allow_html=True)
                st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 40px; color:lightgrey">{round(df_gps_date['accel_decel_over_3_5'][0],1)} m</p></div>""", unsafe_allow_html=True)
                if type_activity == "Training":
                    target = df_gps_date['target_accel_decel_over_3_5'][0]
                    st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 12px; color:lightgrey">Target: {target} m</p></div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 12px; color:lightgrey">Moyenne Match: x m</p></div>""", unsafe_allow_html=True)

                st.markdown("<br>",unsafe_allow_html=True)

                fig, ax = plot_init((30,2))
                
                plot_barh_session(type_activity,target,df_gps_date,'accel_decel_over_3_5',df_matchs,current_season,"darkred")

                st.pyplot(fig)

                pass_br(2)

                fig, ax = plot_init((20,6),spine_color=False)

                plot_last7days_session(df_gps_7_last_days_bis,'accel_decel_over_3_5',"darkred")

                st.pyplot(fig)

                st.divider()

                st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 12px; color:#4b4f53; font-weight: bold">LAST 7 DAYS TOTAL</p></div>""", unsafe_allow_html=True)
                st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 30px; color:#4b4f53">{round(df_gps_7_last_days_bis['accel_decel_over_3_5'].sum()):,} m</p></div>""", unsafe_allow_html=True)

            with e.container(border=True):
                    
                st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 16px; font-weight: bold; color:lightgrey">MAXIMAL SPEED</p></div>""", unsafe_allow_html=True)
                st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 40px; color:lightgrey">{round(df_gps_date['peak_speed'][0],1)} km/h</p></div>""", unsafe_allow_html=True)
                if type_activity == "Training":
                    target = df_gps_date['target_peak_speed'][0]
                    st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 12px; color:lightgrey">Target: {target} km/h</p></div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 12px; color:lightgrey">Moyenne Match: x km/h</p></div>""", unsafe_allow_html=True)

                st.markdown("<br>",unsafe_allow_html=True)

                fig, ax = plot_init((30,2),spine_color=False)

                if type_activity == "Training":

                    plt.barh([""],[target * 1.2],height=0.5,edgecolor='black',color='#eee')
                    plt.barh([""],[target],height=0.49,color='#ddd')
                    plt.barh([""],[target * 0.75],height=0.49,color='#ccc')
                    plt.barh([""],[target * 0.5],height=0.49,color='#bbb')
                    
                    plt.barh([""],[df_gps_date['peak_speed'][0]],height=0.2,color="purple",edgecolor='black') 

                    color_target = "black" if target > df_gps_date['peak_speed'][0] else "gold"
                    plt.axvline(target,linewidth=50,zorder=3,color=color_target)

                    plt.xlim(0,target * 1.2)
                    plt.ylim(-0.2,0.2);
                    liste_x_ticks = [0,target * 0.5,target * 0.75,target,target * 1.2]
                    plt.xticks(liste_x_ticks,["\n0","\n50%","\n75%","\nTarget","\n120%"],fontsize=45);

                else:
                    
                    max_match = df_matchs[df_matchs['season'] == current_season]['peak_speed'].max()

                    plt.barh([""],[max_match],height=0.5,edgecolor='black',color='#eee')
                    plt.barh([""],[max_match * 0.75],height=0.49,color='#ddd')
                    plt.barh([""],[max_match * 0.50],height=0.49,color='#ccc')
                    plt.barh([""],[max_match * 0.25],height=0.49,color='#bbb')
                    
                    plt.barh([""],[df_gps_date['peak_speed'][0]],height=0.2,color="purple",edgecolor='black') 

                    plt.xlim(0,max_match)
                    plt.ylim(-0.2,0.2);
                    liste_x_ticks = [0,max_match*0.25,max_match*0.5,max_match*0.75,max_match]
                    plt.xticks(liste_x_ticks,["\n0","\n25%","\n50%","\n75%","\nMax Match"],fontsize=45);

                st.pyplot(fig)

                st.markdown("<br>",unsafe_allow_html=True)
                st.markdown("<br>",unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(20,6))

                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#0e1117')

                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')

                for spine in ax.spines.values():
                    spine.set_color('black')

                plt.bar(df_gps_7_last_days_bis['date'], df_gps_7_last_days_bis['peak_speed'], width=0.3, color='purple')

                max_date = df_gps_7_last_days_bis[df_gps_7_last_days_bis['peak_speed'] == df_gps_7_last_days_bis['peak_speed'].max()]['date'].values[0]
                max_distance = df_gps_7_last_days_bis['peak_speed'].max()

                scatters = plt.scatter(max_date, max_distance, color='gold', marker='_', s=2000)

                plt.annotate(round(max_distance,1),(max_date, max_distance),color='white',fontsize=30,ha='center',va='bottom',xytext=(0, 10),textcoords='offset points')

                plt.xticks([]);
                plt.yticks([]);

                plt.xlabel('\nLast 7 days', color='white',fontsize=25);
            
                st.pyplot(fig)

                st.divider()

                st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 12px; color:#4b4f53; font-weight: bold">LAST 7 DAYS MAX</p></div>""", unsafe_allow_html=True)
                st.markdown(f"""<div style="display: flex; justify-content: center;"><p style="font-size: 30px; color:#4b4f53">{round(df_gps_7_last_days_bis['peak_speed'].max(),1)} km/h</p></div>""", unsafe_allow_html=True)

            a, b, c = st.columns(3,gap="medium")
            df_next_matchs = df_matchs[df_matchs['date'] > date_session].reset_index(drop=True)
            opponent_next_match, ecart_next_matchs = df_next_matchs['opposition_full'][0], (df_next_matchs['date'][0] - date_session).days

            with a.container(border=True,height=125):
                
                cols = st.columns([2,3])

                cols[0].image("icons/football.png", width=80)
                cols[1].markdown(f"<p align=center><u>Next Game:</u></p>",unsafe_allow_html=True)
                cols[1].markdown(f"""<div style="display: flex; align-items: center; height: 100%; justify-content: center;"><p style="font-size: 14px; color: lightgrey">{opponent_next_match} (in {ecart_next_matchs} days)</p></div>""", unsafe_allow_html=True)

            with b.container(border=True,height=125):

                cols = st.columns([2,3])

                cols[0].image("icons/sleeping.png", width=90)
                cols[1].markdown(f"<p align=center><u>Priority 1:</u></p>",unsafe_allow_html=True)
                cols[1].markdown(f"""<div style="display: flex; align-items: center; height: 100%; justify-content: center;"><p style="font-size: 14px; color: lightgrey">{df_indiv_prio['Target'][0]}</p></div>""", unsafe_allow_html=True)

            with c.container(border=True,height=125):

                cols = st.columns([2,3])

                cols[0].image("icons/diet.png", width=80)
                cols[1].markdown(f"<p align=center><u>Priority 2:</u></p>",unsafe_allow_html=True)
                cols[1].markdown(f"""<div style="display: flex; align-items: center; height: 100%; justify-content: center;"><p style="font-size: 14px; color: lightgrey">{df_indiv_prio['Target'][1]}</p></div>""", unsafe_allow_html=True)

        except:

            pass

if value_tab == "Month Overview":

##### CALENDAR ##### 

    if "start_date" not in st.session_state:

        st.session_state.start_date = today_date

    SLIDING_WINDOW_SIZE = 12

    col1, col2, col3 = st.columns([1, 5, 1])

    with col1:
        pass_br(3)        
        if st.button("⬅ Previous"):
            move_month(today_date,"previous")
        
    with col3:
        pass_br(3)
        if st.button("Next ➡"):
            move_month(today_date,"next")

    date_month = st.session_state.start_date
    start_date, end_date = get_month_date_range(date_month.year, date_month.month - 1)

    with col2:

        col2_cols = st.columns([4.1,2,4])

        with col2_cols[1]:
            if st.button("📆 Last Complete Month"):
                move_month(today_date,"today")
            
            st.markdown("<br>",unsafe_allow_html=True)

            toggle_distance = st.toggle("Distance")
            toggle_distance_high_speed = st.toggle("Distance High Speed (over 21)")
            toggle_distance_peak_over90 = st.toggle("Peak Speed (over 90%)")

            if int(date_month.month) - 1 > 0:
                st.button(calendar.month_name[int(date_month.month) - 1] + " " + str(date_month.year))
            else:
                st.button(calendar.month_name[12] + " " + str(date_month.year - 1))

        st.markdown("<br>",unsafe_allow_html=True)
        cols_date = st.columns(8)
        i_cols = 0
        
        for i in range((end_date - start_date).days + 1): 
            
            current_date = start_date + datetime.timedelta(days=i)
            
            with cols_date[i_cols].container(border=True,height=200):

                df_current_date = df_gps[df_gps['date'] == pd.to_datetime(current_date)].reset_index(drop=True)
                df_current_date_gps_scaler = df_gps_ss[df_gps_ss['date'] == pd.to_datetime(current_date)].reset_index(drop=True)

                if (len(df_current_date) > 0):
                    emoji = ("🏆" if df_current_date['day_duration'][0] > 0 and not pd.isna(df_current_date['opposition_full'][0]) else "⚽️" if df_current_date['day_duration'][0] > 0 else "")
                    color = "royalblue" if pd.to_datetime(current_date) == pd.to_datetime(today_date) else "lightgrey"
                    st.markdown(
                        f"<span style='color:{color}; font-size:10px'>{current_date.strftime('%d %b')}"
                        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
                        f"<span style='font-size:9px'>{emoji}</span></span>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(f"<span style='color:{color}'>{current_date.strftime('%d %b')}</span>", unsafe_allow_html=True)
                
                if (len(df_current_date) > 0):

                    if df_current_date['day_duration'][0] > 0:
                        
                        if toggle_distance:
                            
                            distance_scaler_value = df_current_date_gps_scaler['distance'][0]

                            plot_png(distance_scaler_value,"distance")

                        if toggle_distance_high_speed:

                            high_speed_scaler_value = df_current_date_gps_scaler['distance_over_21'][0]

                            plot_png(high_speed_scaler_value,"high-speed")

                        if toggle_distance_peak_over90:

                            peak_over90 = df_current_date['peak_speed'][0] >= 0.9 * max_speed_ever

                            if peak_over90:
                                
                                png_over_90 = "icons/peak_speed_over90.png"

                                st.markdown(f"<div style='text-align: center; margin-top: 15px'><img src='data:image/png;base64,{base64.b64encode(open(png_over_90, 'rb').read()).decode()}' width='20'/></div>",unsafe_allow_html=True)

            i_cols = i_cols + 1 if i_cols < 7 else 0

    with col1:
        
        year_value = date_month.year - 1 if date_month.month - 1 == 0 else date_month.year
        month_value = date_month.month - 1 if date_month.month - 1 > 0 else 12
        previous_month_value = month_value - 1 if month_value > 1 else 12
        previous_year_value = year_value - 1 if previous_month_value == 12 else year_value
        month_value_name, previous_month_value_name = calendar.month_name[month_value], calendar.month_name[previous_month_value]
        
        df_gps_current_month = df_gps[(df_gps['date'].dt.month == month_value) & ((df_gps['date'].dt.year == year_value))]
        df_gps_previous_month = df_gps[(df_gps['date'].dt.month == previous_month_value) & ((df_gps['date'].dt.year == previous_year_value))]
        
        st.markdown("<br><br><br><br><br>",unsafe_allow_html=True)

        try:
            month_overview_cells("distance","Total Distance",df_gps_current_month,df_gps_previous_month,previous_month_value_name)
        except:
            pass
    
        st.divider()

        try:
            month_overview_cells("distance_over_21","High Speed Distance",df_gps_current_month,df_gps_previous_month,previous_month_value_name)
        except:
            pass

        st.divider()
        
        try:
            month_overview_cells("distance_over_27","Very High Speed Distance",df_gps_current_month,df_gps_previous_month,previous_month_value_name)
        except:
            pass

        st.divider()

    with col3:

        year_value = date_month.year - 1 if date_month.month - 1 == 0 else date_month.year
        month_value = date_month.month - 1 if date_month.month - 1 > 0 else 12
        previous_month_value = month_value - 1 if month_value > 1 else 12
        previous_year_value = year_value - 1 if previous_month_value == 12 else year_value
        month_value_name, previous_month_value_name = calendar.month_name[month_value], calendar.month_name[previous_month_value]
        
        df_gps_current_month = df_gps[(df_gps['date'].dt.month == month_value) & ((df_gps['date'].dt.year == year_value))]
        df_gps_previous_month = df_gps[(df_gps['date'].dt.month == previous_month_value) & ((df_gps['date'].dt.year == previous_year_value))]
        
        st.markdown("<br><br><br><br><br>",unsafe_allow_html=True)

        try:
            month_overview_cells_v2(df_gps_current_month,df_gps_previous_month,previous_month_value_name)
        except:
            pass

        st.divider()  

        with st.expander("Legend:",expanded=False):
            #st.markdown("<p style='margin-left:20px;'><u>Legend:</u></p>",unsafe_allow_html=True)

            st.markdown("<p style='font-size:12px; margin-left:20px;'>⚽️ Training</p>",unsafe_allow_html=True)
            st.markdown("<p style='font-size:12px; margin-left:20px;'>🏆 Game</p>",unsafe_allow_html=True)
            st.markdown("<p style='margin-left:20px;'>""<span style='font-size:16px; color:#d6e6f4;'>◉ </span> ""<span style='font-size:12px; color:white;'>Very Lower than Player Average</span>""</p>",unsafe_allow_html=True)
            st.markdown("<p style='margin-left:20px;'>""<span style='font-size:16px; color:#8fc2de;'>◉ </span> ""<span style='font-size:12px; color:white;'>Lower than Player Average</span>""</p>",unsafe_allow_html=True)
            st.markdown("<p style='margin-left:20px;'>""<span style='font-size:16px; color:#66abd4;'>◉ </span> ""<span style='font-size:12px; color:white;'>Around Player Average</span>""</p>",unsafe_allow_html=True)
            st.markdown("<p style='margin-left:20px;'>""<span style='font-size:16px; color:#3080bd;'>◉ </span> ""<span style='font-size:12px; color:white;'>Higher than Player Average</span>""</p>",unsafe_allow_html=True)
            st.markdown("<p style='margin-left:20px;'>""<span style='font-size:16px; color:#083e81;'>◉ </span> ""<span style='font-size:12px; color:white;'>Very Higher than Player Average</span>""</p>",unsafe_allow_html=True)


        












