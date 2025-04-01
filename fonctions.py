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

from config import CONFIG

def plot_png(variable_value,icon,width=30):

    png_scaler = np.select([variable_value < -1.5,variable_value < -0.5,variable_value < 0.5, variable_value < 1.5],["very_low","low",'middle',"high"],default='very_high')
    png_scaler = f"/Users/maxenceduffuler/Desktop/CHELSEA_PROJECT/CODE/STREAMLIT/icons/{png_scaler}_{icon}.png"

    st.markdown(f"<div style='text-align: center; margin-top: 15px'><img src='data:image/png;base64,{base64.b64encode(open(png_scaler, 'rb').read()).decode()}' width='{width}'/></div>",unsafe_allow_html=True)

def month_overview_cells(col,text,df_gps_current_month,df_gps_previous_month,previous_month_value_name):

    total_current_month, total_previous_month = df_gps_current_month[col].sum(), df_gps_previous_month[col].sum()
    total_evolution = round((total_current_month - total_previous_month) / total_previous_month * 100)
    arrow = "➚" if total_evolution > 0 else "➘" if total_evolution < 0 else "➖"
    color_text = "green" if total_evolution > 0 else "darkred" if total_evolution < 0 else "white"

    st.markdown(f"""<p style="text-align:center; font-weight:600; font-size:14px">{text}</p>""",unsafe_allow_html=True)
    st.markdown(f"""
    <p style="text-align:center; font-weight:600;">
        <span style="font-size:25px; color:lightblue;">{round(total_current_month / 1000, 1)}</span>
        <span style="font-size:12px; color:lightblue;"> km</span>
    </p>""", unsafe_allow_html=True)
    st.markdown(f"""
    <p style="text-align:center;">
        <span style="font-size:14px; color:{color_text}; font-weight:600;">{arrow} {total_evolution}%</span>
        <span style="font-size:10px; color:white;"> vs {previous_month_value_name[:3].lower()}</span>
    </p>""", unsafe_allow_html=True)       

def month_overview_cells_v2(df_gps_current_month,df_gps_previous_month,previous_month_value_name):

        number_of_games_current_month, number_of_minutes_current_month = len(df_gps_current_month.dropna(subset='opposition_full')), df_gps_current_month.dropna(subset='opposition_full')['day_duration'].sum()
        number_of_games_previous_month, number_of_minutes_previous_month = len(df_gps_previous_month.dropna(subset='opposition_full')), df_gps_previous_month.dropna(subset='opposition_full')['day_duration'].sum()
        number_of_games_evolution = round((number_of_games_current_month - number_of_games_previous_month) / number_of_games_previous_month * 100)
        number_of_games_arrow = "➚" if number_of_games_evolution > 0 else "➘" if number_of_games_evolution < 0 else "➖"
        number_of_games_color_text = "green" if number_of_games_evolution > 0 else "darkred" if number_of_games_evolution < 0 else "white"
        number_of_minutes_evolution = round((number_of_minutes_current_month - number_of_minutes_previous_month) / number_of_minutes_previous_month * 100)
        number_of_minutes_arrow = "➚" if number_of_minutes_evolution > 0 else "➘" if number_of_minutes_evolution < 0 else "➖"
        number_of_minutes_color_text = "green" if number_of_minutes_evolution > 0 else "darkred" if number_of_minutes_evolution < 0 else "white"

        st.markdown(f"""<p style="text-align:center; font-weight:600; font-size:14px">Games Played</p>""",unsafe_allow_html=True)
        st.markdown(f"""
        <p style="text-align:center; font-weight:600;">
            <span style="font-size:25px; color:lightblue;">{number_of_games_current_month}</span>
            <span style="font-size: 12px; color:lightblue;"> games</span>
        </p>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <p style="text-align:center;">
            <span style="font-size:14px; color:{number_of_games_color_text}; font-weight:600;">{number_of_games_arrow} {number_of_games_evolution}%</span>
            <span style="font-size:10px; color:white;"> vs {previous_month_value_name[:3].lower()}</span>
        </p>
        """, unsafe_allow_html=True)  

        st.divider()  

        st.markdown(f"""<p style="text-align:center; font-weight:600; font-size:14px">Minutes Played</p>""",unsafe_allow_html=True)
        st.markdown(f"""
        <p style="text-align:center; font-weight:600;">
            <span style="font-size:25px; color:lightblue;">{round(number_of_minutes_current_month)}</span>
            <span style="font-size:12px; color:lightblue;"> minutes</span>
        </p>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <p style="text-align:center;">
            <span style="font-size:14px; color:{number_of_minutes_color_text}; font-weight:600;">{number_of_minutes_arrow} {number_of_minutes_evolution}%</span>
            <span style="font-size:10px; color:white;"> vs {previous_month_value_name[:3].lower()}</span>
        </p>
        """, unsafe_allow_html=True)  

def get_month_date_range(year, month_num):

    year = year - 1 if month_num == 0 else year
    month_num = 12 if month_num == 0 else month_num
    last_day = calendar.monthrange(year, month_num)[1]
    start_date = datetime.date(year, month_num, 1)
    end_date = datetime.date(year, month_num, last_day)
    return start_date, end_date

def move_month(today_date,direction):
    if direction == "previous":
        st.session_state.start_date -= datetime.timedelta(days=30)
    elif direction == "next":
        st.session_state.start_date += datetime.timedelta(days=30)
    elif direction == "today":
        st.session_state.start_date = today_date

def pass_br(number_of_br):
     
    for i in range(number_of_br):
        st.markdown("<br>",unsafe_allow_html=True)

def plot_init(figsize,spine_color=True):

    fig, ax = plt.subplots(figsize=figsize)

    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    
    color = "white" if spine_color else '#0e1117'
    for spine in ax.spines.values():
        spine.set_color(color)
    
    return fig, ax

def plot_barh_session(type_activity,target,df_gps_date,col,df_matchs,current_season,color):
 
    if type_activity == "Training":

        plt.barh([""],[target * 1.2],height=0.5,edgecolor='black',color='#eee')
        plt.barh([""],[target],height=0.49,color='#ddd')
        plt.barh([""],[target * 0.75],height=0.49,color='#ccc')
        plt.barh([""],[target * 0.5],height=0.49,color='#bbb')

        plt.barh([""],[df_gps_date[col][0]],height=0.2,color=color,edgecolor='black') 

        color_target = 'black' if df_gps_date[col][0] < target else 'gold'
        print(col)
        print(df_gps_date[col][0],target)
        plt.axvline(target,linewidth=50,zorder=3,color=color_target)

        plt.xlim(0,target * 1.2)
        plt.ylim(-0.2,0.2);
        liste_x_ticks = [0,target * 0.5,target * 0.75,target,target * 1.2]
        plt.xticks(liste_x_ticks,["\n0","\n50%","\n75%","\nTarget","\n120%"],fontsize=45);

    else:

        max_match = df_matchs[df_matchs['season'] == current_season][col].max()

        plt.barh([""],[max_match],height=0.5,edgecolor='black',color='#eee')
        plt.barh([""],[max_match * 0.75],height=0.49,color='#ddd')
        plt.barh([""],[max_match * 0.50],height=0.49,color='#ccc')
        plt.barh([""],[max_match * 0.25],height=0.49,color='#bbb')

        plt.barh([""],[df_gps_date['distance'][0]],height=0.2,color=color,edgecolor='black') 

        plt.xlim(0,max_match)
        plt.ylim(-0.2,0.2);
        liste_x_ticks = [0,max_match*0.25,max_match*0.5,max_match*0.75,max_match]
        plt.xticks(liste_x_ticks,["\n0","\n25%","\n50%","\n75%","\nMax Match"],fontsize=45);

def plot_last7days_session(df_gps_7_last_days_bis,col,color):

    plt.bar(df_gps_7_last_days_bis['date'], df_gps_7_last_days_bis[col], width=0.3, color=color)

    max_date = df_gps_7_last_days_bis[df_gps_7_last_days_bis[col] == df_gps_7_last_days_bis[col].max()]['date'].values[0]
    max_distance = df_gps_7_last_days_bis[col].max()

    scatters = plt.scatter(max_date, max_distance, color='gold', marker='_', s=2000)

    plt.annotate(round(max_distance),(max_date, max_distance),color='white',fontsize=30,ha='center',va='bottom',xytext=(0, 10),textcoords='offset points')

    plt.xticks([]);
    plt.yticks([]);

    plt.xlabel('\nLast 7 days', color='white',fontsize=25);

def check_targets(df_gps_date):

    target_distance = df_gps_date['target_distance'][0]
    target_distance_over_21 = df_gps_date['target_distance_over_21'][0]
    target_distance_over_27 = df_gps_date['target_distance_over_27'][0]
    target_accel_decel_over_3_5 = df_gps_date['target_accel_decel_over_3_5'][0]
    target_peak_speed = df_gps_date['target_peak_speed'][0]

    real_distance = df_gps_date['distance'][0]
    real_distance_over_21 = df_gps_date['distance_over_21'][0]
    real_distance_over_27 = df_gps_date['distance_over_27'][0]
    real_accel_decel_over_3_5 = df_gps_date['accel_decel_over_3_5'][0]
    real_peak_speed = df_gps_date['peak_speed'][0]

    count = 0

    if real_distance >= target_distance:
        count += 1
    if real_distance_over_21 >= target_distance_over_21:
        count += 1
    if real_distance_over_27 >= target_distance_over_27:
        count += 1
    if real_accel_decel_over_3_5 >= target_accel_decel_over_3_5:
        count += 1
    if real_peak_speed >= target_peak_speed:
        count += 1

    return count
        





























