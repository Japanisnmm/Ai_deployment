# Importing ToolKits
from time import sleep
import pandas as pd
import numpy as np


import streamlit as st
from streamlit.components.v1 import html
from streamlit_option_menu import option_menu

import tensorflow as tf

def run():
    st.set_page_config(
        page_title="Forecasting Loan Risk",
        page_icon="💰",
        layout="wide"
    )

    if "the_df" not in st.session_state:
        st.session_state.the_df = pd.DataFrame()

    # Function To Load Our Dataset
    @st.cache_data
    def load_loan_detection_ann_model(model_path):
        return tf.keras.models.load_model(model_path)

    @st.cache_data
    def load_scaler_transformation(model_path):
        return pd.read_pickle(model_path)

    model = load_loan_detection_ann_model(
        "loan_defualt_risk.h5")

    scaler = load_scaler_transformation(
        "model_min_max_scaler.pkl")

    def check_columns(df, order_columns):
        columns = df.columns.to_list()
        columns = list(map(lambda x: x.lower(), columns))
        order_columns = list(map(lambda x: x.lower(), order_columns))

        return 1 if columns == order_columns else 0

    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    side_bar_options_style = {
        "container": {"padding": "0!important", "background-color": '#121212', "border-radius": "0"},
        "icon": {"color": "#fff", "font-size": "20px"},
        "nav-link": {"color": "#fff", "font-size": "18px", "text-align": "left", "margin": "0px", "margin-bottom": "0px"},
        "nav-link-selected": {"background-color": "#009378", "font-size": "16px", },
    }

    header = st.container()
    content = st.container()

    st.write("")

    with st.sidebar:
        st.write("")
        st.title("KMITLoanRiskPredicted")
        st.write("")

        page = option_menu(
            menu_title=None,
            options=['Preidct'],
            icons=['robot'],
            menu_icon="cast",
            default_index=0,
            styles=side_bar_options_style
        )

        st.write("")
        # st_lottie(lottie_json, height=350, speed=1)

        # Home Page
        if page == "Preidct":
            with header:
                st.header("Loan Default Risk Prediction 💰")

            with content:
                col1, col2 = st.columns([8, 4])

                with col1:
                    with st.form("Preidct"):
                        c1, c2 = st.columns(2)
                        with c1:
                            annual_income = st.number_input('**รายได้รวมในช่วง 1 ปี**', min_value=1,
                                                            max_value=1000000000, value=8000)

                            applicant_age = st.number_input(
                                '**อายุผู้สมัคร**', min_value=21, max_value=85, value=24)

                            marital_status = st.selectbox('**สถานะ**', options=[
                                                          "แต่งงาน", "โสด"], index=1)

                        house_ownership = st.selectbox('**ข้อมูลเกี่ยวกับบ้าน**', options=[
                            "เช่าอยู่", "เจ้าของ", "อาศัยอยู่กับครอบครัว"], index=0)

                        with c2:
                            work_exp = st.number_input(
                                '**ประสบการณ์การทำงาน**', min_value=0, max_value=40, value=0)

                            years_in_current_employment = st.number_input(
                                '**จำนวนปีที่ทำงาน**', min_value=0, max_value=30, value=0)

                            vehicle_ownership = st.selectbox('**ยานพาหนะ**', options=[
                                "มี", "ไม่มี"], index=1)

                        predict_button = st.form_submit_button("**Predict** 🚀")

                with col2:
                    if predict_button:

                        # marital_status
                        marital_status_encodded = 0  # Married
                        if marital_status == "โสด":
                            marital_status_encodded = 1

                        # house_ownership
                        house_ownership_encodded = [0, 0]  # No-Rent No-Own

                        if house_ownership == "เช่าอยู่":
                            house_ownership_encodded = [0, 1]  # Rented

                        elif house_ownership == "เจ้าของ":
                            house_ownership_encodded = [1, 0]  # Owned

                        # vehicle_ownership
                        vehicle_ownership_encodded = 1  # Yes

                        if vehicle_ownership == "ไม่มี":
                            vehicle_ownership_encodded = 0  # No

                        # Create list of all New Data
                        new_data = [annual_income, applicant_age, work_exp,
                                    years_in_current_employment, marital_status_encodded]

                        # Appending All Data
                        new_data.extend(house_ownership_encodded)
                        new_data.append(vehicle_ownership_encodded)

                        scaled_data = scaler.transform([new_data])

                        with st.spinner(text='Predict The Value..'):

                            predicted_value_prop = model.predict(
                                [scaled_data])[0][0]
                            predicted_value = (predicted_value_prop > 0.5) * 1

                            sleep(1.2)

                            st.subheader("Default Risk")
                            st.progress(
                                value=int(predicted_value_prop*100),)
                            st.subheader(f"{predicted_value_prop*100:0.2f}%")

                            if predicted_value == 0:
                                st.success("")
                                st.image("imgs/loan.png",
                                         caption="", width=150)
                                st.subheader("แนะนำให้เสนอสินเชื่อ")
                                st.subheader(":green[มีความเสี่ยงปกติ]")

                            else:
                                st.error("")
                                st.image("imgs/speedometer.png",
                                         caption="", width=105)
                                st.subheader(f"ไม่แนะนำให้เสนอสินเชื่อ")
                                st.subheader(":red[มีความเสี่ยงที่จะผิดชำระหนี้]")


run()
