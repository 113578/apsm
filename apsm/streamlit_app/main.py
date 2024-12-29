import streamlit as st

from functions import create_template


is_uploaded = True


st.set_page_config(page_title='APSM', page_icon='📈')

st.title('Аналитическая платформа биржевых событий')

tab_currency, tab_stocks = st.tabs(tabs=['Котировки валют', 'Акции'])

with tab_currency:
    create_template(
        is_uploaded=is_uploaded,
        template_type='Котировки валют'
    )

with tab_stocks:
    create_template(
        is_uploaded=is_uploaded,
        template_type='Акции'
    )
