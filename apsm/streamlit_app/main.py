import asyncio
import streamlit as st

from components import create_template


async def main():
    """
    Запускает основной цикл Streamlit-приложения, отображает вкладки для валют и акций.
    """
    is_uploaded = True

    st.set_page_config(page_title='APSM', page_icon='📈')

    st.title('Аналитическая платформа биржевых событий')

    tab_currency, tab_stocks = st.tabs(tabs=['Котировки валют', 'Акции'])

    with tab_currency:
        await create_template(
            is_uploaded=is_uploaded,
            template_type='Котировки валют'
        )

    with tab_stocks:
        await create_template(
            is_uploaded=is_uploaded,
            template_type='Акции'
        )


asyncio.run(main())
