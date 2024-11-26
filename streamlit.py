import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib



# Performance and memory optimization
@st.cache_data
def load_data(filepath):
    """Cache data loading to improve performance"""
    return pd.read_csv(filepath, index_col=0)


@st.cache_resource
def load_model(filepath):
    """Cache model loading to prevent repeated loading"""
    return joblib.load(filepath)


def prepare_input_data(inputs):
    """Centralize input data preparation logic"""
    # Label Encoding
    cinsiyet_encoded = 1 if inputs['cinsiyet'] == "Kadın" else 0
    influencer_istegi_encoded = 0 if inputs['influencer_istegi'] == "Evet" else 1

    # Pre-defined column list to avoid repetition
    ALL_COLUMNS = [
        'Cinsiyet', 'Yas', 'Günlük Sosyal Medya Kullanımı',
        'İnfluencer olma istegi',
        'Son bir haftada kaç gün kendinizi depresif veya moraliniz bozuk hissettiniz?'
    ]

    # One-Hot Encoding columns
    ONE_HOT_COLUMNS = {
        'Kullanılan Sosyal Medya': ['Diğer', 'Linkedin', 'Tiktok', 'Twitter', 'İnstagram'],
        'Sosyal Medyada İlgilenilen İçerik': ['Diğer', 'Eğitim', 'Eğlence', 'Haber', 'Kişisel Gelişim'],
        'İçerik paylaşma sıklığı': ['Hiç', 'Nadiren', 'Sık sık'],
        'Sosyal medya kullanırken hissedilen duygu': ['Diğer', 'Kaygı', 'Mutlu', 'Yalnızlık', 'Öfke']
    }

    # Initialize data dictionary
    data = {col: 0 for col in ALL_COLUMNS}
    data.update({f"{category}_{val}": 0 for category, vals in ONE_HOT_COLUMNS.items() for val in vals})

    # Update with actual input values
    data.update({
        'Cinsiyet': cinsiyet_encoded,
        'Yas': inputs['yas'],
        'Günlük Sosyal Medya Kullanımı': inputs['gunluk_kullanim'],
        'İnfluencer olma istegi': influencer_istegi_encoded,
        'Son bir haftada kaç gün kendinizi depresif veya moraliniz bozuk hissettiniz?': inputs['depresif_gunler']
    })

    # Update One-Hot Encoded columns
    for category, col in [
        ('Kullanılan Sosyal Medya', inputs['kullanilan_sosyal_medya']),
        ('Sosyal Medyada İlgilenilen İçerik', inputs['ilgilenilen_icerik']),
        ('İçerik paylaşma sıklığı', inputs['icerik_paylasma_sikligi']),
        ('Sosyal medya kullanırken hissedilen duygu', inputs['hissedilen_duygu'])
    ]:
        data[f"{category}_{col}"] = 1

    return pd.DataFrame([data])



def main():
    st.set_page_config(page_title="Sosyal Medya ve Yalnızlık", layout="wide", initial_sidebar_state="expanded", page_icon="📊")

    # Load data once
    try:
        df = load_data('data.csv')
    except FileNotFoundError:
        st.error("Veri dosyası bulunamadı!")
        return

    # Columns optimization
    columns_to_include = [
        col for col in df.columns
        if df[col].nunique() <= 500 or pd.api.types.is_numeric_dtype(df[col])
    ]

    # Sidebar setup
    st.sidebar.title("Sosyal Medya ve Yalnızlık 📊")
    sidebar_option = st.sidebar.selectbox(
        "Bir seçenek seçin:",
        ["Home", "Data", "Prediction"]
    )

    # Main content
    if sidebar_option == "Home":
        st.subheader("Sosyal Medya Kullanımı ve Yalnızlık Tahmini 📊")
        col1, col2 = st.columns([2, 1])  # [2, 1] sütun genişlik oranlarını belirler

        # Sol sütunda yazıyı göster
        with col1:
            st.write("""
            Bu uygulama, sosyal medya kullanım alışkanlıklarının bireylerin yalnızlık hissi üzerindeki etkisini incelemek için geliştirilmiştir.\n
            Projede, 18-26 yaş aralığındaki 230 kişinin sosyal medya kullanımına dair veriler analiz edilmiş ve yalnızlık hissi tahmini yapılmıştır.

            Veri seti, sosyal medya kullanım süresi, platform tercihleri, etkileşim sıklığı, cinsiyet gibi değişkenlerden oluşmaktadır. Tahmin modelleri, bu değişkenler arasındaki ilişkileri değerlendirerek bireylerin yalnızlık düzeyine dair öngörülerde bulunmayı amaçlar.

            Amaç:

            📌 Sosyal medya ve yalnızlık arasındaki potansiyel ilişkileri anlamak\n
            📌 Daha bilinçli sosyal medya kullanımını teşvik etmek\n
            📌 Veri bilimi ile farkındalık yaratmak

            Uygulamada veriye ilişkin değişkenleri görselleştirebilir, tahmin modellerinin çıktılarıyla veriyi keşfedebilirsiniz. Keyifli bir keşif yolculuğu dilerim! 🌟
            """)

        # Sağ sütunda görseli göster
        with col2:
            st.image("background.png", use_container_width=True)


    elif sidebar_option == "Data":
        data_subsection = st.sidebar.radio(
            "Data:",
            ["Data Information", "Data Visualization"]
        )

        if data_subsection == "Data Information":
            with st.expander("Dataset Overview"):
                st.write(df)
                st.write("First 5 Observations", df.head())
                st.write("Last 5 Observations", df.tail())
                st.write("Descriptive Statistics", df.describe().T)

        elif data_subsection == "Data Visualization":
            st.subheader("Data Visualization")

            # Visualization optimizations
            if columns_to_include:
                variable = st.selectbox("Select a variable to visualize:", columns_to_include)

                col1, col2 = st.columns(2)

                with col1:
                    if pd.api.types.is_numeric_dtype(df[variable]):
                        fig1 = px.box(df, y=variable, title=f"{variable} Boxplot")
                    else:
                        counts = df[variable].value_counts()
                        fig1 = px.bar(
                            counts,
                            x=counts.index,
                            y=counts.values,
                            labels={"x": variable, "y": "Count"},
                            title=f"{variable} Barplot"
                        )
                    st.plotly_chart(fig1, use_container_width=True)

                with col2:
                    if pd.api.types.is_numeric_dtype(df[variable]):
                        fig2 = px.histogram(df, x=variable, title=f"{variable} Histogram")
                    else:
                        fig2 = px.pie(
                            df,
                            names=variable,
                            title=f"{variable} Pie Chart",
                            hole=0.3,
                            template="plotly_dark"
                        )
                    st.plotly_chart(fig2, use_container_width=True)

            # Grouped Visualization
            group_var = st.selectbox(
                "Select a grouping variable:",
                [col for col in df.columns if df[col].dtype in ['object', 'category', 'bool']]
            )
            numeric_var = st.selectbox(
                "Select a numeric variable:",
                [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            )
            agg_func = st.radio("Select an aggregation function:", ["mean", "sum", "count"])

            if group_var and numeric_var:
                grouped_data = df.groupby(group_var)[numeric_var].agg(agg_func).reset_index()

                col1, col2 = st.columns(2)

                with col1:
                    bar_fig = px.bar(
                        grouped_data,
                        x=group_var,
                        y=numeric_var,
                        title=f"{agg_func.title()} of {numeric_var} by {group_var}"
                    )
                    st.plotly_chart(bar_fig, use_container_width=True)

                with col2:
                    pie_fig = px.pie(
                        grouped_data,
                        names=group_var,
                        values=numeric_var,
                        title=f"{agg_func.title()} of {numeric_var} by {group_var}",
                        hole=0.3
                    )
                    st.plotly_chart(pie_fig, use_container_width=True)

    elif sidebar_option == "Prediction":
        option = st.selectbox(
            "Hangi Model İle Tahmin Yapmak İstersiniz ?",
            ("Logistic Regression", "LGBM")
        )

        # Prediction input fields
        col1, col2 = st.columns(2)

        with col1:
            cinsiyet = st.selectbox("Cinsiyetiniz ?", ["Kadın", "Erkek"])
            yas = st.slider("Yaşınız ?", 10, 50, 18)
            gunluk_kullanim = st.number_input(
                "Günlük Kaç Saat Sosyal Medya Kullanıyorsunuz ?",
                min_value=0, max_value=24, value=2, step=1
            )

        with col2:
            influencer_istegi = st.selectbox("Influencer Olmak Ister Misiniz ?", ["Evet", "Hayır"])
            depresif_gunler = st.number_input(
                "Son 1 Haftada Kendinizi Kaç Gün Depresif Hissettiniz ?",
                min_value=0, max_value=7, value=2, step=1
            )

        kullanilan_sosyal_medya = st.selectbox(
            "En Çok Kullandığınız Sosyal Medya ?",
            ["Diğer", "Linkedin", "Tiktok", "Twitter", "İnstagram"]
        )
        ilgilenilen_icerik = st.selectbox(
            "Sosyal Medyada En Çok Hangi İçeriklerle İlgileniyorsunuz ?",
            ["Diğer", "Eğitim", "Eğlence", "Haber", "Kişisel Gelişim"]
        )
        icerik_paylasma_sikligi = st.selectbox(
            "Ne Sıklıkla İçerik Paylaşıyorsunuz ?",
            ["Hiç", "Nadiren", "Sık sık"]
        )
        hissedilen_duygu = st.selectbox(
            "Sosyal Medya Kullanırken En Çok Hangi Duyguyu Hissediyorsunuz ?",
            ["Diğer", "Kaygı", "Mutlu", "Yalnızlık", "Öfke"]
        )

        # Prediction logic
        if st.button("Tahmin Et"):
            inputs = {
                'cinsiyet': cinsiyet,
                'yas': yas,
                'gunluk_kullanim': gunluk_kullanim,
                'influencer_istegi': influencer_istegi,
                'depresif_gunler': depresif_gunler,
                'kullanilan_sosyal_medya': kullanilan_sosyal_medya,
                'ilgilenilen_icerik': ilgilenilen_icerik,
                'icerik_paylasma_sikligi': icerik_paylasma_sikligi,
                'hissedilen_duygu': hissedilen_duygu
            }

            input_data = prepare_input_data(inputs)

            if option == "Logistic Regression":
                log_model = load_model("logistic_model.pkl")
                log_model_accuracy = 77.39  # Replace with actual value
                log_f1 = 84.43  # Replace with actual value

                prediction = log_model.predict(input_data)
                tahmin = "Hayır" if prediction[0] == 1 else "Evet"

            else:  # LGBM
                lgbm_model = load_model("lgbm_model.pkl")
                lgbm_model_acc = 77.82  # Replace with actual value
                lgbm_f1 = 85.57  # Replace with actual value

                prediction = lgbm_model.predict(input_data)
                tahmin = "Hayır" if prediction[0] == 1 else "Evet"

            # Result display
            if tahmin == "Evet":
                st.info(
                    f"Kendinizi Yalnız Hissediyorsunuz 😔\n\n"
                    f"Model Başarı Oranı: %{(log_model_accuracy if option == 'Logistic Regression' else lgbm_model_acc)}\n\n"
                    f"Model F1-Score: %{(log_f1 if option == 'Logistic Regression' else lgbm_f1)}"
                )
            else:
                st.success(
                    f"Kendinizi Yalnız Hissetmiyorsunuz! 🥳\n\n"
                    f"Model Başarı Oranı: %{(log_model_accuracy if option == 'Logistic Regression' else lgbm_model_acc)}\n\n"
                    f"Model F1-Score: %{(log_f1 if option == 'Logistic Regression' else lgbm_f1)}"
                )
                st.balloons()


if __name__ == "__main__":
    main()

