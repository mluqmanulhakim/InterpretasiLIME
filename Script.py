from secrets import choice
import selectors
from turtle import color
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import matplotlib
sns.set_style("whitegrid")
import pandas as pd
import numpy as np
import string
import nltk
import re
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from imblearn.over_sampling import SMOTE
from lime import lime_text

st.title("Interpretasi Model Analisis Sentimen dengan LIME")
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["intro", "Data", 
                                                    "TF-IDF Vectorizer", 
                                                    "Inbalance Data",
                                                    "Klasifikasi (Modeling)", 
                                                    "Interpretasi tiap kalimat", 
                                                    "Interpretasi Keseluruhan",  
                                                    "Hadiah buat yang baca"])
tab1.subheader("Intro")
tab1.image("Black_Lime.png")
tab1.markdown("""Web ini dibuat ketika saya sedang gabut, 
Capek juga menunggu dapat pekerjaan Hehe:D. Okey, jadi 
disini saya akan membahas mengenai penelitian skripsi 
saya kemarin terkait Membuka (Interpretasi) salah satu 
misteriiii Kotak Hitam (Machine Learning) hohoho.. 
Becandaa, okey lanjut serius.""")

tab1.markdown("""Interpretasi Model Analisis Sentimen pada 
penelitian ini menggunakan ***Local Interpretable Model-agnostic 
Explanation (LIME)***. Proses Anlisis Sentimen akan menggunakan 
metode klasifikasi Random Forest. Data yang digunakan diambil dari 
Twitter pada tanggal 7 September dengan kata Kunci 'Pajak' dan 
'Tax Amnesty'. Gimana hasil Interpretasi Analisis Sentimen? Gas aja 
langsung deh yaa ke Tab selanjutnya yaitu ***Data***""")

# Import Data
tab2.subheader("Persiapan Data")
tab2.markdown("""Data telah melewati ***Pre-Processing*** 
data yang meliputi: """)
tab2.markdown("1. Cleansing: Proses membersihkan data teks dari data yang tidak konsisten atau tidak diperlukan")
tab2.markdown("2. Case folding: Proses merubah bentuk teks kembali ke bentuk dasarnya, salah satunya membuat teks atau kalimat menjadi lower case atau huruf kecil semua.")
tab2.markdown("3. Stemming: Merupakan proses pengubahan kata ke kata asalnya dengan menghapus imbuhan awal atau akhir pada kata tersebut.")
tab2.markdown("4. Stopword Removal: Tahap ini dilakukan untuk membuang kata-kata yang cukup umum dan sering muncul namun tidak memiliki pengaruh yang signifikan terhadap makna suatu teks atau kalimat")
tab2.subheader("Data")
tab2.markdown("Terdapat 1000 Data yang di ambil Dari Twitter dengan kata kunci Pajak dan Tax Amnesty. Data telah dilabeli secara manual. Ada dua kelas di data yang saya pakai, yaitu Positif dan Negatif yang masing-masing akan diubah menjadi nilai Boolean yaitu 0 untuk Positif dan 1 untuk Negatif. Berikutnya, data akan melalui proses TF-IDF Vectorizer")
df = pd.read_csv("Datapakai.csv", sep = ";")
df.drop(df.columns[[0]], axis = 1, inplace=True)
tab2.dataframe(df)

def convert(polarity):
    if polarity == 'positive':
        return 0
    else:
        return 1

df['Class'] = df["Class"].apply(convert)

# TF-IDF Vectorizer
tab3.subheader("Kata ke Bobot, dengan TF-IDF Vectorizer")
tab3.markdown("""Kata TF-IDF Vectorizer itu gabugan dari dua metode, 
            yaitu metode ***Term Frequency*** dan 
            ***Inverse Document Frequency (TF-IDF)*** serta metode 
            ***CountVectorizer***. ***Term Frequency*** dan 
            ***Inverse Document Frequency (TF-IDF)***
            itu merupakan suatu proses ekstraksi fitur 
            yang punya tujuan mengubah kata dari suatu 
            text (Dokumen) menjadi angka yang bisa 
            dimengerti sama komputer ges. Lalu, untuk 
            ***CountVectorizer*** itu berfungsi untuk 
            menghitung banyaknya kemunculan kata pada suatu kalimat.""")
tab3.markdown("""Nah kalo diperhatiin nih dari definisi diatas, 
            kita bisa tau, kenapa kok dua metode itu digabungin. 
            Diliat dari definisi ***CountVectorizer***, metode 
            itu hanya bisa melihat banyaknya kemunculan kata di 
            satu kalimat, tapi tidak memberikan bobot ke kelimat itu,
            atau dengan kata lain, kita gatau nih seberapa penting
            kata tersebut pada data yang kita miliki jika hanya dari
            metode ***CountVectorizer***. Naah, Disini fungsinya 
            ***TF-IDF*** Gess. ***TF-IDF*** memungkinkan untuk kita 
            bisa mengetahui bobot suatu kata pada data text yang kita milikii.
            Nah untuk data saya ini, Berikut hasil dari ***TF-IDF Vectorizer*** nyaa.""")
vectorizer = TfidfVectorizer()
vectorizer.fit(df['Tweet'])
vector = vectorizer.transform(df['Tweet'])
df_vector = pd.DataFrame(vector.todense(), 
                        index = [f'D{i+1}' for i in range (len(df['Tweet']))],
                        columns = [vectorizer.get_feature_names_out()])
tab3.dataframe(df_vector)

# SMOTE
tab4.subheader("Sebaran Data Training")
tab4.markdown("""Data yang sudah siap, selanjutnya di split menjadi data 
                Training dan data Testing. Pada penelitian ini, saya 
                membagi data Training 80%  dan data Testing 20%. Setelah di split,
                 kita lihat sebaran jumlah masing masing kelas di data Training, sebagai berikut: """)
X = df['Tweet']
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(vector,y, random_state = 42,
                                                    test_size = 0.2)

kata_kelas = y_train.value_counts()
freq_kelas = pd.DataFrame(kata_kelas)
freq_kelas.reset_index(level=0, inplace=True)
freq_kelas.columns = ['Kelas', 'Jumlah']
fig1 = plt.figure(figsize = (6,3))
ax = sns.barplot(data = freq_kelas, x = 'Kelas', y = 'Jumlah', palette = 'deep')
patches = [matplotlib.patches.Patch(color=sns.color_palette()[i], label=t) for i,t in enumerate(t.get_text() for t in ax.get_xticklabels())]
plt.legend(handles=patches, loc="upper left") 
plt.xticks(rotation = 'vertical')
plt.gca().set_title('Perbandingan Jumlah Kelas data Training')
tab4.pyplot(fig1)

tab4.markdown("""Waw, dari data training yang kita punya
                keliatan kelas negatif (1) lebih banyak 
                ya, kenapa ya banyak banget negatif nya 
                di isu pajak ini? ***yo gak tau kok tanya saya***.
                Nah intinya, Karena pada data training ini
                jumlah kelasnya cukup berbeda signifikasn,
                maka kita perlu menyeimbangkan jumlah datanya. Emang Kenapa?
                Karena, kalau dibiarkan data nya tidak seimbang sebanyak ini, 
                hasil klasifikasi akan condong pada data mayoritas. Nah, Berikut
                hasil penyeimbangan data yang dilakukan: """)

from collections import Counter
sm = SMOTE(random_state = 42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
labels = ['Positif', 'Negatif']
size1 = [107,693]
size2 = [693,693]

def subPlot():
    fig2 = plt.figure(figsize=(10,5), facecolor= "black")

    # Plot sebelum SMOTE
    plt.subplot(1, 2, 1)
    plt.pie(size1, labels=labels, autopct='%1.1f%%', shadow = True, explode = 
            (0.1,0), textprops = dict(color = "white"))
    plt.title("Sebelum Diseimbangkan", color = "white")
    # Plot setelah SMOTE
    plt.subplot(1, 2, 2)
    plt.pie(size2, labels=labels, autopct='%1.1f%%', shadow = True, explode = 
            (0.1,0), textprops = dict(color = "white"))
    plt.title("Setelah Diseimbangkan", color = "white")

    tab4.pyplot(fig2)
subPlot()
tab4.markdown("""Pada proses penyeimbangan data ini, saya
                menggunakan metode ***Synthetic Minority 
                Over-sampling Technique (SMOTE)***. 
                ***SMOTE*** itu salah satu metode untuk menyelesaikan
                masalah data yang gak seimbang. ***SMOTE*** bekerja 
                dengan me-replikasi data minoritas ***(Oversampling)***.
                Metode ***SMOTE*** menggunakan ***K-NN*** untuk menentukan
                sample terdekat lalu dikalikan dengan angka acak antara 0 dan 1.
                Hasil dari pengkalian tersebut ditambahkan ke data sample dan menjadi
                data baru. Nah pada penelitian saya ini, saya mendapatkan komposisi 
                terbaik untuk penyeimbangan jumlah datanya yaitu disamaratakan. Okey 
                setelah data training seimbang, lanjut ke pembentukan model yaaa""")

# TAB 5. Klasifikasi (Modeling)
tab5.subheader("Klasifikasi Analisis Sentimen dengan Random Forest")
tab5.markdown("""Okey, jadi di pembahasan kali ini akan sedikit berbeda. dalam 
                pembentukan model, pembaca silahkan pilih parameter Hyperparameter yang 
                disediakan, biar ada kaya aplikasinya aja dikit hehe.""")
col1, col2 = tab5.columns(2)                
n_estimator = col1.radio("Pilih n_estimator", ["Pilih", 50, 200, 500, 1000])
criterion = col2.radio("Pilih Criterion", ["Pilih", "gini", "entropy"])
random_state = 0
if n_estimator == "Pilih" and criterion == "Pilih":
    tab5.error("Pilih dulu Hyperparameter nya yaa")
elif n_estimator == "Pilih":
    tab5.error("n_estimatornya pilih duluu")
elif criterion == "Pilih":
    tab5.error("Criterionnya pilih duluu")
else:
    with st.spinner("Tunggu ngitung duluu"):
        model_choise = RandomForestClassifier(n_estimators = n_estimator, criterion = criterion,
                                                random_state = random_state)
        y_preds = model_choise.fit(X_train_sm, y_train_sm).predict(X_test)
        accuracy = accuracy_score(y_test,y_preds)
        tab5.markdown("Akurasi model :")
        tab5.success(accuracy)

if n_estimator == 500 and criterion == "entropy":
    tab5.success("""Naahh ini Hyperparameter terbaik gess""")
    with st.spinner("Bentarr ada evaluasi modelnyaa..."):
        model = RandomForestClassifier(n_estimators = n_estimator , criterion = criterion ,
                                        random_state = random_state)
        y_preds_testing = model.fit(X_train_sm, y_train_sm).predict(X_test)
        presisi = round(precision_score(y_test, y_preds_testing), 2)
        recall = round(recall_score(y_test, y_preds_testing), 2)
        f1score = round(f1_score(y_test, y_preds_testing), 2)
        akurasi = accuracy_score(y_test, y_preds_testing)
        col3, col4, col5, col6 = tab5.columns(4)
        col3.metric("Presisi", presisi, "Relatif")
        col4.metric("Recall", recall, "Relatif")
        col5.metric("F1-Score", f1score, "Baik")
        col6.metric("Akurasi", akurasi, "Baik")
        tab5.markdown("""Nah kita udah dapet ya model terbaiknya. 
                        didapat nilai ***Presisi 94%***, 
                        ***Recall 94%*** dan ***Akurasi 94%***. 
                        Kita bisa lihat bahwa nilai presisi dan 
                        akurasi cukup bagus. Model ini dapat 
                        dikatan bekerja dengan baik.""")
elif n_estimator != "Pilih" and criterion != "Pilih":
    tab5.error("Bukan Hyperparameter terbaik")

# INTERPRETASI LIME tiap kalimat
model = RandomForestClassifier(n_estimators = 500 , criterion = "entropy" ,
                                        random_state = random_state)
model.fit(X_train_sm, y_train_sm).predict(X_test)
tab6.subheader("Interpretasi Model Analisis Sentimen")
c = make_pipeline(vectorizer, model)
class_names = ['positive', 'negative']
explainer = LimeTextExplainer(class_names = class_names)
# App per kalimat
idx = tab6.number_input("Masukkan Index text yang mau di Interpretasi (0-999): ", 0, 999)
exp = explainer.explain_instance(df['Tweet'][idx], c.predict_proba, num_features = 30)
prob_neg = round(c.predict_proba([df['Tweet'][idx]])[0,1], 2)
prob_pos = round(c.predict_proba([df['Tweet'][idx]])[0,0], 2)
col7, col8, col9 = tab6.columns(3)
col7.metric("Index", idx)
col8.metric("Peluang Positif", prob_pos)
col9.metric("Peluang Negatif", prob_neg)
col10, col11 = tab6.columns(2)
list_exp = exp.as_list()
frameexp = pd.DataFrame(list_exp, columns= ['Kata', 
                                            'Bobot dan Pengaruh'])
fig3 = exp.as_pyplot_figure()
kalimat = df['Tweet'][idx]
col10.text_area("Tweet", kalimat)
col11.pyplot(fig3)
tab6.markdown("""Cara membaca interpretasi diatas 
                adalah dengan melihat warna pada grafik yang
                ditunjukan. Grafik tersebut menunjukkan pengaruh
                kata pada sentimen negatif. Jika nilai
                suatu katanya minus, maka kata tersebut berpengaruh 
                negatif terhadap sentimen negatif, yang berarti 
                memiliki pengaruh positif. Bagan merah menunjukan 
                kata yang memiliki pengaruh menjadikan kalimat 
                menjadi bersentimen positif. Hijau memiliki pengaruh 
                menjadikan kalimat menjadi bersentimen negatif""")

# INTERPRETASI KESELURUHAN ISU PAJAK
tab7.subheader("Interpretasi Keselurusan Sentimen pada Isu Pajak")
positif_df = pd.read_csv("Positif.csv", sep = ";")
negatif_df = pd.read_csv("Negatif.csv", sep = ";")
positif_sum = pd.read_csv("Positif_Jumlah.csv", sep=";")
negatif_sum = pd.read_csv("Negatif_Jumlah.csv", sep=";")
positif_sum.columns = ['Kata (Positif)', 'Bobot']
negatif_sum.columns = ['Kata (Negatif)', 'Bobot']
freq_pos = pd.DataFrame(positif_df['Kata'].value_counts())
freq_neg = pd.DataFrame(negatif_df['Kata'].value_counts())
freq_pos.reset_index(level=0, inplace=True)
freq_neg.reset_index(level=0, inplace=True)
freq_pos.columns = ['Kata', 'Jumlah']
freq_neg.columns = ['Kata', 'Jumlah']
tab7.subheader("Kata paling sering Muncul pada setiap Sentimen")
col12, col13 = tab7.columns(2)
# Kata Positif
fig4 = plt.figure(figsize = (8,5))
ax = sns.barplot(data = freq_pos.loc[0:20], x = 'Kata', y = 'Jumlah', palette = 'deep')
ax.set(ylabel = 'counts')
plt.xticks(rotation = 'vertical')
plt.gca().set_title('Urutan total kata positif')
col12.pyplot(fig4)

# Kata Negatif
fig5 = plt.figure(figsize = (8,5))
ax = sns.barplot(data = freq_neg.loc[0:20], x = 'Kata', y = 'Jumlah', palette = 'deep')
ax.set(ylabel = 'counts')
plt.xticks(rotation = 'vertical')
plt.gca().set_title('Urutan total kata negatif')
col13.pyplot(fig5)

tab7.subheader("Kata paling berpengaruh pada setiap Sentimen")
col14, col15 = tab7.columns(2)
col14.dataframe(positif_sum.head())
col15.dataframe(negatif_sum.head())
tab7.markdown("""Kata-kata teratas yang memiliki pengaruh pada sentimen negatif yakni
“pajakperasrakyat”, “rakyat”, “kemplang”, “kaya” dan “pajak”, merujuk pada 
beberapa bahasan, seperti: banyaknya kebijakan baru mengenai pajak, 
masyarakat merasa orang kaya diampuni pajak dengan tax amnesty sedangkan 
rakyat kecil tidak, adanya perusahaan yang tidak patuh bayar pajak, hingga 
adanya kebijakan untuk menaikan pajak yang dirasa memeras rakyat. 
Pembahasan tersebut mengartikan bahwa sentimen negatif masyarakat 
terhadap isu pajak mengarah pada pelaksanaan yang dirasa adanya perbedaan 
perlakuan kepada masyarakat yang mengindikasikan bahwa permasalahannya 
adalah pada keadilan.""")
tab7.markdown("""Kata-kata teratas yang memiliki pengaruh pada sentimen positif yakni 
“taxamnesty”, “wajib”, “pajak”, ”uuhpp” dan “bayar”, merujuk pada bahasan 
seperti: target pengumpulan pajak yang lebih besar dengan tax amnesty, 
program tax amnesty sebagai reformasi sistem, penerapan kebijakan UU HPP 
dilakukan dengan asas keadilan bagi masyarakat kecil dengan adanya hastag 
“#ruuhppuntukmasyarakatkecil”, hingga pernyataan bahwa kebijakan pajak ini 
merupakan pajak yang lebih berkeadilan. Pembahasan tersebut mengartikan 
bahwa sentimen positif masyarakat terhadap isu pajak banyak di sasarkan 
langsung kepada kebijakan yang baru saja dibuat. Hal ini mengindikasikan 
bahwa kebijakan-kebijakan baru yang dibuat pemerintah mendapat banyak 
sentimen positif dari masyarakat khususnya di media sosial twitter.""")


# SELESAI
tab8.balloons()
tab8.subheader("SELESAI, MAKASI YANG SUDAH MEMBACA")
tab8.success("Yeaay hadiahnya balon tadi yang lewat. sama salju dikit biar dinginn")
tab8.snow()
nama = tab8.text_input("Masukin nama panggilan kalian")
nama = nama.lower()
if nama == "maula":
    tab8.success("Selamaat kamu dapat gopay 50rb")
elif nama == "lifty":
    tab8.success("Selamat kamu dapat gopay 50rb")
elif nama == "arsy":
    tab8.error("APASIH LUU UDAH KAYA JUGAAA")
elif nama == "":
    tab8.error("Masukin nama panggilan cepet, siapa tau beruntung")
else:
    tab8.error("Maap, gak dulu")