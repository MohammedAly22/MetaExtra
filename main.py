import nltk
import pandas as pd
import streamlit as st
from preprocessing import Preprocessor

my_dict = {
    'محاسبة': 'Finance',
    'طب': 'Medical',
    'سياسة': 'Politics',
    'رياضة': 'Sports',
    'تكنولوجيا': 'Tech',
}

# generate tf_idf for documents
def generate_tf_idf():
    with st.spinner("جار تحميل جداول TF-IDF..."):
        corpus_fileids = corpus.fileids()
        cleaned_documents = []

        for file_id in corpus_fileids:
            doc = preprocessor.get_document_content(file_id)
            title, publisher, content = preprocessor.extract_metadata(doc)
            cleaned_content = preprocessor.clean_document(content)
            cleaned_documents.append(cleaned_content)

        dataframe = preprocessor.extract_keywords(cleaned_documents)
        dataframe2 = preprocessor.extract_keywords(cleaned_documents, ngram_range=(2, 2))

    return dataframe, dataframe2


# for custom CSS style
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("استخراج البيانات الوصفية")
with st.form(key='my_form', clear_on_submit=False):
    option = st.selectbox(
        'ادخل نوع الملف',
        ('محاسبة', 'طب', 'سياسة', 'رياضة', 'تكنولوجيا'))

    corpus_path = 'arabiya/Train/' + my_dict[option]
    corpus = nltk.corpus.PlaintextCorpusReader(corpus_path, r'.+\.txt')

    preprocessor = Preprocessor(corpus)

    uploaded_file = st.file_uploader("اختر ملف")
    submitted = st.form_submit_button(label='استخرج البيانات الوصفية')

    if submitted:
        if uploaded_file is not None:
            file_content = uploaded_file.getvalue().decode("utf-8")
            title, publisher, body = preprocessor.extract_metadata(
                file_content)

            col1, col2 = st.columns(2)
            col2.success("العنوان")
            col2.write(title)
            col1.success("المكان - الناشر")
            col1.write(publisher)

            col2.success("المحتوى")
            col2.write(body)
            col1.success("المحتوى النظيف")
            col1.write(preprocessor.clean_document(body))

            st.success("الكلمات المفتاحية")
            df, df2 = generate_tf_idf()
            # st.table(df)
            try:
                df = pd.DataFrame(df.loc[uploaded_file.name])
                df.columns = ['Tf-IDF scores']

                df2 = pd.DataFrame(df2.loc[uploaded_file.name])
                df2.columns = ['Tf-IDF scores']

                st.number_input(min_value=5, max_value=50, label='ادخل عدد الكلمات المطلوب اظهارها',
                                key='number_of_words')

                df = df.nlargest(st.session_state.number_of_words, 'Tf-IDF scores')
                df2 = df2.nlargest(st.session_state.number_of_words, 'Tf-IDF scores')

                st.table(df)
                st.success("رسم بياني للكلمات المفتاحية الاحادية")
                st.bar_chart(df)

                st.write("الكلمات المفتاحية الزوجية")
                st.table(df2)
                st.success("رسم بياني للكلمات المفتاحية الزوجية")
                st.bar_chart(df2)
            except KeyError:
                st.error(f"من فضلك ادخل نوع الملف الصحيح بدلا من '{option}'!")
        else:
            st.error("من فضلك ادخل الملف!")
