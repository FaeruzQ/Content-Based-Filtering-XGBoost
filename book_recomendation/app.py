# =========================
# STREAMLIT UI
# =========================
st.title("Sistem Rekomendasi Buku")
st.write("Content-Based Filtering + XGBoost")

query = st.text_input("Masukkan judul atau kata kunci buku")

if st.button("Cari Rekomendasi"):
    if query.strip() == "":
        st.warning("Masukkan kata kunci terlebih dahulu.")
    else:
        with st.spinner("Memproses rekomendasi..."):

            # Hitung cosine similarity untuk query
            query_vec = tfidf.transform([query])
            sim_scores = cosine_similarity(
                query_vec, tfidf_matrix
            ).flatten()

            # =====================================================================
            # PERUBAHAN UTAMA: VALIDASI KUERI ASAL-ASALAN (OUT-OF-DISTRIBUTION)
            # =====================================================================
            if np.max(sim_scores) == 0:
                st.error(f"Maaf, kata kunci '{query}' tidak ditemukan atau tidak relevan dengan koleksi buku kami.")
                st.info("Coba masukkan kata kunci lain seperti genre, nama penulis, atau judul spesifik.")
            # =====================================================================
            else:
                # Jika ada minimal satu kecocokan kata (similarity > 0), lanjutkan proses XGBoost
                # Buat label berdasarkan query
                y = create_labels_topk(sim_scores, k=0.2)

                # Gabungkan fitur TF-IDF + cosine similarity
                X_features = hstack([
                    tfidf_matrix,
                    sim_scores.reshape(-1, 1)
                ])

                # Latih model untuk query ini
                model = XGBClassifier(
                    random_state=42,
                    n_estimators=100,
                    eval_metric='logloss'
                )
                model.fit(X_features, y)

                # Prediksi relevansi
                relevance_probabilities = model.predict_proba(
                    X_features
                )[:, 1]

                # Tampilkan hasil
                df_result = df.copy()
                df_result["similarity"] = sim_scores
                df_result["relevance_probability"] = relevance_probabilities

                top_books = df_result.sort_values(
                    by="relevance_probability",
                    ascending=False
                ).head(20)

                st.subheader(f"Rekomendasi untuk: '{query}'")
                st.dataframe(
                    top_books[[
                        "relevance_probability",
                        "title",
                        "author",
                        "genre"
                    ]]
                )
