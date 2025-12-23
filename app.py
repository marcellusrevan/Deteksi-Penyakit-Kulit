# =========================
# AUTO DETECTION + FUSION
# =========================
if image is not None:
    st.image(image, width=400)

    with st.spinner("Menganalisis citra..."):
        result = predict(image)

    # ===== DECISION FUSION =====
    if result["disease_label"] in benign_diseases:
        final_status = "Jinak"
        final_color = "success"
    else:
        final_status = "Ganas"
        final_color = "error"

    # Cek konsistensi binary vs multiclass
    binary_ok = (result["binary_label"] == final_status)

    st.markdown("## 🧾 Hasil Deteksi Akhir")

    # ===== FINAL STATUS =====
    if final_color == "error":
        st.error("🔴 **Status Kanker: GANAS**")
    else:
        st.success("🟢 **Status Kanker: JINAK**")

    # ===== DISEASE =====
    if result["disease_label"] in malignant_diseases:
        st.error(
            f"🔴 **Jenis Penyakit:** {result['disease_label']}  \n"
            f"Confidence: {result['disease_conf']:.2f}%"
        )
    else:
        st.success(
            f"🟢 **Jenis Penyakit:** {result['disease_label']}  \n"
            f"Confidence: {result['disease_conf']:.2f}%"
        )

    # ===== CEK KONSISTENSI =====
    if not binary_ok:
        st.warning(
            "⚠️ Prediksi penyakit tidak konsisten. "
            "Mohon konsultasikan ke dokter untuk pemeriksaan lebih lanjut."
        )
