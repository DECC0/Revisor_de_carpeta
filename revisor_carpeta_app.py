import io
import json
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

# =============================
# Revisor de carpeta – UI (Streamlit)
# =============================
# Ejecuta con:
#   pip install streamlit pandas openpyxl
#   streamlit run revisor_carpeta_app.py
# -----------------------------

st.set_page_config(page_title="Revisión de Carpeta – Física", page_icon="📁", layout="wide")
st.title("📁 Revisión de Carpeta – Física")
st.caption("Carga los archivos, revisa requisitos y descarga el reporte.")

with st.sidebar:
    st.header("⚙️ Configuración")
    st.markdown("Sube el Excel del estudiante y la tabla de secciones.")
    f_est = st.file_uploader("Excel del estudiante (df)", type=["xlsx", "xls", "csv"])
    f_tabla = st.file_uploader("Tabla agrupada (Sección | Materia/examen)", type=["xlsx", "xls", "csv"])    

    st.markdown("---")
    st.subheader("Mínimos por sección")
    default_cursos = {
        'Curso introductorio': 11,
        'Curso disciplinar': 44,
        'Electivos de profundización': 19,
        'Cursos de matemáticas': 15,
        'Cursos de informática externos': 3,
        'Cursos de química': 3,
        'Electivo en ciencias básico': 3,
        'Electivo en ciencias avanzado': 3,
        'Cursos de educación general': 27
    }
    cursos_text = st.text_area(
        "Diccionario JSON de mínimos",
        value=json.dumps(default_cursos, ensure_ascii=False, indent=2),
        height=250
    )
    try:
        cursos: Dict[str, float] = json.loads(cursos_text)
    except Exception as e:
        st.error(f"JSON inválido en mínimos por sección: {e}")
        st.stop()

    st.markdown("---")
    st.subheader("Parámetros")
    exigir_cbus_total_6 = st.checkbox("Exigir total CBUS (CBCA/CBPC/CBCO) ≥ 6", value=True)
    exigir_ecur_epsi = st.checkbox("Exigir 2× ECUR y 1× EPSI", value=True)
    meta_cle = st.number_input("Créditos meta CLE", min_value=0.0, value=6.0, step=1.0)

# ============================
# Helpers
# ============================

def parse_nota(x):
    # Normaliza nota: numérica o "A"; NaN si no es interpretable
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s.upper() == "A":
        return "A"
    try:
        return float(s.replace(",", "."))
    except Exception:
        return np.nan


def is_pass(n):
    # Aprobado si nota >=3 o es "A"
    if isinstance(n, (int, float, np.floating)):
        return n >= 3
    if isinstance(n, str) and n.upper() == "A":
        return True
    return False


def nota_para_orden(n):
    # Valor numérico para ordenar; "A" se toma como 5.0
    if isinstance(n, (int, float, np.floating)):
        return float(n)
    if isinstance(n, str) and n.upper() == "A":
        return 5.0
    return -1.0


def ensure_list_column(df_tabla: pd.DataFrame) -> pd.DataFrame:
    """Asegura que la columna 'Materia/examen' sea una lista por fila."""
    out = df_tabla.copy()
    if 'Materia/examen' not in out.columns:
        raise ValueError("La tabla agrupada debe tener la columna 'Materia/examen'.")
    if 'Sección' not in out.columns:
        raise ValueError("La tabla agrupada debe tener la columna 'Sección'.")
    mask_list = out['Materia/examen'].map(lambda x: isinstance(x, list))
    if mask_list.sum() != len(out):
        out['Materia/examen'] = (
            out['Materia/examen']
            .apply(lambda x: [s.strip() for s in str(x).split(',') if str(s).strip() != ""]) 
        )
    return out


def evaluar(df: pd.DataFrame, tabla_agrupada: pd.DataFrame, cursos: Dict[str, float],
            meta_cle: float = 6.0,
            exigir_cbus_total_6: bool = True,
            exigir_ecur_epsi: bool = True):
    # 0) Mapa curso->sección
    tabla_agrupada = ensure_list_column(tabla_agrupada)
    tabla_map = (
        tabla_agrupada
        .explode("Materia/examen")
        .rename(columns={"Materia/examen": "Curso"})
    )
    tabla_map["Curso"] = tabla_map["Curso"].astype(str).str.strip()
    tabla_map["Sección"] = tabla_map["Sección"].astype(str).str.strip()
    curso_a_seccion = dict(zip(tabla_map["Curso"], tabla_map["Sección"]))

    # 1) Normalizar df del estudiante
    dfw = df.copy()
    for col in ["Materia/examen", "Créditos", "Nota final"]:
        if col not in dfw.columns:
            raise ValueError(f"El Excel del estudiante debe tener la columna '{col}'.")

    dfw['Materia/examen'] = dfw['Materia/examen'].astype(str).str.strip()
    dfw['nota_parsed'] = dfw['Nota final'].apply(parse_nota)
    dfw['Aprobado'] = dfw['nota_parsed'].apply(is_pass)
    dfw['nota_sort'] = dfw['nota_parsed'].apply(nota_para_orden)

    # 2) Mejor intento por curso (mejor nota y aprobado)
    df_best = (
        dfw.sort_values(['Aprobado', 'nota_sort'], ascending=[False, False])
           .drop_duplicates(subset=['Materia/examen'], keep='first')
           .copy()
    )
    df_best['Sección'] = df_best['Materia/examen'].map(curso_a_seccion)

    # 3) Créditos por sección
    aprb = df_best['Aprobado'].fillna(False)
    cred = df_best['Créditos'].fillna(0)
    creditos_por_seccion = (
        df_best.loc[aprb & df_best['Sección'].notna()]
               .groupby('Sección', sort=False)['Créditos']
               .sum()
               .to_dict()
    )
    resultado = {sec: float(creditos_por_seccion.get(sec, 0.0)) for sec in cursos.keys()}
    faltantes = {sec: max(float(cursos[sec]) - resultado[sec], 0.0) for sec in cursos.keys()}

    # 4) reglas específicas
    cod = df_best['Materia/examen'].astype(str).str.strip()
    attr = df_best.get('Atributos sección')
    if attr is not None:
        attr = attr.astype(str).str.strip()

    longname_col = 'Nombre largo curso o examen'
    if longname_col in df_best.columns:
        longn_up = df_best[longname_col].astype(str).str.strip().str.upper()
    else:
        longn_up = pd.Series([""] * len(df_best), index=df_best.index)

    # Educación general: obligatorios + ECUR/EPSI
    SECC_EDU_GEN = 'Cursos de educación general'
    req_names_escr = {"ESCRITURA UNIVERSITARIA I", "ESCRITURA UNIVERSITARIA II"}
    req_dere_code = 'DERE-1300'

    escr_ok = {nm: bool((aprb & (longn_up == nm)).any()) for nm in req_names_escr}
    dere_ok = bool((aprb & (cod == req_dere_code)).any())
    obli_ok = {**escr_ok, req_dere_code: dere_ok}

    if exigir_ecur_epsi:
        if attr is None:
            ecur_cnt = 0
            epsi_cnt = 0
            ecur_ok = False
            epsi_ok = False
        else:
            ecur_cnt = int((aprb & (attr == 'ECUR-CURSO TIPO E')).sum())
            epsi_cnt = int((aprb & (attr == 'EPSI-CURSO EPSILON')).sum())
            ecur_ok = ecur_cnt >= 2
            epsi_ok = epsi_cnt >= 1
    else:
        ecur_cnt = int((aprb & (attr == 'ECUR-CURSO TIPO E')).sum()) if attr is not None else 0
        epsi_cnt = int((aprb & (attr == 'EPSI-CURSO EPSILON')).sum()) if attr is not None else 0
        ecur_ok = True
        epsi_ok = True

    edug_obli_ok = all(obli_ok.values())
    educacion_general_ok = edug_obli_ok and ecur_ok and epsi_ok

    # CBUS: CBCC-1177, 1 de cada tipo y total mínimo
    is_cbus      = cod.str.match(r"^(CBCA|CBPC|CBCO|CBCC)-\d{3,4}$", na=False)
    is_cbcc_1177 = (cod == 'CBCC-1177')
    is_cbca      = cod.str.match(r"^CBCA-\d{3,4}$", na=False)
    is_cbpc      = cod.str.match(r"^CBPC-\d{3,4}$", na=False)
    is_cbco      = cod.str.match(r"^CBCO-\d{3,4}$", na=False)

    cbcc_ok = bool((aprb & is_cbcc_1177).any())
    cbca_cnt = int((aprb & is_cbca).sum())
    cbpc_cnt = int((aprb & is_cbpc).sum())
    cbco_cnt = int((aprb & is_cbco).sum())
    cbus_total_cnt = cbca_cnt + cbpc_cnt + cbco_cnt

    cbus_tipos_min_ok = (cbca_cnt >= 1) and (cbpc_cnt >= 1) and (cbco_cnt >= 1)
    cbus_total_ok     = (cbus_total_cnt >= 6) if exigir_cbus_total_6 else (cbus_total_cnt >= 3)
    cbus_ok           = cbcc_ok and cbus_tipos_min_ok and cbus_total_ok

    # CLE: créditos válidos por reglas específicas
    is_depo = cod.str.match(r"^DEPO-\d{3,4}$", na=False)
    depo_ok_cred = float(cred[aprb & is_depo & (cred == 1)].sum())
    unmapped_ok_cred = float(cred[aprb & df_best['Sección'].isna() & ~is_cbus & ~is_depo].sum())

    SECC_ADV_SCI = 'Electivo en ciencias avanzado'
    min_adv = float(cursos.get(SECC_ADV_SCI, 0))
    mask_adv_noncbus = aprb & (df_best['Sección'] == SECC_ADV_SCI) & (~is_cbus)
    cred_adv_noncbus = float(cred[mask_adv_noncbus].sum())
    excedente_adv = max(0.0, cred_adv_noncbus - min_adv)

    mask_edug = (df_best['Sección'] == SECC_EDU_GEN)
    mask_excluir_escr = longn_up.isin(req_names_escr)
    mask_excluir_attr = (attr.isin({'ECUR-CURSO TIPO E', 'EPSI-CURSO EPSILON'}) if attr is not None else False)
    mask_excluir_dere = (cod == req_dere_code)

    mask_edug_3cred_ok = (
        aprb
        & mask_edug
        & (cred == 3)
        & ~mask_excluir_escr
        & ~mask_excluir_attr
        & ~is_cbus
        & ~mask_excluir_dere
    )
    edug_3cred_para_cle = float(cred[mask_edug_3cred_ok].sum())

    cle_credits = float(depo_ok_cred + unmapped_ok_cred + excedente_adv + edug_3cred_para_cle)
    cle_ok = cle_credits >= float(meta_cle)

    # ECAP / RLEC / RDOM: notas especiales
    nota_raw_up = df_best['Nota final'].astype(str).str.strip().str.upper()
    is_ecap = cod.str.match(r"^ECAP", na=False)
    is_rlec = cod.str.match(r"^RLEC", na=False)
    is_rdom = cod.str.match(r"^RDOM", na=False)

    ecap_total = int(is_ecap.sum())
    ecap_ok = (ecap_total == 0) or bool((nota_raw_up[is_ecap] == 'FISI').all())
    rlec_total = int(is_rlec.sum())
    rdom_total = int(is_rdom.sum())
    rlec_ok = (rlec_total == 0) or bool((df_best.loc[is_rlec, 'nota_parsed'] == 1).all())
    rdom_ok = (rdom_total == 0) or bool((df_best.loc[is_rdom, 'nota_parsed'] == 1).all())

    # ======================
    # 5) Cursos usados / no usados para el conteo
    # ======================
    aprb_main = aprb.copy()
    used = pd.Series(False, index=df_best.index)
    used_reason = pd.Series("", index=df_best.index, dtype="object")

    # 5.0 Obligatorios duros: ESCR I/II, CBCC-1177, DERE-1300, 2×ECUR, 1×EPSI

    # ESCR I/II por nombre largo
    mask_escr_obl = aprb_main & longn_up.isin(
        {"ESCRITURA UNIVERSITARIA I", "ESCRITURA UNIVERSITARIA II"}
    )
    used.loc[mask_escr_obl] = True
    used_reason.loc[mask_escr_obl] = "Obligatorio: ESCRITURA UNIVERSITARIA"

    # CBCC-1177
    mask_cbcc = aprb_main & cod.eq("CBCC-1177")
    used.loc[mask_cbcc] = True
    used_reason.loc[mask_cbcc] = "Obligatorio: CBCC-1177"

    # DERE-1300
    mask_dere = aprb_main & cod.eq("DERE-1300")
    used.loc[mask_dere] = True
    used_reason.loc[mask_dere] = "Obligatorio: DERE-1300"

    # ECUR / EPSI mínimos
    if attr is not None:
        # ECUR: 2 cursos
        mask_ecur = aprb_main & (attr == "ECUR-CURSO TIPO E") & ~used
        ecur_idxs = df_best[mask_ecur].sort_values('Créditos', ascending=False).index
        ecur_count = 0
        for i in ecur_idxs:
            if ecur_count >= 2:
                break
            used.at[i] = True
            used_reason.at[i] = "Obligatorio: ECUR"
            ecur_count += 1

        # EPSI: 1 curso
        mask_epsi = aprb_main & (attr == "EPSI-CURSO EPSILON") & ~used
        epsi_idxs = df_best[mask_epsi].sort_values('Créditos', ascending=False).index
        epsi_count = 0
        for i in epsi_idxs:
            if epsi_count >= 1:
                break
            used.at[i] = True
            used_reason.at[i] = "Obligatorio: EPSI"
            epsi_count += 1

    # 5.1 Créditos por sección hasta el mínimo
    # restamos los créditos de los ya usados obligatorios
    remaining_by_sec = {}
    for sec_name in cursos.keys():
        base_min = float(cursos.get(sec_name, 0.0))
        cred_usados_prev = float(
            df_best.loc[aprb_main & used & (df_best['Sección'] == sec_name), 'Créditos'].sum()
        )
        remaining_by_sec[sec_name] = max(base_min - cred_usados_prev, 0.0)

    for sec_name in cursos.keys():
        rem = remaining_by_sec[sec_name]
        if rem <= 0:
            continue
        mask_sec = aprb_main & (df_best['Sección'] == sec_name) & ~used
        idxs = df_best[mask_sec].sort_values('Créditos', ascending=False).index
        for i in idxs:
            if rem <= 0:
                break
            ccurso = float(df_best.at[i, 'Créditos'] or 0)
            if ccurso <= 0:
                continue
            used.at[i] = True
            used_reason.at[i] = f"Sección: {sec_name}"
            rem -= ccurso
        remaining_by_sec[sec_name] = max(rem, 0.0)

    # 5.2 Asignación a CLE (i..iv) hasta meta_cle
    cle_remaining = float(meta_cle)

    # i) DEPO 1 crédito
    mask_depo_1 = aprb_main & ~used & is_depo & (df_best['Créditos'] == 1)
    for i in df_best[mask_depo_1].index:
        if cle_remaining <= 0:
            break
        used.at[i] = True
        used_reason.at[i] = "CLE: DEPO (1 cr)"
        cle_remaining -= float(df_best.at[i, 'Créditos'])

    # ii) Sin sección (no CBUS, no DEPO)
    mask_unmapped = aprb_main & ~used & df_best['Sección'].isna() & ~is_cbus & ~is_depo
    for i in df_best[mask_unmapped].index:
        if cle_remaining <= 0:
            break
        used.at[i] = True
        used_reason.at[i] = "CLE: sin sección"
        cle_remaining -= float(df_best.at[i, 'Créditos'])

    # iii) Excedente Adv. Science (no CBUS) por encima del mínimo
    mask_adv_candidates = aprb_main & ~used & (df_best['Sección'] == SECC_ADV_SCI) & (~is_cbus)
    adv_idxs = df_best[mask_adv_candidates].sort_values('Créditos', ascending=False).index
    used_adv_cred = float(
        df_best.loc[used & (df_best['Sección'] == SECC_ADV_SCI), 'Créditos'].sum()
    )
    for i in adv_idxs:
        if cle_remaining <= 0:
            break
        ccurso = float(df_best.at[i, 'Créditos'])
        if used_adv_cred < min_adv:
            used_adv_cred += ccurso  # todavía llenando el mínimo de la sección
            continue
        used.at[i] = True
        used_reason.at[i] = "CLE: excedente Adv. Science"
        cle_remaining -= ccurso

    # iv) EduGen extra 3cr (excluye obligatorios/atributos/CBUS/DERE)
    mask_edug_extra = (
        aprb_main
        & ~used
        & (df_best['Sección'] == SECC_EDU_GEN)
        & (df_best['Créditos'] == 3)
        & ~mask_excluir_escr
        & ~mask_excluir_attr
        & ~is_cbus
        & ~mask_excluir_dere
    )
    for i in df_best[mask_edug_extra].index:
        if cle_remaining <= 0:
            break
        used.at[i] = True
        used_reason.at[i] = "CLE: extra EduGen 3 cr"
        cle_remaining -= float(df_best.at[i, 'Créditos'])

    # DataFrames usados / no usados
    df_used = df_best.copy()
    df_used['Usado'] = used
    df_used['Motivo uso'] = used_reason
    #df_used['Sección original'] = df_best['Sección'].fillna('Sin sección')

    df_no_usado = df_used[aprb_main & (~df_used['Usado'])].copy()
    df_no_usado['Sección para conteo'] = df_no_usado['Sección']

    # paquetes de salida
    extras = {
        # educación general
        'EDUG_obligatorios': {**escr_ok, req_dere_code: dere_ok},
        'EDUG_ECUR_cnt(>=2?)': ecur_cnt,
        'EDUG_ECUR_ok': ecur_ok,
        'EDUG_EPSI_cnt(>=1?)': epsi_cnt,
        'EDUG_EPSI_ok': epsi_ok,
        'EDUG_ok_global': educacion_general_ok,
        # CBUS
        'CBUS_cbcc1177_ok': cbcc_ok,
        'CBUS_tipo_min_ok(1 de cada)': cbus_tipos_min_ok,
        'CBUS_total_cursos(CBCA/CBPC/CBCO)': cbus_total_cnt,
        'CBUS_total_ok': cbus_total_ok,
        'CBUS_ok_global': cbus_ok,
        # CLE
        'CLE_DEPO_cred': depo_ok_cred,
        'CLE_Unmapped_noCBUS_cred': unmapped_ok_cred,
        'CLE_Excedente_AdvSci_noCBUS_cred': excedente_adv,
        'CLE_EduGen_3cred_extra_cred': edug_3cred_para_cle,
        'CLE_total_creditos': cle_credits,
        'CLE_ok': cle_ok,
        # ECAP / RLEC / RDOM
        'ECAP_total': ecap_total,
        "ECAP_ok(nota=='FISI')": ecap_ok,
        'RLEC_total': rlec_total,
        'RLEC_ok(nota==1)': rlec_ok,
        'RDOM_total': rdom_total,
        'RDOM_ok(nota==1)': rdom_ok,
    }

    resultado_extendido = {**resultado, 'CLE': cle_credits}

    return df_used, resultado, faltantes, extras, resultado_extendido, df_no_usado


def seleccionar_columnas_detalle(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve sólo las columnas indicadas (1-4, 7, 10-11, 19 y las últimas 5 si existen)."""
    if df.empty:
        return df
    cols = list(df.columns)
    total = len(cols)
    posiciones = list(range(1, 5)) + [7, 10, 11, 19]  # 1-based
    indices = []
    for p in posiciones:
        idx = p - 1
        if 0 <= idx < total and idx not in indices:
            indices.append(idx)
    # últimas 5
    start_last = max(total - 5, 0)
    for idx in range(start_last, total):
        if idx not in indices:
            indices.append(idx)
    return df.iloc[:, indices]


# =============================
# Acción principal
# =============================

if f_est is not None and f_tabla is not None:
    # Lee archivos cargados desde la barra lateral
    try:
        if f_est.name.lower().endswith('.csv'):
            df = pd.read_csv(f_est)
        else:
            df = pd.read_excel(f_est)
    except Exception:
        df = pd.read_csv(f_est)

    try:
        if f_tabla.name.lower().endswith('.csv'):
            tabla_agrupada = pd.read_csv(f_tabla)
        else:
            tabla_agrupada = pd.read_excel(f_tabla)
    except Exception as e:
        st.error(f"Error leyendo la tabla agrupada: {e}")
        st.stop()

    with st.expander("👀 Vista previa – Estudiante (primeras 20 filas)", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)
    with st.expander("👀 Vista previa – Tabla agrupada", expanded=False):
        st.dataframe(tabla_agrupada, use_container_width=True)

    # Ejecuta evaluación con parámetros del sidebar
    try:
        df_best, resultado, faltantes, extras, resultado_ext, df_no_usado = evaluar(
            df, tabla_agrupada, cursos,
            meta_cle=meta_cle,
            exigir_cbus_total_6=exigir_cbus_total_6,
            exigir_ecur_epsi=exigir_ecur_epsi
        )
    except Exception as e:
        st.error(f"Error en la evaluación: {e}")
        st.stop()

    df_detalle = seleccionar_columnas_detalle(df_best)

    # Tarjetas resumen
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📚 Créditos CLE", f"{extras['CLE_total_creditos']:.1f}", "+OK" if extras['CLE_ok'] else "-Falta")
    with c2:
        st.metric("📖 Educación General", "OK" if extras['EDUG_ok_global'] else "FALTA")
    with c3:
        st.metric("🧩 CBUS", "OK" if extras['CBUS_ok_global'] else "FALTA")
    with c4:
        st.metric("✅ ECAP/RLEC/RDOM", "OK" if (extras["ECAP_ok(nota=='FISI')"] and extras['RLEC_ok(nota==1)'] and extras['RDOM_ok(nota==1)']) else "FALTA")

    st.markdown("---")

    # Resumen por sección (para varias pestañas)
    df_res = pd.DataFrame({
        'Sección': list(resultado.keys()),
        'Créditos aprobados': list(resultado.values()),
        'Mínimo requerido': [cursos[k] for k in resultado.keys()],
        'Faltantes': [faltantes[k] for k in resultado.keys()],
    })

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Créditos por sección",
        "Detalles evaluados",
        "Checks específicos",
        "Descargas",
        "Cursos no usados",
    ])

    with tab1:
        st.dataframe(df_res, use_container_width=True)

    with tab2:
        st.dataframe(df_detalle, use_container_width=True)

    with tab3:
        st.subheader("Educación General")
        st.json({
            "Obligatorios": extras['EDUG_obligatorios'],
            "ECUR_cnt": extras['EDUG_ECUR_cnt(>=2?)'],
            "EPSI_cnt": extras['EDUG_EPSI_cnt(>=1?)'],
            "OK_global": extras['EDUG_ok_global'],
        })
        st.subheader("CBUS")
        st.json({
            "CBCC-1177": extras['CBUS_cbcc1177_ok'],
            "tipos_min": extras['CBUS_tipo_min_ok(1 de cada)'],
            "total": extras['CBUS_total_cursos(CBCA/CBPC/CBCO)'],
            "OK_total": extras['CBUS_total_ok'],
            "OK_global": extras['CBUS_ok_global'],
        })
        st.subheader("CLE")
        st.json({
            "DEPO_cred": extras['CLE_DEPO_cred'],
            "Unmapped_cred": extras['CLE_Unmapped_noCBUS_cred'],
            "Excedente_AdvSci": extras['CLE_Excedente_AdvSci_noCBUS_cred'],
            "EduGen_3cred_extra": extras['CLE_EduGen_3cred_extra_cred'],
            "CLE_total": extras['CLE_total_creditos'],
            "CLE_ok": extras['CLE_ok'],
        })
        st.subheader("ECAP / RLEC / RDOM")
        st.json({
            "ECAP_total": extras['ECAP_total'],
            "ECAP_ok": extras["ECAP_ok(nota=='FISI')"],
            "RLEC_total": extras['RLEC_total'],
            "RLEC_ok": extras['RLEC_ok(nota==1)'],
            "RDOM_total": extras['RDOM_total'],
            "RDOM_ok": extras['RDOM_ok(nota==1)'],
        })

    with tab4:
        # Resumen por sección (CSV)
        csv = df_res.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "⬇️ Descargar resumen por sección (CSV)",
            data=csv,
            file_name="resumen_por_seccion.csv",
            mime="text/csv",
        )

        # Detalle por estudiante (Excel multi-hoja)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_detalle.to_excel(writer, index=False, sheet_name='DetalleEstudiante')
            df_res.to_excel(writer, index=False, sheet_name='ResumenSecciones')
            df_no_usado.to_excel(writer, index=False, sheet_name='CursosNoUsados')
        st.download_button(
            "⬇️ Descargar detalle (XLSX)",
            data=buffer.getvalue(),
            file_name="detalle_revision.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with tab5:
        st.subheader("Cursos aprobados NO usados para el conteo")
        st.dataframe(df_no_usado, use_container_width=True)

else:
    st.info("Sube los dos archivos en la barra lateral para comenzar.")
