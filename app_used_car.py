#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local Streamlit UI for the project's trained Rank-1 OLS .joblib model.

Run beside predict_used_car_ols.py and train_used_car_ols.py:
    python3 -m streamlit run app_used_car.py

Dropdown catalog is built IN MEMORY from the latest SQL Server STG_USED_CAR
snapshot every time this Streamlit app process starts. No catalog CSV is read
or written. Model predictions still use the original trained preprocessor;
the catalog is only for Brand → Model → Sub-model UI options. Existing
predict_used_car_ols.py and train_used_car_ols.py remain required imports.
Only load trusted .joblib files (joblib deserialization can execute code).
"""
from __future__ import annotations

import hashlib
import io
import math
import re
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import text
from pandas.errors import PerformanceWarning

import predict_used_car_ols as predictor
import train_used_car_ols as training

TRAIN_DIR = predictor.TRAIN_DIR
MODEL_NAME = re.compile(r"^used_car_models_(\d{8}_\d{6})\.joblib$")
DATE_DIR = re.compile(r"^\d{8}$")
MISSING_LABEL = "ไม่มีข้อมูล / ไม่ทราบ"
SENTINELS = {"__OTHER__", "__MISSING__"}
CASCADE_FIELDS = {"brand", "model", "sub_model"}

LABELS = {
    "brand": "ยี่ห้อรถ (Brand)",
    "model": "รุ่นรถ (Model)",
    "sub_model": "รุ่นย่อย (Sub-model)",
    "model_year": "ปีรถ (Model year)",
    "mileage": "เลขไมล์ (กม.)",
    "fuel_type": "ประเภทเชื้อเพลิง (Fuel type)",
    "engine_size": "ขนาดเครื่องยนต์ (Engine size)",
    "body_type": "ประเภทรถ (Body type)",
    "color": "สีรถ (Color)",
    "number_of_seats": "จำนวนที่นั่ง (Number of seats)",
}


def find_models() -> list[Path]:
    if not TRAIN_DIR.is_dir():
        return []
    files = [p for p in TRAIN_DIR.glob("*/used_car_models_*.joblib")
             if p.is_file() and DATE_DIR.fullmatch(p.parent.name)
             and MODEL_NAME.fullmatch(p.name)]
    return sorted(files, key=lambda p: (p.stem, p.parent.name), reverse=True)


@st.cache_resource(show_spinner="กำลังโหลด Model ที่เลือก …")
def load_selected_model(file_path: str, mtime_ns: int, file_size: int):
    del mtime_ns, file_size
    return predictor.load_bundle(Path(file_path))


@st.cache_data(show_spinner="กำลังดึงข้อมูล Brand / Model / Sub-model จาก STG_USED_CAR …")
def load_vehicle_catalog_from_stg() -> tuple[pd.DataFrame, str]:
    """Query the latest STG snapshot ONCE per app process; do not create files.

    Streamlit reruns the script whenever an input changes. cache_data avoids
    another SQL query per widget interaction, while a new Streamlit process
    starts with an empty cache and fetches the current STG snapshot again.
    """
    engine = training.build_engine()  # Reuse the existing SQL Server config/env.
    try:
        table = (f"{training.quote_sql_identifier(training.DB_SCHEMA)}."
                 f"{training.quote_sql_identifier(training.SOURCE_TABLE)}")
        with engine.connect() as conn:
            # Detect case variations, fail clearly if Alice changes required fields.
            column_names = list(pd.read_sql(text(f"SELECT TOP (0) * FROM {table}"), conn).columns)
            wanted = [training.PCS_DATE_COLUMN, "brand", "model", "sub_model"]
            lookup = {str(col).casefold(): str(col) for col in column_names}
            missing = [name for name in wanted if name.casefold() not in lookup]
            if missing:
                raise ValueError(
                    f"{training.DB_SCHEMA}.{training.SOURCE_TABLE} ไม่พบคอลัมน์ "
                    f"{missing}; คอลัมน์ปัจจุบัน: {column_names}"
                )
            columns = ", ".join(
                f"{training.quote_sql_identifier(lookup[name.casefold()])} "
                f"AS {training.quote_sql_identifier(name)}" for name in wanted
            )
            raw = pd.read_sql(text(f"SELECT DISTINCT {columns} FROM {table}"), conn)
    finally:
        engine.dispose()

    if raw.empty:
        raise ValueError(f"{training.DB_SCHEMA}.{training.SOURCE_TABLE} ไม่มีข้อมูล")
    # STG_USED_CAR must be the latest single-date snapshot (same train rule).
    pcs_date = training.parse_single_pcs_date(raw).strftime("%Y-%m-%d")
    required = ["brand", "model", "sub_model"]
    catalog = raw[required].copy()
    for name in required:
        catalog[name] = catalog[name].astype("string").str.strip().replace("", pd.NA)
    catalog = catalog.dropna(subset=["brand", "model"]).drop_duplicates()
    if catalog.empty:
        raise ValueError("STG_USED_CAR ไม่พบข้อมูล brand/model ที่ใช้ทำ Dropdown")
    return catalog.reset_index(drop=True), pcs_date


def matching_rows(catalog: pd.DataFrame, brand: object = None,
                  model: object = None) -> pd.DataFrame:
    """Filter based on actual source brand/model combinations; case-insensitive."""
    subset = catalog
    for feature, selected in (("brand", brand), ("model", model)):
        if isinstance(selected, str) and selected.strip():
            subset = subset.loc[
                subset[feature].str.casefold().eq(selected.strip().casefold()).fillna(False)
            ]
    return subset


def options_for(catalog: pd.DataFrame, feature: str, *,
                brand: object = None, model: object = None) -> list[str]:
    subset = matching_rows(catalog, brand=brand if feature != "brand" else None,
                           model=model if feature == "sub_model" else None)
    levels = subset[feature].dropna().astype(str).unique().tolist()
    return sorted(levels, key=str.casefold)


def choice_to_value(choice: object) -> object:
    if choice == MISSING_LABEL:
        return None
    return choice


def dependent_widget_key(run_id: str, feature: str, *,
                         brand: object = None, model: object = None) -> str:
    """A change of Brand/Model creates a new dependent widget, clearing stale picks."""
    parents = ((brand,) if feature == "model" else
               (brand, model) if feature == "sub_model" else ())
    signature = "\x1f".join(str(p) for p in parents)
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
    return f"category_{run_id}_{feature}_{digest}"


def numeric_value(raw: str, feature: str) -> float:
    text = raw.strip().replace(",", "")
    if not text:
        raise ValueError(f"กรุณากรอก {LABELS.get(feature, feature)}")
    try:
        number = float(text)
    except ValueError as error:
        raise ValueError(f"{LABELS.get(feature, feature)} ต้องเป็นตัวเลข") from error
    if not math.isfinite(number):
        raise ValueError(f"{LABELS.get(feature, feature)} ต้องเป็นตัวเลขที่มีค่าจำกัด")
    if feature in {"mileage", "engine_size"} and number < 0:
        raise ValueError(f"{feature} ต้องไม่ติดลบ")
    if feature == "number_of_seats" and (number <= 0 or not number.is_integer()):
        raise ValueError("จำนวนที่นั่งต้องเป็นจำนวนเต็มที่มากกว่า 0")
    if feature == "model_year" and (not number.is_integer() or not 1886 <= number <= 2100):
        raise ValueError("ปีรถต้องเป็นปี ค.ศ. ที่สมเหตุสมผล")
    return number


def model_selector_label(path: Path) -> str:
    match = MODEL_NAME.fullmatch(path.name)
    run_id = match.group(1) if match else path.stem
    return f"{run_id}  ·  PCS_DATE {path.parent.name}"


def category_select(feature: str, options: list[str], run_id: str, *,
                    brand: object = None, model: object = None) -> object:
    """One searchable/editable widget; there is NO separate custom text field."""
    label = LABELS.get(feature, feature)
    choices = [*options, MISSING_LABEL]
    key = dependent_widget_key(run_id, feature, brand=brand, model=model)
    selection = st.selectbox(
        label,
        options=choices,
        index=None,
        placeholder="เลือกจากรายการ หรือพิมพ์ค่าใหม่แล้วกด Enter",
        accept_new_options=True,
        key=key,
    )
    return choice_to_value(selection)


def create_form(bundle: dict, state, catalog: pd.DataFrame) -> dict[str, object] | None:
    """Render cascade widgets OUTSIDE st.form so child choices update immediately."""
    required = list(bundle["selected_source_features"])
    numeric = set(state.numeric_features)
    categorical = set(state.categorical_features)
    values: dict[str, object] = {}
    run_id = str(bundle["model_id"])
    st.caption("ข้อมูลรถ · เลือกหรือพิมพ์ใน Dropdown เดียวกัน; รุ่นรถและรุ่นย่อยจะกรองตามที่เลือก")

    # Keep the Brand → Model → Sub-model hierarchy in one compact row.
    # Streamlit stacks columns on narrow screens; dependent widgets still rerun
    # immediately because these inputs are deliberately outside st.form.
    hierarchy = st.columns(3, gap="small")
    with hierarchy[0]:
        if "brand" in required:
            values["brand"] = category_select(
                "brand", options_for(catalog, "brand"), run_id
            )
    with hierarchy[1]:
        if "model" in required:
            if "brand" in required and values.get("brand") is None:
                st.selectbox(
                    LABELS["model"], options=[], index=None,
                    placeholder="เลือกยี่ห้อรถก่อน", disabled=True,
                    key=f"waiting_model_{run_id}",
                )
                values["model"] = None
            else:
                values["model"] = category_select(
                    "model", options_for(catalog, "model", brand=values.get("brand")),
                    run_id, brand=values.get("brand"),
                )
    with hierarchy[2]:
        if "sub_model" in required:
            if "model" in required and values.get("model") is None:
                st.selectbox(
                    LABELS["sub_model"], options=[], index=None,
                    placeholder="เลือกรุ่นรถก่อน", disabled=True,
                    key=f"waiting_submodel_{run_id}",
                )
                values["sub_model"] = None
            elif "brand" in required and values.get("brand") is None:
                st.selectbox(
                    LABELS["sub_model"], options=[], index=None,
                    placeholder="เลือกยี่ห้อรถก่อน", disabled=True,
                    key=f"waiting_brand_submodel_{run_id}",
                )
                values["sub_model"] = None
            else:
                values["sub_model"] = category_select(
                    "sub_model",
                    options_for(
                        catalog, "sub_model", brand=values.get("brand"),
                        model=values.get("model"),
                    ),
                    run_id, brand=values.get("brand"), model=values.get("model"),
                )

    # Three compact columns instead of two full-width columns. Keep a predictable
    # order for the current model while accommodating new feature sets later.
    preferred_order = (
        "model_year", "mileage", "engine_size",
        "fuel_type", "body_type", "color", "number_of_seats",
    )
    other_features = [feature for feature in required if feature not in CASCADE_FIELDS]
    other_features.sort(key=lambda name: (
        preferred_order.index(name) if name in preferred_order else len(preferred_order),
        required.index(name),
    ))
    feature_columns = st.columns(3, gap="small")
    for idx, feature in enumerate(other_features):
        with feature_columns[idx % 3]:
            label = LABELS.get(feature, feature)
            if feature in numeric:
                median = state.numeric_medians.get(feature)
                hint = f"ค่ากลางตอน Train: {median:g}" if median is not None else "กรอกตัวเลข"
                values[feature] = st.text_input(
                    label, value="", placeholder=hint,
                    key=f"numeric_{run_id}_{feature}",
                )
            elif feature in categorical:
                model_levels = state.category_levels[feature]
                options = [lv for lv in model_levels if lv not in SENTINELS]
                values[feature] = category_select(feature, options, run_id)
            else:
                st.error(f"Feature {feature!r} ไม่มีชนิด Numeric/Categorical ใน Model")
                return None

    if not st.button("ทำนายราคา", type="primary", use_container_width=False):
        return None

    cleaned: dict[str, object] = {}
    for feature in required:
        raw = values.get(feature)
        if feature in numeric:
            cleaned[feature] = numeric_value(str(raw), feature)
        elif feature in categorical:
            if raw is None:
                if feature == "sub_model":
                    # Explicit missing is allowed for sub_model (as at training).
                    # Unlike a missing selection, the user must have selected
                    # the special MISSING_LABEL; see mandatory choice below.
                    selected_widget = st.session_state.get(
                        dependent_widget_key(run_id, feature,
                                             brand=values.get("brand"), model=values.get("model"))
                    )
                    if selected_widget != MISSING_LABEL:
                        raise ValueError(f"กรุณาเลือก {LABELS.get(feature, feature)}")
                elif feature in CASCADE_FIELDS:
                    raise ValueError(f"กรุณาเลือกหรือพิมพ์ {LABELS.get(feature, feature)}")
                else:
                    selected_widget = st.session_state.get(
                        dependent_widget_key(run_id, feature)
                    )
                    if selected_widget != MISSING_LABEL:
                        raise ValueError(f"กรุณาเลือก {LABELS.get(feature, feature)}")
                cleaned[feature] = None
            elif isinstance(raw, str) and not raw.strip():
                raise ValueError(f"กรุณาเลือกหรือพิมพ์ {LABELS.get(feature, feature)}")
            else:
                cleaned[feature] = raw
    return cleaned


def main() -> None:
    st.set_page_config(page_title="Used Car Price Prediction", page_icon="🚗", layout="wide")
    # Streamlit's default wide layout stretches each input on large monitors.
    # Constrain content width without constraining the browser window itself.
    st.markdown(
        """<style>
        .stMainBlockContainer, .block-container {
            max-width: 1080px !important;
            padding-top: 1.35rem !important;
            padding-bottom: 2rem !important;
        }
        @media (max-width: 720px) {
            .stMainBlockContainer, .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
        }
        </style>""",
        unsafe_allow_html=True,
    )
    st.title("🚗 Used Car Price Prediction")
    st.write("ทดลองทำนายราคาขายรถมือสองจาก OLS Model ที่ Train และบันทึกไว้ใน `.joblib`")
    st.caption("เลือกข้อมูลรถจาก STG_USED_CAR · ทำนายด้วย .joblib · ไม่ Train ใหม่ และไม่สร้าง Catalog CSV")

    models = find_models()
    if not models:
        st.error(f"ไม่พบไฟล์ Model ใน {TRAIN_DIR}/YYYYMMDD/ กรุณา Train Model ก่อน")
        st.stop()
    selected = st.selectbox("เลือก Model Run", options=models, format_func=model_selector_label)
    with st.expander("รายละเอียดไฟล์ Model"):
        st.code(str(selected), language=None)

    try:
        info = selected.stat()
        bundle, training = load_selected_model(str(selected), info.st_mtime_ns, info.st_size)
        state = predictor.make_state(bundle, training)
        required = list(bundle["selected_source_features"])
        if set(required) != set(state.numeric_features) | set(state.categorical_features):
            raise ValueError("รายชื่อ Features ไม่ตรงกับ Preprocessing State")
    except Exception as error:
        st.error(f"โหลด Model ไม่สำเร็จ: {error}")
        st.stop()

    st.caption(f"MODEL_ID: {bundle['model_id']}  ·  PCS_DATE: {bundle['pcs_date']}  ·  Features: {len(required)}")
    with st.expander("ดู Feature และ Category Mapping ที่ Model รองรับ"):
        st.write("Input ที่ Model ใช้จริง: " + ", ".join(f"`{name}`" for name in required))
        for name in state.categorical_features:
            st.write(f"**{name}** — Reference: `{state.reference_categories[name]}`; "
                     f"Retained: {', '.join(str(v) for v in state.category_levels[name])}")
        st.caption("ตัวเลือก Brand/Model/Sub-model แสดงจาก Catalog ต้นทาง; "
                   "Model อาจจัดค่าที่ไม่ได้เก็บไว้ตอน Train เป็น __OTHER__")

    if CASCADE_FIELDS.intersection(required):
        # Explicit refresh is optional; normally query once after every app start.
        if st.button("รีเฟรชรายการรถจาก STG_USED_CAR"):
            load_vehicle_catalog_from_stg.clear()
        try:
            catalog, catalog_pcs_date = load_vehicle_catalog_from_stg()
        except Exception as error:
            st.error(f"ดึงรายการรถจาก SQL Server ไม่สำเร็จ: {error}")
            st.info("ตรวจสอบ Docker SQL Server, การตั้งค่าการเชื่อมต่อใน "
                    "train_used_car_ols.py และ USED_CAR_DB_SERVER=127.0.0.1 "
                    "จากนั้นรีเฟรชหน้าเว็บ")
            st.stop()
        st.caption(
            f"Dropdown: STG_USED_CAR (PCS_DATE {catalog_pcs_date}) · "
            f"{len(catalog):,} คู่ข้อมูลยี่ห้อ/รุ่น/รุ่นย่อย · อ่านในหน่วยความจำ ไม่สร้าง CSV"
        )
        model_pcs_date = pd.to_datetime(bundle["pcs_date"]).strftime("%Y-%m-%d")
        if catalog_pcs_date != model_pcs_date:
            st.warning(
                f"ข้อมูล Dropdown เป็น STG วันที่ {catalog_pcs_date} "
                f"แต่ Model ใช้ข้อมูล Train วันที่ {model_pcs_date}: "
                "เลือกค่าที่เพิ่งปรากฏใน STG อาจถูกเข้ารหัสเป็น __OTHER__ "
                "ตาม Mapping ของ Model เดิม"
            )
    else:
        catalog = pd.DataFrame(columns=["brand", "model", "sub_model"])

    try:
        car = create_form(bundle, state, catalog)
    except ValueError as error:
        st.error(str(error))
        return
    if car is None:
        return

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PerformanceWarning)
            result = predictor.predict(pd.DataFrame([car]), bundle, training)
    except Exception as error:
        st.error(f"ทำนายไม่สำเร็จ: {error}")
        return

    row = result.iloc[0]
    price = float(row["PREDICTED_PRICE_THB"])
    st.divider()
    st.subheader("ผลทำนาย")
    st.metric("Predicted Price (THB)", f"฿{price:,.2f}")
    if price < 0:
        st.error("Model ทำนายราคาออกมาติดลบ — ไม่ควรนำค่าดังกล่าวไปใช้เป็นราคาขายโดยตรง")
    st.caption(f"MODEL_ID: {bundle['model_id']} · Point prediction ไม่ใช่ราคาขายที่รับประกัน")

    mapped = []
    for name in state.categorical_features:
        used = row.get(f"{name}_MODEL_CATEGORY")
        mapped.append({"Feature": name, "Input": car.get(name), "Model category": used})
        if used == "__OTHER__":
            st.warning(f"{name}: ข้อมูลที่เลือกถูกจัดเข้า `__OTHER__` ตาม Mapping ของ Model")
    if mapped:
        with st.expander("ตรวจสอบ Category ที่ Model ใช้จริง"):
            st.dataframe(pd.DataFrame(mapped), hide_index=True, use_container_width=True)

    result_csv = io.StringIO()
    result.to_csv(result_csv, index=False, float_format="%.2f")
    st.download_button(
        "ดาวน์โหลดผล Predict (CSV)", result_csv.getvalue().encode("utf-8-sig"),
        file_name=f"used_car_prediction_{bundle['run_id']}.csv", mime="text/csv",
    )


if __name__ == "__main__":
    main()
