# USECAR_PREDICT_PRICE — Work Log

ทุกครั้งที่แก้โค้ด ให้เพิ่มรายการงานพร้อมวันที่/ขอบเขต, baseline Git, ไฟล์ที่แก้, ผลทดสอบ และสิ่งที่ยังไม่ได้ทดสอบ ห้ามบันทึก secret values หรือข้อมูลรถรายแถว เก็บประวัติเก่าไว้และเพิ่มข้อมูลใหม่ตามจริง

## 2026-09-20 — สำรวจ Workspace และจัดทำ Project Context

### ขอบเขตและฐานก่อนทำงาน

ผู้ใช้อนุญาตเฉพาะการตรวจสถานะและจัดทำ `AGENTS.md`, `Docs/PROJECT_CONTEXT.md`, `Docs/WORK_LOG.md` งานนี้ไม่แก้ Application Code/OLS/Schema/Predict Logic, ไม่แก้ STG/ข้อมูลต้นทาง/Data Dictionary และไม่ Commit/Push

- Workspace: `/Users/krissanap/Document/KMUTT/USECAR_PREDICT_PRICE`
- Initial Git: clean `master...origin/master`, staged/unstaged diff ว่าง, ahead/behind `0/0`
- `HEAD` = local `origin/master` = live GitHub HEAD/master = `3bd4cadb55280535d56999ddeb5503fcd4291084`
- Commit: `train model`, 2026-09-20 08:59:53 +07:00; remote <https://github.com/krissanapipukmutt/USECAR_PREDICT_PRICE.git>
- ตรวจ live refs ด้วย read-only `git ls-remote`; ครั้งแรกใน sandbox มี DNS failure แล้ว read-only escalation ที่ได้รับอนุมัติสำเร็จ ไม่มี fetch/pull/merge/reset หรือเปลี่ยน Git refs
- ไม่พบ Workspace/GitHub tracked difference ก่อนทำงาน จึงใช้ Workspace โดยไม่ต้องเปลี่ยนเวอร์ชัน
- ไม่พบ ancestor/root AGENTS.md เดิม; `Docs` และ Data Dictionary มีอยู่แล้ว ไม่สร้าง `docs` ตัวเล็ก

### สิ่งที่อ่านและตรวจ

- Python application ครบ 5 ไฟล์: `train_used_car_ols.py` (1,810 lines), `predict_used_car_ols.py` (266), `app_used_car.py` (438), `analyze_used_car_ols.py` (235), `plot_used_car_ols.py` (338)
- Data Dictionary `Docs/Used_Car_OLS_Data_Dictionary_Thai_Clear.xlsx` แบบ read-only XML extraction ครบสอง worksheets; ไม่ save/export workbook ใหม่
- Local tracked/ignored files, staged/unstaged diff, branch/remotes และ reachable history 3 commits; secrets ตรวจแบบไม่แสดงค่า รวม multiline defaults และ tracked bytecode
- Existing CSV schemas, model/result keys, aggregate metrics, selection/outlier counts; ไม่แสดงหรือบันทึกข้อมูลรถรายแถว และไม่ deserialize joblib
- `.venv/pyvenv.cfg` และ package METADATA แบบอ่านอย่างเดียว ไม่ import application หรือติดตั้ง packages
- ใช้ skill `developing-with-streamlit` และ best-practices สำหรับอ่าน UI และ `spreadsheets` สำหรับอ่าน Dictionary; จำกัดตามขอบเขตที่ผู้ใช้อนุญาต

### ไฟล์ที่สร้าง

| ไฟล์ | เนื้อหา |
| --- | --- |
| `AGENTS.md` | กติกาเริ่มงาน, Git/version comparison, Docs casing, ขอบเขตห้ามแก้, secrets, approval และการอัปเดต Work Log |
| `Docs/PROJECT_CONTEXT.md` | Architecture/DB, Dictionary mapping, Training/CV, Price Filter/Outlier, Prediction/UI, artifacts, environment, ปัญหาและงานถัดไป พร้อมแหล่งอ้างอิง |
| `Docs/WORK_LOG.md` | บันทึกการตรวจครั้งนี้ ผลตรวจจริงและสิ่งที่ยังไม่ได้ทดสอบ |

### ข้อค้นพบสำคัญ

- Application ทั้งห้าอ่าน source DB อย่างเดียว ไม่พบ DB write/DDL; training export Top 3 Regression CSV และ Rank‑1 joblib ไม่มี DWH loader/approve workflow ที่ implement ใน scripts
- Dictionary ชื่อ/ลำดับคอลัมน์ตรง 18 RESULT และ 13 COEFFICIENT แต่ datatype/keys/rules ระบุ proposed design; actual DDL ยังไม่ยืนยัน
- Code default Price Filter = BASELINE; EXPERIMENT ตัดราคา ≤1,000/confirmed bad IDs จาก CV train และ final fit เท่านั้น; validation ยังใช้ full cleaned positive-price cohort; Outlier flags ไม่ตัด train เอง
- Latest run `20260920_085412` เป็น EXPERIMENT: clean 34,848, ตัด 598, final fit 34,250; previous `20260920_085346` BASELINE ใช้ 34,848 ทั้งหมด
- Rank‑1 latest CSV: CV RMSE 1,110,814.34 บาท / MAE 456,889.75 บาท; previous baseline 1,020,440.92 / 314,887.12 บาท เป็นผลจากไฟล์เดิม ไม่ใช่ผล retrain และยังไม่ตัดสินเลือก model
- พบ hardcoded `USED_CAR_DB_PASSWORD` default ที่ `train_used_car_ols.py:35`, tracked bytecode และประวัติ 3 commits รวม remote commit ปัจจุบัน ไม่เปิดเผยค่าและไม่ตรวจ credential validity
- ไม่มี root `.gitignore`; tracked CSV 35, joblib 5 รวม 235,956,222 bytes, pyc 2 และ raw/listing-level outputs; `.venv` ignored โดยไฟล์ภายในเองเท่านั้น
- Analyze default run เก่า, diagnostics ยังไม่จำกัด final-fit experiment cohort, UI catalog ไม่มี TTL; บันทึกไว้โดยไม่แก้ application

### การตรวจสอบที่ทำและผล

| การตรวจ | ผล | ข้อจำกัด |
| --- | --- | --- |
| Git status/branch/diff และ live refs | PASS: initial Workspace ตรง live master | เป็น snapshot ณ เวลาตรวจ ไม่รับรอง remote ในอนาคต |
| Python syntax ผ่าน `ast.parse` | PASS ทั้ง 5 scripts | ไม่ import/execute; ไม่ใช่ unit/integration test |
| Dictionary เทียบ CSV headers/order | PASS ทั้ง 10 Regression CSV จาก 5 runs: 18/13 columns | ไม่ยืนยัน SQL datatypes/constraints |
| RESULT IDs / coefficient FK / composite key | PASS ทั้ง 5 runs; RESULT run ละ 3 rows; COEFFICIENT 317/522/522/522/317 rows | ตรวจภายในแต่ละ run; ไม่ยืนยัน concurrent ID uniqueness |
| Artifact inventory | PASS: 5 runs ครบ RESULT/COEFFICIENT/joblib | ตรวจชื่อ/ขนาด/CSV เท่านั้น ไม่ตรวจ internal joblib compatibility |
| Selection / Outlier counts | PASS: final rows + exclusions = 34,848; 25,844 + 8,400 + 604 = 34,848; 598 + 2 + 4 = 604 | เป็น aggregate ของ reports เดิม ไม่ได้ query source DB ใหม่ |
| Secrets / ignore audit | พบ issues: hardcoded secret/history/bytecode, missing root ignore, tracked data/models | Targeted inspection ไม่ใช่ exhaustive forensic scan |
| Application / Dictionary preservation | ตรวจ SHA‑256 เทียบ baseline หลังสร้างเอกสาร | ดู final verification ด้านล่าง |

### ยังไม่ได้ทดสอบ

- Live SQL Server connectivity, ODBC driver, table schema/keys/permissions, source freshness/row counts หรือ DWH ETL
- Training/retraining, recomputation ของ CV/OLS/coefficient CI, row-selection edge cases, eligible validation metrics หรือ independent holdout
- Joblib deserialization, dependency imports/runtime compatibility, CLI inference หรือ equivalence กับ coefficient-based calculation
- Streamlit rendering, dropdowns จาก DB, cache refresh, input edge cases, session/download interactions หรือ end-to-end prediction
- Password validity, rotation, repository visibility และ full forensic scan ทุก historical binary
- ไม่เปลี่ยน `.gitignore`, secret configuration, tracking หรือ history เพราะอยู่นอกขอบเขตงานนี้

### Final verification

- PASS: SHA‑256 ของ Python scripts ทั้ง 5 ไฟล์และ Data Dictionary ตรง baseline ก่อนเขียนเอกสารทุกไฟล์
- PASS: ตรวจรหัสผ่านที่ดึงจาก AST กับเอกสารทั้งสามแล้วไม่พบค่าลับ โดยไม่แสดงค่าในการตรวจ
- PASS: เอกสารเป็น UTF‑8, Markdown fences ปิดครบ, local document links มีปลายทางจริง และผ่าน whitespace checks
- PASS: มีเพียงโฟลเดอร์ `Docs` เดิม ไม่มี `docs` ซ้ำ
- PASS: หลังสร้างเอกสาร Git มีเพียง `?? AGENTS.md`, `?? Docs/PROJECT_CONTEXT.md`, `?? Docs/WORK_LOG.md`; tracked diff และ staged diff ว่าง ไม่มีการเปลี่ยน source/artifact ที่ track อยู่
- ตรวจทานข้ามกับ source อีกครั้งและแก้ข้อความเรื่องลำดับ feature inference, infinity handling ของ outlier report, หลักฐาน price mode ของ run `083158` และบทบาท shared preprocessing ก่อนจบงาน
- ไม่มี application changes, Data Dictionary edits, DB operations, stage, Commit หรือ Push

### งานถัดไปที่เสนอ

เริ่มจาก credential rotation/secret configuration และ Git hygiene เป็นงานที่อนุญาตแยก แล้วตรวจ DDL/Dictionary semantics, environment/bundle compatibility และ diagnostics/runtime ตามลำดับ รายละเอียดและข้อจำกัดอยู่ใน [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) ทุกการเปลี่ยน OLS/Database Schema/Output Schema/Predict Logic ต้องได้รับอนุญาตก่อน

## 2026-09-20 — แผน Training Evaluation V4 (Plan only)

### ขอบเขตและสถานะที่รับช่วงต่อ

รอบนี้วางแผนและตรวจแบบ read-only ต่อจากผลเดิม ยกเว้นการเพิ่มบันทึกส่วนนี้ตามคำสั่งผู้ใช้ ไม่แก้ Application Code, Database Schema, `STG_USED_CAR`, Data Dictionary หรือ artifacts และไม่รัน Training/Prediction/UI, Commit หรือ Push

- Branch `master`; local `HEAD`, local `origin/master` และ live GitHub `master` ที่ตรวจแบบ read-only ตรงกันที่ `3bd4cadb55280535d56999ddeb5503fcd4291084`
- Application scripts ทั้ง 5 ไฟล์ไม่มี diff เทียบ `HEAD`; ใช้ผลอ่าน source เดิมร่วมกับการตรวจ `train_used_car_ols.py` ล่าสุดต่อได้ ไม่สำรวจโปรเจกต์ใหม่ทั้งหมด
- งาน V4 ก่อน Usage Limit ทำเสร็จถึงขั้นตรวจ flow/functions, compatibility และร่างแนวทาง split/CV/holdout แต่ยัง **ไม่ได้** บันทึกแผน V4 ลง Work Log และยังไม่มี implementation
- Working tree ก่อนเพิ่มบันทึกนี้มี user/external changes ที่รักษาไว้: `.DS_Store` 4 ไฟล์ถูกแก้, เอกสาร 3 ไฟล์ยัง untracked และมี artifacts run `20260920_092442` ใหม่ 6 ไฟล์ที่ยัง untracked ไม่มี staged changes
- Artifacts `092442` ที่อ่านเฉพาะ aggregate ยืนยัน EXPERIMENT: selection 34,848 แถว, ใช้ train 34,250, ตัด 598; RESULT มี 3 rows และ COEFFICIENT 317 rows ข้อมูลนี้เป็นผล run ก่อนหน้า ไม่ใช่ผลที่สร้างในรอบวางแผนนี้
- ไม่พบ `logs/` หรือ `*.log` ของ run `092442`; จึงยืนยัน Terminal Output ฉบับเต็มย้อนหลังไม่ได้
- จาก `price_quality_all_20260920_092442.csv` ซึ่งเป็น artifact proxy ไม่ใช่ query STG ใหม่: `listing_id` และ `source_url` ไม่ว่างและ unique ทั้ง 34,848 แถว แต่ยังไม่ตัดความเสี่ยงรถคันเดียวถูกประกาศใหม่ด้วย ID/URL ต่างกัน

### A. Implementation ปัจจุบันที่ยืนยันจากโค้ด

ลำดับใน `main()` (`train_used_car_ols.py:1528`) คือ:

1. สร้าง `run_datetime/run_id`, `build_engine()` และ `load_source_data()` อ่าน `SELECT *` จาก source แล้ว `normalize_required_column_names()` และ `parse_single_pcs_date()` ตรวจ target/snapshot
2. `export_outlier_flag_only_reports()` เรียก `build_group_price_outlier_report()` บน source ก่อน clean target; Group-based outlier เป็นรายงาน flag-only และไม่เลือกแถว Train
3. `clean_target()` แปลง `price` เป็น numeric แล้วเก็บ finite `price > 0`; `maybe_convert_numeric_like_columns()` แปลง predictor text ที่ parse ได้อย่างน้อย 98%
4. `select_train_rows()` ใช้ `price_filter_exclusion_reason()`: BASELINE เก็บ positive-price ทั้งหมด; EXPERIMENT ตัด `price <= 1,000`/confirmed IDs จาก final fit และแต่ละ CV train fold จากนั้น `export_train_row_selection_report()` เขียน audit
5. `infer_eligible_features()` ทำ data-driven eligibility บน `final_train_df`; `build_candidate_experiments()` ใช้ `generate_candidate_feature_sets()` สร้าง BASELINE สูงสุด 8 ชุดและ EXPANDED_BRAND_MODEL ของ 2 ชุดแรก
6. `evaluate_candidate_cv()` สร้าง `KFold(5, shuffle=True, random_state=42)` ใหม่ต่อ candidate แต่ได้ indices เหมือนกันเพราะ input/order/seed เดียวกัน; EXPERIMENT filter เฉพาะ fold train ส่วน validation ยังเป็นทุก positive-price row
7. ในแต่ละ fold `fit_preprocessor()` เรียน numeric medians, category levels/reference/OTHER จาก fold train; `transform_with_preprocessor()` ใช้ state เดิมกับ validation; `backward_eliminate_source_features()` ทำ joint F-test ราย source feature บน fold train และ `fit_ols()` fit OLS
8. CV RMSE/MAE หลักเป็นค่าเฉลี่ยราย foldบน validation ทุก positive-price row ส่วน eligible-only RMSE/MAE เป็น pooled OOF diagnostics จาก `price_filter_eligible_validation_mask()` และยังไม่ใช้ ranking
9. `fit_final_candidate()` refit preprocessing, backward feature selection และ OLS บน `final_train_df` ของทุก candidate; `rank_candidates()` deduplicate structure แล้วเรียง CV RMSE, CV MAE, final-fit adjusted R²
10. `build_result_dataframe()` และ `build_coefficient_dataframe()` export Top 3; `build_top1_model_bundle()` + `joblib.dump()` export Rank 1 เท่านั้น โครงสร้างเดิมคือ RESULT 18 columns, COEFFICIENT 13 columns และหนึ่ง `.joblib`

ดังนั้น behavior ที่ระบุใน prompt ถูกต้อง: Candidate 10 มี eligible-only OOF ดีกว่า Candidate 2 ตาม log ที่ให้มา แต่แพ้ ranking เพราะ current `cv_rmse` วัด all-positive validation. ตัวเลข run `092442` เป็น OOF/CV ไม่ใช่ untouched holdout และไม่รับประกันผลกับรถใหม่

### B. แผน Training Evaluation V4

#### B1. Target cohort และ audit

- กำหนด `TARGET_PRICE_MIN_EXCLUSIVE_THB = 1_000.0` เป็น training/evaluation cohort rule: หลัง `clean_target()` สร้าง `eligible_df = df[df.price > 1_000]` เพียงครั้งเดียวก่อน split
- ใช้ `eligible_df` เท่านั้นสำหรับ Development, CV train, CV validation, Holdout และ Full-data Refit; 598 แถวในหลักฐานปัจจุบันอยู่ใน exclusion audit แต่ไม่เข้า metric หรือ model fit
- Rule นี้อาศัย actual target และห้ามนำไป validate/gate input ใน `predict_used_car_ols.py` หรือ UI เพราะเวลาทำนายยังไม่รู้ราคาจริง
- `CONFIRMED_BAD_LISTING_IDS` ถ้ายังใช้ ต้องตัดก่อน split และบันทึกเหตุผลแยก; Group-based `SUSPECTED_OUTLIER` คง flag-only ทุกสถานะและไม่ถูกลบอัตโนมัติ
- ปรับ row-selection report แยก `ELIGIBLE_FOR_EVALUATION`, split assignment และ exclusion reason โดยยังเป็นไฟล์ analysis แยก ไม่เพิ่มคอลัมน์ใน Regression tables

#### B2. Holdout ก่อน Candidate Selection และ duplicate control

- หลังสร้าง eligible cohort ให้ audit duplicate identity ก่อน split โดยลำดับ key: normalized `listing_id`; fallback normalized `source_url`; หากมี business-approved vehicle fingerprint จึงใช้เป็น fallback เพิ่ม ห้ามใช้ target ใน duplicate key
- หลักฐาน `092442` ไม่พบ duplicate listing ID/URL แต่ต้องตรวจทุก run; brand/model/sub_model/year ซ้ำจำนวนมากและไม่ใช่หลักฐานว่าเป็นรถคันเดียว จึงห้ามใช้สี่ฟิลด์นี้เป็น group key เดี่ยว ๆ
- สร้าง deterministic 80% Development / 20% Holdout ด้วย seed 42 ก่อน candidate generation/preprocessing ถ้ามี duplicate group ให้ทั้ง group อยู่ฝั่งเดียว; ถ้า identity ทุกแถว unique ให้ split แบบ reproducible และ stratify ด้วย price bands ที่กำหนดล่วงหน้าเพื่อรักษาการกระจายราคา
- บันทึก row counts, group counts, price-band distribution และ checksum ของ split assignment ใน evaluation sidecar ห้ามแสดง listing IDs ใน console/log
- Holdout ถูก seal ระหว่าง candidate generation/CV/ranking ใช้เพียงครั้งเดียวกับ Rank 1 ที่เลือกจาก Development CV แล้ว ไม่ใช้เปรียบเทียบ candidates, encoding profiles, features, thresholds หรือแก้ config หากผลไม่ดี

#### B3. Common 5-fold CV บน Eligible Development

- สร้าง folds หนึ่งครั้งจาก Development แล้วส่ง immutable/precomputed `(train_idx, valid_idx)` ชุดเดียวให้ทุก candidate; ตรวจว่าแต่ละ development row เป็น validation ครั้งเดียว, train/validation ไม่ overlap และ duplicate group ไม่ข้าม fold
- เมื่อมี groups ใช้ group-aware split และตรวจ price-band balance; เมื่อทุก identity unique ใช้ shuffled 5-fold แบบ fixed seed พร้อม balance report การ stratify regression ใช้ fixed business price bands ไม่เรียน threshold จาก holdout
- Validation ของทุก foldมีเฉพาะ `price > 1,000`; ยกเลิกความหมายเดิมที่ EXPERIMENT filter เฉพาะ train และ eligible-only diagnostics ซ้ำซ้อน
- ทุก operation ที่เรียนจากข้อมูลต้องใช้ fold train เท่านั้น: numeric-like type decision, missing/cardinality eligibility, numeric medians, retained category levels, reference/OTHER mapping และ backward feature selection Validation ทำเฉพาะ transform/predict/metric
- เพื่อให้ candidate identity คงที่ แยก schema/name exclusions ออกจาก data-driven eligibility: candidate universe มาจาก Development schema/config เท่านั้น ส่วน data-driven usability เรียนในแต่ละ fold; candidate ที่ fold ใดไม่มี usable required structure ต้อง fail พร้อมเหตุผล ไม่ปรับด้วย validation

#### B4. Ranking และ diagnostics

- Primary rank = pooled OOF RMSE บน Eligible Development (`sqrt(sum squared error / OOF n)`); Secondary = pooled OOF MAE; tie-break ที่สามต้องกำหนดล่วงหน้า แนะนำ candidate ID เพื่อ deterministic ordering ไม่ใช้ final-fit adjusted R² ลดการผสม evaluation bases
- เก็บ fold mean/std RMSE/MAE เพิ่มเพื่อดู stability แต่ไม่ใช้แทน pooled OOF ranking
- รายงาน OOF แยก price bands ที่ล็อกก่อน run: count, RMSE, MAE และถ้าต้องการ median absolute error; พร้อม Negative Prediction Rate และ `model=__OTHER__` mapping rate/denominator
- OTHER rate ต้องคำนวณด้วย fold-train encoder ของแต่ละ prediction และกำหนดกรณี candidate ไม่มี `model` เป็น N/A ไม่ใช่ 0
- Holdout report ของ Rank 1 ใช้ metrics และ diagnostics ชุดเดียวกัน แยก section/stage ชัดเจน และไม่ย้อนกลับไป rerank

#### B5. CV, Holdout และ Full-data Refit

| Stage | ข้อมูล | เรียนอะไร | ใช้ทำอะไร |
| --- | --- | --- | --- |
| Development CV | 5 folds ใน eligible Development | ทุก fold เรียน preprocessing + feature selection + OLS จาก fold train | เลือก candidate/profile ด้วย OOF RMSE/MAE |
| Holdout test | eligible Holdout ที่ไม่เคยถูกใช้ | ไม่ fit; ใช้ Development-fit checkpoint ของ Rank 1 transform/predict ครั้งเดียว | ประเมิน generalization เท่านั้น |
| Full-data refit | eligible Development + Holdout หลังประเมิน | coefficient-only OLS refit ภายใต้ structure/preprocessor ที่ freeze จาก Development | สร้าง deployable Top 3 artifacts โดยไม่ใช้ Holdout เลือก feature/encoding/threshold |

ข้อเสนอหลักคือ หลัง CV ให้ fit Top 3 บน Development เพื่อ freeze candidate/profile, selected source/encoded features และ preprocessor state; ประเมิน Holdout เฉพาะ Rank 1; จากนั้น Full-data Refit เปลี่ยนเฉพาะ OLS coefficients บน encoded eligible full data โดยไม่ rerun category selection, medians/reference หรือ backward elimination ด้วย Holdout วิธีนี้ใช้แถวเต็มเพื่อประมาณ coefficients แต่ Holdout metrics อ้าง Development checkpoint ไม่ใช่ exported refit model ต้องระบุให้ชัดใน report

- RESULT CSV `RMSE`/`MAE` คงเป็น Development OOF metrics เพื่อรักษา schema/ความหมายสำหรับ ranking; `R_SQUARED`, `ADJ_R_SQUARED`, F statistics และ COEFFICIENT statistics มาจาก Full-data Refit
- `N_OBSERVATION` ต้องดึง/ตรวจเทียบ `int(ols_result.nobs)` ของ model แต่ละ rank ไม่ส่งค่ารวมจากภายนอกแบบเดียวทุก rowโดยไม่ตรวจ และสำหรับข้อเสนอ coefficient-only full refitควรเท่ากับ eligible full-data count
- หาก matrix ของ rank ใดทำให้ statsmodels drop row ต้อง fail export หรือบันทึก mismatch เป็น error ไม่เขียน N ที่ไม่ตรง model จริง
- เพิ่ม evaluation files ใน `output/analysis/<PCS_DATE>/` เท่านั้น เช่น `training_evaluation_<RUN_ID>.csv`, `holdout_predictions_<RUN_ID>.csv` และ `training_evaluation_metadata_<RUN_ID>.json`; ไม่เปลี่ยน 18/13-column SQL contract หรือชื่อสาม artifacts หลัก
- Sidecar ต้องระบุ stage, cohort rule, seed, split/fold checksum, counts, candidate/profile, metric aggregation, checkpoint/refit distinction และ source/code provenance โดยหลีกเลี่ยง secret/connection string และควบคุม row-level prediction file

### C. ไฟล์และฟังก์ชันที่จะได้รับผลกระทบเมื่อได้รับอนุมัติ

| ไฟล์ | ฟังก์ชัน/ส่วน | แผนผลกระทบ |
| --- | --- | --- |
| `train_used_car_ols.py` | config/dataclasses | เพิ่ม V4 cohort, holdout, split metadata และ metrics fields โดยคง OLS/output columns |
|  | `clean_target`, `price_filter_exclusion_reason`, `select_train_rows`, `price_filter_eligible_validation_mask`, `export_train_row_selection_report` | รวม semantics ให้ `price > 1,000` เป็น eligible cohort ทั้ง train/eval; เก็บ audit และไม่ใช้ rule ตอน predict |
|  | `maybe_convert_numeric_like_columns`, `infer_eligible_features` | แยก schema-only discovery กับ fold-train learned decisionsเพื่อป้องกัน leakage |
|  | `generate_candidate_feature_sets`, `build_candidate_experiments` | สร้าง candidate universe จาก Development/config โดยไม่เห็น Holdout |
|  | `evaluate_candidate_cv` | รับ common folds, ใช้ eligible Development ทั้งสองฝั่ง, คืน pooled OOF metrics/price-band/negative/OTHER diagnostics |
|  | `fit_preprocessor`, `transform_with_preprocessor`, `backward_eliminate_source_features`, `fit_ols` | logic OLS เดิม; เพิ่ม validation ว่า fit เฉพาะ fold train และรองรับ frozen refit |
|  | `fit_final_candidate`, `rank_candidates` | แยก Development checkpoint, ranking RMSE/MAE และ coefficient-only full refit; ไม่ใช้ adjusted R² tie-break |
|  | `build_result_dataframe`, `build_coefficient_dataframe`, `build_top1_model_bundle`, `main` | N จาก model.nobs, คง schemas/joblib required keys, orchestrate split/holdout/sidecars/log-safe output |
| `predict_used_car_ols.py` | `load_bundle`, `make_state`, `predict_dataframe` | คาดว่าไม่ต้องเปลี่ยน predict formula/target gate; เพิ่มเฉพาะ compatibility validation ถ้า bundle metadata V4 เป็น additive และต้องยืนยัน legacy + V4 load |
| `app_used_car.py` | model loading/UI prediction path | ไม่ควรต้องเปลี่ยน behavior; regression testว่า V4 bundle ใช้ selected features เดิมได้และไม่มี price input/gate |
| `analyze_used_car_ols.py` | bundle state restoration, `main` | แยก in-sample refit diagnostics จาก CV/Holdout sidecars; ห้ามเรียก Holdout ว่า in-sample และแก้ eliminated-feature state issueเมื่อได้รับอนุมัติ |
| `plot_used_car_ols.py` | artifact loading/model comparison/`main` | อ่าน stage-labeled evaluation report สำหรับ CV/Holdout plots; คง coefficient plots และไม่ผสม in-sample/OOF/test |
| `Docs/PROJECT_CONTEXT.md`, `Docs/WORK_LOG.md` | V4 behavior/log | อัปเดตหลัง implementation/test จริงทุกครั้ง |
| `.gitignore` (เสนอสร้าง) | secrets/artifacts/log rules | อย่างน้อย ignore `logs/*.log`, `.DS_Store`, `__pycache__/`, `*.pyc`, `.env*`, `.streamlit/secrets.toml`; การ untrack/history cleanup เป็นงานแยก |
| Data Dictionary | ไม่แก้ | Regression 18/13 columns เดิม; evaluation schema อยู่ sidecar และเอกสารใหม่เท่านั้น |

### D. Test Plan และ Acceptance Criteria

1. **Target/cohort unit tests:** invalid/NaN/infinity/nonpositive/1/1,000/1,000.01/positive values; PASS เมื่อ train, every CV validation, holdout และ refitมี `price > 1,000` เท่านั้น และ predictorไม่รับ/อ้าง target
2. **Split tests:** deterministic seed, 80/20 tolerance, full coverage, no overlap, group containment, price-band distribution report; PASS เมื่อทุก eligible rowอยู่ stage เดียวและ checksumซ้ำได้
3. **Duplicate leakage tests:** duplicated listing ID/URL synthetic rowsต้องอยู่ partition/fold เดียว; blank/conflicting IDs failหรือใช้ fallbackตาม policy; PASS เมื่อไม่มี groupข้าม boundaries
4. **Common-fold tests:** inject candidate orderต่างกันแล้ว fold indices/checksumต้องเหมือนเดิม; validation unionเท่ากับ Development หนึ่งครั้งและทุก fold nonempty
5. **Preprocessing leakage tests:** ค่า median/category ที่มีเฉพาะ validation/holdoutต้องไม่ปรากฏใน train state; unseen category map OTHER/reference ตามเดิม; numeric-like dtype/missing/cardinality decisionsมาจาก fold trainเท่านั้น
6. **Feature-selection tests:** synthetic source groupsที่ P-value ผ่าน/ไม่ผ่าน, last-feature rule, fold isolation; Holdout target perturbationต้องไม่เปลี่ยน candidate/profile/selected structure/preprocessor ที่ freeze แล้ว
7. **Metric/ranking tests:** hand-calculated OOF residualsตรวจ pooled RMSE/MAE, fold statistics, price bands, negative rate, OTHER numerator/denominator; Candidate 10-like caseต้อง rankด้วย eligible OOF RMSE แล้ว MAE ไม่ใช้ all-positiveหรือ Holdout
8. **Holdout protocol tests:** Holdoutถูกอ่านเพื่อ predictionหลัง rank lockเพียงครั้งเดียว; เปลี่ยน Holdout valuesต้องเปลี่ยน test metricsแต่ไม่เปลี่ยน ranking/frozen structure; report labelต้องเป็น HOLDOUT_TEST
9. **Refit/export tests:** Full refitใช้ eligible full rows, frozen design,ไม่ rerun selection; `N_OBSERVATION == int(model.nobs)` ทุก RESULT row; RESULT/COEFFICIENT headers/order/key/NULL rulesตรง Dictionary; Top 3 CSV + Rank 1 joblib filenamesเดิม
10. **Bundle compatibility tests:** legacy bundleและ V4 bundleผ่าน `load_bundle`; CLI single/JSON/CSV, unseen category, invalid numeric, negative prediction flag และ UI shared predictorให้ output schemaเดิม ไม่มี target-price gate
11. **Diagnostics tests:** analyze/plotแยก CV, Holdout, Full-refit labelsและ counts; ไม่เรียก holdoutว่า in-sample; sidecarsอยู่ `output/analysis/YYYYMMDD/`
12. **Security/log tests:** scan source/log/artifactsสำหรับ password/credentialed connection stringsและ row-level sensitive fieldsก่อน staging; PASS เมื่อ raw logไม่ถูก track,ไม่มี secret, exit statusของ pipelineถูกเก็บถูกต้อง
13. **Integration test ที่ได้รับอนุญาตภายหลัง:** runกับ controlled snapshotเพียงครั้งเดียว ตรวจ counts 34,848→34,250ถ้า snapshotเดิม, split/fold invariants, metrics, artifactsและ prediction smoke test; ตัวเลขเป็น expectation ไม่ใช่ hard-codeถ้า snapshotเปลี่ยน

Acceptance ทั้งหมดต้องผ่านก่อนเสนอ Commit; บันทึก command/result/สิ่งที่ไม่ได้ทดสอบใน Work Log ไม่มี test ใดเขียนกลับ `STG_USED_CAR`

### Training log และ RUN_ID ที่เสนอ

ปัจจุบัน `main()` สร้าง `run_id = datetime.now()` ภายใน process จึงใช้ `tee logs/train_$(date ...).log` แล้วรับประกันชื่อเดียวกับ artifactไม่ได้ โดยเฉพาะข้ามวินาที วิธี V4 ที่เสนอ:

- เพิ่ม optional `USED_CAR_RUN_ID` ซึ่ง validate รูป `YYYYMMDD_HHMMSS`, สร้างครั้งเดียวก่อนเริ่ม และใช้ค่าเดียวกับ artifact/report ทุกไฟล์; หากไม่กำหนดให้ generate แบบเดิม
- พิมพ์ `[RUN] RUN_ID=<id>` เป็นบรรทัดต้น ๆ และไม่พิมพ์ password/ODBC connection string; ตรวจ collision ก่อนเขียนเพื่อไม่ overwrite run เดิม
- คำสั่งครั้งถัดไปใช้ shell `pipefail`, สร้าง ID ภายนอกหนึ่งครั้ง, ส่งเข้า process และ `tee logs/train_<RUN_ID>.log` เพื่อเห็น Terminal พร้อมบันทึก stdout+stderr; ตรวจ exit code, parse RUN_ID line และยืนยันว่า filenames ทั้งสาม/sidecarsตรงกัน
- Raw log อาจมี server/database metadataและรายละเอียดการประมวลผล จึงเสนอ ignore `logs/*.log` โดย default และ Commit เฉพาะ sanitized summary ใน Work Log หากต้องเก็บ log ต้องตรวจ secret patterns/connection strings/ข้อมูลราย listing ก่อนและขออนุมัติ
- รอบนี้ไม่สร้าง `logs/`, ไม่เพิ่ม `.gitignore`, ไม่รันคำสั่ง Train และไม่สร้าง logย้อนหลัง

### E. ประเด็นรออนุมัติก่อน Implementation

1. อนุมัติ cohort ถาวร `price > 1,000` สำหรับ Train/CV/Holdout/Refit และเลิก BASELINE/EXPERIMENT dual evaluation ใน V4 หรือจะรักษา modeเก่าไว้เพื่อ backward comparisonเท่านั้น
2. อนุมัติ split 80/20, seed 42, duplicate identity policy และ fixed price bands/วิธี group-aware balancing
3. อนุมัติ ranking เป็น pooled eligible Development OOF RMSE → pooled MAE → candidate ID โดยไม่ใช้ adjusted R²/holdout
4. อนุมัติให้ Holdoutประเมิน Rank 1 เพียงครั้งเดียว หรือจะรายงาน Top 3 โดยยืนยันว่าจะไม่เลือกจาก Holdout; ข้อเสนอคือ Rank 1 เท่านั้น
5. อนุมัติ Full-data coefficient-only refitด้วย frozen Development preprocessing/feature structure ข้อแลกเปลี่ยนคือ Holdout metricsวัด Development checkpoint ไม่ใช่ exported refit model; ทางเลือกคือ export Development-fit modelซึ่ง N จะประมาณ 80% และไม่ใช้ Holdoutในการ fit
6. อนุมัติให้ RESULT `RMSE/MAE` คงเป็น Development OOF, `N_OBSERVATION`/fit statisticsเป็น Full-data Refit และ Holdout metricsอยู่ sidecarเท่านั้น
7. อนุมัติชื่อ/ระดับรายละเอียด evaluation sidecars โดยเฉพาะ `holdout_predictions` ที่เป็น row-level sensitive dataและไม่ควร Commitโดย default
8. อนุมัติ optional environment-controlled RUN_ID, การสร้าง `logs/`, และ root `.gitignore`; ต้องกำหนดว่าจะเก็บ raw logs/artifactsนอก Gitหรือ Commitเฉพาะ sanitized summaries
9. อนุมัติขอบเขต diagnostics: แก้ analyze/plotใน V4พร้อม training หรือแยก phaseหลัง core evaluationผ่าน; predict/UIควรเป็น compatibility testsเป็นหลัก
10. ก่อน Commit ใด ๆ ต้องทบทวน dirty `.DS_Store`, untracked artifacts `092442`, เอกสารและ credential/history issue แยกกัน ห้ามใช้ `git add .`

### ลำดับ Implementation ที่เสนอหลังอนุมัติ

1. Freeze decisions/tests และสร้าง synthetic unit testsก่อนแก้ flow
2. Refactor cohort + duplicate audit + deterministic Development/Holdout/common folds
3. ทำ fold-local preprocessing/eligibility/feature selection และ OOF metrics/ranking
4. เพิ่ม sealed Holdout evaluation, frozen coefficient-only refit และ sidecar exports
5. รักษา Regression CSV/joblib contractและตรวจ `N_OBSERVATION` ต่อ model
6. ทดสอบ legacy/V4 predictor + UI compatibility แล้วปรับ analyze/plot stage labels/readers
7. เพิ่ม RUN_ID/loggingและ ignore policyตามที่อนุมัติ จากนั้นจึงทำ controlled training run
8. อัปเดต Project Context/Work Log ด้วยผลจริง ตรวจ secrets/diffs และรออนุมัติก่อน Commit/Push

### Handoff สำหรับ ChatGPT

อ่าน `AGENTS.md`, `Docs/PROJECT_CONTEXT.md`, section “2026-09-20 — แผน Training Evaluation V4” ในไฟล์นี้ และ source `train_used_car_ols.py` ล่าสุดก่อนทำงาน Application ยังไม่ถูกแก้และยังไม่มี V4 train run สิ่งที่ต้องให้เจ้าของโครงการตัดสินใจคือ 10 ข้อในหัวข้อ E โดยเฉพาะ split/duplicate policy, Rank-1-only Holdout, coefficient-only full refit, metric semantics และ raw log/artifact Git policy รักษา dirty `.DS_Store` กับ untracked run `092442` ไว้ ห้าม Commit/Push หรือเปิดเผย credential

## 2026-09-20 — Training Evaluation V4 Implementation

### Baseline และขอบเขต

- เริ่มจาก clean `master` ที่ `75f1ea6b2bd3c8ac80de48529c040b101abad144`; local `HEAD`, `origin/master` และ live GitHub `master` ตรงกันทั้งตอนเริ่มและ final verification
- ได้รับอนุมัติ Implementation, synthetic automated tests, additive sidecars, RUN_ID/log wrapper, `.gitignore` และเอกสาร
- ไม่ได้รับอนุมัติ Full Training กับ SQL Server, Commit หรือ Push จึงไม่ได้ทำสามรายการนี้
- ไม่แก้ Database Schema, `STG_USED_CAR`, Data Dictionary, Predict Logic หรือ Streamlit UI behavior

### Agent coordination

| Agent | สิทธิ์ | สถานะ/ผล |
| --- | --- | --- |
| Main Agent | แก้ Training/integration/new support files/docs | Implementation และ integration checks เสร็จ |
| Agent 1 — Training Audit | Read-only | ตรวจ flow/leakage/group split/ranking/holdout/refit/output/RUN_ID และส่ง edge casesครบ ไม่มีการแก้ไฟล์ |
| Agent 2 — Testing & Evaluation | แก้ `tests/` เท่านั้น | สร้าง synthetic test suite; ผลล่าสุดบันทึกด้านล่าง |
| Agent 3 — Compatibility & Security | Read-only | ยืนยัน additive bundle compatibility, 18/13 schema, no target gate และระบุ diagnostics/security risks ไม่มีการแก้ไฟล์ |

### Implementation ที่เปลี่ยน

- เพิ่ม permanent eligible cohort `price > 1,000` หลัง `clean_target()` และใช้ cohortเดียวสำหรับ Development/CV/Holdout/Refit Outlier flagsยังเป็น review-only
- เพิ่ม transitive duplicate groupingจาก normalized listing ID/URL โดยสร้าง groupก่อนตัด cohortเพื่อรักษา identity bridge, deterministic group-safe splitที่ใกล้ 20% ที่สุดภายใน tolerance ±5 percentage points และ fixed business price-band metadata
- เพิ่ม common group-safe 5 folds; preprocessing/type/eligibility/category/feature selectionเรียนจาก fold train Validation transform/predictเท่านั้น
- เปลี่ยน CV เป็น pooled OOF RMSE/MAE และ ranking `RMSE → MAE → candidate_id`; เพิ่ม fold mean/std, price-band, negative และ model OTHER diagnostics
- Fit Top 3 Development checkpointsก่อนเปิด Holdout ประเมิน Holdoutเฉพาะ Rank 1 แล้วทำ coefficient-only refitบน Eligible Full Dataด้วย frozen design
- RESULT/COEFFICIENT schemasเดิม; `N_OBSERVATION` จาก `ols_result.nobs`; bundle 2.0 เพิ่ม V4 metadataแบบ additiveและคง Predictor required keys
- เพิ่ม evaluation/holdout/metadata sidecarsใต้ `output/analysis/YYYYMMDD/` แยก metric stagesชัดเจน และเก็บ Development OOF diagnosticsของทุก successful candidate
- เพิ่ม strict optional `USED_CAR_RUN_ID`, collision refusalก่อน outputแรก และไม่พิมพ์ DB endpoint/password/connection stringใน normal config output
- เอา nonempty password defaultออกจาก current source; environmentต้องให้ `USED_CAR_DB_PASSWORD` ตอน controlled run
- เพิ่ม `scripts/run_training_v4.sh` สำหรับ `pipefail`, terminal+logผ่าน `tee`, RUN_ID/artifact verificationและ credential-pattern scan
- เพิ่ม root `.gitignore`; ไม่ untrack, delete หรือ rewrite tracked files/historyเดิม

### Additional Requirement — Dynamic Source Schema

- ตรวจยืนยันว่า `load_source_data()` ใช้ `SELECT *` และไม่มี snapshot-specific feature list; `infer_candidate_universe()` อ่านเฉพาะ Development schema ส่วน missing/cardinality/numeric-like/category eligibility เรียนใหม่ใน fold trainผ่าน `infer_eligible_features()`/`fit_preprocessor()` จึงไม่ใช้ Holdoutกำหนด eligibilityหรือ datatype
- คง optional schema behavior: คอลัมน์เพิ่ม/ลบ/เปลี่ยนชื่อถือเป็น current-run schemaและเข้า exclusion/eligibility/candidate generationตามกฎเดิม ไม่มีการ hardcode snapshotปัจจุบัน
- กำหนด mandatory source columnsชัดเจนเป็น `price` และ `PCS_DATE`; normalizeชื่อแบบ case-insensitiveและ failก่อน trainingหากขาดหรือมีชื่อซ้ำแบบ case-insensitiveที่กำกวม
- เพิ่ม additive `training_feature_schema` ใน Rank-1 bundle พร้อมคง `selected_source_features` และ frozen preprocessorเดิม Predictorจึงใช้ feature schemaตอน train, เพิกเฉย extra input columns และ rejectเมื่อ selected featureหายไป
- ไม่มีการแก้ `STG_USED_CAR`, Database Schema, Predictor logic หรือ Output Schemaหลัก

### ไฟล์ที่แก้/สร้าง

- Modified: `train_used_car_ols.py`
- New: `.gitignore`
- New: `scripts/run_training_v4.sh`
- New: `tests/test_training_evaluation_v4.py`
- Modified: `Docs/PROJECT_CONTEXT.md`
- Modified: `Docs/WORK_LOG.md`
- ไม่แก้: `predict_used_car_ols.py`, `app_used_car.py`, `analyze_used_car_ols.py`, `plot_used_car_ols.py`, Data Dictionary และ source/artifacts

### Automated tests และ checks

- PASSED: `.venv/bin/python -m unittest discover -s tests -v` — 11/11 synthetic tests ณ integration passแรก
- PASSED: Python AST parse ทั้ง 5 application scriptsและ test file
- PASSED: `bash -n scripts/run_training_v4.sh`
- PASSED: `git diff --check`
- PASSED: no DB/training/artifact write/joblib deserializationใน tests
- PASSED: tracked pycacheที่ test importทำให้เปลี่ยนถูกคืนเป็น HEAD; ไม่รวม generated bytecodeใน intended diff
- PASSED final rerun: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -v` — 25/25 tests ครอบคลุม exact schema order, RUN_ID/collision, legacy bundle contract, duplicate bridgeผ่าน ineligible row, unseen→`__OTHER__`, holdout tolerance, categorical numeric-like coercion และ dynamic schema cases
- PASSED dynamic-schema regression: optional columnเพิ่ม/ลบ/เปลี่ยนชื่อ, numeric↔numeric-like string datatype, mandatory `price`/`PCS_DATE` หาย, case-insensitive duplicate mandatory name, extra prediction column และ missing saved model feature
- PASSED: synthetic end-to-end V4 integrationใน temporary directory — Development/Holdout 80/20, common OOF coverage, Rank checkpoint, Holdout evaluation, frozen Full-data refit `nobs=100`, RESULT/COEFFICIENT/bundle และ sidecar exportsครบ; statsmodelsออก covariance-rank warningsจาก synthetic collinearityแต่ไม่มี test failure
- PASSED: prior credential literalไม่อยู่ใน changed text/current source defaultว่าง โดยตรวจแบบไม่แสดงค่า
- PASSED: final intended Git statusมีเฉพาะ Training, tests, support files และ docsที่ระบุ ไม่มี pycache/artifact/logใหม่

### ข้อจำกัดและสิ่งที่ยังไม่ได้ทดสอบ

- NOT TESTED: SQL Server connectivity, actual current snapshot, ODBC/runtime, full candidate runtime/memory และตัวเลข metrics V4จริง
- NOT TESTED: การสร้าง artifacts/sidecars/logจริงจาก end-to-end controlled run และ collisionกับไฟล์จริง
- NOT TESTED: Streamlit DB catalog/rendering; Predictor compatibilityตรวจด้วย synthetic bundle ไม่ใช่ production joblib
- NOT IMPLEMENTED ในรอบ core V4: ปรับ `analyze_used_car_ols.py`/`plot_used_car_ols.py` ให้อ่าน sidecarsและเปลี่ยน historical in-sample/mean-fold labels เพราะ approved agent reviewเป็น read-onlyและ main phaseจำกัด Training/new files เอกสาร V4กำกับความหมายไว้แล้ว ควรทำ phaseถัดไปก่อนใช้ diagnosticsกับ V4 run
- SECURITY LIMITATION: current sourceไม่มี hardcoded default แต่ secretใน Git history/tracked bytecodeเดิมยังอยู่ การ rotate/untrack/history cleanupยังต้องขออนุมัติแยก
- DATA LIMITATION: duplicate policyตรวจ exact normalized listing ID/URL; รถคันเดิมที่ถูก relistด้วย identityใหม่ยังอาจข้าม splitได้
- STATISTICAL LIMITATION: Holdoutวัด Development checkpoint ไม่ใช่ exported Full-data refit modelตามที่ metadataระบุ; Holdoutห้ามใช้ปรับ modelหลังดูผล
- OPERATIONAL LIMITATION: principal outputsและ sidecarsยังเขียนทีละไฟล์ ไม่ใช่ atomic transaction หาก controlled runหยุดกลางทางต้องเก็บหลักฐาน, ไม่ใช้ partial artifacts และเริ่มใหม่ด้วย RUN_ID ใหม่

### Controlled training command — ยังไม่รัน

เมื่อเจ้าของอนุมัติและตั้ง DB environmentครบ ให้รันจาก project root:

```bash
USED_CAR_RUN_ID="$(date '+%Y%m%d_%H%M%S')" ./scripts/run_training_v4.sh
```

Scriptใช้ RUN_IDเดียวกับ `logs/train_<RUN_ID>.log` และ artifacts ตรวจไม่ overwrite, รักษา Python exit status และตรวจชื่อไฟล์หลังจบ Raw logถูก `.gitignore` และต้องตรวจ secret/row-level contentก่อนแชร์เสมอ

### สถานะการส่งมอบ

Implementation และ synthetic validationเสร็จ แต่สถานะยังเป็น **awaiting owner approval for controlled V4 training run** ห้าม Commit/Pushจนได้รับอนุมัติ
