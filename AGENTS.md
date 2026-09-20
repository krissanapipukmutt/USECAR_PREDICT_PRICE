# USECAR_PREDICT_PRICE — Working instructions

## เริ่มงานจากหลักฐานจริง

- ใช้ไฟล์จริงใน Workspace เป็นฐาน อ่าน [Docs/PROJECT_CONTEXT.md](Docs/PROJECT_CONTEXT.md) และ [Docs/WORK_LOG.md](Docs/WORK_LOG.md) ก่อนทำงาน
- ตรวจ `git status --short --branch`, branch/upstream และ diff ก่อนแก้ไฟล์ อย่าเขียนทับงานที่ผู้ใช้ทำค้างไว้
- เมื่อต้องเทียบ GitHub ให้ตรวจ remote ปัจจุบันด้วย read-only operation; `origin/master` ในเครื่องอย่างเดียวไม่ยืนยันสถานะ GitHub ล่าสุด หากต่างกัน ให้รายงานความแตกต่างก่อนเลือกเวอร์ชัน ห้าม pull/reset/merge เพื่อทำให้ตรงกันเอง
- ก่อนแก้ Application Code ต้องอ่าน Python scripts ปัจจุบันทั้งหมดของโปรเจกต์และ Data Dictionary ใน `Docs` รวมถึงตรวจ secrets และไฟล์ที่ไม่ควร Commit โดยไม่แสดงค่าลับ
- แยกข้อเท็จจริงจากโค้ด, ผลจาก artifacts เดิม, ผลทดสอบที่รันใหม่ และสิ่งที่ยังไม่ตรวจ ให้ชัดเจน อย่าอ้างว่า database หรือ model ใช้งานได้จาก syntax check เพียงอย่างเดียว

## โครงสร้างเอกสารที่ต้องรักษา

- `AGENTS.md`: กติกาการทำงานร่วมกัน
- `Docs/PROJECT_CONTEXT.md`: Architecture, Database, Training, Prediction, UI, artifacts และประเด็นคงค้าง
- `Docs/WORK_LOG.md`: ความคืบหน้า ไฟล์ที่แก้ ผลตรวจ/ทดสอบ และสิ่งที่ยังไม่ได้ทดสอบ
- ใช้ `Docs` ตัว D ใหญ่เท่านั้น ห้ามสร้าง `docs` ซ้ำ
- Data Dictionary เดิมคือ `Docs/Used_Car_OLS_Data_Dictionary_Thai_Clear.xlsx` ห้ามย้าย เปลี่ยนชื่อ แก้ หรือเขียนทับโดยไม่ได้รับอนุญาต
- เมื่อแก้โค้ดทุกครั้ง ต้องอัปเดต `Docs/WORK_LOG.md` ในงานเดียวกัน พร้อมเหตุผล รายชื่อไฟล์ ผลทดสอบ และข้อจำกัด; อัปเดต Project Context เมื่อพฤติกรรมหรือโครงสร้างเปลี่ยน

## ขอบเขตที่ต้องได้รับอนุญาต

- ห้ามเปลี่ยน OLS, target/formula, preprocessing/feature selection, Price Filter/Outlier rules, Database Schema, Output Schema หรือ Predict Logic โดยไม่ได้รับอนุญาต
- ห้ามแก้ `STG_USED_CAR` หรือข้อมูลต้นทาง รวมถึงไฟล์ raw data ใน `ONE2CAR`
- อย่าเพิ่ม DDL/DML หรือรัน training เพื่อประกอบการสำรวจสถานะ งาน training จะอ่าน DB และสร้าง artifacts/report ใหม่ จึงต้องอยู่ในขอบเขตงานที่ได้รับมอบหมายจริง
- งานเริ่มต้นวันที่ 2026-09-20 อนุญาตเฉพาะการตรวจสถานะและสร้างเอกสารสามไฟล์นี้ ไม่ใช่การอนุญาตแก้ application หรือแก้ประเด็นคงค้างทั้งหมด
- ห้าม Commit หรือ Push อัตโนมัติ ต้องรอผู้ใช้อนุมัติ ห้ามลบไฟล์ออกจาก tracking หรือ rewrite Git history โดยไม่ได้รับอนุญาต

## Secrets และข้อมูลที่ไม่ควรเผยแพร่

- ห้ามแสดงหรือคัดลอกรหัสผ่าน/token/connection string ที่มี credentials ลงข้อความ เอกสาร logs หรือ tests; รายงานเฉพาะชื่อ key, ชื่อไฟล์ และบรรทัด
- การอ่านไฟล์ต้อง redact secret assignment ทั้ง expression รวมถึง string ที่อยู่คนละบรรทัดกับชื่อ key และไม่ dump environment หรือ bytecode
- พบ hardcoded default ของ `USED_CAR_DB_PASSWORD` ใน `train_used_car_ols.py` และการติดตาม bytecode/data/artifacts เดิมแล้ว ดูรายละเอียดใน Project Context; ห้ามตีความว่าไฟล์ที่ถูก track อยู่แล้วปลอดภัยสำหรับเผยแพร่
- ห้าม Commit secrets, ข้อมูลรถที่ไม่ควรเผยแพร่, `.venv` หรือ Model ขนาดใหญ่โดยไม่ได้รับอนุญาต ตรวจทั้ง staged paths และ staged content ก่อนเสนอ Commit
- อย่าใช้ `git add .` โดยไม่ตรวจรายการ ปัจจุบันไม่มี root `.gitignore`; การเพิ่ม ignore ในอนาคตไม่ลบไฟล์ที่ถูก track หรือ secret ในประวัติย้อนหลัง
- ห้ามเปลี่ยน credential, ลบ artifacts หรือ rewrite history เพื่อแก้ผลตรวจนี้เอง ให้จัดเป็นงานแยกตามการอนุญาตของผู้ใช้

## จุดเริ่มต้นของโค้ดและการตรวจสอบ

- `train_used_car_ols.py`: SQL Server read, preprocessing, OLS/CV/ranking, CSV/joblib exports
- `predict_used_car_ols.py`: ใช้ saved Rank‑1 bundle สำหรับ CLI และฟังก์ชันที่ UI เรียก
- `app_used_car.py`: Streamlit UI และ catalog จาก STG แบบอ่านอย่างเดียว
- `analyze_used_car_ols.py`, `plot_used_car_ols.py`: diagnostics ของ artifacts กับ snapshot ที่อ่านจาก DB
- สำหรับงาน Streamlit ใช้ skill `developing-with-streamlit` ที่มีอยู่ แต่กติกาของผู้ใช้เรื่องขอบเขตและการอนุญาตมีลำดับสูงกว่า คำแนะนำ cleanup จาก skill ไม่ใช่สิทธิแก้โค้ดนอกงาน
- เลือกการตรวจที่ไม่มี side effects สำหรับงานเอกสาร เช่น AST parse, CSV header/key checks, Git diff และ checksum; อย่า import/run entry points หรือ deserialize joblib เพียงเพื่อยืนยัน syntax
- เมื่อได้รับอนุญาตแก้โค้ด ให้ทดสอบตามผลกระทบจริงโดยป้องกัน DB/source data และการเขียนทับ artifacts ที่มีอยู่ บันทึกผลที่ผ่าน/ไม่ผ่าน/ไม่ได้รันตามจริง
