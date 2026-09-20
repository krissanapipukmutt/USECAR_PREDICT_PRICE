# Feature Approval Registry — Initial Proposal

ไฟล์ที่ระบบใช้จริงคือ `config/feature_approval_registry.json` รุ่น `2026-09-20-draft-1` และมีสถานะ `PENDING_OWNER_APPROVAL` เอกสารนี้เป็น inventory สำหรับตรวจรับรองครั้งแรก ไม่ใช่หลักฐานว่าได้ตรวจ schema ปัจจุบันจาก SQL Server และยังไม่อนุญาตให้ Full Training

## หลักฐานและขอบเขต

รายการเริ่มต้นมาจาก exclusion rules ใน source code, input controls ของ UI, historical selected-feature metadata และ header ของ local raw CSV เท่านั้น รอบนี้ไม่ได้เชื่อม SQL Server จึงยังไม่ทราบว่าคอลัมน์ใดเพิ่ม หาย หรือเปลี่ยน datatype ใน `dbo.STG_USED_CAR` ปัจจุบัน

| สถานะเสนอ | คอลัมน์ | เหตุผลย่อ |
| --- | --- | --- |
| TARGET | `price` | Target ที่ทราบเฉพาะ Train/Evaluation ห้ามใช้เป็น predictor |
| MANDATORY_METADATA | `PCS_DATE` | ระบุ latest snapshot ห้ามใช้เป็น predictor |
| PENDING_REVIEW — มี input path ปัจจุบัน | `brand`, `model`, `sub_model`, `model_year`, `mileage`, `fuel_type`, `transmission`, `engine_size`, `body_type`, `color`, `number_of_seats` | เคยเป็น candidate/feature หรือมี CLI/UI input แต่ยังไม่ได้รับ business approval ครั้งแรก |
| PENDING_REVIEW — input contract ยังไม่ชัด | `province`, `location`, `seller_name`, `seller_type` | ต้องยืนยันความหมาย, cardinality และความพร้อมตอน Predict ก่อนอนุมัติ |
| EXCLUDED | `listing_id`, `source_url` | Technical identity ใช้ duplicate grouping/audit เท่านั้น |
| EXCLUDED — leakage risk | `title`, `raw_price`, `description` | อาจมี target price หรือข้อมูลที่ไม่เหมาะกับ prediction-time contract |
| EXCLUDED — raw/technical metadata | `raw_mileage`, `seller_url`, `image_url`, `scraped_at`, `dealer_slug` | Raw duplicate, URL หรือ collection/technical metadata ตาม exclusion policy เดิม |

คอลัมน์จริงที่ไม่อยู่ใน registry จะปรากฏใน Schema Review Report เป็น `PENDING_REVIEW` เสมอ แม้ข้อมูลสมบูรณ์หรือสัมพันธ์กับ target สูง ระบบไม่เขียนชื่อใหม่กลับเข้า registry และไม่ส่งเข้า preprocessing, feature selection, CV หรือ refit

## ขั้นตอนตรวจรับรองของเจ้าของ Project

1. อนุญาต read-only schema review กับ SQL Server เพื่อสร้าง `schema_review_<RUN_ID>.json`; รอบนี้ยังไม่ได้ทำ
2. ตรวจ New/Missing/Type-changed columns, leakage risk และ prediction-time availability จากรายงาน โดยไม่ตรวจจากข้อมูลรถรายแถว
3. เปลี่ยนเฉพาะ feature ที่ยืนยันแล้วจาก `PENDING_REVIEW` เป็น `APPROVED`; คง target/metadata/exclusions และเหตุผลไว้
4. ตรวจ `allowed_schema_types` โดยเฉพาะ numeric feature ที่อนุญาต text ได้เฉพาะ numeric-like conversion ซึ่งจะเรียนและตรวจใน fold train
5. เปลี่ยน `initial_approval_status` เป็น `APPROVED` ผ่านการแก้ไฟล์ที่ review ย้อนหลังได้ แล้วรัน tests ใหม่
6. จึงพิจารณาอนุมัติ Controlled V4 Training แยกต่างหาก ระบบจะบันทึก registry version/checksum, schema fingerprint และ approved features ลง evaluation metadata และ model bundle

การอนุมัติใน registry เป็นเพียงสิทธิให้ feature เข้าสู่ candidate universe Feature ยังต้องผ่าน fold-local eligibility และอาจไม่ถูกเลือกเข้าโมเดล
