# Feature Approval Registry — Initial Owner Approval

Registry ที่ระบบใช้จริงคือ `config/feature_approval_registry.json` รุ่น `2026-09-20-initial-owner-approved-1` สถานะ `APPROVED` การอนุมัตินี้ให้สิทธิเฉพาะการเข้าสู่ Candidate Universe แต่ละ feature ยังต้องผ่าน fold-local eligibility, preprocessing และ feature selection ตาม V4 และยังไม่มีการอนุมัติให้รัน Controlled/Full Training

## หลักฐาน Actual Schema

- Local metadata-only report: `output/analysis/schema_review/actual_schema_20260920_130140.json`
- Source: `dbo.STG_USED_CAR`
- ตรวจเมื่อ: 2026-09-20 13:01:40 +07:00
- Columns: 27
- Schema fingerprint: `ef6ea4a011dac3266e727440d547efa018aa9a787a491a755077f467aec53a5c`
- Report fingerprint ถูกคำนวณซ้ำจาก column metadataแล้วตรงกัน ไม่มีข้อมูลรถรายแถว

รายงานนี้ยืนยันโครงสร้าง ณ เวลาที่ตรวจเท่านั้น Schema Gate ต้องตรวจใหม่ทุก run

## สถานะที่เจ้าของอนุมัติ

| สถานะ | คอลัมน์ | SQL type จริง | Prediction-time evidence / เหตุผล |
| --- | --- | --- | --- |
| TARGET | `price` | `bigint` | Target เท่านั้น ไม่ใช่ predictor |
| MANDATORY_METADATA | `PCS_DATE` | `date` | Snapshot metadata ไม่ใช่ predictor |
| APPROVED | `brand`, `model`, `sub_model` | `nvarchar` | มี CLI/UI input; ยังต้องผ่าน fold-local eligibility |
| APPROVED | `model_year` | `smallint` | มี CLI/UI input; numeric compatible |
| APPROVED | `mileage` | `bigint` | มี CLI/UI input; numeric compatible |
| APPROVED | `fuel_type`, `transmission` | `nvarchar` | มี CLI/UI input; categorical compatible |
| APPROVED | `engine_size` | `decimal` | มี CLI/UI input; numeric compatible |
| APPROVED | `body_type`, `color` | `nvarchar` | มี CLI/UI input; categorical compatible |
| APPROVED | `number_of_seats` | `int` | มี CLI/UI input; numeric compatible |
| PENDING_REVIEW | `province`, `location`, `seller_name`, `seller_type` | `nvarchar` | Input contract/business meaningยังไม่อนุมัติ จึงห้ามเข้า Candidate Universe |
| EXCLUDED — identity | `listing_id`, `source_url` | `bigint`, `nvarchar` | ใช้ duplicate grouping/audit ไม่ใช่ predictor |
| EXCLUDED — leakage risk | `title`, `raw_price`, `description` | `nvarchar` | อาจมี target priceหรือข้อมูลที่ไม่เหมาะกับ prediction contract |
| EXCLUDED — raw/technical | `raw_mileage`, `dealer_slug`, `seller_url`, `image_url`, `scraped_at` | `nvarchar`, `datetime2` | Raw duplicate, URL หรือ collection/technical metadata |

ไม่มีคอลัมน์ New/Missing เมื่อเทียบ Actual Schema กับ Registry ฉบับนี้

## Type Compatibility

- SQL `nvarchar` → registry type `text`
- SQL `smallint`, `bigint`, `int`, `decimal` → registry type `numeric`
- SQL `date`/`datetime2` → SQL metadata type `datetime`
- `PCS_DATE` จาก SQL `date` อาจถูก Pandas materializeเป็น `datetime64` หรือ Python `date` ใน object series Runtime gateจึงรองรับ registry typesทั้ง `datetime` และ `text`; regression testยืนยันสองเส้นทางนี้
- Numeric feature ที่ภายหลังเปลี่ยนเป็น textยังต้องผ่าน numeric-like inferenceจาก fold train ห้ามถือว่า type allowanceบังคับให้ featureผ่าน eligibility

## Offline Validation

คำสั่งที่ใช้โดยไม่เชื่อม SQL Serverและไม่เรียก Training `main()`:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/validate_schema_registry.py \
  output/analysis/schema_review/actual_schema_20260920_130140.json
```

ผล: valid, fingerprintตรง, mandatoryครบ, 11 Approved compatible, 4 Pendingไม่ authorized, Target/Metadata/Excludedไม่ authorized และไม่มี New/Missing columns

หาก Approved featureหายหรือชนิดไม่รองรับ validatorและ runtime gateต้องรายงานและไม่อนุญาตให้ใช้ ไม่มีการแก้ Registryหรืออนุมัติ featureอัตโนมัติ
