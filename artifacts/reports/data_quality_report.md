# Data-quality report

**Fallback used:** Yes. No qualifying original pharma CSV was available; all downstream results are demo-only.

## Provenance

- Source type: `fallback_synthetic`
- Source path: `data/synthetic/fallback_pharma_sales.csv`
- Reason: No qualifying pharma CSV was found in configured workspace search roots.

## Structure

- Shape: **9,000 rows × 18 columns**
- Exact duplicates: **0 (0.00%)**
- Total missing cells: **0**
- Date coverage: **2024-01-01 to 2025-06-01** (18 months)
- Invalid dates: **0**

## Highest missingness

- `distributor`: 0 (0.00%)
- `customer`: 0 (0.00%)
- `city`: 0 (0.00%)
- `country`: 0 (0.00%)
- `latitude`: 0 (0.00%)
- `longitude`: 0 (0.00%)
- `channel`: 0 (0.00%)
- `subchannel`: 0 (0.00%)
- `product_name`: 0 (0.00%)
- `product_class`: 0 (0.00%)

## Coverage

- `country`: 4 distinct (Brazil, India, United Kingdom, United States)
- `city`: 12 distinct (Bengaluru, Birmingham, Brasilia, Chicago, Delhi, Houston, London, Manchester)
- `product_name`: 8 distinct (Cardiovex, Dermasol, Glycoban, Immunara, Neurocalm, Oncora, Renapro, Respira)
- `product_class`: 8 distinct (Cardiovascular, Dermatology, Diabetes, Immunology, Neurology, Oncology, Renal, Respiratory)
- `sales_manager`: 8 distinct (MGR_01, MGR_02, MGR_03, MGR_04, MGR_05, MGR_06, MGR_07, MGR_08)
- `sales_team`: 4 distinct (Team_Apex, Team_Bridge, Team_Catalyst, Team_Delta)
- `channel`: 3 distinct (Clinic, Hospital, Retail)
- `subchannel`: 6 distinct (Independent Pharmacy, Pharmacy Chain, Primary Care, Private Hospital, Public Hospital, Specialist Clinic)

## Field lineage

Original source columns and their canonical mappings are recorded in `data_quality_report.json`. Synthetic enrichment and anomaly-label fields are added only after this source-level profile.
