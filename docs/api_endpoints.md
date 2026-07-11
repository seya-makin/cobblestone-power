# Data Source API Reference

## SMARD API (Primary — no authentication required)

**Base URL:** `https://www.smard.de/app/chart_data/`

**How it works:** First fetch the index file to get available timestamps, then fetch each data chunk.

| Series | Filter ID | Index Endpoint | Data Endpoint |
|---|---|---|---|
| DA Price DE-LU | 4169 | /4169/DE-LU/index_en.json | /4169/DE-LU/4169_DE-LU_hour_{ts}.json |
| Total Load | 410 | /410/DE/index_en.json | /410/DE/410_DE_hour_{ts}.json |
| Wind Onshore | 123 | /123/DE/index_en.json | /123/DE/123_DE_hour_{ts}.json |
| Wind Offshore | 3791 | /3791/DE/index_en.json | /3791/DE/3791_DE_hour_{ts}.json |
| Solar PV | 125 | /125/DE/index_en.json | /125/DE/125_DE_hour_{ts}.json |

**Response format:** `[[unix_timestamp_ms, value], ...]`

**Example:**
GET https://www.smard.de/app/chart_data/4169/DE-LU/index_en.json
Returns list of available timestamps.

GET https://www.smard.de/app/chart_data/4169/DE-LU/4169_DE-LU_hour_1672531200000.json
Returns [[1672531200000, 89.42], [1672534800000, 81.33], ...]

**Units:** Prices in EUR/MWh, generation and load in MW

---

## ENTSO-E Transparency API (Fallback — API key required)

**Base URL:** `https://transparency.entsoe.eu/api`
**Authentication:** securityToken={ENTSOE_API_KEY} as query parameter
**Registration:** transparency.entsoe.eu (free, approval 24-48 hours)

| Data | Document Type | Description |
|---|---|---|
| DA Prices | A44 | Day-ahead Prices 12.1.D |
| Load Forecast | A65 | Day-ahead Total Load Forecast 6.1.B |
| Wind/Solar | A69 | Day-ahead Generation Forecasts 14.1.C |
| Nuclear | A80 | Unavailability of Production Units 15.1.A |
| Cross-border | A09 | Scheduled Commercial Exchanges 12.1.G |

| Bidding Zone | Code | Valid From |
|---|---|---|
| DE-LU | 10Y1001A1001A82H | 2018-10-01 |
| DE-AT-LU | 10Y1001A1001A63L | Pre 2018-10-01 |
| DE country | 10Y1001A1001A83F | All periods |

**Note:** SMARD retrieves data directly from ENTSO-E under EU Regulation 543/2013 — both sources provide equivalent data quality.
