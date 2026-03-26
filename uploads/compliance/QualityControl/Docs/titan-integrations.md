# Titan link-outs (UI only)

This module is intentionally limited to recording site inspections. SWMS/document workflows live in Titan Docs, and compliance workflows live in Titan Compliance.

Config keys:
- config('inspection_integrations.titan_docs.*')
- config('inspection_integrations.titan_compliance.*')

All link-outs are guarded with Route::has() and fall back to URL paths.
