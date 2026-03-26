# ComplianceIQ — Workflows

**Lifecycle:** Capture → Evidence Log → Filter → Report → Review → Sign-off

- **Capture:** Upstream systems write payloads; store immutable digests in `compliance_hashes`.
- **Evidence Log:** View/filter in Admin → Compliance Logs.
- **Report:** Build by period + template; export CSV; AI summary job can enrich report.
- **Review & Annotate:** Team adds notes on the report thread.
- **Sign-off:** Authorized roles finalize; `signed_off_by/at` recorded.

**Jobs**
- `GenerateComplianceSummary(reportId)` — builds executive summary via AI provider hook.
- `AnalyzeComplianceAnomalies(reportId)` — flags issues as annotations.
- `TamperCheckHashes()` — recomputes digests and appends verification results.

**RBAC**
- Admin, Compliance Officer: full access; Auditor: read-only (reports + logs).
