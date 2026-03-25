# Treasury — Master-Treasury-V1.0
Scope: Bank accounts, reconciliations, payment runs, liquidity forecasts. AI-ready and permissioned.
Install: place under Modules/Treasury, migrate, seed PermissionSeeder & MenuSeeder, grant `treasury.access`.
API: GET /api/treasury/accounts, GET /api/treasury/unreconciled
AI: POST /treasury/ai/forecast, POST /treasury/ai/recon-suggest


## V1.1 Additions
- Bank feed CSV import (`POST /treasury/feed/upload`)
- Reconciliation suggestions API (`POST /api/treasury/match-suggest`)
- Payment Runs (`POST /treasury/payment-runs`, `GET /treasury/payment-runs/{id}`)
- Contracts for `LedgerPosterInterface` + `BankFeedImporterInterface`
- Services: `CsvBankFeedImporter`, `MatchingService`, `PaymentRunService`


## V1.2 Additions
- **Ledger posting bridge**: binds `LedgerPosterInterface` to Accounting's `LedgerInterface` (when present).
- **Payment file export**: ABA (AU) export for payment runs → `GET /treasury/payment-runs/{id}/export/aba`.
- **Reconciliation Rules UI**: create simple LIKE-match rules at `/treasury/rules`.
- Notes: Poster is no-op if AccountingSuite not installed; safe to deploy standalone.


## V1.3 Additions
- Auto-categorization on bank feed import using `reconciliation_rules`.
- New columns on `bank_transactions`: `account_code`, `category`.


## V1.4 Additions
- Exporters: **SEPA pain.001** and **CSV batch** alongside ABA.
- Payment Runs UI: list and detail pages with export buttons.


## V1.5 Additions (milestone)
- Console command `treasury:post-payment-run {run_id}` to post runs to GL via Accounting.
- Registered console provider to expose commands.
- Unit test skeleton for `MatchingService`.
- Treasury now reaches Milestone **V1.5** — export-ready, rule-driven, and GL-integrated.
