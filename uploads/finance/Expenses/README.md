# Expenses — Master-Expenses-V1.0
Ported from donor repo (schema/models) and wrapped as an nWidart module with RBAC, routes, views, AI stubs.
Install: place under Modules/Expenses, migrate, seed PermissionSeeder & MenuSeeder, grant `expenses.access`.


## V1.1 Additions (2025-10-01)
- Approvals workflow (submit → approve → reimburse) with timestamps.
- Receipt uploads with optional AI OCR stub (stores to `storage/app/public/receipts` when default disk=public).
- Show page for each expense with actions and receipts list.
- Routes added under `/expenses/{id}/*`.
