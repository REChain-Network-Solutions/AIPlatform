# JobManagement (nWidart Module)

Job Management suite extracted from the Engineerings module.
Features: Work Requests → Work Orders → Recurring Jobs → Services Catalog → Meters → Files.

## Quick Install
1. Copy `Modules/JobManagement` into your Laravel app (nWidart-enabled).
2. Run:
   ```bash
   php artisan module:list
   php artisan migrate
   php artisan route:list | grep -E 'engineerings|work|recurring-work|work-calendar|meter'
   ```
3. Visit the Job Management area via the `engineerings.*` routes (see below).

## Routes (key)
- engineerings (resource) — main CRUD
- work (resource) — work orders
- recurring-work (resource) — recurring jobs
- work-calendar (resource)
- meter.* — export, multiple-upload, scan-barcode

## Tables
- workrequests, workrequests_items, workrequests_services
- workorders, workorders_files, workorders_recurring
- services, services_category, services_sub_category
- meters

## Notes
- Keep route names/URIs unchanged to avoid controller breakage.
- Permissions: see `Config/permissions.php` and wire to your RBAC.
