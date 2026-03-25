# WorkOrders (Smart FSM → nWidart)

This module is a **first-pass port** of Smart FSM's Field Service domain into an nWidart module.

### Included (usable code first)
- Eloquent models: WorkOrder, WOType, WORequest, WOServiceAppointment, WOServiceTask, WOServicePart, ServiceTask, ServicePart
- Migrations: work_orders, service_tasks, service_parts (others to add as you validate)
- Controllers: WorkOrderController, WOTypeController, WORequestController, ServicePartController (namespaced)
- Views: related Blade templates pulled over (20 files)

### Install
1. Ensure your Laravel app has `nwidart/laravel-modules` installed.
2. Copy `Modules/WorkOrders` into your app root.
3. Run migrations:
   ```bash
   php artisan migrate
   ```
4. Visit `/workorders` while logged in.

### Next Pass Targets
- Add migrations for `wo_types`, `wo_requests`, `wo_service_appointments`, `wo_service_tasks`, `wo_service_parts` (if not already covered).
- Port Form Request classes and Policies; wire permissions (Spatie).
- Replace any remaining `view()` calls to use namespace `workorders::`.
- Add routes for controllers (index/create/store/show/edit/update/destroy).
- Write seeders for demo data and factories for testing.


---
## Pass 2
- Added migrations: wo_types, wo_requests, wo_service_appointments, wo_service_tasks, wo_service_parts
- Expanded routes: full resource routes for orders/types/requests/appointments/tasks/parts (+ service catalog)
- Created controller stubs for appointments, tasks, parts, and service-task controller


---
## Pass 3
- Added FormRequests for Work Orders, Appointments, Tasks, Parts
- Added permissive WorkOrderPolicy stub (swap to Gate/Spatie as needed)
- Implemented real CRUD handlers for Work Orders + line items + appointments
- Added minimal views (index/create/edit/show)
- Added WorkOrdersSeeder with demo data (labor + part + appointment)

**Seed command:**
```bash
php artisan db:seed --class="Modules\\WorkOrders\\Database\\Seeders\\WorkOrdersSeeder"
```


---
## Pass 4
- Registered `WorkOrderPolicy` via Gate in ServiceProvider
- Added Spatie roles/permissions seeder (admin/technician/viewer)
- Implemented totals recalculation service; hooked into line-item changes
- Added model factories and a demo dataset seeder

**Seed commands:**
```bash
php artisan db:seed --class="Modules\\WorkOrders\\Database\\Seeders\\WorkOrdersPermissionSeeder"
php artisan db:seed --class="Modules\\WorkOrders\\Database\\Seeders\\WorkOrdersDemoSeeder"
```


---
## Pass 5 (Production polish)
- **Route protection**: added Spatie `permission:*` middleware to all resources
- **Domain events**: Created `WorkOrderCreated/Updated/Completed` + listener `SendWorkOrderWebhook`
- **Webhook config**: set `WORKORDERS_WEBHOOK_URL` in `.env` to receive JSON payloads
- **Event provider**: auto-registered to wire events → webhook
- **SSO/JWT middleware stub**: `Http/Middleware/AcceptSsoJwt.php` for Worksuite handoff
- **Tests**: basic Feature test scaffold (adjust to your base TestCase)

**Enable webhook**: set `WORKORDERS_WEBHOOK_URL=https://your-listener.example/webhooks/workorders`

**Note**: For per-action permission middleware on `Route::resource`, if your Laravel version doesn’t support arrays, define explicit routes per action.


## Settings
- UI: `/admin/workorders/settings` (requires `workorders.settings`)
- Toggle API auth, set webhook URL + retry/backoff.
- Values cache for convenience. For permanence: `php artisan vendor:publish --tag=workorders-config` then edit `config/workorders.php` or .env:

```
WORKORDERS_WEBHOOK_URL=
WORKORDERS_WEBHOOK_RETRIES=3
WORKORDERS_WEBHOOK_BACKOFF=5
```

## CSV
- Export: `php artisan workorders:export-csv --path=exports/wo.csv`
- Import: `php artisan workorders:import-csv exports/wo.csv` (add `--dry-run` to preview)


## Queueable Webhooks
- Webhooks now dispatch via a **queue job** with retries/backoff.
- Failures are recorded in `workorders_failed_webhooks`.
- Run migrations after upgrading.

## Self test
```
php artisan workorders:selftest
php artisan workorders:selftest --queue   # also dispatches a test job
```


## Assignments widget
- Include in any Work Order page (replace `{{ $workOrder->id }}` accordingly):
```
<x-workorders::assignments :work_order_id="$workOrder->id" />
```
This renders a live widget and a button that jumps to the Contractors assignment screen pre-filled with the Work Order ID.
