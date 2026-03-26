# Inspection Module

## Pass 1 changes
- Adds canonical, sidebar-safe route names.
- Guards sidebar links with `Route::has()` to prevent dashboard crashes.

## Canonical routes
- `recurring-inspection_schedules.*`
- `inspection_schedules.*`
- `schedule-inspection.*`

## Deploy
```bash
cd /home/saassmar/domains/admin.buildsm.art/public_html
php artisan optimize:clear
php artisan route:clear
```
