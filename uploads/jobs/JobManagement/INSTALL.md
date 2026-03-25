# Install & Smoke Test

## Prereqs
- Laravel app with `nwidart/laravel-modules` installed and enabled.

## Install
```bash
php artisan module:enable JobManagement || true
php artisan migrate
php artisan route:list | grep -E 'engineerings|work|recurring-work|work-calendar|meter'
```

## Smoke
- GET the index routes:
  - `/engineerings`
  - `/work`
  - `/recurring-work`
- Trigger an export: `/engineerings/export`
- Try barcode scan endpoint: `/meter/scan-barcode`

## Menu Integration (hint)
- Read `Config/config.php['menu']` for suggested label/icon/routes.
- In apps like Worksuite/QuickAI bridges, inject a sidebar link to `engineerings.index`.
