# ComplianceIQ — Install

1. Copy to `Modules/ComplianceIQ`.
2. `php artisan module:enable ComplianceIQ`
3. Run migrations & seed permissions:
   - `php artisan migrate`
   - `php artisan db:seed --class="Modules\\ComplianceIQ\\Database\\Seeders\\CompliancePermissionSeeder"`
4. Ensure roles exist: **Admin**, **Compliance Officer**, **Auditor**.
5. Add scheduler entries for:
   - GenerateComplianceSummary (cron: `0 2 * * *`)
   - TamperCheckHashes (cron: `15 2 * * *`)
6. Visit **Admin → Compliance** to use reports & logs.


## Optional: PDF & AI Provider
- For PDF export, install a renderer:
  - `composer require barryvdh/laravel-dompdf` (recommended) **or** ensure `dompdf/dompdf` is available.
  - Set `report_export.default=pdf` in `config/complianceiq.php`.
- AI provider:
  - Default is a Null provider (placeholders). To enable OpenAI adapter, set:
    - `COMPLIANCEIQ_AI_DRIVER=openai`
  - Then wire your OpenAI client in `Services/AI/OpenAIComplianceAI.php`.
