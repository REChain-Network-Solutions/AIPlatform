# EInvoice AI Additions

## Invoice Page Integration
Add the AI note widget to your invoice show blade:
```blade
@include('einvoice::components.invoice.ai-note', ['invoiceId' => $invoice->id])
```
This renders a card with the latest AI note and a **Generate AI Note** button.


## AI Invoice Creation
Include a creation widget on a client or invoice-create page:
```blade
@include('einvoice::components.invoice.ai-create', ['clientId' => $client->id ?? null])
```
This lets staff describe the bill in plain English → AI returns a structured draft (JSON). You can then attempt to auto-create a real invoice. If auto-creation fails due to schema mismatch, the draft is still shown so staff can paste it into the native form.


## Module-Local Invoices
This module ships with its own invoice tables when your host app lacks them:
- `einvoice_invoices` (header)
- `einvoice_invoice_items` (lines)

The AI Create flow now inserts items and computes totals in these tables.


## Menu Seeder
This module now ships with a MenuSeeder that inserts an **E-Invoices** link in the `menus` table.

Run:
```bash
php artisan module:seed EInvoice --class=Modules\\EInvoice\\Database\\Seeders\\MenuSeeder
```
Adjust the table name/structure in `MenuSeeder.php` if your host app uses a different menu system.


## Auto-Seed Menu
The EInvoiceServiceProvider now auto-runs the MenuSeeder when you run:
```bash
php artisan module:seed EInvoice
```
No need to specify the class manually.


## Menu Seeder (Idempotent)
Running `php artisan module:seed EInvoice` will upsert the **E-Invoices** menu entry:
- If it exists (by `slug = einvoice`), it will be **updated** (title, url, icon, order).
- If it does not exist, it will be **inserted**.


## Menu Entries
Seeding now upserts **two** entries in `menus`:
- `einvoice` → `/einvoice` (icon `fa fa-file-invoice`, order 50)
- `einvoice-ai` → `/einvoice/settings/ai` (icon `fa fa-robot`, order 51)
