<?php

namespace Modules\WorkOrders\Console;

use Illuminate\Console\Command;
use Illuminate\Support\Facades\Storage;
use League\Csv\Reader;
use League\Csv\Statement;
use Modules\WorkOrders\Entities\WorkOrder;

class WorkOrdersImportCsvCommand extends Command
{
    protected $signature = 'workorders:import-csv {path} {--dry-run} {--validate-only}';
    protected $description = 'Import Work Orders from a CSV with headers: id,title,status,priority,customer_id,scheduled_at,created_at,updated_at';

    public function handle(): int
    {
        $path = $this->argument('path');
        if (!Storage::exists($path)) {
            $this->error("CSV not found at storage/app/{$path}");
            return 1;
        }
        $stream = Storage::readStream($path);
        $csv = Reader::createFromStream($stream);
        $csv->setHeaderOffset(0);
        $stmt = (new Statement());
        $records = $stmt->process($csv);

        $count = 0;
        foreach ($records as $row) {
            $count++;
            if ($this->option('dry-run')) continue;

            $wo = WorkOrder::firstOrNew(['id' => $row['id'] ?? null]);
            $wo->title = $row['title'] ?? '';
            $wo->status = $row['status'] ?? 'draft';
            $wo->priority = $row['priority'] ?? 'normal';
            $wo->customer_id = $row['customer_id'] ?? null;
            $wo->scheduled_at = $row['scheduled_at'] ?? null;
            $wo->created_at = $row['created_at'] ?? now();
            $wo->updated_at = $row['updated_at'] ?? now();
            $wo->save();
        }

        $creates=0; $updates=0; $errors=[];
        // Re-read for validation/diff
        $stream = \Illuminate\Support\Facades\Storage::readStream($path);
        $csv2 = \League\Csv\Reader::createFromStream($stream); $csv2->setHeaderOffset(0);
        foreach ($csv2->getRecords() as $row) {
            $row_errors = [];
            $id = $row['id'] ?? null;
            if (!$id) { $errors[] = 'Missing id'; continue; }
            $exists = \Modules\WorkOrders\Entities\WorkOrder::find($id);
            if ($exists) { $updates++; } else { $creates++; }
            // basic validation
            foreach (['title','status','priority'] as $k) {
                if (!isset($row[$k]) || $row[$k]==='') { $errors[] = "Row id={$id}: missing {$k}"; }
            }
        }
        $this->info("Processed {$count} rows from {$path}" . ($this->option('dry-run') ? ' (dry-run)' : ''));
        $this->info("Would create: {$creates}, would update: {$updates}");
        // Write error CSV
        if ($errors) {
            $errorRows = ["id,error"];
            foreach ($errors as $e) {
                $errorRows[] = ",".str_replace(',', ';', $e);
            }
            \Illuminate\Support\Facades\Storage::put('exports/workorders_import_errors.csv', implode("\n", $errorRows));
            $this->warn('Wrote error CSV to storage/app/exports/workorders_import_errors.csv');
            $this->warn("Validation issues (showing up to 10):");
            foreach (array_slice($errors,0,10) as $e) { $this->line(" - ".$e); }
        }
        return 0;
    }
}
