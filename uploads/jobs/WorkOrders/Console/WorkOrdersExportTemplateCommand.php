<?php

namespace Modules\WorkOrders\Console;

use Illuminate\Console\Command;
use Illuminate\Support\Facades\Storage;

class WorkOrdersExportTemplateCommand extends Command
{
    protected $signature = 'workorders:export-template {--path=exports/workorders_template.csv}';
    protected $description = 'Export a CSV template for Work Orders import.';

    public function handle(): int
    {
        $headers = 'id,title,status,priority,customer_id,scheduled_at,created_at,updated_at';
        $example = '1,Change fan belt,scheduled,normal,42,2025-10-10 09:00:00,2025-10-01 12:00:00,2025-10-01 12:00:00';
        $csv = $headers."\n".$example."\n";
        $path = $this->option('path');
        Storage::put($path, $csv);
        $this->info("Template written to storage/app/{$path}");
        return 0;
    }
}
