<?php

namespace App\Console\Commands\Clear;

use App\Helpers\Classes\Helper;
use Illuminate\Console\Command;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Log;

class ClearJobTableCommand extends Command
{
    /**
     * The name and signature of the console command.
     *
     * @var string
     */
    protected $signature = 'app:clear-job-table';

    /**
     * The console command description.
     *
     * @var string
     */
    protected $description = 'Command description';

    /**
     * Execute the console command.
     */
    public function handle()
    {
        if (Helper::appIsNotDemo()) {
            return;
        }

        // Log::info('Clearing job table');

        DB::table('jobs')->truncate();

        DB::table('failed_jobs')->truncate();

        // Log::info('Job table cleared successfully');
    }
}
