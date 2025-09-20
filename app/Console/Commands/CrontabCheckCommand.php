<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use Illuminate\Support\Facades\Log;

class CrontabCheckCommand extends Command
{
    protected $signature = 'app:crontab-check';

    protected $description = 'Check if the cron job is running';

    public function handle(): void
    {

        $currentTime = now()->toDateTimeString();

        Log::info("Cron job is running at: {$currentTime}");

        cache()->put('crontab_check', now());
    }
}
