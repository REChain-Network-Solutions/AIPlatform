<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use Illuminate\Support\Facades\Cache;

class CrontabCheckCommand extends Command
{
    protected $signature = 'app:crontab-check';

    protected $description = 'Check if the cron job is running';

    public function handle(): void
    {

        $currentTime = now()->toDateTimeString();
        Cache::put('crontab_check', now());
    }
}
