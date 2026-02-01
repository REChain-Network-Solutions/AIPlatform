<?php

namespace App\Extensions\AISocialMedia\System\Console\Commands;

use App\Extensions\AISocialMedia\System\Console\Commands\Concerns\HasDynamicHandle;
use App\Extensions\AISocialMedia\System\Console\Commands\Concerns\HasGenerateQuery;
use Illuminate\Console\Command;

class GeneratePostDailyCommand extends Command
{
    use HasDynamicHandle;
    use HasGenerateQuery;

    protected $signature = 'app:generate-post-daily';

    protected $description = 'Generate post for all users that choose daily post';

    public function handle(): void
    {
        $this->dynamicHandle('day');

        $this->info('Posts generation daily jobs dispatched for all users.');
    }
}
