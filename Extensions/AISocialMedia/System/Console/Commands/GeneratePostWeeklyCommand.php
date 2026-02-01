<?php

namespace App\Extensions\AISocialMedia\System\Console\Commands;

use App\Extensions\AISocialMedia\System\Console\Commands\Concerns\HasDynamicHandle;
use App\Extensions\AISocialMedia\System\Console\Commands\Concerns\HasGenerateQuery;
use Illuminate\Console\Command;

class GeneratePostWeeklyCommand extends Command
{
    use HasDynamicHandle;
    use HasGenerateQuery;

    protected $signature = 'app:generate-post-weekly';

    protected $description = 'Generate post for all users that choosed weekly post';

    public function handle(): void
    {
        $this->dynamicHandle('week');

        $this->info('Posts generation weekly jobs dispatched for all users.');
    }
}
