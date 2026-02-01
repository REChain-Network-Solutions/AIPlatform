<?php

namespace App\Extensions\AISocialMedia\System\Console\Commands;

use App\Extensions\AISocialMedia\System\Console\Commands\Concerns\HasDynamicHandle;
use App\Extensions\AISocialMedia\System\Console\Commands\Concerns\HasGenerateQuery;
use Illuminate\Console\Command;

class GeneratePostMonthlyCommand extends Command
{
    use HasDynamicHandle;
    use HasGenerateQuery;

    protected $signature = 'app:generate-post-monthly';

    protected $description = 'Generate post for all users that choose monthly post';

    public function handle(): void
    {
        $this->dynamicHandle('month');

        $this->info('Posts generation monthly jobs dispatched for all users.');
    }
}
