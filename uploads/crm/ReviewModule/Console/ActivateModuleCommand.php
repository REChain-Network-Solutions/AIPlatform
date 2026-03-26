<?php

namespace Modules\ReviewModule\Console;

use Illuminate\Console\Command;

class ActivateModuleCommand extends Command
{
    /**
     * The name and signature of the console command.
     *
     * You can run: php artisan module:activate ReviewModule
     */
    protected $signature = 'module:activate {module_slug?}';

    /**
     * The console command description.
     */
    protected $description = 'Activate ReviewModule module (Worksuite-compatible stub).';

    public function handle(): int
    {
        $this->info('ReviewModule: activation stub complete.');
        return self::SUCCESS;
    }
}
