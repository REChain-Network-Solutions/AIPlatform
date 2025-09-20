<?php

namespace App\Console\Commands\Extension;

use Illuminate\Console\Command;
use Illuminate\Support\Facades\Artisan;
use Illuminate\Support\Facades\Storage;

class ExtensionMigrationCommand extends Command
{
    protected $signature = 'ext:migration {name} {--f|for= : Extension Name}';

    protected $description = 'Create a migration for a specific extension';

    public function handle(): void
    {
        $extension = $this->option('for');

        if (! $extension) {
            $this->error('You must specify an extension name using --for or -f.');

            return;
        }

        $migrationName = $this->argument('name');

        $basePath = base_path("app/Extensions/{$extension}/database/migrations");

        if (! is_dir($basePath)) {
            Storage::disk('extension')->makeDirectory("{$extension}/database/migrations");
        }

        Artisan::call('make:migration', [
            'name'   => $migrationName,
            '--path' => "app/Extensions/{$extension}/Database/Migrations",
        ]);

        $this->info("$migrationName created for extension: {$extension}");
    }
}
