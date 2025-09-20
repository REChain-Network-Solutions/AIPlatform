<?php

namespace App\Console\Commands\Extension;

use Illuminate\Console\Command;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Str;

class ExtensionControllerCommand extends Command
{
    protected $signature = 'ext:controller {name} {--f|for= : Extension Name}';

    protected $description = 'Create a controller for a specific extension';

    public function handle()
    {
        $extension = Str::studly($this->option('for'));
        $controllerName = Str::studly($this->argument('name')) . 'Controller';

        if (! $extension) {
            $this->error('You must specify an extension name using --for or -f.');

            return;
        }

        $controllerBasePath = base_path("app/Extensions/{$extension}/System/Http/Controllers");

        if (! is_dir($controllerBasePath)) {
            Storage::disk('extension')->makeDirectory("{$extension}/System/Http/Controllers");
        }

        $controllerPath = "{$controllerBasePath}/{$controllerName}.php";
        $namespace = "App\\Extensions\\{$extension}\\System\\Http\\Controllers";

        if (file_exists($controllerPath)) {
            $this->error("Controller already exists at: {$controllerPath}");

            return;
        }

        $content = file_put_contents($controllerPath, $content);

        $this->info("Controller created at: {$controllerPath}");
    }
}
