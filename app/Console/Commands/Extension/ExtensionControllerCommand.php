<?php

namespace App\Console\Commands\Extension;

use Illuminate\Console\Command;
use Illuminate\Filesystem\Filesystem;
use Illuminate\Support\Str;
use Symfony\Component\Console\Command\Command as CommandAlias;

class ExtensionControllerCommand extends Command
{
    protected $signature = 'ext:controller {name} {--f|for= : Extension name to create the controller for}';

    protected $description = 'Create a controller for a specific extension';

    protected Filesystem $files;

    public function __construct(Filesystem $files)
    {
        parent::__construct();
        $this->files = $files;
    }

    public function handle(): int
    {
        $extension = $this->getExtensionName();
        if (! $extension) {
            return CommandAlias::FAILURE;
        }

        $controllerName = $this->getControllerName();
        $controllerPath = $this->getControllerPath($extension, $controllerName);

        if ($this->files->exists($controllerPath)) {
            $this->error("Controller already exists: {$controllerPath}");

            return CommandAlias::FAILURE;
        }

        $this->ensureDirectoryExists(dirname($controllerPath));
        $this->createController($controllerPath, $extension, $controllerName);

        $this->info("✓ Controller created: {$controllerPath}");

        return CommandAlias::SUCCESS;
    }

    protected function getExtensionName(): ?string
    {
        $extension = $this->option('for');

        if (empty($extension)) {
            $this->error('Extension name is required. Use --for or -f to specify the extension.');

            return null;
        }

        return Str::studly($extension);
    }

    protected function getControllerName(): string
    {
        $name = $this->argument('name');

        return Str::studly($name) . 'Controller';
    }

    protected function getControllerPath(string $extension, string $controllerName): string
    {
        return base_path("app/Extensions/{$extension}/System/Http/Controllers/{$controllerName}.php");
    }

    protected function ensureDirectoryExists(string $directory): void
    {
        if (! $this->files->isDirectory($directory)) {
            $this->files->makeDirectory($directory, 0755, true);
        }
    }

    protected function createController(string $path, string $extension, string $controllerName): void
    {
        $stub = $this->getControllerStub();
        $content = $this->populateStub($stub, $extension, $controllerName);

        $this->files->put($path, $content);
    }

    protected function getControllerStub(): string
    {
        return <<<'STUB'
		<?php

		namespace {{ namespace }};

		use Illuminate\Http\Request;
		use App\Http\Controllers\Controller;

		class {{ class }} extends Controller
		{
			/**
			 * Display a listing of the resource.
			 */
			public function index()
			{
				//
			}

			/**
			 * Show the form for creating a new resource.
			 */
			public function create()
			{
				//
			}

			/**
			 * Store a newly created resource in storage.
			 */
			public function store(Request $request)
			{
				//
			}

			/**
			 * Display the specified resource.
			 */
			public function show(string $id)
			{
				//
			}

			/**
			 * Show the form for editing the specified resource.
			 */
			public function edit(string $id)
			{
				//
			}

			/**
			 * Update the specified resource in storage.
			 */
			public function update(Request $request, string $id)
			{
				//
			}

			/**
			 * Remove the specified resource from storage.
			 */
			public function destroy(string $id)
			{
				//
			}
		}
		STUB;
    }

    protected function populateStub(string $stub, string $extension, string $controllerName): string
    {
        $namespace = "App\\Extensions\\{$extension}\\System\\Http\\Controllers";

        return str_replace(
            ['{{ namespace }}', '{{ class }}'],
            [$namespace, $controllerName],
            $stub
        );
    }
}
