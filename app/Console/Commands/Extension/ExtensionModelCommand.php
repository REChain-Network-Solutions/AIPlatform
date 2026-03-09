<?php

namespace App\Console\Commands\Extension;

use Illuminate\Console\Command;
use Illuminate\Support\Facades\Artisan;
use Illuminate\Support\Str;

class ExtensionModelCommand extends Command
{
    protected $signature = 'ext:model {name} {--f|for= : Extension Name} {--m : Also create a migration}';

    protected $description = 'Create a model for a specific extension';

    public function handle()
    {
        $extension = Str::studly($this->option('for'));

        if (! $extension) {
            $this->error('You must specify an extension name using --for or -f.');

            return;
        }

        $modelName = Str::studly($this->argument('name'));

        $basePath = base_path("app/Extensions/{$extension}/System/Models");

        if (! mkdir($basePath, 0755, true) && ! is_dir($basePath)) {
            $this->error("Failed to create directory: {$basePath}");

            return;
        }

        $modelPath = "{$basePath}/{$modelName}.php";

        if (file_exists($modelPath)) {
            $this->error("Model already exists at: {$modelPath}");

            return;
        }

        // Model template'i
        $namespace = "App\\Extensions\\{$extension}\\System\\Models";
        $tableName = 'ext_' . Str::snake(Str::pluralStudly($modelName));

        $content = $this->getModelTemplate($namespace, $modelName, $tableName);

        if (file_put_contents($modelPath, $content) === false) {
            $this->error("Failed to create model file: {$modelPath}");

            return;
        }

        $this->info("Model created at: {$modelPath}");

        // Check if -m flag is present, then create migration
        if ($this->option('m')) {
            $migrationName = 'create_' . Str::snake(Str::pluralStudly($modelName)) . '_table';

            Artisan::call('make:migration', [
                'name'     => $migrationName,
                '--create' => $tableName,
                '--path'   => "app/Extensions/{$extension}/database/migrations",
            ]);

            $this->info("Migration created for model: {$modelName}");
        }
    }

    /**
     * Get the model template content
     */
    private function getModelTemplate(string $namespace, string $modelName, string $tableName): string
    {
        return "<?php

namespace {$namespace};

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class {$modelName} extends Model
{
    use HasFactory;

    /**
     * The table associated with the model.
     *
     * @var string
     */
    protected \$table = '{$tableName}';

    /**
     * The attributes that are mass assignable.
     *
     * @var array<int, string>
     */
    protected \$fillable = [
        //
    ];

    /**
     * The attributes that should be hidden for serialization.
     *
     * @var array<int, string>
     */
    protected \$hidden = [
        //
    ];

    /**
     * The attributes that should be cast.
     *
     * @var array<string, string>
     */
    protected \$casts = [
        //
    ];
}
";
    }
}
