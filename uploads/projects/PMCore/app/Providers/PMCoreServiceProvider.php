<?php

namespace Modules\PMCore\app\Providers;

use App\Models\Task;
use App\Services\Settings\SettingsRegistry;
use Illuminate\Support\Facades\Blade;
use Illuminate\Support\Facades\Gate;
use Illuminate\Support\ServiceProvider;
use Modules\PMCore\app\Models\Project;
use Modules\PMCore\app\Models\Timesheet;
use Modules\PMCore\app\Policies\ProjectPolicy;
use Modules\PMCore\app\Policies\ProjectTaskPolicy;
use Modules\PMCore\app\Policies\TimesheetPolicy;
use Modules\PMCore\app\Settings\PMCoreSettings;
use Nwidart\Modules\Traits\PathNamespace;
use RecursiveDirectoryIterator;
use RecursiveIteratorIterator;

class PMCoreServiceProvider extends ServiceProvider
{
    use PathNamespace;

    protected string $name = 'PMCore';

    protected string $nameLower = 'pmcore';

    /**
     * Boot the application events.
     */
    public function boot(): void
    {
        $this->registerCommands();
        $this->registerCommandSchedules();
        $this->registerTranslations();
        $this->registerConfig();
        $this->registerViews();
        $this->loadMigrationsFrom(module_path($this->name, 'database/migrations'));

        // Register policies
        $this->registerPolicies();

        // Register module settings
        $this->registerModuleSettings();
    }

    /**
     * Register the service provider.
     */
    public function register(): void
    {
        $this->app->register(EventServiceProvider::class);
        $this->app->register(RouteServiceProvider::class);
    }

    /**
     * Register commands in the format of Command::class
     */
    protected function registerCommands(): void
    {
        $this->commands([
            \Modules\PMCore\app\Console\UpdateTimesheetAmounts::class,
        ]);
    }

    /**
     * Register command Schedules.
     */
    protected function registerCommandSchedules(): void
    {
        // $this->app->booted(function () {
        //     $schedule = $this->app->make(Schedule::class);
        //     $schedule->command('inspire')->hourly();
        // });
    }

    /**
     * Register translations.
     */
    public function registerTranslations(): void
    {
        $langPath = resource_path('lang/modules/'.$this->nameLower);

        if (is_dir($langPath)) {
            $this->loadTranslationsFrom($langPath, $this->nameLower);
            $this->loadJsonTranslationsFrom($langPath);
        } else {
            $this->loadTranslationsFrom(module_path($this->name, 'resources/lang'), $this->nameLower);
            $this->loadJsonTranslationsFrom(module_path($this->name, 'resources/lang'));
        }
    }

    /**
     * Register config.
     */
    protected function registerConfig(): void
    {
        $relativeConfigPath = config('modules.paths.generator.config.path');
        $configPath = module_path($this->name, $relativeConfigPath);

        if (is_dir($configPath)) {
            $iterator = new RecursiveIteratorIterator(new RecursiveDirectoryIterator($configPath));

            foreach ($iterator as $file) {
                if ($file->isFile() && $file->getExtension() === 'php') {
                    $relativePath = str_replace($configPath.DIRECTORY_SEPARATOR, '', $file->getPathname());
                    $configKey = $this->nameLower.'.'.str_replace([DIRECTORY_SEPARATOR, '.php'], ['.', ''], $relativePath);
                    $key = ($relativePath === 'config.php') ? $this->nameLower : $configKey;

                    $this->publishes([$file->getPathname() => config_path($relativePath)], 'config');
                    $this->mergeConfigFrom($file->getPathname(), $key);
                }
            }
        }
    }

    /**
     * Register views.
     */
    public function registerViews(): void
    {
        $viewPath = resource_path('views/modules/'.$this->nameLower);
        $sourcePath = module_path($this->name, 'resources/views');

        $this->publishes([$sourcePath => $viewPath], ['views', $this->nameLower.'-module-views']);

        $this->loadViewsFrom(array_merge($this->getPublishableViewPaths(), [$sourcePath]), $this->nameLower);

        $componentNamespace = $this->module_namespace($this->name, $this->app_path(config('modules.paths.generator.component-class.path')));
        Blade::componentNamespace($componentNamespace, $this->nameLower);
    }

    /**
     * Get the services provided by the provider.
     */
    public function provides(): array
    {
        return [];
    }

    private function getPublishableViewPaths(): array
    {
        $paths = [];
        foreach (config('view.paths') as $path) {
            if (is_dir($path.'/modules/'.$this->nameLower)) {
                $paths[] = $path.'/modules/'.$this->nameLower;
            }
        }

        return $paths;
    }

    /**
     * Register model policies
     */
    protected function registerPolicies(): void
    {
        Gate::policy(Project::class, ProjectPolicy::class);
        Gate::policy(Timesheet::class, TimesheetPolicy::class);
        Gate::policy(Task::class, ProjectTaskPolicy::class);
    }

    /**
     * Register module settings
     */
    private function registerModuleSettings(): void
    {
        if (class_exists(SettingsRegistry::class)) {
            $settingsRegistry = app(SettingsRegistry::class);

            $settingsRegistry->registerModule('pmcore', [
                'name' => __('Project Management'),
                'description' => __('Configure project defaults, workflow rules, and management preferences'),
                'icon' => 'bx bx-briefcase',
                'handler' => PMCoreSettings::class,
                'permissions' => [], // Can add permissions if needed
                'order' => 30, // Display order in settings menu
            ]);
        }
    }
}
