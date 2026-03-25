<?php
namespace Modules\WorkOrders\Providers;

use Illuminate\Support\Facades\Blade;
use Illuminate\Support\ServiceProvider;

// import any console commands AFTER the namespace
use Modules\WorkOrders\Console\Commands\GenerateRecurringWorkOrders;

class WorkOrdersServiceProvider extends ServiceProvider
{
    public function register(): void
    {
        // Merge config if present
        $configPath = __DIR__ . '/../Config/config.php';
        if (file_exists($configPath)) {
            $this->mergeConfigFrom($configPath, 'workorders');
        }

        // Register console commands only in console
        if ($this->app->runningInConsole()) {
            $commands = [];

            $maybe = [
                \Modules\WorkOrders\Console\WorkOrdersSelfTestCommand::class,
                \Modules\WorkOrders\Console\WorkOrdersExportTemplateCommand::class,
                \Modules\WorkOrders\Console\WorkOrdersImportCsvCommand::class,
                \Modules\WorkOrders\Console\WorkOrdersExportCsvCommand::class,
                \Modules\WorkOrders\Console\WorkOrdersUninstallCommand::class,
                // add real FQCN if GenerateRecurringWorkOrders exists:
                \Modules\WorkOrders\Console\Commands\GenerateRecurringWorkOrders::class,
            ];

            // Register only classes that actually exist
            foreach ($maybe as $fqcn) {
                if (class_exists($fqcn)) {
                    $commands[] = $fqcn;
                }
            }

            if ($commands) {
                $this->commands($commands);
            }
        }
    }

    public function boot(): void
    {
        // Routes
        $web = __DIR__ . '/../Routes/web.php';
        if (file_exists($web)) {
            $this->loadRoutesFrom($web);
        }
        $api = __DIR__ . '/../Routes/api.php';
        if (file_exists($api)) {
            $this->loadRoutesFrom($api);
        }

        // Views
        $views = __DIR__ . '/../Resources/views';
        if (is_dir($views)) {
            $this->loadViewsFrom($views, 'workorders');
            // Point Blade components to this module (fix incorrect WorksuiteWorkOrders path)
            Blade::componentNamespace('Modules\\WorkOrders\\Resources\\views\\components', 'workorders');
        }

        // Migrations
        $migrations = __DIR__ . '/../Database/Migrations';
        if (is_dir($migrations)) {
            $this->loadMigrationsFrom($migrations);
        }

        // Translations
        $lang = __DIR__ . '/../Resources/lang';
        if (is_dir($lang)) {
            $this->loadTranslationsFrom($lang, 'workorders');
        }

        // Publish config (optional)
        $configPath = __DIR__ . '/../Config/config.php';
        if (file_exists($configPath)) {
            $this->publishes([
                $configPath => config_path('workorders.php'),
            ], 'workorders-config');
        }
    }
}