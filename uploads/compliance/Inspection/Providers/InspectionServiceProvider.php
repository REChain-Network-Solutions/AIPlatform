<?php

namespace Modules\Inspection\Providers;

use Illuminate\Console\Scheduling\Schedule;
use Illuminate\Support\ServiceProvider;
use Modules\Inspection\Console\Commands\AutoCreateRecurringSchedules;

class InspectionServiceProvider extends ServiceProvider
{
    protected string $moduleName = 'Inspection';

    public function register(): void
    {
        $this->registerConfig();
        $this->registerCommands();
    }

    public function boot(): void
    {
        $this->registerTranslations();
        $this->registerViews();
        $this->scheduleCommands();
    
        // Titan Zero + Titan Go integration (capabilities registry)
        if (class_exists(\Modules\TitanZero\Services\CapabilityRegistry::class)) {
            \Modules\TitanZero\Services\CapabilityRegistry::registerModuleFromConfig('Inspection');
        }
}

    protected function registerConfig(): void
    {
        // module config.php (if present)
        $this->publishes([
            __DIR__ . '/../Config/config.php' => config_path('inspection.php'),
        ], 'config');

        $this->mergeConfigFrom(__DIR__ . '/../Config/config.php', 'inspection');

        // integrations (Titan link-outs only)
        $this->publishes([
            __DIR__ . '/../Config/integrations.php' => config_path('inspection_integrations.php'),
        ], 'config');

        $this->mergeConfigFrom(__DIR__ . '/../Config/integrations.php', 'inspection_integrations');
    }

    protected function registerTranslations(): void
    {
        $langPath = resource_path('lang/modules/' . strtolower($this->moduleName));

        if (is_dir($langPath)) {
            $this->loadTranslationsFrom($langPath, 'inspection');
        } else {
            $this->loadTranslationsFrom(__DIR__ . '/../Resources/lang', 'inspection');
        }
    }

    protected function registerViews(): void
    {
        $this->loadViewsFrom(__DIR__ . '/../Resources/views', 'inspection');

        $this->publishes([
            __DIR__ . '/../Resources/views' => resource_path('views/modules/' . strtolower($this->moduleName)),
        ], 'views');
    }

    protected function registerCommands(): void
    {
        $this->commands([
            AutoCreateRecurringSchedules::class,
        ]);
    }

    protected function scheduleCommands(): void
    {
        // Worksuite commonly defines app.cron_timezone; fall back to app.timezone.
        $timezone = config('app.cron_timezone') ?: config('app.timezone', 'UTC');

        /** @var Schedule $schedule */
        $schedule = $this->app->make(Schedule::class);

        $schedule->command('recurring-schedule-create')
            ->daily()
            ->timezone($timezone);
    }
}
