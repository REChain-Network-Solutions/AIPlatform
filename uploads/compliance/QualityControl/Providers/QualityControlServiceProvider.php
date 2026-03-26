<?php

namespace Modules\QualityControl\Providers;

use Illuminate\Console\Scheduling\Schedule;
use Illuminate\Support\Facades\Event;
use Illuminate\Support\ServiceProvider;
use Modules\QualityControl\Console\Commands\AutoCreateRecurringSchedules;
use Modules\QualityControl\Console\Commands\ActivateQualityControlModuleCommand;
use Modules\QualityControl\Listeners\JobCompletedListener;

class QualityControlServiceProvider extends ServiceProvider
{
    protected string $moduleName = 'QualityControl';

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

        // Optional: auto-create Quality Checks when a job is completed.
        if (config('quality_control.auto_create_on_job_complete', false)) {
            $listener = app(JobCompletedListener::class);

            // String events (loose coupling) — other modules can dispatch these.
            Event::listen('job.completed', [$listener, 'handle']);
            Event::listen('cleaning.job.completed', [$listener, 'handle']);
            Event::listen('jobs.completed', [$listener, 'handle']);
        }
    
        // Titan Zero + Titan Go integration (capabilities registry)
        if (class_exists(\Modules\TitanZero\Services\CapabilityRegistry::class)) {
            \Modules\TitanZero\Services\CapabilityRegistry::registerModuleFromConfig('QualityControl');
        }
    }

    protected function registerConfig(): void
    {
        // module config.php (if present)
        $this->publishes([
            __DIR__ . '/../Config/config.php' => config_path('quality_control.php'),
        ], 'config');

        $this->mergeConfigFrom(__DIR__ . '/../Config/config.php', 'quality_control');

        // integrations (Titan link-outs only)
        $this->publishes([
            __DIR__ . '/../Config/integrations.php' => config_path('quality_control_integrations.php'),
        ], 'config');

        $this->mergeConfigFrom(__DIR__ . '/../Config/integrations.php', 'quality_control_integrations');
    }

    protected function registerTranslations(): void
    {
        $langPath = resource_path('lang/modules/' . strtolower($this->moduleName));

        if (is_dir($langPath)) {
            $this->loadTranslationsFrom($langPath, 'quality_control');
        } else {
            $this->loadTranslationsFrom(__DIR__ . '/../Resources/lang', 'quality_control');
        }
    }

    protected function registerViews(): void
    {
        $this->loadViewsFrom(__DIR__ . '/../Resources/views', 'quality_control');

        $this->publishes([
            __DIR__ . '/../Resources/views' => resource_path('views/modules/' . strtolower($this->moduleName)),
        ], 'views');
    }

    protected function registerCommands(): void
    {
        $this->commands([
            AutoCreateRecurringSchedules::class,
            ActivateQualityControlModuleCommand::class,
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