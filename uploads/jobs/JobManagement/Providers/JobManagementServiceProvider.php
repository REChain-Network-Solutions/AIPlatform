<?php

namespace Modules\JobManagement\Providers;

use Illuminate\Support\ServiceProvider;
use Illuminate\Support\Facades\Config;

class JobManagementServiceProvider extends ServiceProvider
{
    public function boot(): void
    {
        $this->loadMigrationsFrom(__DIR__.'/../Database/Migrations');
        $this->loadRoutesFrom(__DIR__.'/../Routes/web.php');
        $this->loadRoutesFrom(__DIR__.'/../Routes/api.php');
        $this->loadViewsFrom(__DIR__.'/../Resources/views', 'jobmanagement');
        $this->mergeConfigFrom(__DIR__.'/../Config/config.php', 'jobmanagement');
    }

    public function register(): void
    {
        $this->publishes([
            __DIR__.'/../Config/config.php' => config_path('jobmanagement.php'),
        ], 'config');
    }
}
