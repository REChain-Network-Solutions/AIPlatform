<?php

namespace Modules\Quotes\Providers;

use Illuminate\Support\ServiceProvider;
use Illuminate\Support\Facades\Blade;

class QuotesServiceProvider extends ServiceProvider
{
    public function register(): void
    {
        $this->mergeConfigFrom(__DIR__.'/../Config/quotes.php', 'quotes');
        $this->app->register(RouteServiceProvider::class);
    }

    public function boot(): void
    {
        if ($this->app->runningInConsole()) {
            $this->callAfterResolving('seeder', function ($seeder) {
                $seeder->call(\Modules\Quotes\Database\Seeders\MenuSeeder::class);
            });
        }
        $this->loadMigrationsFrom(__DIR__.'/../Database/Migrations');
        $this->loadViewsFrom(__DIR__.'/../Resources/views', 'quotes');
        Blade::componentNamespace('Modules\\Quotes\\Views\\Components', 'quotes');
    }
}
