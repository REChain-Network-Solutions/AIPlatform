<?php

declare(strict_types=1);

namespace App\Extensions\Maintenance\System;

use App\Domains\Marketplace\Contracts\UninstallExtensionServiceProviderInterface;
use App\Extensions\Maintenance\System\Http\Controllers\MaintenanceController;
use Illuminate\Contracts\Http\Kernel;
use Illuminate\Routing\Router;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\Route;
use Illuminate\Support\ServiceProvider;
use Mcamara\LaravelLocalization\Facades\LaravelLocalization;

class MaintenanceServiceProvider extends ServiceProvider implements UninstallExtensionServiceProviderInterface
{
    public function register(): void {}

    public function boot(Kernel $kernel): void
    {
        $this->registerViews()
            ->registerRoutes();

    }

    public function registerViews(): static
    {
        $this->loadViewsFrom([__DIR__ . '/../resources/views'], 'maintenance');

        return $this;
    }

    public function registerMigrations(): static
    {
        $this->loadMigrationsFrom(__DIR__ . '/../database/migrations');

        return $this;
    }

    private function registerRoutes(): void
    {
        $this->router()
            ->group([
                'prefix'     => LaravelLocalization::setLocale(),
                'middleware' => ['web', 'auth', 'localeSessionRedirect', 'localizationRedirect', 'localeViewPath'],
            ], function (Router $router) {
                $router
                    ->controller(MaintenanceController::class)
                    ->prefix('dashboard/admin/settings')
                    ->middleware(['auth', 'admin'])
                    ->as('dashboard.admin.settings.maintenance.')
                    ->group(function (Router $router) {
                        $router->get('maintenance', 'index')->name('index');
                        $router->post('maintenance', 'update')->name('update');
                    });
            });
    }

    private function router(): Router|Route
    {
        return $this->app['router'];
    }

    public static function uninstall(): void
    {
        Cache::forget('maintenance');
    }
}
