<?php

declare(strict_types=1);

namespace App\Extensions\Hubspot\System;

use App\Extensions\Hubspot\System\Http\Controllers\HubspotController;
use Illuminate\Contracts\Http\Kernel;
use Illuminate\Routing\Router;
use Illuminate\Support\Facades\Route;
use Illuminate\Support\ServiceProvider;
use Mcamara\LaravelLocalization\Facades\LaravelLocalization;

class HubspotServiceProvider extends ServiceProvider
{
    public function register(): void {}

    public function boot(Kernel $kernel): void
    {
        $this->registerViews()
            ->registerRoutes();

    }

    public function registerViews(): static
    {
        $this->loadViewsFrom([__DIR__ . '/../resources/views'], 'hubspot');

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
                    ->prefix('dashboard/admin')
                    ->middleware('admin')
                    ->name('dashboard.admin.')
                    ->group(function (Router $router) {
                        $router->resource('hubspot', HubspotController::class)->only(['index', 'store']);
                    });

            });
    }

    private function router(): Router|Route
    {
        return $this->app['router'];
    }
}
