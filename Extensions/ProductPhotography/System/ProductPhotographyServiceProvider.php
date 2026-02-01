<?php

declare(strict_types=1);

namespace App\Extensions\ProductPhotography\System;

use App\Extensions\ProductPhotography\System\Http\Controllers\PebblelySettingController;
use App\Extensions\ProductPhotography\System\Http\Controllers\ProductPhotographyController;
use Illuminate\Contracts\Http\Kernel;
use Illuminate\Routing\Router;
use Illuminate\Support\Facades\Route;
use Illuminate\Support\ServiceProvider;
use Mcamara\LaravelLocalization\Facades\LaravelLocalization;

class ProductPhotographyServiceProvider extends ServiceProvider
{
    public function register(): void
    {
        $this->registerConfig();
    }

    //pebblely

    public function boot(Kernel $kernel): void
    {
        $this->registerTranslations()
            ->registerViews()
            ->registerRoutes()
            ->registerMigrations();

    }

    public function registerConfig(): static
    {
        $this->mergeConfigFrom(__DIR__ . '/../config/product-photography.php', 'product-photography');

        return $this;
    }

    protected function registerTranslations(): static
    {
        $this->loadTranslationsFrom(__DIR__ . '/../resources/lang', 'product-photography');

        return $this;
    }

    public function registerViews(): static
    {
        $this->loadViewsFrom([__DIR__ . '/../resources/views'], 'product-photography');

        return $this;
    }

    public function registerMigrations(): static
    {
        $this->loadMigrationsFrom(__DIR__ . '/../database/migrations');

        return $this;
    }

    private function registerRoutes(): static
    {
        $this->router()
            ->group([
                'prefix'     => LaravelLocalization::setLocale(),
                'middleware' => ['web', 'auth', 'localeSessionRedirect', 'localizationRedirect', 'localeViewPath'],
            ], function (Router $router) {
                $router
                    ->prefix('dashboard')
                    ->name('dashboard.')
                    ->group(function (Router $router) {
                        $router
                            ->prefix('user')
                            ->name('user.')
                            ->group(function (Router $router) {
                                $router->resource('ai-product-shot', ProductPhotographyController::class)->only(['index', 'store']);
                                $router->get('ai-product-shot-delete/{id}', [ProductPhotographyController::class, 'delete'])->name('ai-product-shot.delete');
                            });

                        $router
                            ->controller(PebblelySettingController::class)
                            ->prefix('admin/settings')
                            ->name('admin.settings.')
                            ->group(function (Router $router) {
                                $router->get('pebblely', 'index')->name('pebblely');
                                $router->post('pebblely', 'update')->name('pebblely.update');
                            });
                    });

            });

        return $this;
    }

    private function router(): Router|Route
    {
        return $this->app['router'];
    }
}
