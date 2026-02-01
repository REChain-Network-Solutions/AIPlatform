<?php

declare(strict_types=1);

namespace App\Extensions\PhotoStudio\System;

use App\Extensions\PhotoStudio\System\Http\Controllers\ClipdropSettingController;
use App\Extensions\PhotoStudio\System\Http\Controllers\NovitaSettingController;
use App\Extensions\PhotoStudio\System\Http\Controllers\PhotoStudioController;
use App\Http\Middleware\CheckTemplateTypeAndPlan;
use Illuminate\Contracts\Http\Kernel;
use Illuminate\Routing\Router;
use Illuminate\Support\Facades\Route;
use Illuminate\Support\ServiceProvider;
use Mcamara\LaravelLocalization\Facades\LaravelLocalization;

/**
 * Author: Rauf Abbas <info@liquid-themes.com>
 *
 * @note When you create a new service provider, make sure to add it to the "MarketplaceServiceProvider". Otherwise, your Laravel application won’t recognize this provider, and the related functions won’t work properly.
 */
class PhotoStudioServiceProvider extends ServiceProvider
{
    public function register(): void
    {
        $this->registerConfig();
    }

    public function boot(Kernel $kernel): void
    {
        $this->registerTranslations()
            ->registerViews()
            ->registerRoutes()
            ->registerMigrations();
    }

    public function registerConfig(): static
    {
        $this->mergeConfigFrom(__DIR__ . '/../config/photo-studio.php', 'photo-studio');

        return $this;
    }

    protected function registerTranslations(): static
    {
        $this->loadTranslationsFrom(__DIR__ . '/../resources/lang', 'photo-studio');

        return $this;
    }

    public function registerViews(): static
    {
        $this->loadViewsFrom([__DIR__ . '/../resources/views'], 'photo-studio');

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
                            ->controller(PhotoStudioController::class)
                            ->prefix('user')
                            ->name('user.photo-studio.')
                            ->group(function (Router $router) {
                                $router->get('photo-studio', 'index')->name('index')->middleware(CheckTemplateTypeAndPlan::class);
                                $router->post('photo-studio', 'store')->name('store');
                                $router->get('photo-studio/{photoStudio}/delete', 'delete')->name('delete');
                            });

                        $router
                            ->controller(ClipdropSettingController::class)
                            ->prefix('admin/settings')
                            ->name('admin.settings.')
                            ->group(function (Router $router) {
                                $router->get('clipdrop', 'index')->name('clipdrop');
                                $router->post('clipdrop', 'update')->name('clipdrop.update');
                            });

                        $router
                            ->controller(NovitaSettingController::class)
                            ->prefix('admin/settings')
                            ->name('admin.settings.')
                            ->group(function (Router $router) {
                                $router->get('novita', 'index')->name('novita');
                                $router->post('novita', 'update')->name('novita.update');
                                $router->get('photo-studio-check', 'checkImageStatus')->name('photo-studio.check');

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
