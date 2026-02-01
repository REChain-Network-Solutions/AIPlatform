<?php

declare(strict_types=1);

namespace App\Extensions\AiAvatarPro\System;

use App\Extensions\AiAvatarPro\System\Http\Controllers\AiAvatarProController;
use App\Extensions\AiAvatarPro\System\Http\Controllers\HeygenSettingController;
use Illuminate\Contracts\Http\Kernel;
use Illuminate\Routing\Router;
use Illuminate\Support\Facades\Route;
use Illuminate\Support\ServiceProvider;
use Mcamara\LaravelLocalization\Facades\LaravelLocalization;

class AiAvatarProServiceProvider extends ServiceProvider
{
    public function register(): void {}

    public function boot(Kernel $kernel): void
    {
        $this->registerTranslations()
            ->registerViews()
            ->registerRoutes()
            ->registerMigrations();
    }

    protected function registerTranslations(): static
    {
        $this->loadTranslationsFrom(__DIR__ . '/../resources/lang', 'ai-avatar-pro');

        return $this;
    }

    public function registerViews(): static
    {
        $this->loadViewsFrom([__DIR__ . '/../resources/views'], 'ai-avatar-pro');

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
                        $router->prefix('user')
                            ->name('user.')
                            ->group(function (Router $router) {
                                $router->resource('ai-avatar-pro', AiAvatarProController::class)->except('destroy', 'show');
                                $router->get('ai-avatar-pro-delete/{id}', [AiAvatarProController::class, 'delete'])->name('ai-avatar-pro.delete');
                                $router->get('ai-avatar-pro-check', [AiAvatarProController::class, 'checkVideoStatus'])->name('ai-avatar-pro.check');
                            });
                        $router
                            ->controller(HeygenSettingController::class)
                            ->prefix('admin/settings')
                            ->name('admin.settings.')
                            ->group(function (Router $router) {
                                $router->get('heygen', 'index')->name('heygen');
                                $router->post('heygen', 'update')->name('heygen.update');
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
