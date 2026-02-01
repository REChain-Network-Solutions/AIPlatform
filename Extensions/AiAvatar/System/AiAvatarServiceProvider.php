<?php

declare(strict_types=1);

namespace App\Extensions\AiAvatar\System;

use App\Extensions\AiAvatar\System\Http\Controllers\AiAvatarController;
use App\Extensions\AiAvatar\System\Http\Controllers\SynthesiaSettingController;
use Illuminate\Contracts\Http\Kernel;
use Illuminate\Routing\Router;
use Illuminate\Support\Facades\Route;
use Illuminate\Support\ServiceProvider;
use Mcamara\LaravelLocalization\Facades\LaravelLocalization;

class AiAvatarServiceProvider extends ServiceProvider
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
        $this->loadTranslationsFrom(__DIR__ . '/../resources/lang', 'ai-avatar');

        return $this;
    }

    public function registerViews(): static
    {
        $this->loadViewsFrom([__DIR__ . '/../resources/views'], 'ai-avatar');

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
                                $router->resource('ai-avatar', AiAvatarController::class)->except('destroy', 'show');
                                $router->get('ai-avatar-delete/{id}', [AiAvatarController::class, 'delete'])->name('ai-avatar.delete');
                                $router->get('ai-avatar-check', [AiAvatarController::class, 'checkVideoStatus'])->name('ai-avatar.check');
                            });
                        $router
                            ->controller(SynthesiaSettingController::class)
                            ->prefix('admin/settings')
                            ->name('admin.settings.')
                            ->group(function (Router $router) {
                                $router->get('synthesia', 'index')->name('synthesia');
                                $router->post('synthesia', 'update')->name('synthesia.update');
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
