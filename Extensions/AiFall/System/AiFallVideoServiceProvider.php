<?php

declare(strict_types=1);

namespace App\Extensions\AiFall\System;

use App\Extensions\AiFall\System\Http\Controllers\FalAISettingController;
use App\Extensions\AiFall\System\Http\Controllers\FallVideoController;
use Illuminate\Contracts\Http\Kernel;
use Illuminate\Routing\Router;
use Illuminate\Support\Facades\Route;
use Illuminate\Support\ServiceProvider;
use Mcamara\LaravelLocalization\Facades\LaravelLocalization;

class AiFallVideoServiceProvider extends ServiceProvider
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
        $this->loadTranslationsFrom(__DIR__ . '/../resources/lang', 'ai-fall-video');

        return $this;
    }

    public function registerViews(): static
    {
        $this->loadViewsFrom([__DIR__ . '/../resources/views'], 'ai-fall-video');

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
                                $router->resource('fall-video', FallVideoController::class)->except('destroy', 'show');
                                $router->get('fall-video-delete/{id}', [FallVideoController::class, 'delete'])->name('fall-video.delete');
                                $router->get('fall-video-check', [FallVideoController::class, 'checkVideoStatus'])->name('fall-video.check');
                            });
                        $router->controller(FalAISettingController::class)
                            ->prefix('admin/settings')
                            ->middleware(['auth', 'admin'])
                            ->name('admin.settings.')->group(function (Router $router) {
                                $router->get('fal-ai', 'index')->name('fal-ai');
                                $router->post('fal-ai', 'update')->name('fal-ai.update');
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
