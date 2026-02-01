<?php

declare(strict_types=1);

namespace App\Extensions\AiVideoPro\System;

use App\Extensions\AiVideoPro\System\Http\Controllers\AiVideoProController;
use App\Extensions\AiVideoPro\System\Http\Controllers\FalAISettingController;
use Illuminate\Contracts\Http\Kernel;
use Illuminate\Routing\Router;
use Illuminate\Support\Facades\Route;
use Illuminate\Support\ServiceProvider;

class AiVideoProServiceProvider extends ServiceProvider
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
        $this->loadTranslationsFrom(__DIR__ . '/../resources/lang', 'ai-video-pro');

        return $this;
    }

    public function registerViews(): static
    {
        $this->loadViewsFrom([__DIR__ . '/../resources/views'], 'ai-video-pro');

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
                'middleware' => ['web', 'auth'],
            ], function (Router $router) {
                $router
                    ->prefix('dashboard')
                    ->name('dashboard.')
                    ->group(function (Router $router) {
                        $router->prefix('user')
                            ->name('user.')
                            ->group(function (Router $router) {
                                $router->resource('ai-video-pro', AiVideoProController::class)->except('destroy', 'show');
                                $router->get('ai-video-pro-delete/{id}', [AiVideoProController::class, 'delete'])->name('ai-video-pro.delete');
                                $router->get('ai-video-pro-check', [AiVideoProController::class, 'checkVideoStatus'])->name('ai-video-pro.check');
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
