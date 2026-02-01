<?php

declare(strict_types=1);

namespace App\Extensions\AdvancedImage\System;

use App\Extensions\AdvancedImage\System\Http\Controllers\AdvancedImageController;
use App\Extensions\AdvancedImage\System\Http\Controllers\ClipdropSettingController;
use App\Extensions\AdvancedImage\System\Http\Controllers\FreepikSettingController;
use App\Extensions\AdvancedImage\System\Http\Controllers\GeneratePromptController;
use App\Extensions\AdvancedImage\System\Http\Controllers\NovitaSettingController;
use Illuminate\Contracts\Http\Kernel;
use Illuminate\Routing\Router;
use Illuminate\Support\Facades\Route;
use Illuminate\Support\ServiceProvider;

class AdvancedImageServiceProvider extends ServiceProvider
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
        $this->loadTranslationsFrom(__DIR__ . '/../resources/lang', 'advanced-image');

        return $this;
    }

    public function registerViews(): static
    {
        $this->loadViewsFrom([__DIR__ . '/../resources/views'], 'advanced-image');

        return $this;
    }

    public function registerMigrations(): static
    {
        $this->loadMigrationsFrom(__DIR__ . '/../database/migrations');

        return $this;
    }

    private function registerRoutes(): static
    {
        $router = $this->router()
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
                                $router->any('image-to-prompt', GeneratePromptController::class)->name('generate.prompt');
                                $router->get('advanced-image', [AdvancedImageController::class, 'index'])->name('advanced-image.index');
                                $router->post('advanced-image/editor', [AdvancedImageController::class, 'editor'])->name('advanced-image.editor');
                                $router->get('advanced-image/editor/{task}/status', [AdvancedImageController::class, 'status'])->name('advanced-image.editor.task');
                            });

                        $router
                            ->controller(NovitaSettingController::class)
                            ->prefix('admin/settings')
                            ->name('admin.settings.')
                            ->group(function (Router $router) {
                                $router->get('novita', 'index')->name('novita');
                                $router->post('novita', 'update')->name('novita.update');
                            });

                        $router
                            ->controller(FreepikSettingController::class)
                            ->prefix('admin/settings')
                            ->name('admin.settings.')
                            ->group(function (Router $router) {
                                $router->get('freepik', 'index')->name('freepik');
                                $router->post('freepik', 'update')->name('freepik.update');
                            });

                        $router
                            ->controller(ClipdropSettingController::class)
                            ->prefix('admin/settings')
                            ->name('admin.settings.')
                            ->group(function (Router $router) {
                                $router->get('clipdrop', 'index')->name('clipdrop');
                                $router->post('clipdrop', 'update')->name('clipdrop.update');
                            });

                    });
            });

        $router->any('api/webhook/advanced-image/{model?}', [AdvancedImageController::class, 'webhook'])
            ->middleware('api')
            ->name('webhook.advanced-image');

        return $this;
    }

    private function router(): Router|Route
    {
        return $this->app['router'];
    }
}
