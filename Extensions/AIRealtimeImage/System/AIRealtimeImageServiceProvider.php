<?php

declare(strict_types=1);

namespace App\Extensions\AIRealtimeImage\System;

use App\Domains\Marketplace\Contracts\UninstallExtensionServiceProviderInterface;
use App\Extensions\AIRealtimeImage\System\Http\Controllers\AIRealtimeImageController;
use App\Extensions\AIRealtimeImage\System\Http\Controllers\TogetherSettingController;
use Illuminate\Contracts\Http\Kernel;
use Illuminate\Routing\Router;
use Illuminate\Support\Facades\Route;
use Illuminate\Support\ServiceProvider;
use Mcamara\LaravelLocalization\Facades\LaravelLocalization;

class AIRealtimeImageServiceProvider extends ServiceProvider implements UninstallExtensionServiceProviderInterface
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
            ->registerMigrations()
            ->publishAssets()
            ->registerComponents();
    }

    public function registerComponents(): static
    {
        return $this;
    }

    public function publishAssets(): static
    {
        $this->publishes([
            __DIR__ . '/../resources/assets/js'     => public_path('vendor/ai-realtime-image/js'),
            __DIR__ . '/../resources/assets/images' => public_path('vendor/ai-realtime-image/images'),
        ], 'extension');

        return $this;
    }

    public function registerConfig(): static
    {
        $this->mergeConfigFrom(__DIR__ . '/../config/ai-realtime-image.php', 'ai-realtime-image');

        return $this;
    }

    protected function registerTranslations(): static
    {
        $this->loadTranslationsFrom(__DIR__ . '/../resources/lang', 'ai-realtime-image');

        return $this;
    }

    public function registerViews(): static
    {
        $this->loadViewsFrom([__DIR__ . '/../resources/views'], 'ai-realtime-image');

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
                                $router->resource('ai-realtime-image', AIRealtimeImageController::class)->only('index', 'store');
                                $router->get('ai-realtime-image/gallery', [AIRealtimeImageController::class, 'gallery'])->name('ai-realtime-image.gallery');
                            });

                        $router
                            ->controller(TogetherSettingController::class)
                            ->prefix('admin/settings')
                            ->name('admin.settings.')
                            ->group(function (Router $router) {
                                $router->get('together', 'index')->name('together');
                                $router->post('together', 'update')->name('together.update');
                            });
                    });
            });

        return $this;
    }

    private function router(): Router|Route
    {
        return $this->app['router'];
    }

    public static function uninstall(): void
    {
        // TODO: Implement uninstall() method.
    }
}
