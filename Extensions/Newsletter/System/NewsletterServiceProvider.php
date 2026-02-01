<?php

declare(strict_types=1);

namespace App\Extensions\Newsletter\System;

use App\Extensions\Newsletter\System\Http\Controllers\NewsletterController;
use Illuminate\Contracts\Http\Kernel;
use Illuminate\Routing\Router;
use Illuminate\Support\Facades\Route;
use Illuminate\Support\ServiceProvider;
use Mcamara\LaravelLocalization\Facades\LaravelLocalization;

class NewsletterServiceProvider extends ServiceProvider
{
    public function register(): void {}

    public function boot(Kernel $kernel): void
    {
        $this->registerViews()
            ->registerRoutes();

    }

    public function registerViews(): static
    {
        $this->loadViewsFrom([__DIR__ . '/../resources/views'], 'news-letter');

        return $this;
    }

    private function registerRoutes(): static
    {
        $this->router()
            ->group([
                'prefix'     => LaravelLocalization::setLocale(),
                'middleware' => ['web', 'auth', 'localeSessionRedirect', 'localizationRedirect', 'localeViewPath'],
            ], function (Router $route) {
                Route::prefix('dashboard')
                    ->middleware('auth')
                    ->name('dashboard.')
                    ->group(function () {
                        Route::resource(
                            'newsletter',
                            NewsletterController::class
                        )->only('create', 'store');
                    });
            });

        return $this;
    }

    private function router(): Router|Route
    {
        return $this->app['router'];
    }
}
