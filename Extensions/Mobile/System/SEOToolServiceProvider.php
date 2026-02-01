<?php

declare(strict_types=1);

namespace App\Extensions\SEOTool\System;

use App\Extensions\SEOTool\System\Http\Controllers\SeoController;
use App\Http\Middleware\CheckTemplateTypeAndPlan;
use Illuminate\Contracts\Http\Kernel;
use Illuminate\Routing\Router;
use Illuminate\Support\Facades\Route;
use Illuminate\Support\ServiceProvider;
use Mcamara\LaravelLocalization\Facades\LaravelLocalization;

class SEOToolServiceProvider extends ServiceProvider
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
            ->registerComponents();

    }

    public function registerComponents(): static
    {
        //        $this->loadViewComponentsAs('example', []);

        return $this;
    }

    public function registerConfig(): static
    {
        $this->mergeConfigFrom(__DIR__ . '/../config/seo-tool.php', 'seo-tool');

        return $this;
    }

    protected function registerTranslations(): static
    {
        $this->loadTranslationsFrom(__DIR__ . '/../resources/lang', 'seo-tool');

        return $this;
    }

    public function registerViews(): static
    {
        $this->loadViewsFrom([__DIR__ . '/../resources/views'], 'seo-tool');

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
                                $router
                                    ->controller(SeoController::class)
                                    ->prefix('seo')
                                    ->name('seo.')
                                    ->group(function () {
                                        Route::get('', 'index')->name('index')->middleware(CheckTemplateTypeAndPlan::class);
                                        Route::post('suggestKeywords', 'suggestKeywords')->name('suggestKeywords');
                                        Route::post('genkeywords', 'generateKeywords')->name('genkeywords');
                                        Route::post('genSearchQuestions', 'genSearchQuestions')->name('genSearchQuestions');
                                        Route::post('genSEO', 'genSEO')->name('genSEO');
                                        Route::post('analyseArticle', 'analyseArticle')->name('analyseArticle');
                                        Route::post('improveArticle', 'improveArticle')->name('improveArticle');
                                    });
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
