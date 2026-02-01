<?php

declare(strict_types=1);

namespace App\Extensions\AIPlagiarism\System;

use App\Extensions\AIPlagiarism\System\Http\Controllers\PlagiarismController;
use App\Http\Middleware\CheckTemplateTypeAndPlan;
use Illuminate\Contracts\Http\Kernel;
use Illuminate\Routing\Router;
use Illuminate\Support\Facades\Route;
use Illuminate\Support\ServiceProvider;
use Mcamara\LaravelLocalization\Facades\LaravelLocalization;

class AIPlagiarismServiceProvider extends ServiceProvider
{
    public function register(): void {}

    public function boot(Kernel $kernel): void
    {
        $this->registerViews()
            ->registerRoutes()
            ->registerMigrations();

    }

    public function registerViews(): static
    {
        $this->loadViewsFrom([__DIR__ . '/../resources/views'], 'ai-plagiarism');

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
                Route::prefix('dashboard')
                    ->middleware('auth')
                    ->name('dashboard.')
                    ->group(function () {
                        Route::prefix('user')
                            ->name('user.')
                            ->group(function () {
                                Route::prefix('openai')->name('openai.')->group(function () {
                                    Route::get('detectaicontent', [PlagiarismController::class, 'detectAIContent'])->name('detectaicontent.index')->middleware(CheckTemplateTypeAndPlan::class);
                                    Route::post('aicontentcheck', [PlagiarismController::class, 'detectAIContentCheck'])->name('detectaicontent.check');
                                    Route::post('aicontentsave', [PlagiarismController::class, 'detectAIContentSave'])->name('detectaicontent.save');
                                    Route::get('plagiarism', [PlagiarismController::class, 'plagiarism'])->name('plagiarism.index')->middleware(CheckTemplateTypeAndPlan::class);
                                    Route::post('plagiarismcheck', [PlagiarismController::class, 'plagiarismCheck'])->name('plagiarism.check');
                                    Route::post('plagiarismsave', [PlagiarismController::class, 'plagiarismSave'])->name('plagiarism.save');
                                });
                            });
                        Route::prefix('admin')
                            ->middleware('admin')
                            ->name('admin.')
                            ->group(function () {
                                Route::prefix('settings')
                                    ->name('settings.')
                                    ->group(function () {
                                        Route::get('plagiarism', [PlagiarismController::class, 'plagiarismSetting'])->name('plagiarism');
                                        Route::post('plagiarismapi-save', [PlagiarismController::class, 'plagiarismSettingSave'])->name('plagiarism.setting.save');
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
