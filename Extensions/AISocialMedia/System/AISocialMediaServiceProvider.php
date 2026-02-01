<?php

declare(strict_types=1);

namespace App\Extensions\AISocialMedia\System;

use App\Domains\Marketplace\Contracts\UninstallExtensionServiceProviderInterface;
use App\Extensions\AISocialMedia\System\Http\Controllers\Api\InstagramController;
use App\Extensions\AISocialMedia\System\Http\Controllers\AutomationController;
use App\Extensions\AISocialMedia\System\Http\Controllers\AutomationPlatformController;
use App\Extensions\AISocialMedia\System\Http\Controllers\AutomationSettingController;
use App\Extensions\AISocialMedia\System\Http\Controllers\AutomationStepController;
use App\Extensions\AISocialMedia\System\Http\Controllers\GenerateContentController;
use App\Extensions\AISocialMedia\System\Http\Controllers\UploadController;
use App\Extensions\AISocialMedia\System\Http\Middleware\AutomationCacheMiddleware;
use App\Http\Middleware\CheckTemplateTypeAndPlan;
use Illuminate\Console\Scheduling\Schedule;
use Illuminate\Contracts\Http\Kernel;
use Illuminate\Routing\Router;
use Illuminate\Support\Facades\Route;
use Illuminate\Support\ServiceProvider;
use Mcamara\LaravelLocalization\Facades\LaravelLocalization;

class AISocialMediaServiceProvider extends ServiceProvider implements UninstallExtensionServiceProviderInterface
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
            ->registerCommand();

    }

    public function publishAssets(): static
    {
        $this->publishes([
            __DIR__ . '/../resources/assets/images' => public_path('vendor/ai-social-media/images'),
        ], 'extension');

        return $this;
    }

    public function registerConfig(): static
    {
        $this->mergeConfigFrom(__DIR__ . '/../config/ai-social-media.php', 'ai-social-media');

        return $this;
    }

    public function registerCommand(): static
    {
        if ($this->app->runningInConsole()) {
            $this->commands([
                Console\Commands\GeneratePostDailyCommand::class,
                Console\Commands\GeneratePostMonthlyCommand::class,
                Console\Commands\GeneratePostWeeklyCommand::class,
            ]);

            $this->app->booted(function () {
                $schedule = $this->app->make(Schedule::class);
                $schedule->command('app:generate-post-daily')->everyTwoMinutes();
                $schedule->command('app:generate-post-weekly')->everyTwoMinutes();
                $schedule->command('app:generate-post-monthly')->everyTwoMinutes();
            });
        }

        return $this;
    }

    protected function registerTranslations(): static
    {
        $this->loadTranslationsFrom(__DIR__ . '/../resources/lang', 'ai-social-media');

        return $this;
    }

    public function registerViews(): static
    {
        $this->loadViewsFrom([__DIR__ . '/../resources/views'], 'ai-social-media');

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

                $router->group([
                    'prefix'     => 'oauth',
                    'controller' => InstagramController::class,
                ], function (Router $route) {
                    $route->get('redirect/instagram', 'redirect')->name('oauth.connect.instagram');
                    $route->get('callback/instagram', 'callback')->name('oauth.callback.instagram');
                });

                Route::prefix('dashboard')->name('dashboard.')
                    ->group(function () {
                        Route::prefix('user')
                            ->name('user.')
                            ->group(function () {

                                Route::controller(AutomationPlatformController::class)
                                    ->prefix('automation/platform')
                                    ->name('automation.platform.')
                                    ->group(function () {
                                        Route::get('', 'index')->name('list');
                                        Route::post('update/{platform}', 'update')->name('update');
                                        Route::get('disconnect/{automationPlatform}', 'disconnect')->name('disconnect');
                                    });

                                Route::controller(AutomationController::class)
                                    ->prefix('automation')
                                    ->name('automation.')
                                    ->group(function () {

                                        Route::post('upload', UploadController::class)->name('upload');

                                        Route::post('genPost', [GenerateContentController::class, 'generateContent'])->name('generate-content');

                                        Route::group([
                                            'controller' => AutomationStepController::class,
                                            'middleware' => AutomationCacheMiddleware::class,
                                        ], static function (Router $router) {
                                            $router->get('', 'stepFirst')->name('index')->middleware(CheckTemplateTypeAndPlan::class)->withoutMiddleware(AutomationCacheMiddleware::class);
                                            $router->any('step/two', 'stepSecond')->name('step.second');
                                            $router->any('step/third', 'stepThird')->name('step.third');
                                            $router->any('step/fourth', 'stepFourth')->name('step.fourth');
                                            $router->any('step/fifth', 'stepFifth')->name('step.fifth');
                                            $router->any('step/last', 'stepLast')->name('step.last');
                                            $router->any('step/store', 'storeScheduledPost')->name('step.store');
                                        });

                                        Route::post('', 'nextStep')->name('postindex');

                                        Route::get('scheduled-posts', 'scheduledPosts')->name('list')->middleware(CheckTemplateTypeAndPlan::class);
                                        Route::get('scheduled-posts/delete/{id}', 'scheduledPostsDelete')->name('delete');
                                        Route::post('scheduled-posts/edit/{id}', 'scheduledPostsEdit')->name('edit');

                                        Route::prefix('platform')
                                            ->name('platform.')
                                            ->group(function () {});

                                        Route::prefix('company')
                                            ->name('company.')
                                            ->group(function () {
                                                Route::get('get-products/{company_id}', 'getProducts')->name('getProducts');
                                            });

                                        Route::prefix('campaign')
                                            ->name('campaign.')
                                            ->group(function () {
                                                Route::get('', 'campaignList')->name('list');
                                                Route::get('add-or-update/{id?}', 'campaignAddOrUpdate')->name('addOrUpdate');
                                                Route::get('delete/{id?}', 'campaignDelete')->name('delete');
                                                Route::post('save', 'campaignAddOrUpdateSave')->name('campaignAddOrUpdateSave');
                                                Route::get('get-target/{campaign_id}', 'getCampaignTarget')->name('getCampaignTarget');
                                                Route::post('genContent', 'generateCampaignContent')->name('genContent');
                                                Route::post('genTopics', 'generateCampaignTopics')->name('genTopics');
                                            });

                                        Route::post('update', 'updateAutomation')->name('update');
                                        Route::post('getCompany', 'getCompany')->name('getCompany');
                                        Route::post('getSelectedProducts', 'getSelectedProducts')->name('getSelectedProducts');

                                    });
                            });

                        Route::prefix('admin')
                            ->middleware('admin')
                            ->name('admin.')
                            ->group(function () {
                                Route::controller(AutomationSettingController::class)
                                    ->prefix('automation')
                                    ->name('automation.')
                                    ->group(function () {
                                        Route::get('settings', 'index')->name('settings');
                                        Route::post('settings/update', 'update')->name('settings.update');
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

    public static function uninstall(): void
    {
        setting([
            'ai_automation' => 0,
        ])->save();
    }
}
