<?php

declare(strict_types=1);

namespace App\Extensions\Xero\System;

use App\Domains\Marketplace\Contracts\UninstallExtensionServiceProviderInterface;
use App\Extensions\Xero\System\Http\Controllers\XeroSettingController;
use Dcblogdev\Xero\Facades\Xero;
use Illuminate\Console\Scheduling\Schedule;
use Illuminate\Contracts\Http\Kernel;
use Illuminate\Routing\Router;
use Illuminate\Support\Facades\Route;
use Illuminate\Support\ServiceProvider;
use Mcamara\LaravelLocalization\Facades\LaravelLocalization;
use Throwable;

class XeroServiceProvider extends ServiceProvider implements UninstallExtensionServiceProviderInterface
{
    public function register(): void
    {
        $this->registerConfig();
    }

    public function boot(Kernel $kernel): void
    {
        if (! app()->runningUnitTests()) {
            $this->registerTranslations()
                ->registerViews()
                ->registerRoutes()
                ->registerCommand()
                ->registerMigrations();
        }
    }

    public function registerConfig(): static
    {
        $this->mergeConfigFrom(__DIR__ . '/../config/xero.php', 'xero');

        return $this;
    }

    public function registerCommand(): static
    {
        if ($this->app->runningInConsole()) {
            $this->app->booted(function () {
                $schedule = $this->app->make(Schedule::class);
                $schedule->command('xero:keep-alive')->everyFiveMinutes();
            });
        }

        return $this;
    }

    protected function registerTranslations(): static
    {
        $this->loadTranslationsFrom(__DIR__ . '/../resources/lang', 'xero');

        return $this;
    }

    public function registerViews(): static
    {
        $this->loadViewsFrom([__DIR__ . '/../resources/views'], 'xero');

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
                            ->controller(XeroSettingController::class)
                            ->prefix('admin/settings')
                            ->name('admin.settings.')
                            ->group(function (Router $router) {
                                $router->get('xero', 'index')->name('xero');
                                $router->post('xero/update', 'update')->name('xero.update');
                                $router->post('xero', static function () {
                                    return redirect('xero/connect');
                                })->name('xero.connect');
                                $router->post('create-contacts', 'createContacts')->name('xero.create-contacts');
                            });
                    });
            });
        $this->router()
            ->group([
                'prefix'     => LaravelLocalization::setLocale(),
                'middleware' => ['web', 'localeSessionRedirect', 'localizationRedirect', 'localeViewPath'],
            ], function (Router $router) {

                if (! app()->runningInConsole()) {
                    config([
                        'xero.clientId'     => setting('XERO_CLIENT_ID'),
                        'xero.clientSecret' => setting('XERO_CLIENT_SECRET'),
                        'xero.redirectUri'  => setting('XERO_REDIRECT_URI'),
                        'xero.landingUri'   => setting('XERO_LANDING_URL'),
                    ]);
                }

                $router->get('xero', function () {
                    try {
                        if (! Xero::isConnected()) {
                            return redirect('xero/connect');
                        } else {
                            return back()->with([
                                'type'     => 'success',
                                'message'  => 'Xero connected successfully',
                            ]);
                        }
                    } catch (Throwable $e) {
                    }
                });
                $router->get('xero/connect', function () {
                    try {
                        return Xero::connect();
                    } catch (Throwable $e) {
                    }
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
