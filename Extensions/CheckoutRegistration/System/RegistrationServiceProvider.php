<?php

declare(strict_types=1);

namespace App\Extensions\CheckoutRegistration\System;

use App\Extensions\CheckoutRegistration\System\Http\Controllers\CheckoutSettingsController;
use App\Extensions\CheckoutRegistration\System\Http\Controllers\RegisterCheckoutController;
use App\Extensions\CheckoutRegistration\System\View\Components\CheckoutCard;
use Illuminate\Contracts\Http\Kernel;
use Illuminate\Routing\Router;
use Illuminate\Support\Facades\Route;
use Illuminate\Support\ServiceProvider;
use Mcamara\LaravelLocalization\Facades\LaravelLocalization;

class RegistrationServiceProvider extends ServiceProvider
{
    public function register(): void {}

    public function boot(Kernel $kernel): void
    {
        $this->registerViews()
            ->registerComponents()
            ->registerMigrations()
            ->registerRoutes();
    }

    public function registerComponents(): static
    {
        $this->loadViewComponentsAs('registration', [
            CheckoutCard::class,
        ]);

        return $this;
    }

    public function registerViews(): static
    {
        $this->loadViewsFrom([__DIR__ . '/../resources/views'], 'checkout-registration');

        return $this;
    }

    public function registerMigrations(): static
    {
        $this->loadMigrationsFrom(__DIR__ . '/../database/migrations');

        return $this;
    }

    private function registerRoutes(): void
    {
        $this->router()
            ->group([
                'prefix'     => LaravelLocalization::setLocale(),
                'middleware' => ['web', 'auth', 'localeSessionRedirect', 'localizationRedirect', 'localeViewPath'],
            ], function (Router $router) {
                Route::prefix('dashboard/admin/checkout-registration')
                    ->middleware('admin')
                    ->name('dashboard.admin.checkout.registration.')
                    ->group(function () {
                        Route::prefix('settings')
                            ->controller(CheckoutSettingsController::class)
                            ->name('settings.')->group(function () {
                                Route::get('/', 'index')->name('index');
                                Route::post('/', 'store')->name('store');
                            });
                    });
            });

        $this->router()
            ->group([
                'prefix'     => LaravelLocalization::setLocale(),
                'middleware' => ['web', 'localeSessionRedirect', 'localizationRedirect', 'localeViewPath'],
            ], function (Router $router) {
                Route::middleware('guest')->group(function () {
                    Route::get('register', [RegisterCheckoutController::class, 'index'])->name('register');
                    Route::post('register-user', [RegisterCheckoutController::class, 'store'])
                        ->name('register-user')->middleware('throttle:15,15');
                    Route::match(['get', 'post'], 'register/checkout/{referral?}', [RegisterCheckoutController::class, 'checkout'])
                        ->name('register.checkout');
                });
            });
    }

    private function router(): Router|Route
    {
        return $this->app['router'];
    }
}
