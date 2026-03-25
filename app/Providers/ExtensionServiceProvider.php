<?php

declare(strict_types=1);

namespace App\Providers;

use App\Services\Extension\ExtensionProviderResolver;
use Illuminate\Support\ServiceProvider;
use Illuminate\Support\Facades\Log;

class ExtensionServiceProvider extends ServiceProvider
{
    public function register(): void
    {
        $providers = app(ExtensionProviderResolver::class)->resolve();

        foreach ($providers as $provider) {
            if (class_exists($provider)) {
                $this->app->register($provider);
            } else {
                Log::warning('Extension provider class not found during register', ['provider' => $provider]);
            }
        }
    }
}
