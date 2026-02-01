<?php

declare(strict_types=1);

namespace App\Extensions\AzureTTS\System;

use Illuminate\Contracts\Http\Kernel;
use Illuminate\Support\ServiceProvider;

class AzureTTSServiceProvider extends ServiceProvider
{
    public function register(): void {}

    public function boot(Kernel $kernel): void
    {
        $this->registerViews();

    }

    public function registerViews(): static
    {
        $this->loadViewsFrom([__DIR__ . '/../resources/views'], 'azure-tts');

        return $this;
    }
}
