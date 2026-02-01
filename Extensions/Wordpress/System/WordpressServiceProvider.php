<?php

declare(strict_types=1);

namespace App\Extensions\Wordpress\System;

use Illuminate\Contracts\Http\Kernel;
use Illuminate\Support\ServiceProvider;

class WordpressServiceProvider extends ServiceProvider
{
    public function register(): void {}

    public function boot(Kernel $kernel): void
    {
        $this->loadMigrationsFrom(__DIR__ . '/../database/migrations');
    }
}
