<?php

declare(strict_types=1);

namespace App\Extensions\Announcement\System;

use App\Domains\Marketplace\Contracts\UninstallExtensionServiceProviderInterface;
use Illuminate\Contracts\Http\Kernel;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Schema;
use Illuminate\Support\ServiceProvider;

/**
 * Author: MagicAI Team <info@liquid-themes.com>
 *
 * @note When you create a new service provider, make sure to add it to the "MarketplaceServiceProvider". Otherwise, your Laravel application won’t recognize this provider, and the related functions won’t work properly.
 * @note If you want to perform a specific action when an extension is uninstalled, you can use the UninstallExtensionServiceProviderInterface. By implementing this interface, you can define custom operations that will be triggered during the uninstallation of the extension.
 */
class AnnouncementServiceProvider extends ServiceProvider implements UninstallExtensionServiceProviderInterface
{
    public function register(): void {}

    public function boot(Kernel $kernel): void
    {
        $this->registerViews()->registerMigrations()->registerRoutes();
    }

    public function registerViews(): static
    {
        $this->loadViewsFrom([__DIR__ . '/../resources/views'], 'announcement');

        return $this;
    }

    public function registerMigrations(): static
    {
        $this->loadMigrationsFrom(__DIR__ . '/../database/migrations');

        return $this;
    }

    public function registerRoutes(): static
    {
        $this->loadRoutesFrom(__DIR__ . '/route.php');

        return $this;
    }

    public static function uninstall(): void
    {
        Schema::dropIfExists('announcements');
        DB::table('migrations')->where('migration', '=', '2025_03_24_184920_create_announcements_table')->delete();
    }
}
