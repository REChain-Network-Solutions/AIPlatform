<?php

namespace Modules\ComplianceIQ\Providers;

use Illuminate\Support\ServiceProvider;
use Illuminate\Support\Facades\Config;

class ComplianceIQServiceProvider extends ServiceProvider
{
    public function register()
    {
        $this->mergeConfigFrom(__DIR__.'/../Config/config.php', 'complianceiq');
        $this->app->bind(\Modules\ComplianceIQ\Services\AI\ComplianceAIInterface::class, function(){
            $driver = config('complianceiq.ai.driver', 'null');
            return $driver === 'openai'
                ? new \Modules\ComplianceIQ\Services\AI\OpenAIComplianceAI()
                : new \Modules\ComplianceIQ\Services\AI\NullComplianceAI();
        });
        $this->mergeConfigFrom(__DIR__.'/../Config/permissions.php', 'complianceiq.permissions');
    }

    public function boot()
    {
        $modulePath = __DIR__.'/..';

        $this->loadMigrationsFrom($modulePath.'/Database/Migrations');
        $this->loadViewsFrom($modulePath.'/Resources/views', 'complianceiq');
        $this->loadTranslationsFrom($modulePath.'/Resources/lang', 'complianceiq');

        $this->publishes([
            $modulePath.'/Config/config.php' => config_path('complianceiq.php'),
        ], 'config');

        // Example menu hook — replace with Worksuite's actual menu builder if different
        app('events')->listen('app.menu.build', function ($menu) {
            if (method_exists($menu, 'add')) {
                $menu->add('Compliance', [
                    'route' => 'admin.compliance.reports.index',
                    'icon' => 'fa fa-shield-check',
                    'permission' => 'compliance.view',
                ]);
                $menu->add('Compliance Logs', [
                    'route' => 'admin.compliance.logs.index',
                    'icon' => 'fa fa-list-check',
                    'permission' => 'compliance.logs.view',
                ]);
            }
        });
    }
}
