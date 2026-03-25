<?php

namespace Modules\EInvoice\Providers;

use Illuminate\Support\Facades\Blade;
use Illuminate\Support\Facades\Config;
use Illuminate\Support\ServiceProvider;
use Modules\EInvoice\Console\ActivateModuleCommand;
use Modules\EInvoice\AI\ClientInterface;
use Modules\EInvoice\AI\Adapters\OpenAIClient;

class EInvoiceServiceProvider extends ServiceProvider
{

    /**
     * Indicates if loading of the provider is deferred.
     *
     * @var bool
     */
    protected $defer = false;

    /**
     * Boot the application events.
     *
     * @return void
     */
    public function boot()
        // Auto-run MenuSeeder when seeding the module
        if ($this->app->runningInConsole()) {
            $this->callAfterResolving('seeder', function ($seeder) {
                $seeder->call(\Modules\EInvoice\Database\Seeders\MenuSeeder::class);
            });
        }
    
    {
        $this->registerTranslations();
        $this->registerConfig();
        $this->registerViews();
        $this->loadMigrationsFrom(__DIR__.'/../Database/Migrations');
        Blade::componentNamespace('Modules\\EInvoice\\Views\\Components', 'einvoice');
        $this->commands([
            ActivateModuleCommand::class,
        ]);
    }

    /**
     * Register the service provider.
     *
     * @return void
     */
    \1
        $this->app->bind(ClientInterface::class, function($app){
            return new OpenAIClient(config('einvoice.ai'));
        });
$this->app->register(RouteServiceProvider::class);
    }

    /**
     * Register config.
     *
     * @return void
     */
    protected function registerConfig()
    {
        $this->publishes([
            __DIR__.'/../Config/config.php' => config_path('einvoice.php'),
        ]);

        $this->mergeConfigFrom(
            __DIR__.'/../Config/config.php', 'einvoice'
        );

        $this->mergeConfigFrom(
            module_path('einvoice', 'Config/xss_ignore.php'),
            'einvoice::xss_ignore'
        );
    }

    /**
     * Register views.
     *
     * @return void
     */
    public function registerViews()
    {
        $viewPath = base_path('resources/views/modules/einvoice');

        $sourcePath = __DIR__.'/../Resources/views';

        $this->publishes([
            $sourcePath => $viewPath
        ]);

        $this->loadViewsFrom(array_merge(array_map(function ($path) {
            return $path . '/modules/einvoice';
        }, Config::get('view.paths')), [$sourcePath]), 'einvoice');
    }

    /**
     * Register translations.
     *
     * @return void
     */
    public function registerTranslations()
    {
        $langPath = base_path('resources/lang/modules/einvoice');

        if (is_dir($langPath)) {
            $this->loadTranslationsFrom($langPath, 'einvoice');
        }
        else {
            $this->loadTranslationsFrom(__DIR__ .'/../Resources/lang', 'einvoice');
        }
    }

    /**
     * Get the services provided by the provider.
     *
     * @return array
     */
    public function provides()
    {
        return array();
    }

}
