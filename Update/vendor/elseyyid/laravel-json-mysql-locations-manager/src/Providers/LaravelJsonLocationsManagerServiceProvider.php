<?php namespace Elseyyid\LaravelJsonLocationsManager\Providers;

use Illuminate\Support\ServiceProvider;
use Elseyyid\LaravelJsonLocationsManager\Services\Helper;

use Elseyyid\LaravelJsonLocationsManager\Commands\InstallCommand;
use Elseyyid\LaravelJsonLocationsManager\Commands\PublishAllCommand;

class LaravelJsonLocationsManagerServiceProvider extends ServiceProvider
{
    /**
     * Indicates if loading of the provider is deferred.
     *
     * @var bool
     */
    protected $defer = false;

    /**
     * Bootstrap the application events.
     *
     * @return void
     */
    public function boot()
    {
        \Config::set('database.connections.mysql', config('lang-manager.connections.mysql'));

        /*Load views*/
        $this->loadViewsFrom(__DIR__ . '/../views', 'langs');
        /*Load routes*/
        $this->loadRoutesFrom(__DIR__.'/../routes/web.php');

        $this->publishes([
            __DIR__ . '/../config/elseyyid-location.php' => base_path('config/elseyyid-location.php'),
        ], 'elseyyid-location');

        $this->publishes([
            __DIR__ . '/../views' => base_path('/resources/views/vendor/langs')], 'views');
    }

    /**
     * Register the service provider.
     *
     * @return void
     */
    public function register()
    {
        //Load helpers
        Helper::loadModuleHelpers(__DIR__);
        $this->mergeConfigFrom(__DIR__.'/../config/database.php','lang-manager');
        $this->mergeConfigFrom(__DIR__.'/../config/elseyyid-location.php', 'elseyyid-location');

        $this->commands($this->commands);
    }

    /**
     * The Artisan commands provided by your application.
     *
     * @var array
     */
    protected $commands = [
        InstallCommand::class,
        PublishAllCommand::class,

    ];
}
