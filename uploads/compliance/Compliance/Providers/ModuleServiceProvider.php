<?php
namespace Modules\Compliance\Providers;
use Illuminate\Support\ServiceProvider;
class ModuleServiceProvider extends ServiceProvider {
  public function boot(): void {
    $this->loadMigrationsFrom(__DIR__.'/../Database/Migrations');
    $this->loadViewsFrom(__DIR__.'/../Resources/views', 'Compliance');
    if (file_exists(__DIR__.'/../Config/config.php')) { $this->mergeConfigFrom(__DIR__.'/../Config/config.php', 'compliance'); }
  }
}
