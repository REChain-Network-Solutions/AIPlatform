<?php
namespace Modules\WorkOrders\Providers;
use Illuminate\Support\ServiceProvider;
class ModuleServiceProvider extends ServiceProvider {
  public function register(): void {
    $this->app->register(ClientPortalServiceProvider::class);
        $this->app->register(\Modules\WorkOrders\Providers\TerminologyServiceProvider::class);
  }
  public function boot(): void {}
}