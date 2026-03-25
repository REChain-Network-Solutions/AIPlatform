<?php
namespace Modules\WorkOrders\Providers;
use Illuminate\Support\ServiceProvider;
use Illuminate\Support\Facades\View;
use Illuminate\Support\Facades\Route;
class ClientPortalServiceProvider extends ServiceProvider {
  public function boot(): void {
    // Inject 'Work Orders' item into Worksuite's client sidebar if the view exists
    View::composer(['client.sections.menu','client.section.menu','client.layouts.sidebar'], function ($view) {
      $menu = $view->getData()['menu'] ?? [];
      $menu[] = [
        'name' => __('Work Orders'),
        'url'  => Route::has('client.workorders.index') ? route('client.workorders.index') : '/account/workorders',
        'icon' => 'fa fa-clipboard-list',
      ];
      $view->with('menu', $menu);
    });
  }
}