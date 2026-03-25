<?php
namespace Modules\WorkOrders\Providers;
use Illuminate\Foundation\Support\Providers\AuthServiceProvider as ServiceProvider;
use Illuminate\Support\Facades\Gate;
use Modules\WorkOrders\Policies\WorkOrderPolicy;

class AuthServiceProvider extends ServiceProvider {
  protected $policies = [
    'workorders' => WorkOrderPolicy::class,
  ];
  public function boot(): void {
    $this->registerPolicies();
    Gate::define('workorder.view', [WorkOrderPolicy::class, 'view']);
    Gate::define('workorder.start', [WorkOrderPolicy::class, 'start']);
    Gate::define('workorder.complete', [WorkOrderPolicy::class, 'complete']);
    Gate::define('workorder.invoice', [WorkOrderPolicy::class, 'invoice']);
    Gate::define('workorder.approve', [WorkOrderPolicy::class, 'approve']);
  }
}
