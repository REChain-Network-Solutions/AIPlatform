<?php
namespace Modules\WorkOrders\Providers;
use Illuminate\Support\ServiceProvider;
use Illuminate\Support\Facades\Blade;
class TerminologyServiceProvider extends ServiceProvider {
  public function boot(): void {
    Blade::directive('term', function($key) {
      $key = trim($key, "'\" ");
      return "<?php echo e(trans('fsm/terms.' . $key)); ?>";
    });
  }
}
