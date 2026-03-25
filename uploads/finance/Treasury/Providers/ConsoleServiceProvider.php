<?php
namespace Modules\Treasury\Providers;
use Illuminate\Support\ServiceProvider;
use Modules\Treasury\Console\Commands\PostPaymentRun;

class ConsoleServiceProvider extends ServiceProvider
{
    public function register(): void
    {
        $this->commands([PostPaymentRun::class]);
    }
}
