<?php

namespace Modules\ComplianceIQ\Providers;

use Illuminate\Foundation\Support\Providers\RouteServiceProvider as ServiceProvider;
use Illuminate\Support\Facades\Route;

class RouteServiceProvider extends ServiceProvider
{
    protected string $namespace = 'Modules\\ComplianceIQ\\Http\\Controllers';

    public function map()
    {
        $this->mapAdminRoutes();
        $this->mapApiRoutes();
    }

    \1
        // Donor bridge: mount donor routes under /admin/compliance/donor
        Route::middleware(['web','auth','admin'])
            ->prefix('admin/compliance/donor')
            ->as('admin.compliance.donor.')
            ->namespace($this->namespace.'\Donor')
            ->group(__DIR__.'/../Routes/donor_admin_bridge.php');
    
        Route::middleware(['web','auth','admin']) // adapt guard if needed
            ->prefix('admin/compliance')
            ->as('admin.compliance.')
            ->namespace($this->namespace.'\\Admin')
            ->group(__DIR__.'/../Routes/admin.php');
    }

    protected function mapApiRoutes()
    {
        Route::middleware(['api','auth:sanctum'])
            ->prefix('api/compliance')
            ->as('api.compliance.')
            ->namespace($this->namespace.'\\Api')
            ->group(__DIR__.'/../Routes/api.php');
    }
}
