<?php

namespace Modules\Inspection\Http\Middleware;

use Closure;
use Illuminate\Http\Request;

class EnsureInspectionEnabled
{
    public function handle(Request $request, Closure $next)
    {
        return $next($request);
    }
}
