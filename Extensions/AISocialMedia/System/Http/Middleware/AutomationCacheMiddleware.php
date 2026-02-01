<?php

namespace App\Extensions\AISocialMedia\System\Http\Middleware;

use Closure;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Cache;
use Symfony\Component\HttpFoundation\Response;

class AutomationCacheMiddleware
{
    public function handle(Request $request, Closure $next): Response
    {
        $hasAutomationData = Cache::has('automation:data');
        $isAutomationStepSecond = $request->routeIs('dashboard.user.automation.step.second');

        if ($isAutomationStepSecond) {
            if (($request->isMethod('GET') && $hasAutomationData) || $request->isMethod('POST')) {
                return $next($request);
            }

            return redirect()->route('dashboard.user.automation.index');
        }

        if ($hasAutomationData) {
            return $next($request);
        }

        return redirect()->route('dashboard.user.automation.index');
    }
}
