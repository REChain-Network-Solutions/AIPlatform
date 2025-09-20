<?php

namespace App\Http\Middleware;

use App\Helpers\Classes\Helper;
use Closure;
use Illuminate\Http\Request;
use Symfony\Component\HttpFoundation\Response;

class DemoCheckMiddleware
{
    /**
     * Handle an incoming request.
     *
     * @param  Closure(\Illuminate\Http\Request): (\Symfony\Component\HttpFoundation\Response)  $next
     */
    public function handle(Request $request, Closure $next): Response
    {
        if (Helper::appIsDemo()) {
            $accept = $request->header('accept', 'text/html');
            if (str_contains($accept, 'application/json')) {
                response()->json([
                    'status'  => 'error',
                    'message' => trans('This feature is disabled in demo mode.'),
                ], 403);
            } else {
                return to_route('dashboard.user.index')->with([
                    'status'  => 'error',
                    'message' => trans('This feature is disabled in demo mode.'),
                ]);
            }
        }

        return $next($request);
    }
}
