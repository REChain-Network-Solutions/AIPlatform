<?php

namespace App\Http\Middleware;

use App\Helpers\Classes\Helper;
use Closure;
use Illuminate\Http\Request;
use Symfony\Component\HttpFoundation\Response;

class EnsureVideoAgentIsNotDemo
{
    /**
     * Handle an incoming request.
     *
     * @param  Closure(Request): (Response)  $next
     */
    public function handle(Request $request, Closure $next): Response
    {
        if (Helper::appIsNotDemo() || $request->isMethodSafe()) {
            return $next($request);
        }

        $message = __('This feature is disabled in demo mode.');

        if ($request->expectsJson()) {
            return response()->json(['message' => $message], Response::HTTP_FORBIDDEN);
        }

        return back()->withInput()->with([
            'type'    => 'error',
            'message' => $message,
        ]);
    }
}
