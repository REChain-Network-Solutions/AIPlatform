<?php

namespace App\Domains\Marketplace\Http\Middleware;

use App\Domains\Marketplace\Repositories\ExtensionManagerRepository;
use Closure;
use Illuminate\Http\Request;
use Symfony\Component\HttpFoundation\Response;

class ExtensionManagerMiddleware
{
    public function handle(Request $request, Closure $next, string $registerKey): Response
    {
        return app(ExtensionManagerRepository::class)->next($request, $next, $registerKey);
    }
}
