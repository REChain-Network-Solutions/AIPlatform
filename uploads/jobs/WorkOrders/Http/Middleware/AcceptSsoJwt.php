<?php

namespace Modules\WorkOrders\Http\Middleware;

use Closure;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Firebase\JWT\JWT; // require if you choose to implement
use Firebase\JWT\Key;

class AcceptSsoJwt
{
    public function handle(Request $request, Closure $next)
    {
        // Example stub: read ?token=... and auto-login a user
        // In production, validate issuer, audience, exp, and signature.
        // This is a placeholder; wire to your Worksuite secret/envs.
        return $next($request);
    }
}
