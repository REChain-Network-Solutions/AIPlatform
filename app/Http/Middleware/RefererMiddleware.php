<?php

namespace App\Http\Middleware;

use App\Models\Referer;
use Carbon\Carbon;
use Closure;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Cookie;
use Illuminate\Support\Facades\Session;
use Illuminate\Support\Str;
use Symfony\Component\HttpFoundation\Response;

class RefererMiddleware
{
    /**
     * Handle an incoming request.
     *
     * @param  Closure(\Illuminate\Http\Request): (\Symfony\Component\HttpFoundation\Response)  $next
     */
    public function handle(Request $request, Closure $next): Response
    {
        if (Cookie::has('referer_session_id')) {
            return $next($request);
        }

        $referer = $request->header('referer');
        $domain = parse_url($referer)['host'] ?? '';
        $randSessionId = Str::random(40);

        if ($domain) {
            Referer::create([
                'session_id' => Session::getId(),
                'referer'    => $referer,
                'domain'     => $domain,
                'created_at' => Carbon::now(),
            ]);
        }

        Cookie::queue('referer_session_id', $randSessionId, 199999);

        return $next($request);
    }
}
