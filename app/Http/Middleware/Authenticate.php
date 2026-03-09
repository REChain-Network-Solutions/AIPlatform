<?php

namespace App\Http\Middleware;

use Closure;
use Google2FA;
use Illuminate\Auth\AuthenticationException;
use Illuminate\Auth\Middleware\Authenticate as Middleware;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;

class Authenticate extends Middleware
{
    public function handle($request, Closure $next, ...$guards)
    {
        $this->authenticate($request, $guards);

        $user = Auth::user();

        // Skip early if no authenticated user
        if (! $user) {
            return $next($request);
        }

        // Handle 2FA check
        if (Google2FA::isActivated() && ! session()->has('save_login_2fa')) {
            Auth::logout();

            // Store only ID, not full user object (lighter in session)
            session()->put('user_id', $user->getAuthIdentifier());

            return redirect()->route('2fa.login');
        }

        return $next($request);
    }

    protected function unauthenticated($request, array $guards)
    {
        $message = $request->routeIs('dashboard.user.openai.chat.*')
            ? 'Please log in to your account to start using Live Chat.'
            : 'Unauthenticated.';

        throw new AuthenticationException($message, $guards, $this->redirectTo($request));
    }

    protected function redirectTo(Request $request): ?string
    {
        return $request->expectsJson() ? null : route('login');
    }
}
