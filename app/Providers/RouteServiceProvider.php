<?php

declare(strict_types=1);

namespace App\Providers;

use App\Domains\Marketplace\Http\Middleware\NewExtensionInstalled;
use App\Http\Middleware\ViewSharedMiddleware;
use Illuminate\Cache\RateLimiting\Limit;
use Illuminate\Foundation\Support\Providers\RouteServiceProvider as ServiceProvider;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\RateLimiter;
use Illuminate\Support\Facades\Route;
use Illuminate\Support\Str;

class RouteServiceProvider extends ServiceProvider
{
    public const HOME = '/dashboard';

    public function boot(): void
    {
        $this->configureRateLimiting();

        $this->routes(function () {
            Route::middleware('api')
                ->prefix('api')
                ->group(base_path('routes/api.php'));

            Route::middleware([
                'web',  ViewSharedMiddleware::class, NewExtensionInstalled::class,
            ])->group(base_path('routes/web.php'));
        });

        $this->app->booted(function (): void {
            foreach (Route::getRoutes() as $route) {
                $uri = $route->uri();

                if (Str::is([
                    'translations/lang/update/*',
                    'translations/lang/update-all',
                    'translations/lang-save',
                ], $uri)) {
                    $route->middleware(['auth', 'admin', 'throttle:30,1']);
                }

                if (Str::is([
                    'image/upload',
                    'images/upload',
                    'files/upload',
                    'pdf/getContent',
                    'rss/fetch',
                ], $uri)) {
                    $route->middleware(['auth', 'throttle:30,1']);
                }

                if (Str::is([
                    'dashboard/*',
                    'ai-image-pro/*',
                    'ai-chat-pro-memory/*',
                    'chatbot-voice/checkVoiceBalance',
                    'share/link',
                    'broadcasting/auth',
                ], $uri)) {
                    $route->middleware('auth');
                }

                if (Str::is('dashboard/admin/*', $uri)) {
                    $route->middleware('admin');
                }
            }
        });
    }

    protected function configureRateLimiting(): void
    {
        RateLimiter::for('api', static function (Request $request) {
            return Limit::perMinute(60)->by($request->user()?->id ?: $request->ip());
        });
    }
}
