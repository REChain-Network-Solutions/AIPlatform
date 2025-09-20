<?php

namespace App\Http\Middleware\Custom;

use App\Helpers\Classes\Helper;
use App\Helpers\Classes\TableSchema;
use Closure;
use Igaster\LaravelTheme\Facades\Theme;
use Illuminate\Http\Request;
use Symfony\Component\HttpFoundation\Response;

class ThemeMiddleware
{
    public function handle(Request $request, Closure $next): Response
    {
        if (Helper::dbConnectionStatus() && TableSchema::hasTable('app_settings', app('magicai_tables'))) {
            $this->setDefaultSettings();

            $activated_front_theme = setting('front_theme');
            $activated_dash_theme = setting('dash_theme');

            $sameTheme = $activated_front_theme === $activated_dash_theme;

            $isDashboard = request()->is('dashboard*', '*/dashboard*');

            $themeToSet = match (true) {
                $sameTheme   => $activated_front_theme,
                $isDashboard => $activated_dash_theme,
                default      => $activated_front_theme,
            };

            Theme::set($themeToSet);
        }

        return $next($request);
    }

    protected function setDefaultSettings(): void
    {
        if (setting('front_theme') === null) {
            setting(['front_theme' => 'default'])->save();
        }
        if (setting('dash_theme') === null) {
            setting(['dash_theme' => 'default'])->save();
        }
    }
}
