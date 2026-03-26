<?php

namespace Modules\Inspection\Support;

use Illuminate\Support\Facades\Route;

final class InspectionMenu
{
    public static function url(string $routeName): ?string
    {
        return Route::has($routeName) ? route($routeName) : null;
    }
}
