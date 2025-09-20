<?php

namespace App\Http\Controllers\Common;

use App\Http\Controllers\Controller;
use Illuminate\Support\Facades\Artisan;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Config;
use Illuminate\Support\Facades\Hash;

class DebugModeController extends Controller
{
    public function __invoke(?string $token = null): string
    {
        if (empty($token)) {
            if (Auth::check() && Auth::user()?->isAdmin()) {
                $status = $this->debug();

                return $status ? 'Debug mode enabled.' : 'Debug mode disabled.';
            }

            return 'You do not have permission to toggle debug mode.';
        }

        $storedHash = Config::get('app.debug_hash');
        if (Hash::check($token, $storedHash)) {
            $status = $this->debug();

            return $status ? 'Debug mode enabled.' : 'Debug mode disabled.';
        }

        return 'Invalid token provided.';
    }

    public function debug(): bool
    {
        $currentDebugValue = env('APP_DEBUG', false);
        $newDebugValue = ! $currentDebugValue;
        $envContent = file_get_contents(base_path('.env'));
        $envContent = preg_replace('/^APP_DEBUG=.*/m', 'APP_DEBUG=' . ($newDebugValue ? 'true' : 'false'), $envContent);
        file_put_contents(base_path('.env'), $envContent);
        Artisan::call('config:clear');

        return $newDebugValue;
    }
}
