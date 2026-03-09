<?php

namespace App\Domains\Marketplace\Http\Middleware;

use App\Domains\Marketplace\Services\ExtensionInstallService;
use Closure;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Artisan;
use Illuminate\Support\Facades\Cache;
use Symfony\Component\HttpFoundation\Response;

class NewExtensionInstalled
{
    public function handle(Request $request, Closure $next): Response
    {
        $extensionInstaller = app(ExtensionInstallService::class);
        $cacheKey = $extensionInstaller->getExtensionInstallCache();

        $value = Cache::get($cacheKey);

        if ($value) {
            Cache::forget($cacheKey);
            $this->runExtensionInstallTasks();
        }

        return $next(
            $request->merge([
                'credit-list-cache' => 'credit-list-cache-' . now()->timestamp,
            ])
        );
    }

    /**
     * Run all extension install related Artisan commands.
     */
    protected function runExtensionInstallTasks(): void
    {
        Artisan::call('optimize:clear');
        Artisan::call('cache:clear');
        Artisan::call('migrate', ['--force' => true]);
        Artisan::call('vendor:publish', [
            '--tag'   => 'extension',
            '--force' => true,
        ]);
    }
}
