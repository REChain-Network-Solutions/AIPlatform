<?php

namespace App\Http\Controllers\Dashboard;

use App\Http\Controllers\Controller;
use Exception;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Artisan;
use Illuminate\Support\Facades\Log;

class DebugController extends Controller
{
    public function __invoke()
    {
        $currentDebugValue = env('APP_DEBUG', false);
        $newDebugValue = ! $currentDebugValue;
        $envContent = file_get_contents(base_path('.env'));
        $envContent = preg_replace('/^APP_DEBUG=.*/m', 'APP_DEBUG=' . ($newDebugValue ? 'true' : 'false'), $envContent);
        file_put_contents(base_path('.env'), $envContent);
        Artisan::call('config:clear');

        return redirect()->back()->with('message', 'Debug mode updated successfully.');
    }

    public function convertChannel(Request $request): JsonResponse
    {
        try {
            if (! auth()->user()->isAdmin()) {
                return response()->json([
                    'success' => false,
                    'message' => 'Unauthorized action.',
                ], 403);
            }

            $enableBeta = $request->input('enable_beta', false) === true || $request->input('enable_beta') === 'true';

            // Path to .env file
            $envPath = base_path('.env');

            if (! file_exists($envPath)) {
                return response()->json([
                    'success' => false,
                    'message' => 'Environment file not found. Please contact system administrator.',
                ], 500);
            }

            // Check if file is writable
            if (! is_writable($envPath)) {
                return response()->json([
                    'success' => false,
                    'message' => '.env file is not writable. Please check file permissions.',
                ], 500);
            }

            // Read current .env content
            $envContent = file_get_contents($envPath);

            // Determine new value - use empty string for stable, 'tester' for beta
            $newValue = $enableBeta ? 'tester' : 'normal';

            // Check if MAGICAI_USER_TYPE exists in .env
            if (preg_match('/^MAGICAI_USER_TYPE=.*$/m', $envContent)) {
                // Update existing value - wrap in quotes if empty to ensure it's set
                $envContent = preg_replace(
                    '/^MAGICAI_USER_TYPE=.*$/m',
                    'MAGICAI_USER_TYPE="' . $newValue . '"',
                    $envContent
                );
            } else {
                // Add new line at the end of file
                $envContent = rtrim($envContent) . "\nMAGICAI_USER_TYPE=\"" . $newValue . "\"\n";
            }

            // Write back to .env file
            $writeResult = file_put_contents($envPath, $envContent);

            if ($writeResult === false) {
                return response()->json([
                    'success' => false,
                    'message' => 'Failed to write to .env file. Please check file permissions.',
                ], 500);
            }

            // Clear all caches to ensure new env value is loaded
            try {
                Artisan::call('optimize:clear');
            } catch (Exception $e) {
                Log::warning('Failed to clear cache after channel change: ' . $e->getMessage());
            }

            $channelName = $enableBeta ? 'Beta' : 'Stable';

            return response()->json([
                'success' => true,
                'message' => "Successfully switched to {$channelName} update channel. Page will reload to apply changes.",
                'channel' => $enableBeta ? 'beta' : 'stable',
                'reload'  => true, // Signal frontend to reload
            ]);

        } catch (Exception $e) {
            Log::error('Error changing update channel: ' . $e->getMessage(), [
                'trace' => $e->getTraceAsString(),
            ]);

            return response()->json([
                'success' => false,
                'message' => 'An unexpected error occurred. Please try again or contact support.',
            ], 500);
        }
    }
}
