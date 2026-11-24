<?php

namespace App\Http\Controllers\Common;

use App\Http\Controllers\Controller;
use Carbon\Carbon;
use Illuminate\Contracts\View\View;
use Illuminate\Http\JsonResponse;
use Illuminate\Support\Facades\App;
use Illuminate\Support\Facades\Artisan;
use Illuminate\Support\Facades\Log;
use Spatie\Health\Commands\RunHealthChecksCommand;
use Spatie\Health\ResultStores\ResultStore;
use Throwable;

class HealthController extends Controller
{
    public function index(): View
    {
        abort_unless(auth()->user()?->isAdmin(), 403);

        $resultStore = App::make(ResultStore::class);
        $checkResults = $resultStore->latestResults();

        // call new status when visit the page
        Artisan::call(RunHealthChecksCommand::class);

        return view('panel.admin.health.index', [
            'lastRanAt'    => new Carbon($checkResults?->finishedAt),
            'checkResults' => $checkResults,
        ]);
    }

    public function logs(): View
    {
        abort_unless(auth()->user()?->isAdmin(), 403);

        return view('panel.admin.health.logs', [
            'logContent' => $this->redactedLogTail(500, 200 * 1024),
        ]);
    }

    public function cacheClear(): JsonResponse
    {
        abort_unless(auth()->user()?->isAdmin(), 403);

        try {
            Artisan::call('optimize:clear');

            Log::info('Health cache cleared', ['user_id' => auth()->id()]);

            return response()->json(['success' => true]);
        } catch (Throwable $th) {
            Log::error('Health cache clear failed', ['user_id' => auth()->id(), 'error' => $th->getMessage()]);

            return response()->json(['success' => false, 'message' => __('An error occurred while clearing the cache.')], 500);
        }
    }

    private function redactedLogTail(int $maxLines = 500, int $maxBytes = 204800): string
    {
        $logFile = storage_path('logs/laravel.log');

        if (! is_readable($logFile)) {
            return __('No logged any data.');
        }

        $content = $this->readFileTail($logFile, $maxBytes);

        $lines = explode("\n", $content);

        if (count($lines) > $maxLines) {
            $lines = array_slice($lines, -$maxLines);
        }

        $content = implode("\n", $lines);
        $content = (string) mb_convert_encoding($content, 'UTF-8', 'UTF-8');

        $content = (string) preg_replace('/\bsk(?:-proj)?[-_][A-Za-z0-9\-_]{8,}\b/', '[REDACTED]', $content);
        $content = (string) preg_replace('/\b(?:ghp_[A-Za-z0-9]{8,}|gho_[A-Za-z0-9]{8,}|github_pat_[A-Za-z0-9_]{8,}|xox[bpas]-[A-Za-z0-9\-]{8,}|AIza[0-9A-Za-z\-_]{35})\b/', '[REDACTED]', $content);
        $content = (string) preg_replace('/\bBearer\s+[A-Za-z0-9\-._~+\/=]{8,}/i', 'Bearer [REDACTED]', $content);
        $content = (string) preg_replace(
            '/((api[_-]?key|api[_-]?secret|secret|token|password|authorization)["\']?\s*[:=]\s*["\']?)([^\s"\'`,;}\]]+)/i',
            '$1[REDACTED]',
            $content
        );

        return $content;
    }

    private function readFileTail(string $path, int $maxBytes): string
    {
        $size = filesize($path);

        if ($size === false || $size <= 0) {
            return '';
        }

        $handle = fopen($path, 'rb');

        if ($handle === false) {
            return '';
        }

        try {
            $offset = max(0, $size - $maxBytes);

            if ($offset > 0) {
                fseek($handle, $offset);
            }

            $content = stream_get_contents($handle);

            if (! is_string($content)) {
                return '';
            }

            if ($offset > 0) {
                $firstNewline = strpos($content, "\n");

                if ($firstNewline !== false) {
                    $content = substr($content, $firstNewline + 1);
                }
            }

            return $content;
        } finally {
            fclose($handle);
        }
    }
}
