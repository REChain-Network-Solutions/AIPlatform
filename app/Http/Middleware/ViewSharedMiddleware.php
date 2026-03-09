<?php

namespace App\Http\Middleware;

use App\Models\UserOpenai;
use Closure;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\View;
use Symfony\Component\HttpFoundation\Response;

class ViewSharedMiddleware
{
    /**
     * Handle an incoming request.
     *
     * @param  Closure(Request): (Response)  $next
     */
    public function handle(Request $request, Closure $next): Response
    {
        if (Auth::check()) {
            $userId = Auth::id();
            $cacheKeys = [
                'total_words'            => "total_words_{$userId}",
                'total_documents'        => "total_documents_{$userId}",
                'total_text_documents'   => "total_text_documents_{$userId}",
                'total_image_documents'  => "total_image_documents_{$userId}",
            ];

            $missingKeys = collect($cacheKeys)->filter(function ($key) {
                return ! Cache::has($key);
            });

            if ($missingKeys->isNotEmpty()) {
                $documents = UserOpenai::where('user_id', $userId);

                $totalWords = (clone $documents)->sum('credits');
                $totalDocuments = (clone $documents)->count();
                $totalTextDocuments = (clone $documents)->where('credits', '!=', 1)->count();
                $totalImageDocuments = (clone $documents)->where('credits', 1)->count();

                foreach ([
                    $cacheKeys['total_words']           => $totalWords,
                    $cacheKeys['total_documents']       => $totalDocuments,
                    $cacheKeys['total_text_documents']  => $totalTextDocuments,
                    $cacheKeys['total_image_documents'] => $totalImageDocuments,
                ] as $key => $value) {
                    Cache::put($key, $value, now()->addMinutes(360));
                }
            }

            foreach ($cacheKeys as $varName => $cacheKey) {
                View::share($varName, Cache::get($cacheKey, 0));
            }
        }

        return $next($request);
    }
}
