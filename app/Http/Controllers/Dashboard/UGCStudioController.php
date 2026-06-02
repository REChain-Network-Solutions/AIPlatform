<?php

declare(strict_types=1);

namespace App\Http\Controllers\Dashboard;

use App\Http\Controllers\Controller;
use App\Services\UGCStudio\UGCSourceRegistry;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\View\View;

class UGCStudioController extends Controller
{
    private const PER_PAGE = 12;

    private const BUCKETS = ['today', 'previous'];

    public function __invoke(Request $request, UGCSourceRegistry $registry): View
    {
        $user = $request->user();
        $sources = $registry->all();

        $todayEntries = collect();
        $previousEntries = collect();
        $paging = [];

        foreach ($sources as $source) {
            $today = $source->pagedOutputs($user, 'today', 1, self::PER_PAGE);
            $previous = $source->pagedOutputs($user, 'previous', 1, self::PER_PAGE);

            $todayEntries = $todayEntries->concat($today['entries']);
            $previousEntries = $previousEntries->concat($previous['entries']);

            $paging[$source->key] = [
                'today'    => ['has_more' => $today['has_more'], 'next_page' => 2],
                'previous' => ['has_more' => $previous['has_more'], 'next_page' => 2],
            ];
        }

        return view('panel.user.ugc-studio.index', [
            'sources'         => $sources,
            'todayEntries'    => $todayEntries->values(),
            'previousEntries' => $previousEntries->values(),
            'paging'          => $paging,
            'perPage'         => self::PER_PAGE,
        ]);
    }

    public function outputs(Request $request, UGCSourceRegistry $registry): JsonResponse
    {
        $data = $request->validate([
            'source' => 'required|string',
            'bucket' => 'required|string|in:today,previous',
            'page'   => 'required|integer|min:2',
        ]);

        $source = $registry->get($data['source']);

        if ($source === null) {
            return response()->json(['error' => __('Unknown source.')], 422);
        }

        $page = (int) $data['page'];
        $result = $source->pagedOutputs($request->user(), $data['bucket'], $page, self::PER_PAGE);

        return response()->json([
            'entries'   => $result['entries'],
            'has_more'  => $result['has_more'],
            'next_page' => $page + 1,
        ]);
    }
}
