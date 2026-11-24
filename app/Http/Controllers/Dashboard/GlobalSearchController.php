<?php

namespace App\Http\Controllers\Dashboard;

use App\Domains\Entity\Facades\Entity;
use App\Helpers\Classes\ApiHelper;
use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use App\Models\RecentSearchKey;
use App\Models\Usage;
use App\Services\Common\GlobalSearchService;
use Exception;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Collection;
use Illuminate\Support\Facades\DB;
use OpenAI\Laravel\Facades\OpenAI;
use Symfony\Component\HttpFoundation\StreamedResponse;

class GlobalSearchController extends Controller
{
    public function __construct(private readonly GlobalSearchService $searchService) {}

    public function search(Request $request): JsonResponse
    {
        $request->validate(['search' => ['nullable', 'string', 'max:200']]);

        $query = trim((string) $request->string('search'));
        $user = auth()->user();

        if (empty($query)) {
            return response()->json(['groups' => [], 'keywords' => []]);
        }

        $groups = $this->searchService->search($user, $query);
        $keywords = $this->recordAndFetchKeywords($query);

        return response()->json(compact('groups', 'keywords'));
    }

    public function aiStream(Request $request): StreamedResponse
    {
        $request->validate(['query' => ['required', 'string', 'max:1000']]);

        $query = (string) $request->string('query');
        $user = auth()->user();
        $entityEnum = Helper::defaultWordModel();
        $driver = Entity::driver($entityEnum)->forUser($user);

        if (! $driver->hasCreditBalance()) {
            return response()->stream(function () {
                echo "event: data\n";
                echo 'data: ' . json_encode(['error' => __('Insufficient credits.')]) . "\n\n";
                echo "event: stop\n";
                echo "data: [DONE]\n\n";
                flush();
            }, 200, [
                'Cache-Control'     => 'no-cache',
                'X-Accel-Buffering' => 'no',
                'Content-Type'      => 'text/event-stream',
            ]);
        }

        ApiHelper::setOpenAiKey();
        $model = $entityEnum->value;

        return response()->stream(function () use ($query, $model, $driver) {
            $outputText = '';

            try {
                $stream = OpenAI::chat()->createStreamed([
                    'model'    => $model,
                    'messages' => [
                        ['role' => 'system', 'content' => 'You are a helpful assistant embedded in a SaaS application. Give concise, direct answers. Use plain text, no markdown headers.'],
                        ['role' => 'user', 'content' => $query],
                    ],
                    'stream' => true,
                ]);

                foreach ($stream as $response) {
                    $chunk = $response->choices[0]->delta->content ?? null;

                    if ($chunk !== null) {
                        if (connection_aborted()) {
                            break;
                        }
                        $outputText .= $chunk;
                        echo PHP_EOL;
                        echo "event: data\n";
                        echo 'data: ' . json_encode(['text' => $chunk]);
                        echo "\n\n";
                        flush();
                    }
                }
            } catch (Exception $e) {
                echo "event: data\n";
                echo 'data: ' . json_encode(['error' => $e->getMessage()]) . "\n\n";
                flush();
            }

            if ($outputText !== '') {
                $wordCount = countWords($outputText);
                $driver->decreaseCredit($wordCount);
                Usage::getSingle()->updateWordCounts($wordCount);
            }

            echo "event: stop\n";
            echo "data: [DONE]\n\n";
            flush();
        }, 200, [
            'Cache-Control'     => 'no-cache',
            'X-Accel-Buffering' => 'no',
            'Content-Type'      => 'text/event-stream',
        ]);
    }

    private function recordAndFetchKeywords(string $keyword): Collection
    {
        $userId = auth()->id();

        if (! $userId) {
            return collect();
        }

        DB::transaction(static function () use ($userId, $keyword) {
            RecentSearchKey::where('user_id', $userId)->where('keyword', $keyword)->delete();
            RecentSearchKey::create(['user_id' => $userId, 'keyword' => $keyword]);

            $keepIds = RecentSearchKey::where('user_id', $userId)
                ->orderByDesc('created_at')
                ->limit(10)
                ->pluck('id');

            if ($keepIds->isNotEmpty()) {
                RecentSearchKey::where('user_id', $userId)
                    ->whereNotIn('id', $keepIds)
                    ->delete();
            }
        });

        return RecentSearchKey::where('user_id', $userId)
            ->orderByDesc('created_at')
            ->limit(10)
            ->get(['id', 'keyword', 'created_at']);
    }
}
