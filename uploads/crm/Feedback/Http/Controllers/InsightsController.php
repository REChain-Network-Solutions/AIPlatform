<?php

namespace Modules\Feedback\Http\Controllers;

use Illuminate\Routing\Controller;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Cache;
use Modules\Feedback\AI\OpenAIClient;

class InsightsController extends Controller
{
    public function index()
    {
        // Aggregate last 30 days
        $since = now()->subDays(30);
        $nps = DB::table('nps_responses')
            ->where('created_at', '>=', $since)
            ->selectRaw('
                SUM(CASE WHEN score >= 9 THEN 1 ELSE 0 END) as promoters,
                SUM(CASE WHEN score BETWEEN 7 AND 8 THEN 1 ELSE 0 END) as passives,
                SUM(CASE WHEN score <= 6 THEN 1 ELSE 0 END) as detractors,
                COUNT(*) as total
            ')->first();

        $csat = DB::table('csat_responses')
            ->where('created_at', '>=', $since)
            ->selectRaw('
                SUM(CASE WHEN rating=5 THEN 1 ELSE 0 END) as r5,
                SUM(CASE WHEN rating=4 THEN 1 ELSE 0 END) as r4,
                SUM(CASE WHEN rating=3 THEN 1 ELSE 0 END) as r3,
                SUM(CASE WHEN rating=2 THEN 1 ELSE 0 END) as r2,
                SUM(CASE WHEN rating=1 THEN 1 ELSE 0 END) as r1,
                COUNT(*) as total
            ')->first();

        $data = [
            'nps' => [
                'promoters' => (int)($nps->promoters ?? 0),
                'passives' => (int)($nps->passives ?? 0),
                'detractors' => (int)($nps->detractors ?? 0),
                'total' => (int)($nps->total ?? 0),
            ],
            'csat' => [
                'r5' => (int)($csat->r5 ?? 0),
                'r4' => (int)($csat->r4 ?? 0),
                'r3' => (int)($csat->r3 ?? 0),
                'r2' => (int)($csat->r2 ?? 0),
                'r1' => (int)($csat->r1 ?? 0),
                'total' => (int)($csat->total ?? 0),
            ],
        ];

        $summary = Cache::remember('feedback.ai.insights', 600, function () use ($data) {
            $client = new OpenAIClient();
            $smoke = $client->smoke();
            if (!$smoke['ok']) {
                return ['enabled' => false, 'reason' => $smoke['details'], 'bullets' => []];
            }
            $prompt = "You are a customer experience analyst. Given 30-day aggregates, write 3 concise bullet points: 1) key signal, 2) likely root causes, 3) one action. Keep each bullet <= 18 words. Data: "
                . json_encode($data);
            $out = $client->suggestReply(['text' => $prompt]);
            $text = $out['reply'] ?? '';
            // Split into bullets by line
            $bullets = array_values(array_filter(array_map('trim', preg_split('/\r?\n+/', $text))));
            if (count($bullets) > 5) { $bullets = array_slice($bullets, 0, 5); }
            return ['enabled' => true, 'bullets' => $bullets, 'raw' => $text];
        });

        return response()->json(['data' => $data, 'summary' => $summary]);
    }
}
