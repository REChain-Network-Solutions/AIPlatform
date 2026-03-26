<?php

namespace Modules\Feedback\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Routing\Controller;
use Illuminate\Support\Facades\DB;

class NpsController extends Controller
{
    public function index()
    {
        $surveys = DB::table('nps_surveys')->orderByDesc('id')->limit(20)->get();
        // quick aggregate for each survey
        $agg = [];
        foreach ($surveys as $s) {
            $row = DB::table('nps_responses')->selectRaw('
                SUM(CASE WHEN score >= 9 THEN 1 ELSE 0 END) as promoters,
                SUM(CASE WHEN score BETWEEN 7 AND 8 THEN 1 ELSE 0 END) as passives,
                SUM(CASE WHEN score <= 6 THEN 1 ELSE 0 END) as detractors,
                COUNT(*) as total
            ')->where('survey_id', $s->id)->first();
            $agg[$s->id] = $row;
        }
        return view('feedback::nps.index', compact('surveys','agg'));
    }

    public function store(Request $request)
    {
        $id = DB::table('nps_surveys')->insertGetId([
            'title' => $request->string('title'),
            'question' => $request->string('question', 'How likely are you to recommend us to a friend or colleague?'),
            'meta' => json_encode([]),
            'created_at' => now(),
            'updated_at' => now(),
        ]);
        return redirect()->route('feedback.nps.index')->with('status', 'NPS survey created: '.$id);
    }
}
