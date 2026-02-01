<?php

namespace App\Extensions\SEOTool\System\Http\Controllers;

use App\Extensions\SEOTool\System\Services\SEOService;
use App\Http\Controllers\Controller;
use Exception;
use Illuminate\Contracts\View\View;
use Illuminate\Http\Request;

class SeoController extends Controller
{
    public function index(): View
    {
        return view('seo-tool::index');
    }

    public function suggestKeywords(Request $request): \Illuminate\Http\JsonResponse
    {
        $keywords = SEOService::getKeywords($request->keyword);

        return response()->json(['result' => $keywords])->header('Content-Type', 'application/json');
    }

    public function analyseArticle(Request $request): \Illuminate\Http\JsonResponse
    {
        $analizingResult = SEOService::analiyzeWithAI($request);
        $percentage = $analizingResult['percentage'];
        $competitorList = $analizingResult['competitorList'];
        $longTailList = $analizingResult['longTailList'];

        return response()->json(['competitorList' => $competitorList, 'percentage'=> $percentage, 'longTailList' => $longTailList])->header('Content-Type', 'application/json');
    }

    public function improveArticle(Request $request)
    {
        return SEOService::improveWithAI($request);
    }

    // article wizard
    public function generateKeywords(Request $request): ?\Illuminate\Http\JsonResponse
    {
        try {
            $keywords = SEOService::getKeywords($request->topic);
            $jsonKeywords = collect($keywords)->map(function ($keyword) {
                return json_encode($keyword, JSON_THROW_ON_ERROR);
            });
            $jsonKeywords = "[\n" . $jsonKeywords->implode(",\n") . "\n]";

            return response()->json(['result' => $jsonKeywords])->header('Content-Type', 'application/json');
        } catch (Exception $e) {
            return response()->json([
                'message' => $e->getMessage(),
            ], 500);
        }
    }

    public function genSearchQuestions(Request $request): ?\Illuminate\Http\JsonResponse
    {
        try {
            $stringQuestions = SEOService::getSearchQuestions($request);

            return response()->json(['result' => $stringQuestions])->header('Content-Type', 'application/json');
        } catch (Exception $e) {
            return response()->json([
                'message' => $e->getMessage(),
            ], 500);
        }
    }

    // admin seo helpers
    public function genSEO(Request $request): ?\Illuminate\Http\JsonResponse
    {
        try {
            $result = SEOService::generateSEO($request);

            return response()->json(['result' => $result])->header('Content-Type', 'application/json');
        } catch (Exception $e) {
            return response()->json([
                'message' => $e->getMessage(),
            ], 500);
        }
    }
}
