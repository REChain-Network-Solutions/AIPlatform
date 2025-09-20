<?php

namespace App\Http\Controllers\Dashboard;

use App\Http\Controllers\Controller;
use App\Models\OpenAIGenerator;
use App\Models\OpenaiGeneratorChatCategory;
use App\Models\UserOpenai;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Throwable;

class SearchController extends Controller
{
    public function search(Request $request)
    {
        $word = $request->search;
        $result = '';
        $keywords = null;

        if ($word == 'delete') {
            $template_search = [];
            $workbook_search = [];
            $ai_chat_search = [];
        } else {
            $keywords = $this->addSearchKey($word);
            $template_search = OpenAIGenerator::where('title', 'like', "%$word%")->get();

            $workbook_search = UserOpenai::where('user_id', auth()->user()->id)->where('title', 'like', "%$word%")->with('generator')->get();

            $ai_chat_search = OpenaiGeneratorChatCategory::where('slug', '<>', 'ai_webchat')->where('slug', '<>', 'ai_vision')->where('slug', '<>', 'ai_pdf')->where('name', 'like', "%$word%")->orWhere('description', 'like', "%$word%")->get();

            if (count($template_search) == 0 and count($workbook_search) == 0 and count($ai_chat_search) == 0) {
                $result = 'null';
            }
        }
        $html = view('panel.layout.includes.search-results', compact('template_search', 'workbook_search', 'ai_chat_search', 'result'))->render();

        return response()->json(compact('html', 'keywords'));
    }

    // add search key
    public function addSearchKey(string $keyword)
    {
        $user = auth()->user();

        $user->recentSearchKeys()->where('keyword', $keyword)->delete();
        $user->recentSearchKeys()->create(['keyword' => $keyword]);

        if ($user->recentSearchKeys()->count() > 10) {
            $user->recentSearchKeys()->orderBy('created_at')->limit(1)->delete();
        }

        return $user->recentSearchKeys()->orderByDesc('created_at')->get();
    }

    // delete search key
    public function deleteSearchkey(string $key): JsonResponse
    {
        try {
            auth()->user()->recentSearchKeys()->where('keyword', $key)->delete();

            return response()->json([
                'status' => 'success',
            ]);
        } catch (Throwable $th) {
            return response()->json([
                'status'  => 'error',
                'message' => 'Failed to delete key',
            ], 500);
        }
    }

    // recent search keys
    public function recentSearchKeys(): JsonResponse
    {
        $keys = auth()->user()->recentSearchKeys()->get();

        return response()->json(compact('keys'));
    }

    // recent lunch
    public function recentLunch()
    {
        $recently_launched = UserOpenai::query()
            ->where('user_id', Auth::id())
            ->orderBy('updated_at', 'desc')
            ->limit(5)
            ->with('generator')
            ->get();
        $html = view('panel.layout.includes.recently-lunched', compact('recently_launched'))->render();

        return response()->json(compact('html'));
    }
}
