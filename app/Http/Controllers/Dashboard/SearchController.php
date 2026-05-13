<?php

namespace App\Http\Controllers\Dashboard;

use App\Enums\AiImageStatusEnum;
use App\Extensions\AiChatProImageChat\System\Models\AiChatProImageModel;
use App\Extensions\AIImagePro\System\Models\AiImageProModel;
use App\Extensions\AiVideoPro\System\Models\UserFall;
use App\Helpers\Classes\MarketplaceHelper;
use App\Http\Controllers\Controller;
use App\Models\OpenAIGenerator;
use App\Models\OpenaiGeneratorChatCategory;
use App\Models\RecentSearchKey;
use App\Models\UserOpenai;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Collection;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\DB;
use Throwable;

class SearchController extends Controller
{
    public function search(Request $request)
    {
        $word = $request->search;
        $result = '';
        $keywords = null;
        $video_search = collect();
        $ai_image_pro_search = collect();
        $ai_chat_pro_image_search = collect();

        if ($word == 'delete') {
            $template_search = [];
            $workbook_search = [];
            $ai_chat_search = [];
        } else {
            // Add search key asynchronously or optimize it
            $keywords = $this->addSearchKeyOptimized($word);

            // Run searches in parallel if possible, or at least optimize queries
            $userId = auth()->id();

            $template_search = OpenAIGenerator::where('title', 'like', "%$word%")
                ->select('id', 'title', 'slug', 'color', 'image', 'active', 'premium', 'type')
                ->get();

            $workbook_search = UserOpenai::where('user_id', $userId)
                ->where(function ($query) use ($word) {
                    return $query->where('title', 'like', "%$word%")
                        ->orWhere('output', 'like', "%$word%")
                        ->orWhere('response', 'like', "%$word%")
                        ->orWhere('input', 'like', "%$word%");
                })
                ->select('id', 'title', 'slug', 'user_id', 'openai_id', 'input', 'output', 'response')
                ->with('generator:id,title,slug,color,image,type')
                ->get();

            $ai_chat_search = OpenaiGeneratorChatCategory::whereNotIn('slug', ['ai_webchat', 'ai_vision', 'ai_pdf'])
                ->where(function ($query) use ($word) {
                    $query->where('name', 'like', "%$word%")
                        ->orWhere('description', 'like', "%$word%");
                })
                ->select('id', 'name', 'slug', 'short_name', 'description', 'color')
                ->get();

            // AI Video Pro search
            if (class_exists(UserFall::class)) {
                $video_search = UserFall::where('user_id', $userId)
                    ->where('status', 'complete')
                    ->whereNotNull('video_url')
                    ->where('video_url', '!=', '')
                    ->where('prompt', 'like', "%$word%")
                    ->select('id', 'prompt', 'video_url', 'model', 'created_at')
                    ->limit(10)
                    ->get();
            }

            // AI Image Pro search
            if (class_exists(AiImageProModel::class) && MarketplaceHelper::isRegistered('ai-image-pro')) {
                $ai_image_pro_search = AiImageProModel::where('user_id', $userId)
                    ->where('status', AiImageStatusEnum::COMPLETED->value)
                    ->whereNotNull('generated_images')
                    ->where('prompt', 'like', "%$word%")
                    ->select('id', 'prompt', 'generated_images', 'created_at')
                    ->limit(10)
                    ->get();
            }

            // AI Chat Pro Image Chat search
            if (class_exists(AiChatProImageModel::class) && MarketplaceHelper::isRegistered('ai-chat-pro-image-chat')) {
                $ai_chat_pro_image_search = AiChatProImageModel::where('user_id', $userId)
                    ->where('status', AiImageStatusEnum::COMPLETED->value)
                    ->whereNotNull('generated_images')
                    ->where('prompt', 'like', "%$word%")
                    ->select('id', 'prompt', 'generated_images', 'created_at')
                    ->limit(10)
                    ->get();
            }

            if ($template_search->isEmpty() && $workbook_search->isEmpty() && $ai_chat_search->isEmpty() && $video_search->isEmpty() && $ai_image_pro_search->isEmpty() && $ai_chat_pro_image_search->isEmpty()) {
                $result = 'null';
            }
        }

        $html = view('panel.layout.includes.search-results', compact('template_search', 'workbook_search', 'ai_chat_search', 'video_search', 'ai_image_pro_search', 'ai_chat_pro_image_search', 'result'))->render();

        return response()->json(compact('html', 'keywords', 'workbook_search'));
    }

    /**
     * Optimized version - Single query with upsert
     */
    public function addSearchKeyOptimized(string $keyword): Collection
    {
        $userId = auth()->id();

        if (! $userId) {
            return collect();
        }

        DB::transaction(static function () use ($userId, $keyword) {
            // Delete existing keyword if it exists
            RecentSearchKey::where('user_id', $userId)
                ->where('keyword', $keyword)
                ->delete();

            // Insert new keyword
            RecentSearchKey::create([
                'user_id' => $userId,
                'keyword' => $keyword,
            ]);

            // Keep only the 10 most recent - delete older ones in a single query
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

        // Return recent keys with a single query
        return RecentSearchKey::where('user_id', $userId)
            ->orderByDesc('created_at')
            ->limit(10)
            ->get(['id', 'keyword', 'created_at']); // Only select needed columns
    }

    /**
     * Delete search key - optimized
     */
    public function deleteSearchkey(string $key): JsonResponse
    {
        try {
            $userId = auth()->id();

            RecentSearchKey::where('user_id', $userId)
                ->where('keyword', $key)
                ->delete();

            return response()->json(['status' => 'success']);
        } catch (Throwable $th) {
            return response()->json([
                'status'  => 'error',
                'message' => 'Failed to delete key',
            ], 500);
        }
    }

    /**
     * Recent search keys - optimized
     */
    public function recentSearchKeys(): JsonResponse
    {
        $keys = RecentSearchKey::where('user_id', auth()->id())
            ->orderByDesc('created_at')
            ->limit(10)
            ->get(['id', 'keyword', 'created_at']);

        return response()->json(compact('keys'));
    }

    /**
     * Recent launch - optimized
     */
    public function recentLunch(): JsonResponse
    {
        $recently_launched = UserOpenai::where('user_id', Auth::id())
            ->orderByDesc('updated_at')
            ->limit(5)
            ->with('generator:id,title,slug') // Only load needed columns
            ->get(['id', 'title', 'openai_id', 'updated_at']); // Only select needed columns

        $html = view('panel.layout.includes.recently-lunched', compact('recently_launched'))->render();

        return response()->json(compact('html'));
    }
}
