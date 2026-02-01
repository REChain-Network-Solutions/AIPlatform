<?php

namespace App\Extensions\AIPlagiarism\System\Http\Controllers;

use App\Extensions\AIPlagiarism\System\Services\AIPlagiarismService;
use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use App\Models\OpenAIGenerator;
use App\Models\SettingTwo;
use App\Models\UserOpenai;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;

class PlagiarismController extends Controller
{
    public function plagiarismCheck(Request $request): ?JsonResponse
    {
        return (new AIPlagiarismService)->checkPlagiarism($request->text ?? '');
    }

    public function detectAIContentCheck(Request $request): ?JsonResponse
    {
        return (new AIPlagiarismService)->detectAIContent($request->text ?? '');
    }

    public function plagiarism()
    {
        return view('ai-plagiarism::index');
    }

    public function detectAIContent()
    {
        return view('ai-plagiarism::detectaicontent');
    }

    public function plagiarismSave(Request $request)
    {
        $input = $request->input;
        $text = $request->text;
        $percent = $request->percent;

        $user = Auth::user();

        $post = OpenAIGenerator::where('slug', 'ai_plagiarism')->first();

        $entry = new UserOpenai;
        $entry->title = str($percent) . '% Plagiarism Document';
        $entry->slug = str()->random(7) . str($user->fullName())->slug() . '-workbook';
        $entry->user_id = Auth::id();
        $entry->openai_id = $post->id;
        $entry->input = $input;
        $entry->hash = str()->random(256);
        $entry->credits = 0;
        $entry->words = 0;
        $entry->output = $text;
        $entry->storage = '';
        $entry->response = $text;

        $entry->save();

        return response()->json(['success' => true]);
    }

    public function detectAIContentSave(Request $request)
    {
        $input = $request->input;
        $text = $request->text;
        $percent = $request->percent;

        $user = Auth::user();

        $post = OpenAIGenerator::where('slug', 'ai_content_detect')->first();

        $entry = new UserOpenai;
        $entry->title = str($percent) . '% AI Content Document';
        $entry->slug = str()->random(7) . str($user->fullName())->slug() . '-workbook';
        $entry->user_id = Auth::id();
        $entry->openai_id = $post->id;
        $entry->input = $input;
        $entry->hash = str()->random(256);
        $entry->credits = 0;
        $entry->words = 0;
        $entry->output = $text;
        $entry->storage = '';
        $entry->response = $text;

        $entry->save();

        return response()->json(['success' => true]);
    }

    public function plagiarismSetting(Request $request)
    {
        return view('ai-plagiarism::setting');
    }

    public function plagiarismSettingSave(Request $request)
    {
        $settings = SettingTwo::first();
        // TODO SETTINGS
        if (Helper::appIsNotDemo()) {
            $settings->plagiarism_key = $request->plagiarism_api_key;
            $settings->save();
        }

        return response()->json([], 200);
    }
}
