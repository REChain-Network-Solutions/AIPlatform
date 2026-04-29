<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Http\Controllers\TTSController as TTS;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;

class TTSController extends Controller
{
    public function preview(Request $request): JsonResponse
    {
        $request->merge(['preview' => true]);

        return app(TTS::class)->generateSpeech($request);
    }

    public function generate(Request $request): JsonResponse
    {
        $request->merge(['from_api' => true]);

        return app(TTS::class)->generateSpeech($request);
    }
}
