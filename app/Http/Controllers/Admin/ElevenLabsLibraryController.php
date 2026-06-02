<?php

namespace App\Http\Controllers\Admin;

use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use App\Http\Requests\Voice\AddSharedVoiceRequest;
use App\Models\Voice\ElevenlabVoice;
use App\Services\Ai\ElevenLabsService;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;

class ElevenLabsLibraryController extends Controller
{
    public function __construct(private ElevenLabsService $service) {}

    public function index(Request $request): JsonResponse
    {
        return $this->service->getSharedVoices($request->query());
    }

    public function add(AddSharedVoiceRequest $request): JsonResponse
    {
        if (Helper::appIsDemo()) {
            return response()->json([
                'status'  => 'error',
                'message' => __('This feature is disabled in demo mode.'),
            ]);
        }

        $payload = $request->validated();

        $response = $this->service->addSharedVoice(
            $payload['public_user_id'],
            $payload['voice_id'],
            $payload['name'],
        );

        $body = $response->getData(true);

        if (($body['status'] ?? null) !== 'success') {
            $code = $body['code'] ?? null;
            $apiMessage = $body['message'] ?? null;

            if ($code === 'paid_plan_required' || $code === 'payment_required') {
                return response()->json([
                    'status'  => 'error',
                    'code'    => $code,
                    'message' => __('ElevenLabs rejected this voice: a paid plan is required to add library voices via the API. Please upgrade your ElevenLabs subscription.') . ($apiMessage ? ' (' . $apiMessage . ')' : ''),
                ]);
            }

            return response()->json([
                'status'  => 'error',
                'code'    => $code,
                'message' => $apiMessage ?: __('Failed to add voice.'),
            ]);
        }

        $newVoiceId = $body['resData']['voice_id'] ?? null;

        if (! $newVoiceId) {
            return response()->json([
                'status'  => 'error',
                'message' => __('ElevenLabs did not return a voice id.'),
            ]);
        }

        $voice = ElevenlabVoice::query()->updateOrCreate(
            ['voice_id' => $newVoiceId, 'user_id' => null],
            [
                'name'     => $payload['name'],
                'language' => $payload['language'] ?? null,
                'path'     => null,
                'status'   => true,
            ],
        );

        return response()->json([
            'status' => 'success',
            'voice'  => [
                'id'          => $voice->id,
                'name'        => $voice->name,
                'voice_id'    => $voice->voice_id,
                'preview_url' => $request->input('preview_url'),
            ],
        ]);
    }

    public function remove(Request $request): JsonResponse
    {
        if (Helper::appIsDemo()) {
            return response()->json([
                'status'  => 'error',
                'message' => __('This feature is disabled in demo mode.'),
            ]);
        }

        $voiceId = (string) $request->input('voice_id', '');
        if ($voiceId === '') {
            return response()->json([
                'status'  => 'error',
                'message' => __('Voice id is required.'),
            ]);
        }

        $voice = ElevenlabVoice::query()
            ->whereNull('user_id')
            ->where('voice_id', $voiceId)
            ->first();

        $this->service->deleteVoice($voiceId);

        if ($voice) {
            $voice->delete();
        }

        return response()->json([
            'status'   => 'success',
            'voice_id' => $voiceId,
        ]);
    }
}
