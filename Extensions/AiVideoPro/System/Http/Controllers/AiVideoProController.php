<?php

namespace App\Extensions\AiVideoPro\System\Http\Controllers;

use App\Domains\Engine\Services\FalAIService;
use App\Domains\Entity\Enums\EntityEnum;
use App\Domains\Entity\Facades\Entity;
use App\Extensions\AiVideoPro\System\Models\UserFall;
use App\Helpers\Classes\ApiHelper;
use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use Exception;
use Illuminate\Contracts\View\View;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Str;

class AiVideoProController extends Controller
{
    /**
     * Display a listing of the resource.
     */
    public function index(): View
    {
        $list = UserFall::query()->where('user_id', auth()->user()->id)->get()->toArray();

        $inProgress = collect($list)->filter(function ($entry) {
            return $entry['status'] === 'IN_QUEUE';
        })->pluck('id')->toArray();

        return view('ai-video-pro::index', compact(['list', 'inProgress']));
    }

    /**
     * Store a newly created resource in storage.
     */
    public function store(Request $request): JsonResponse|RedirectResponse
    {
        if (Helper::appIsDemo()) {
            return back()->with(['message' => 'This feature is disabled in demo mode.', 'type' => 'error']);
        }
        if (! ApiHelper::setFalAIKey()) {
            return back()->with(['message' => 'Please set FAL AI key.', 'type' => 'error']);
        }

        $request->validate([
            'action' => 'required',
            'prompt' => 'required',
            'photo'  => 'required_if:action,klingImage',
        ]);

        $driver = Entity::driver(EntityEnum::fromSlug($request->get('action')))->inputVideoCount(1)->calculateCredit();

        try {
            $driver->redirectIfNoCreditBalance();
        } catch (Exception $e) {
            return redirect()->back()->with([
                'message' => $e->getMessage(),
                'type'    => 'error',
            ]);
        }

        $action = $request->get('action');
        $prompt = $request->get('prompt');
        $userId = auth()->id();

        switch ($action) {
            case 'klingImage':
            case 'haiper':
                $image = $request->file('photo');
                $name = Str::random(12) . '.' . $image?->getClientOriginalExtension();
                Storage::disk('public')->put($name, file_get_contents($image->getRealPath()));
                $url = Helper::parseUrl(config('app.url') . '/uploads/' . $name);

                $response = FalAIService::{$action . 'Generate'}($prompt, $url);

                if (isset($response['status']) && $response['status'] === 'error') {
                    return back()->with(['message' => $response['message'], 'type' => 'error']);
                }

                $this->createUserFall($userId, $prompt, $action, $response, $url);
                $driver->decreaseCredit();

                return back()->with(['message' => 'Created Successfully.', 'type' => 'success']);
            case 'luma-dream-machine':
            case 'kling':
            case 'minimax':
                $response = FalAIService::{$action . 'Generate'}($prompt);

                $this->createUserFall($userId, $prompt, $action, $response);
                $driver->decreaseCredit();

                return back()->with(['message' => 'Created Successfully.', 'type' => 'success']);
            case 'veo2':
                $response = FalAIService::veo2Generate($prompt);
                if ($response->failed()) {
                    return back()->with([
                        'message' => $response->status() . ' ' . $response->reason() . ': ' .
                            $response->json('detail', __('Unknown error occurred')),
                        'type' => 'error',
                    ]);
                }
                $jsonRes = $response->json();
                if (isset($jsonRes['status']) && $jsonRes['status'] === 'error') {
                    return back()->with(['message' => $jsonRes['message'], 'type' => 'error']);
                }
                $this->createUserFall($userId, $prompt, $action, $jsonRes);
                $driver->decreaseCredit();

                return back()->with(['message' => 'Created Successfully.', 'type' => 'success']);
            default:
                return back()->with(['message' => 'Api key Error.', 'type' => 'error']);
        }
    }

    private function createUserFall($userId, $prompt, $action, $response, $imageUrl = null): void
    {
        UserFall::create([
            'user_id'          => $userId,
            'prompt'           => $prompt,
            'prompt_image_url' => $imageUrl,
            'status'           => $response['status'] ?? 'IN_QUEUE',
            'request_id'       => $response['request_id'] ?? null,
            'response_url'     => $response['response_url'] ?? null,
            'model'            => $action,
        ]);
    }

    /**
     * Remove the specified resource from storage.
     */
    public function delete(string $id): RedirectResponse
    {
        $model = UserFall::query()->findOrFail($id);
        $model->delete();

        return back()->with(['message' => 'Deleted Successfully.', 'type' => 'success']);
    }

    public function checkVideoStatus(Request $request): JsonResponse
    {
        $list = UserFall::query()
            ->where('status', '!=', 'complete')
            ->get()
            ->toArray();

        if (! count($list)) {
            return response()->json(['data' => []]);
        }

        $errorEntries = collect($list)->filter(function ($entry) {
            return $entry['status'] === 'error';
        })->pluck('id')->toArray();

        if (count($errorEntries)) {
            UserFall::query()->whereIn('id', $errorEntries)->delete();
        }

        $data = [];
        $ids = $request->get('ids');
        if (! is_array($ids)) {
            $ids = [];
        }

        foreach ($list as $entry) {
            if (! in_array($entry['id'], $ids)) {
                continue;
            }
            $response = FalAIService::getStatus($entry['response_url']);
            if (isset($response['video']['url'])) {
                $data[] = [
                    'divId' => 'video-' . $entry['id'],
                    'html'  => view('ai-video-pro::video-item', ['entry' => $entry])->render(),
                ];

                UserFall::query()->where('id', $entry['id'])->update([
                    'status'    => 'complete',
                    'video_url' => $response['video']['url'],
                ]);

                continue;
            }
            if (
                isset($response['detail']) &&
                in_array($response['detail'], [
                    'Internal Server Error',
                    'Luma API timed out',
                    "Luma API failed: generation.state='failed' generation.failure_reason='400: prompt not allowed because advanced moderation failed'",
                ])
            ) {
                UserFall::query()->where('id', $entry['id'])->delete();
            }

            if (
                isset($response['detail'][0]['type']) &&
                $response['detail'][0]['type'] === 'image_load_error'
            ) {
                UserFall::query()->where('id', $entry['id'])->delete();
            }
        }

        return response()->json(['data' => $data]);
    }
}
