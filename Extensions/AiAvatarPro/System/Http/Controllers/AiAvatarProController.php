<?php

namespace App\Extensions\AiAvatarPro\System\Http\Controllers;

use App\Domains\Entity\Enums\EntityEnum;
use App\Domains\Entity\Facades\Entity;
use App\Extensions\AiAvatarPro\System\Models\AiAvatarPro;
use App\Extensions\AiAvatarPro\System\Services\AiAvatarProService;
use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use Exception;
use Illuminate\Contracts\View\View;
use Illuminate\Http\Request;

class AiAvatarProController extends Controller
{
    public function __construct(
        public AiAvatarProService $service
    ) {}

    public function index(): View
    {
        $userAvatars = AiAvatarPro::query()->where('user_id', auth()->id())->pluck('avatar_id')->toArray();

        $allVideos = $this->service->listVideos()['data']['videos'] ?? [];
        $userVideos = array_filter($allVideos, function ($video) use ($userAvatars) {
            return isset($video['video_id']) && in_array($video['video_id'], $userAvatars, true);
        });

        $detailedVideos = array_map(function ($video) {
            $detail = $this->service->retrieveVideo($video['video_id'])['data'];

            return array_merge($video, $detail);
        }, $userVideos);

        $inProgress = AiAvatarPro::query()->where('user_id', auth()->id())
            ->where('status', 'in_progress')->get()->pluck('avatar_id')->toArray();

        return view('ai-avatar-pro::index', [
            'list'       => $detailedVideos,
            'inProgress' => $inProgress,
        ]);
    }

    public function create(): View
    {
        return view('ai-avatar-pro::create', [
            'avatars' => $this->service->listAvatars()['data']['avatars'] ?? [],
            'voices'  => $this->service->listVoices()['data']['voices'] ?? [],
        ]);
    }

    public function store(Request $request)
    {
        if (Helper::appIsDemo()) {
            return response()->json([
                'status'  => 'error',
                'message' => trans('This feature is disabled in demo mode.'),
            ]);
        }

        $avatarSettings = [
            'type'         => 'avatar',
            'avatar_id'    => $request->get('avatar_id'),
            'avatar_style' => $request->get('avatar_style'),
            'matting'      => $request->get('matting'),
        ];

        $voiceSettings = [
            'type'       => 'text',
            'input_text' => $request->get('input_text'),
            'voice_id'   => $request->get('voice_id'),
        ];

        $videoInputs = [
            'character' => $avatarSettings,
            'voice'     => $voiceSettings,
        ];

        $body = [
            'video_inputs' => [$videoInputs],
            'caption'      => $request->get('caption'),
            'dimension'    => [
                'width'  => 1920,
                'height' => 1080,
            ],
        ];

        $driver = Entity::driver(EntityEnum::HEYGEN)->inputVideoCount(1)->calculateCredit();

        try {
            $driver->redirectIfNoCreditBalance();
        } catch (Exception $e) {
            return redirect()->back()->with([
                'message' => $e->getMessage(),
                'type'    => 'You have no credits left. Please consider upgrading your plan.',
            ]);
        }

        $service = new AiAvatarProService;
        $response = $service->createVideo($body);

        if ($response['error'] == null) {

            AiAvatarPro::query()->create([
                'user_id'   => auth()->user()->id,
                'avatar_id' => $response['data']['video_id'],
                'status'    => 'in_progress',
            ]);

            $driver->decreaseCredit();

            return redirect()->route('dashboard.user.ai-avatar-pro.index')->with([
                'message' => __('Video Created Successfully'),
                'type'    => 'success',
            ]);
        } else {
            return redirect()->back()->with([
                'message' => $response['error']['message'],
                'type'    => 'error',
            ]);
        }

    }

    /**
     * Remove the specified resource from storage.
     */
    public function delete(string $id)
    {
        if (Helper::appIsDemo()) {
            return response()->json([
                'status'  => 'error',
                'message' => trans('This feature is disabled in demo mode.'),
            ]);
        }

        $model = $this->service->deleteVideo($id);

        if ($model->getStatusCode() === 200 || $model->getStatusCode() === 204) {

            $builder = AiAvatarPro::query()->where('avatar_id', $id)->first();

            $builder->delete();

            return back()->with(['message' => __('Deleted Successfully'), 'type' => 'success']);
        } else {
            return back()->with(['message' => __('Delete Failed'), 'type' => 'danger']);
        }
    }

    public function checkVideoStatus(Request $request): \Illuminate\Http\JsonResponse
    {
        $ids = $request->get('ids');

        if (! $ids) {
            return response()->json(['data' => []]);
        }

        $service = new AiAvatarProService;

        $list = $service->listVideos()['data']['videos'];

        $data = [];

        foreach ($list as $entry) {
            if (in_array($entry['video_id'], $ids) && $entry['status'] === 'completed') {
                $detail = $this->service->retrieveVideo($entry['video_id'])['data'];
                $data = array_merge($detail, $entry);

                $data[] = [
                    'divId' => 'video-' . $data['video_id'],
                    'html'  => view('ai-avatar-pro::video-item', ['entry' => $data])->render(),
                ];

                AiAvatarPro::query()->where('avatar_id', $entry['video_id'])->update([
                    'status' => 'completed',
                ]);
            }
        }

        return response()->json(['data' => $data]);
    }
}
