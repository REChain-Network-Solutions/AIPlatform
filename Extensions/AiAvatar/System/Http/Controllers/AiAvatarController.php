<?php

namespace App\Extensions\AiAvatar\System\Http\Controllers;

use App\Domains\Entity\Enums\EntityEnum;
use App\Domains\Entity\Facades\Entity;
use App\Extensions\AiAvatar\System\Http\Requests\AiAvatarRequest;
use App\Extensions\AiAvatar\System\Models\AiAvatar;
use App\Extensions\AiAvatar\System\Services\AiAvatarService;
use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use Exception;
use Illuminate\Contracts\View\View;
use Illuminate\Http\Request;

class AiAvatarController extends Controller
{
    public function __construct(
        public AiAvatarService $service
    ) {}

    public function index(): View
    {
        $list = $this->service->listVideos();

        $model = AiAvatar::query()->where('user_id', auth()->user()->id)->get();

        $list = collect($list)->filter(function ($item) use ($model) {
            $avatarIds = $model->pluck('avatar_id')->toArray();

            return isset($item['id']) && in_array($item['id'], $avatarIds, true);
        })->toArray();

        $inProgress = collect($list)->filter(function ($entry) {
            return $entry['status'] === 'in_progress';
        })->pluck('id')->toArray();

        return view('ai-avatar::index', compact('list', 'inProgress'));
    }

    public function create(): View
    {
        return view('ai-avatar::create', [
            'avatars'     => $this->service->listAvatars(),
            'backgrounds' => $this->service->listBackgrounds(),
        ]);
    }

    public function store(AiAvatarRequest $request)
    {
        if (Helper::appIsDemo()) {
            return response()->json([
                'status'  => 'error',
                'message' => trans('This feature is disabled in demo mode.'),
            ]);
        }

        $avatarSettings = [
            'style'           => $request->get('style'),
            'backgroundColor' => $request->get('backgroundColor'),
        ];

        if ($request->get('style') === 'rectangular') {
            $avatarSettings['horizontalAlign'] = $request->get('horizontalAlign');
        }

        $body = [
            [
                'avatarSettings' => $avatarSettings,
                'scriptText'     => $request->get('scriptText'),
                'avatar'         => $request->get('avatar'),
                'background'     => $request->get('background'),
            ],
        ];

        $driver = Entity::driver(EntityEnum::SYNTHESIA)->inputVideoCount(1)->calculateCredit();

        try {
            $driver->redirectIfNoCreditBalance();
        } catch (Exception $e) {
            return redirect()->back()->with([
                'message' => $e->getMessage(),
                'type'    => 'You have no credits left. Please consider upgrading your plan.',
            ]);
        }

        $service = new AiAvatarService;
        $response = $service->createVideo($body, $request->get('visibility'),
            $request->get('title'), $request->get('description'), $request->get('test'));

        if (! empty($response['status']) && $response['id']) {

            AiAvatar::query()->create([
                'user_id'   => auth()->user()->id,
                'avatar_id' => $response['id'],
                'status'    => $response['status'],
            ]);

            $driver->decreaseCredit();

            return redirect()->route('dashboard.user.ai-avatar.index')->with([
                'message' => __('Video Created Successfully'),
                'type'    => 'success',
            ]);
        } else {
            return redirect()->back()->with([
                'message' => __('AiAvatar API Key Error'),
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

            $builder = AiAvatar::query()->where('avatar_id', $id)->first();

            $builder->delete();

            return back()->with(['message' => __('Deleted Successfully'), 'type' => 'success']);
        } else {
            return back()->with(['message' => __('Delete Failed'), 'type' => 'danger']);
        }
    }

    public function checkVideoStatus(Request $request): \Illuminate\Http\JsonResponse
    {
        $ids = AiAvatar::query()->where('status', 'in_progress')->pluck('avatar_id')->toArray();

        if (! count($ids)) {
            return response()->json(['data' => []]);

        }

        $service = new AiAvatarService;

        $list = $service->listVideos();

        $data = [];

        foreach ($list as $entry) {
            if (in_array($entry['id'], $ids) && $entry['status'] === 'complete') {
                $data[] = [
                    'divId' => 'video-' . $entry['id'],
                    'html'  => view('ai-avatar::video-item', ['entry' => $entry])->render(),
                ];

                AiAvatar::query()->where('avatar_id', $entry['id'])->update([
                    'status' => 'complete',
                ]);
            }
        }

        return response()->json(['data' => $data]);
    }
}
