<?php

namespace App\Extensions\AiFall\System\Http\Controllers;

use App\Domains\Engine\Services\FalAIService;
use App\Domains\Entity\Enums\EntityEnum;
use App\Domains\Entity\Facades\Entity;
use App\Extensions\AiFall\System\Models\UserFall;
use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use Exception;
use Illuminate\Contracts\View\View;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Str;

class FallVideoController extends Controller
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

        return view('ai-fall-video::index', compact(['list', 'inProgress']));
    }

    /**
     * Store a newly created resource in storage.
     */
    public function store(Request $request): JsonResponse|RedirectResponse
    {
        if (! Helper::setFalAIKey()) {
            return back()->with(['message' => 'Please set FAL AI key.', 'type' => 'error']);
        }

        if (Helper::appIsDemo()) {
            return back()->with(['message' => 'This feature is disabled in demo mode.', 'type' => 'error']);
        }

        $driver = Entity::driver(EntityEnum::fromSlug($request->get('action')))->inputVideoCount(1)->calculateCredit();

        try {
            $driver->redirectIfNoCreditBalance();
        } catch (Exception $e) {
            return redirect()->back()->with([
                'message' => $e->getMessage(),
                'type'    => 'You have no credits left. Please consider upgrading your plan.',
            ]);
        }

        $request->validate([
            'action' => 'required',
            'prompt' => 'required',
            'photo'  => 'required_if:action,runway-gen3',
        ]);

        if ($request->get('action') == 'runway-gen3') {
            $image = $request->file('photo');

            $nameOfImage = Str::random(12) . '.png';
            $contents = file_get_contents($image->getRealPath());

            Storage::disk('public')->put($nameOfImage, $contents);

            $path = '/uploads/' . $nameOfImage;
            $url = Helper::parseUrl(config('app.url') . $path);

            $response = FalAIService::runwayGenerate($request->get('prompt'), 'https://gratisography.com/wp-content/uploads/2024/03/gratisography-funflower-800x525.jpg');

            if ($response['status'] === 'error') {
                return back()->with(['message' => $response['message'], 'type' => 'error']);
            }

            $model = new UserFall;
            $model->create([
                'user_id'          => auth()->user()->id,
                'prompt'           => $request->get('prompt'),
                'prompt_image_url' => $url,
                'status'           => $response['status'],
                'request_id'       => $response['request_id'],
                'response_url'     => $response['response_url'],
                'model'            => $request->get('action'),
            ]);

            $driver->decreaseCredit();

            return back()->with(['message' => 'Created Successfully.', 'type' => 'success']);
        } elseif ($request->get('action') == 'luma-dream-machine') {

            $response = FalAIService::lumaGenerate($request->get('prompt'));

            $model = new UserFall;
            $model->create([
                'user_id'      => auth()->user()->id,
                'prompt'       => $request->get('prompt'),
                'status'       => $response['status'],
                'request_id'   => $response['request_id'],
                'response_url' => $response['response_url'],
                'model'        => $request->get('action'),
            ]);

            $driver->decreaseCredit();

            return back()->with(['message' => 'Created Successfully.', 'type' => 'success']);

        } elseif ($request->get('action') == 'kling') {

            $response = FalAIService::klingGenerate($request->get('prompt'));

            $model = new UserFall;
            $model->create([
                'user_id'      => auth()->user()->id,
                'prompt'       => $request->get('prompt'),
                'status'       => $response['status'],
                'request_id'   => $response['request_id'],
                'response_url' => $response['response_url'],
                'model'        => $request->get('action'),
            ]);

            $driver->decreaseCredit();

            return back()->with(['message' => 'Created Successfully.', 'type' => 'success']);
        } elseif ($request->get('action') == 'minimax') {

            $response = FalAIService::minimaxGenerate($request->get('prompt'));

            $model = new UserFall;
            $model->create([
                'user_id'      => auth()->user()->id,
                'prompt'       => $request->get('prompt'),
                'status'       => $response['status'],
                'request_id'   => $response['request_id'],
                'response_url' => $response['response_url'],
                'model'        => $request->get('action'),
            ]);

            $driver->decreaseCredit();

            return back()->with(['message' => 'Created Successfully.', 'type' => 'success']);
        }

        return back()->with(['message' => 'Api key Error.', 'type' => 'error']);
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
        $list = UserFall::query()->get()->toArray();

        if (! count($list)) {
            return response()->json(['data' => []]);

        }

        $data = [];

        foreach ($list as $entry) {
            if (in_array($entry['id'], $request->get('ids'))) {
                $response = FalAIService::getStatus($entry['response_url']);

                if (isset($response['video']['url'])) {
                    $data[] = [
                        'divId' => 'video-' . $entry['id'],
                        'html'  => view('ai-fall-video::video-item', ['entry' => $entry])->render(),
                    ];

                    UserFall::query()->where('id', $entry['id'])->update([
                        'status'    => 'complete',
                        'video_url' => $response['video']['url'],
                    ]);
                }
            }

        }

        return response()->json(['data' => $data]);
    }
}
