<?php

namespace App\Extensions\AISocialMedia\System\Http\Controllers;

use App\Extensions\AISocialMedia\System\Enums\Platform;
use App\Extensions\AISocialMedia\System\Helpers\AutomationHelper;
use App\Extensions\AISocialMedia\System\Jobs\UserPostJob;
use App\Extensions\AISocialMedia\System\Models\AutomationCampaign;
use App\Extensions\AISocialMedia\System\Services\AutomationService;
use App\Extensions\AISocialMedia\System\Services\ScheduledPostService;
use App\Http\Controllers\Controller;
use App\Models\Company;
use Illuminate\Contracts\View\View;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Cache;

class AutomationStepController extends Controller
{
    public function __construct(
        public ScheduledPostService $scheduledPostService
    ) {}

    public function stepFirst(Request $request): View
    {
        return view('ai-social-media::automation-steps.first', $this->data([
            'step'      => 1,
            'platforms' => AutomationService::platforms(),
        ]));
    }

    public function stepSecond(Request $request): View
    {
        if ($request->isMethod('POST')) {
            $request->validate(['platform_id' => 'required|integer']);

            $request->merge(['platform' => AutomationService::find($request->get('platform_id'))->toArray()]);
        }

        $data = self::setCache($request, $hasCache = false);

        $data = array_merge($data, [
            'step'      => 2,
            'companies' => Company::query()->where('user_id', Auth::id())->get(),
        ]);

        return view('ai-social-media::automation-steps.second', $this->data($data));
    }

    public function stepThird(Request $request): View
    {
        $data = self::setCache($request);

        $data['campaigns'] = AutomationCampaign::query()->where('user_id', Auth::id())->get();

        return view('ai-social-media::automation-steps.third', $this->data($data));
    }

    public function stepFourth(Request $request): View
    {
        $data = self::setCache($request);

        return view('ai-social-media::automation-steps.fourth', $this->data($data));
    }

    public function stepFifth(Request $request)
    {
        $request->merge([
            'seo' => (bool) $request->get('seo'),
        ]);

        $data = self::setCache($request);

        return view('ai-social-media::automation-steps.fifth', $this->data($data));
    }

    public function stepLast(Request $request)
    {
        if ($request->isMethod('POST')) {
            $request->validate(['content' => 'required']);
        }

        $request->merge([
            'sendMail'      => (bool) $request->get('sendMail'),
            'auto_generate' => (bool) $request->get('auto_generate'),
        ]);

        $data = self::setCache($request);

        if ($data['platform']['key'] === Platform::instagram->value) {
            $request->validate(['image' => 'required|string']);
        }

        return view('ai-social-media::automation-steps.last', $this->data($data));
    }

    public function storeScheduledPost(Request $request): \Illuminate\Http\RedirectResponse
    {
        $data = self::setCache($request);

        $post = $this->scheduledPostService->store($data);

        if ($request->has('share_test')) {
            UserPostJob::dispatchSync($post);
        }

        return redirect()->route('dashboard.user.automation.list')->with([
            'message' => 'Scheduled post created successfully',
            'type'    => 'success',
        ]);
    }

    public static function setCache(Request $request, bool $hasCache = true)
    {
        if ($request->isMethod('POST')) {
            $data = Cache::get('automation:data') ?: [];

            if ($data) {
                $data = array_merge($data, $request->all());
            } else {
                $data = $request->all();
            }

            Cache::put('automation:data', $data, 6000);

            return $data;
        }

        return Cache::get('automation:data') ?: [];
    }

    private function data(array $array): array
    {
        return array_merge(AutomationHelper::apiKeys(), $array);
    }
}
