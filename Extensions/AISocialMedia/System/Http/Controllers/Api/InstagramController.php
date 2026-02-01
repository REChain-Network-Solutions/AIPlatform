<?php

namespace App\Extensions\AISocialMedia\System\Http\Controllers\Api;

use App\Extensions\AISocialMedia\System\Enums\Platform;
use App\Extensions\AISocialMedia\System\Helpers\Instagram;
use App\Extensions\AISocialMedia\System\Models\AutomationPlatform;
use App\Http\Controllers\Controller;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;

class InstagramController extends Controller
{
    public function redirect()
    {
        return Instagram::authRedirect([
            'ads_management',
            'business_management',
            'instagram_basic',
            'instagram_content_publish',
            'pages_read_engagement',
        ]);
    }

    public function callback(Request $request)
    {
        $code = $request->get('code');

        if (! $code) {
            return redirect()->route('dashboard.user.automation.platform.list')
                ->with([
                    'type'    => 'error',
                    'message' => trans('Something went wrong, please try again.'),
                ]);
        }

        $instagram = new Instagram;
        $token = $instagram->getAccessToken($code)->throw()->json('access_token');
        $instagram->setToken($token);

        $page = $instagram->getAccountInfo(['connected_instagram_account,name,access_token'])
            ->throw()
            ->json('data.0');

        if (! isset($page['connected_instagram_account'])) {
            return redirect()->route('dashboard.user.automation.platform.list')
                ->with([
                    'type'    => 'error',
                    'message' => trans('Something went wrong, please try again.'),
                ]);
        }

        $igAccount = $instagram->getInstagramInfo($page['connected_instagram_account']['id'], ['id,name,username,profile_picture_url'])
            ->throw()
            ->json();

        AutomationPlatform::query()->updateOrCreate([
            'user_id'  => Auth::id(),
            'platform' => Platform::instagram->value,
        ], [
            'credentials' => [
                'id'                     => $igAccount['id'],
                'name'                   => $igAccount['name'],
                'username'               => $igAccount['username'],
                'picture'                => $igAccount['profile_picture_url'],
                'access_token'           => $igAccount['access_token'] ?? $token,
            ],
            'expires_at' => now()->addMonths(2),
        ]);

        return to_route('dashboard.user.automation.platform.list')->with([
            'type'    => 'success',
            'message' => trans('Instagram account connected successfully.'),
        ]);
    }
}
