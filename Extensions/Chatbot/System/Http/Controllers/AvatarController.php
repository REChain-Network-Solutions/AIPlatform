<?php

namespace App\Extensions\Chatbot\System\Http\Controllers;

use App\Extensions\Chatbot\System\Http\Requests\AvatarRequest;
use App\Extensions\Chatbot\System\Http\Resources\Admin\ChatbotAvatarResource;
use App\Extensions\Chatbot\System\Models\ChatbotAvatar;
use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;

class AvatarController extends Controller
{
    public function __invoke(AvatarRequest $request)
    {
        if (Helper::appIsDemo()) {
            return response()->json([
                'type'    => 'error',
                'message' => 'This feature is disabled in Demo version.',
            ], 403);
        }

        $file = $request->file('avatar')->store('avatars', ['disk' => 'public']);

        $chatbotAvatar = ChatbotAvatar::query()->create([
            'user_id' => $request->user()->getAttribute('id'),
            'avatar'  => 'uploads/' . $file,
        ]);

        return ChatbotAvatarResource::make($chatbotAvatar);
    }
}
