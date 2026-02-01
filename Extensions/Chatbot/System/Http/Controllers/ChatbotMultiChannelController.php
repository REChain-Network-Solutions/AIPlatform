<?php

namespace App\Extensions\Chatbot\System\Http\Controllers;

use App\Extensions\Chatbot\System\Http\Resources\Admin\ChatbotChannelResource;
use App\Extensions\Chatbot\System\Models\ChatbotChannel;
use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Http\Resources\Json\AnonymousResourceCollection;

class ChatbotMultiChannelController extends Controller
{
    public function index(Request $request): AnonymousResourceCollection
    {
        $items = ChatbotChannel::query()->where('chatbot_id', $request['chatbot_id'])->get();

        return ChatbotChannelResource::collection($items)->additional([
            'status' => 'success',
        ]);
    }

    public function delete(Request $request): JsonResponse
    {
        if (Helper::appIsDemo()) {
            return response()->json([
                'status'  => 'error',
                'message' => trans('This feature is disabled in demo mode.'),
            ]);
        }

        $channel = ChatbotChannel::query()->findOrFail($request['channel_id']);

        $channel->delete();

        return response()->json([
            'status'  => 'success',
            'message' => trans('Chatbot channel successfully deleted'),
        ]);
    }
}
