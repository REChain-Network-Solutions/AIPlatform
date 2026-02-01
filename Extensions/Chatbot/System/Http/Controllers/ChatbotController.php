<?php

namespace App\Extensions\Chatbot\System\Http\Controllers;

use App\Extensions\Chatbot\System\Http\Requests\ChatbotCustomizeRequest;
use App\Extensions\Chatbot\System\Http\Requests\ChatbotStoreRequest;
use App\Extensions\Chatbot\System\Http\Resources\Admin\ChatbotConversationResource;
use App\Extensions\Chatbot\System\Http\Resources\Admin\ChatbotResource;
use App\Extensions\Chatbot\System\Services\ChatbotService;
use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use Illuminate\Contracts\View\View;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Artisan;
use Illuminate\Support\Facades\Auth;

class ChatbotController extends Controller
{
    public function __construct(public ChatbotService $service) {}

    public function index(Request $request): View
    {
        if (method_exists(Helper::class, 'appIsDemoForChatbot')) {
            if (Helper::appIsDemoForChatbot()) {
                // Clear chatbot for demo mode
                Artisan::call('app:clear-chatbot-demo-mode');
            }
        }

        $externalChatbots = $request->user()->externalChatbots->pluck('id')->toArray();
        $unreadAgentMessagesCount = $this->service->unreadAgentMessagesCount($externalChatbots);
        $unreadAiBotMessagesCount = $this->service->unreadAiBotMessagesCount($externalChatbots);
        $allMessagesCount = $this->service->allMessagesCount($externalChatbots);

        return view('chatbot::index', [
            'chatbots' => $this->service->query()
                ->where('user_id', Auth::id())
                ->orderBy('created_at', 'desc')->paginate(perPage: 10),
            'avatars'                  => $this->service->avatars(),
            'unreadAgentMessagesCount' => $unreadAgentMessagesCount,
            'unreadAiBotMessagesCount' => $unreadAiBotMessagesCount,
            'allMessagesCount'         => $allMessagesCount,
        ]);
    }

    public function store(ChatbotStoreRequest $request): JsonResponse|ChatbotResource
    {
        $chatbot = $this->service->query()->create($request->validated());

        return ChatbotResource::make($chatbot);
    }

    public function update(ChatbotCustomizeRequest $request): JsonResponse|ChatbotResource
    {
        $data = $request->validated();

        $chatbot = $this->service->query()->findOrFail($data['id']);

        if ($chatbot->getAttribute('is_demo')) {
            return response()->json([
                'type'    => 'error',
                'message' => 'This feature is disabled in Demo version.',
            ], 403);
        }

        $chatbot = $this->service->update($data['id'], $data);

        return ChatbotResource::make($chatbot);
    }

    public function conversations(Request $request): \Illuminate\Http\Resources\Json\AnonymousResourceCollection
    {
        $chatbots = $request->user()->externalChatbots->pluck('id')->toArray();

        $conversations = $this->service->conversations($chatbots);

        return ChatbotConversationResource::collection($conversations);
    }

    public function conversationsWithPaginate(Request $request): \Illuminate\Http\Resources\Json\AnonymousResourceCollection
    {
        $chatbots = $request->user()->externalChatbots->pluck('id')->toArray();

        $conversations = $this->service->conversationsWithPaginate($chatbots);

        return ChatbotConversationResource::collection($conversations);
    }

    public function delete(Request $request): JsonResponse
    {
        $request->validate(['id' => 'required']);

        $chatbot = $this->service->query()->findOrFail($request->get('id'));

        if ($chatbot->getAttribute('is_demo')) {
            return response()->json([
                'type'    => 'error',
                'message' => 'This feature is disabled in Demo version.',
            ], 403);
        }

        if ($chatbot->getAttribute('user_id') === Auth::id()) {
            $chatbot->delete();
        } else {
            abort(403);
        }

        return response()->json([
            'message' => 'Chatbot deleted successfully',
            'type'    => 'success',
            'status'  => 200,
        ]);
    }
}
