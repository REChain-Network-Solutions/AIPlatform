<?php

namespace App\Extensions\Chatbot\System\Http\Controllers\Api;

use App\Extensions\Chatbot\System\Enums\InteractionType;
use App\Extensions\Chatbot\System\Http\Requests\ChatbotHistoryStoreRequest;
use App\Extensions\Chatbot\System\Http\Resources\Api\ChatbotConversationResource;
use App\Extensions\Chatbot\System\Http\Resources\Api\ChatbotHistoryResource;
use App\Extensions\Chatbot\System\Http\Resources\Api\ChatbotResource;
use App\Extensions\Chatbot\System\Models\Chatbot;
use App\Extensions\Chatbot\System\Models\ChatbotConversation;
use App\Extensions\Chatbot\System\Models\ChatbotHistory;
use App\Extensions\Chatbot\System\Services\GeneratorService;
use App\Extensions\ChatbotAgent\System\Services\ChatbotForPanelEventAbly;
use App\Helpers\Classes\Helper;
use App\Helpers\Classes\MarketplaceHelper;
use App\Http\Controllers\Controller;
use App\Models\Setting;
use Exception;
use Illuminate\Http\Request;
use Illuminate\Http\Resources\Json\AnonymousResourceCollection;
use Illuminate\Support\Facades\Log;

class ChatbotApplicationController extends Controller
{
    public Setting $setting;

    public function __construct(
        public GeneratorService $service
    ) {
        $this->setting = Setting::getCache();
    }

    public function index(Chatbot $chatbot): ChatbotResource
    {
        return ChatbotResource::make($chatbot);
    }

    public function indexSession(Chatbot $chatbot, string $sessionId): ChatbotResource
    {
        $conversations = ChatbotConversation::query()
            ->where('chatbot_id', $chatbot->getAttribute('id'))
            ->where('session_id', $sessionId)
            ->with('lastMessage')
            ->get();

        return ChatbotResource::make($chatbot)->additional([
            'conversations' => ChatbotConversationResource::collection($conversations),
        ]);
    }

    public function connectSupport(Request $request, Chatbot $chatbot, string $sessionId)
    {
        if (MarketplaceHelper::isRegistered('chatbot-agent')) {
            $request->validate(['conversation_id' => 'required|integer|exists:ext_chatbot_conversations,id']);

            /** @var ChatbotConversation $conversation */
            $conversation = ChatbotConversation::find($request->get('conversation_id'));

            if ($chatbot->getAttribute('interaction_type') === InteractionType::SMART_SWITCH) {
                $conversation->update(['connect_agent_at' => now()]);

                $chatbotHistory = null;

                if ($chatbot->getAttribute('connect_message')) {
                    $chatbotHistory = $this->insertMessage(
                        conversation: $conversation,
                        message: trans($chatbot->getAttribute('connect_message')),
                        role: 'assistant',
                        model: $chatbot->getAttribute('ai_model'),
                        forcePanelEvent: true
                    );
                }

                try {
                    ChatbotForPanelEventAbly::dispatch($chatbot, $conversation, $chatbotHistory);
                } catch (Exception $e) {
                    Log::error($e->getMessage());
                }

                return ChatbotConversationResource::make($conversation)->additional([
                    'history' => $chatbotHistory ? ChatbotHistoryResource::make($chatbotHistory) : null,
                ]);
            }

            abort(404);
        }
    }

    public function conversionStore(Chatbot $chatbot, string $sessionId): ChatbotConversationResource
    {
        $chatbotConversation = ChatbotConversation::query()
            ->create([
                'is_showed_on_history' => false,
                'ip_address'           => request()->header('cf-connecting-ip') ?: request()->ip(),
                'chatbot_id'           => $chatbot->getAttribute('id'),
                'session_id'           => $sessionId,
                'connect_agent_at'     => $chatbot->getAttribute('interaction_type') === InteractionType::HUMAN_SUPPORT ? now() : null,
                'last_activity_at'     => now(),
            ]);

        $this->insertMessage(
            $chatbotConversation,
            $chatbot->getAttribute('welcome_message'),
            'assistant', $chatbot->getAttribute('ai_model'),
            (bool) $chatbotConversation->getAttribute('connect_agent_at')
        );

        return ChatbotConversationResource::make($chatbotConversation);
    }

    public function conversion(Chatbot $chatbot, string $sessionId, ChatbotConversation $chatbotConversation): ChatbotConversationResource
    {
        if ($chatbotConversation->getAttribute('chatbot_id') !== $chatbot->getAttribute('id')) {
            abort(404);
        }

        if ($chatbotConversation->getAttribute('session_id') !== $sessionId) {
            abort(404);
        }

        return ChatbotConversationResource::make($chatbotConversation);
    }

    public function messages(Chatbot $chatbot, string $sessionId, ChatbotConversation $chatbotConversation): AnonymousResourceCollection
    {
        if ($chatbotConversation->getAttribute('chatbot_id') !== $chatbot->getAttribute('id')) {
            abort(404);
        }

        if ($chatbotConversation->getAttribute('session_id') !== $sessionId) {
            abort(404);
        }

        $messages = ChatbotHistory::query()
            ->where('conversation_id', $chatbotConversation->getAttribute('id'))
            ->orderByDesc('id')
            ->paginate(perPage: request('per_page', 10));

        return ChatbotHistoryResource::collection($messages);
    }

    public function storeMessage(ChatbotHistoryStoreRequest $request, Chatbot $chatbot, string $sessionId, ChatbotConversation $chatbotConversation): ChatbotHistoryResource
    {
        if ($chatbotConversation->getAttribute('chatbot_id') !== $chatbot->getAttribute('id')) {
            abort(404);
        }

        if ($chatbotConversation->getAttribute('session_id') !== $sessionId) {
            abort(404);
        }

        $userMessage = $this->insertMessage($chatbotConversation, $request->validated('prompt'), 'user', $chatbot->getAttribute('ai_model'));

        if (! $chatbotConversation->getAttribute('is_showed_on_history')) {
            $chatbotConversation->update(['is_showed_on_history' => true]);
        }

        if ($chatbotConversation->getAttribute('connect_agent_at')) {
            return ChatbotHistoryResource::make($userMessage)->additional([
                'connection' => 'panel',
            ]);
        }

        if (Helper::appIsNotDemo()) {
            $response = $this->service
                ->setChatbot($chatbot)
                ->setConversation($chatbotConversation)
                ->setPrompt(
                    $request->validated('prompt')
                )
                ->generate();

            if (empty($response)) {
                $response = trans('Sorry, I can\'t answer right now.');
            }
        } else {
            $response = 'This feature is disabled in Demo version.';
        }

        $message = $this->insertMessage($chatbotConversation, $response, 'assistant', $chatbot->getAttribute('ai_model'));

        return ChatbotHistoryResource::make($message)->additional([
            'connection' => 'ai',
        ]);
    }

    protected function insertMessage(ChatbotConversation $conversation, string $message, string $role, string $model, bool $forcePanelEvent = false)
    {
        $chatbot = $conversation->getAttribute('chatbot');

        $chatbotHistory = ChatbotHistory::query()->create([
            'chatbot_id'      => $conversation->getAttribute('chatbot_id'),
            'conversation_id' => $conversation->getAttribute('id'),
            'role'            => $role,
            'model'           => $this->setting->openai_default_model,
            'message'         => $message,
            'created_at'      => now(),
            'read_at'         => $conversation->getAttribute('connect_agent_at') ? null : now(),
        ]);

        $sendEvent = $conversation->getAttribute('connect_agent_at') && $chatbot->getAttribute('interaction_type') !== InteractionType::AUTOMATIC_RESPONSE && $role === 'user';

        if ($sendEvent || $forcePanelEvent) {
            $conversation->touch();
            if (MarketplaceHelper::isRegistered('chatbot-agent')) {
                ChatbotForPanelEventAbly::dispatch(
                    $chatbot,
                    $conversation->load('lastMessage'),
                    $chatbotHistory
                );
            }
        }

        return $chatbotHistory;
    }
}
