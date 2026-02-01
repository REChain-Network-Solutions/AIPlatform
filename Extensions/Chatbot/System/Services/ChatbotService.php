<?php

namespace App\Extensions\Chatbot\System\Services;

use App\Extensions\Chatbot\System\Models\Chatbot;
use App\Extensions\Chatbot\System\Models\ChatbotAvatar;
use App\Extensions\Chatbot\System\Models\ChatbotConversation;
use Illuminate\Contracts\Pagination\LengthAwarePaginator;
use Illuminate\Database\Eloquent\Builder;
use Illuminate\Database\Eloquent\Collection;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Support\Facades\Auth;

class ChatbotService
{
    public function agentConversations(array $chatbots, ?string $orderBy = null): Collection|array
    {
        $agentFilter = filter_var(request()->get('agentFilter', false), FILTER_VALIDATE_BOOLEAN);

        return ChatbotConversation::query()
            ->with('chatbot:id,uuid,avatar')
            ->with(['histories.user:id,avatar', 'lastMessage'])
            ->when(($agentFilter == false), function (Builder $query) {
                $query->whereNull('connect_agent_at');
            }, function (Builder $query) {
                $query->whereNotNull('connect_agent_at');
            })
            ->whereIn('chatbot_id', $chatbots)
            ->when($orderBy, function (Builder $query) use ($orderBy) {
                $query->orderBy($orderBy ?: 'id', 'desc');
            })
            ->get();
    }

    public function unreadAgentMessagesCount(array $chatbots): int
    {
        return ChatbotConversation::query()
            ->whereNotNull('connect_agent_at')
            ->whereIn('chatbot_id', $chatbots)
            ->whereHas('histories', function (Builder $query) {
                $query->where('role', 'user')->where('read_at', null);
            })
            ->count();
    }

    public function unreadAiBotMessagesCount(array $chatbots): int
    {
        return ChatbotConversation::query()
            ->whereNull('connect_agent_at')
            ->whereIn('chatbot_id', $chatbots)
            ->whereHas('histories', function (Builder $query) {
                $query->where('role', 'user')->where('read_at', null);
            })
            ->count();
    }

    public function allMessagesCount(array $chatbots): int
    {
        return ChatbotConversation::query()
            ->whereIn('chatbot_id', $chatbots)
            ->whereHas('histories', function (Builder $query) {
                $query->where('role', 'user');
            })
            ->count();
    }

    public function agentConversationsWithPaginate(array $chatbots, ?string $orderBy = null): LengthAwarePaginator
    {
        $agentFilter = filter_var(request()->get('agentFilter', false), FILTER_VALIDATE_BOOLEAN);

        return ChatbotConversation::query()
            ->when(request('chatbot_channel') && request('chatbot_channel') !== 'all', function (Builder $query) {
                $query->where('chatbot_channel', request('chatbot_channel'));
            })
            ->where('is_showed_on_history', true)
            ->with('chatbot:id,uuid,avatar')
            ->with(['histories.user:id,avatar', 'lastMessage'])
            ->when(($agentFilter === false), function (Builder $query) {
                $query->whereNull('connect_agent_at');
            }, function (Builder $query) {
                $query->whereNotNull('connect_agent_at');
            })
            ->whereIn('chatbot_id', $chatbots)
            ->when($orderBy, function (Builder $query) use ($orderBy) {
                $query->orderBy($orderBy ?: 'id', 'desc');
            })
            ->orderBy(
                function ($query) {
                    $query->select('created_at')
                        ->from('ext_chatbot_histories')
                        ->whereColumn('ext_chatbot_histories.conversation_id', 'ext_chatbot_conversations.id')
                        ->where('ext_chatbot_histories.role', 'user')
                        ->latest()
                        ->limit(1);
                },
                'desc'
            )
            ->paginate(request('per_page', request('perPage', 30)));
    }

    public function agentConversationsBySearch(array $chatbots, string $search)
    {
        return ChatbotConversation::query()
            ->with('chatbot:id,uuid,avatar')
            ->with(['histories.user:id,avatar', 'lastMessage'])
            ->whereNotNull('connect_agent_at')
            ->whereIn('chatbot_id', $chatbots)
            ->whereHas('histories', function (Builder $query) use ($search) {
                $query->where('message', 'like', "%$search%");
            })
            ->get();
    }

    public function conversations(array $chatbots, ?string $orderBy = null): Collection|array
    {
        return ChatbotConversation::query()
            ->with('chatbot:id,uuid,avatar')
            ->with(['histories', 'lastMessage'])
            ->whereIn('chatbot_id', $chatbots)
            ->when($orderBy, function (Builder $query) use ($orderBy) {
                $query->orderBy($orderBy ?: 'id', 'desc');
            })
            ->get();
    }

    public function conversationsWithPaginate(array $chatbots, ?string $orderBy = null): LengthAwarePaginator
    {
        return ChatbotConversation::query()
            ->when(request('chatbot_channel'), function (Builder $query) {
                $query->where('chatbot_channel', request('chatbot_channel'));
            })
            ->where('is_showed_on_history', true)
            ->with('chatbot:id,uuid,avatar')
            ->with(['histories', 'lastMessage'])
            ->whereIn('chatbot_id', $chatbots)
            ->when($orderBy, function (Builder $query) use ($orderBy) {
                $query->orderBy($orderBy ?: 'id', 'desc');
            })
            ->paginate(request('per_page', request('perPage', 30)));
    }

    public function update(Model|int $model, array $data): Model
    {
        if (is_int($model)) {
            $model = $this->query()->findOrFail($model);
        }

        $model->update($data);

        return $model;
    }

    public function avatars(): Collection|array
    {
        return ChatbotAvatar::query()
            ->where(function (Builder $query) {
                return $query->where('user_id', Auth::id())->orWhereNull('user_id');
            })
            ->get();
    }

    public function query(): \Illuminate\Database\Eloquent\Builder
    {
        return Chatbot::query();
    }
}
