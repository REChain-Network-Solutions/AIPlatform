<?php

declare(strict_types=1);

use App\Extensions\Chatbot\System\Models\Chatbot;
use App\Extensions\Chatbot\System\Models\ChatbotConversation;
use App\Extensions\Chatbot\System\Models\ChatbotHistory;
use App\Extensions\Chatbot\System\Services\ChatbotAnalyticsService;
use App\Helpers\Classes\MarketplaceHelper;
use App\Models\Setting;
use App\Models\SettingTwo;
use App\Models\User;
use Illuminate\Support\Facades\View;
use Illuminate\Support\Str;

beforeEach(function () {
    if (! MarketplaceHelper::isRegistered('chatbot')) {
        $this->markTestSkipped('Chatbot extension is not registered.');
    }
    if (! defined('LARAVEL_START')) {
        define('LARAVEL_START', microtime(true));
    }
    $setting = Setting::factory()->create();
    $settingTwo = SettingTwo::factory()->create();
    View::share('setting', $setting);
    View::share('settings_two', $settingTwo);
    View::share('good_for_now', true);
    $this->user = User::factory()->create();
    $this->chatbot = Chatbot::query()->create([
        'uuid'               => Str::uuid()->toString(),
        'user_id'            => $this->user->id,
        'title'              => 'Test Bot',
        'bubble_message'     => 'Hello',
        'welcome_message'    => 'Welcome',
        'interaction_type'   => 'automatic_response',
        'instructions'       => 'Be helpful',
        'ai_model'           => 'gpt-4o-mini',
        'ai_embedding_model' => 'text-embedding-3-small',
    ]);
});

test('analytics page loads successfully for authenticated user', function () {
    $this->actingAs($this->user)
        ->get(route('dashboard.chatbot.analytics.index'))
        ->assertSuccessful()
        ->assertViewIs('chatbot::analytics.index')
        ->assertViewHas('stats')
        ->assertViewHas('newConversationsData')
        ->assertViewHas('agentRepliesData')
        ->assertViewHas('topAgents')
        ->assertViewHas('topChannels');
});

test('analytics page redirects unauthenticated users', function () {
    $this->get(route('dashboard.chatbot.analytics.index'))
        ->assertRedirect();
});

test('total conversations count is correct', function () {
    ChatbotConversation::query()->create([
        'chatbot_id' => $this->chatbot->id,
        'session_id' => 'sess-1',
    ]);

    ChatbotConversation::query()->create([
        'chatbot_id' => $this->chatbot->id,
        'session_id' => 'sess-2',
    ]);

    $service = app(ChatbotAnalyticsService::class);
    $count = $service->totalConversations([$this->chatbot->id]);

    expect($count)->toBe(2);
});

test('average rating is calculated from reviewed conversations', function () {
    ChatbotConversation::query()->create([
        'chatbot_id'               => $this->chatbot->id,
        'session_id'               => 'sess-1',
        'review_submitted_at'      => now(),
        'review_selected_response' => '5',
    ]);

    ChatbotConversation::query()->create([
        'chatbot_id'               => $this->chatbot->id,
        'session_id'               => 'sess-2',
        'review_submitted_at'      => now(),
        'review_selected_response' => '3',
    ]);

    ChatbotConversation::query()->create([
        'chatbot_id' => $this->chatbot->id,
        'session_id' => 'sess-3',
    ]);

    $service = app(ChatbotAnalyticsService::class);
    $rating = $service->averageRating([$this->chatbot->id]);

    expect($rating)->toBe(4.0);
});

test('average rating returns zero when no reviews exist', function () {
    $service = app(ChatbotAnalyticsService::class);
    $rating = $service->averageRating([$this->chatbot->id]);

    expect($rating)->toBe(0.0);
});

test('average response time measures first assistant reply after connect_agent_at', function () {
    // Conversation 1: agent replies 30s after takeover (valid)
    $conv1 = ChatbotConversation::query()->create([
        'chatbot_id'       => $this->chatbot->id,
        'session_id'       => 'sess-rt-1',
        'connect_agent_at' => now(),
    ]);

    ChatbotHistory::query()->create([
        'chatbot_id'      => $this->chatbot->id,
        'conversation_id' => $conv1->id,
        'role'            => 'assistant',
        'message'         => 'Agent reply',
        'created_at'      => now()->addSeconds(30),
    ]);

    // Conversation 2: agent replies 90s after takeover (valid)
    $conv2 = ChatbotConversation::query()->create([
        'chatbot_id'       => $this->chatbot->id,
        'session_id'       => 'sess-rt-2',
        'connect_agent_at' => now(),
    ]);

    ChatbotHistory::query()->create([
        'chatbot_id'      => $this->chatbot->id,
        'conversation_id' => $conv2->id,
        'role'            => 'assistant',
        'message'         => 'Agent reply',
        'created_at'      => now()->addSeconds(90),
    ]);

    // Conversation 3: agent replies after 3 hours (stale — excluded)
    $conv3 = ChatbotConversation::query()->create([
        'chatbot_id'       => $this->chatbot->id,
        'session_id'       => 'sess-rt-3',
        'connect_agent_at' => now(),
    ]);

    ChatbotHistory::query()->create([
        'chatbot_id'      => $this->chatbot->id,
        'conversation_id' => $conv3->id,
        'role'            => 'assistant',
        'message'         => 'Very late reply',
        'created_at'      => now()->addHours(3),
    ]);

    $service = app(ChatbotAnalyticsService::class);
    $result = $service->averageResponseTime([$this->chatbot->id]);

    // Only conv1 (30s) and conv2 (90s) count → avg = 60s = 1m
    expect($result)->toBe('1m');
});

test('new conversations chart data includes all channel and per-channel data', function () {
    ChatbotConversation::query()->create([
        'chatbot_id'      => $this->chatbot->id,
        'session_id'      => 'sess-1',
        'chatbot_channel' => 'livechat',
    ]);

    ChatbotConversation::query()->create([
        'chatbot_id'      => $this->chatbot->id,
        'session_id'      => 'sess-2',
        'chatbot_channel' => 'whatsapp',
    ]);

    $service = app(ChatbotAnalyticsService::class);
    $result = $service->getNewConversationsChartData([$this->chatbot->id]);

    expect($result)->toHaveKeys(['chart_data', 'months'])
        ->and($result['chart_data'])->toHaveCount(3)
        ->and($result['chart_data'][0]['id'])->toBe('all')
        ->and(collect($result['chart_data'])->pluck('id')->toArray())->toContain('livechat', 'whatsapp');
});

test('agent replies chart data returns monthly data', function () {
    $conversation = ChatbotConversation::query()->create([
        'chatbot_id'       => $this->chatbot->id,
        'session_id'       => 'sess-1',
        'connect_agent_at' => now(),
    ]);

    ChatbotHistory::query()->create([
        'chatbot_id'      => $this->chatbot->id,
        'conversation_id' => $conversation->id,
        'user_id'         => $this->user->id,
        'role'            => 'assistant',
        'message'         => 'Agent reply',
        'created_at'      => now(),
    ]);

    $service = app(ChatbotAnalyticsService::class);
    $result = $service->getAgentRepliesChartData([$this->chatbot->id]);

    expect($result)->toHaveKeys(['chart_data', 'months'])
        ->and($result['chart_data'])->toHaveCount(1)
        ->and($result['chart_data'][0]['name'])->toBe('agent_replies')
        ->and(array_sum($result['chart_data'][0]['data']))->toBeGreaterThanOrEqual(1);
});

test('top agents returns chatbots ordered by conversation count', function () {
    $chatbot2 = Chatbot::query()->create([
        'uuid'               => Str::uuid()->toString(),
        'user_id'            => $this->user->id,
        'title'              => 'Second Bot',
        'bubble_message'     => 'Hello',
        'welcome_message'    => 'Welcome',
        'interaction_type'   => 'automatic_response',
        'instructions'       => 'Be helpful',
        'ai_model'           => 'gpt-4o-mini',
        'ai_embedding_model' => 'text-embedding-3-small',
    ]);

    // 3 conversations for the first chatbot
    foreach (range(1, 3) as $i) {
        ChatbotConversation::query()->create([
            'chatbot_id' => $this->chatbot->id,
            'session_id' => "sess-bot1-$i",
        ]);
    }

    // 1 conversation for the second chatbot
    ChatbotConversation::query()->create([
        'chatbot_id' => $chatbot2->id,
        'session_id' => 'sess-bot2-1',
    ]);

    $service = app(ChatbotAnalyticsService::class);
    $topAgents = $service->getTopAgents([$this->chatbot->id, $chatbot2->id]);

    expect($topAgents)->toHaveCount(2)
        ->and($topAgents[0]['name'])->toBe('Test Bot')
        ->and($topAgents[0]['conversations_count'])->toBe(3)
        ->and($topAgents[1]['name'])->toBe('Second Bot')
        ->and($topAgents[1]['conversations_count'])->toBe(1);
});

test('top channels returns channels ordered by conversation count', function () {
    foreach (range(1, 3) as $i) {
        ChatbotConversation::query()->create([
            'chatbot_id'      => $this->chatbot->id,
            'session_id'      => "sess-livechat-$i",
            'chatbot_channel' => 'livechat',
        ]);
    }

    ChatbotConversation::query()->create([
        'chatbot_id'      => $this->chatbot->id,
        'session_id'      => 'sess-whatsapp-1',
        'chatbot_channel' => 'whatsapp',
    ]);

    $service = app(ChatbotAnalyticsService::class);
    $topChannels = $service->getTopChannels([$this->chatbot->id]);

    expect($topChannels)->toHaveCount(2)
        ->and($topChannels[0]['channel'])->toBe('Livechat')
        ->and($topChannels[0]['conversations_count'])->toBe(3)
        ->and($topChannels[1]['channel'])->toBe('Whatsapp')
        ->and($topChannels[1]['conversations_count'])->toBe(1);
});

test('average conversation duration excludes idle gaps over 5 minutes', function () {
    $conversation = ChatbotConversation::query()->create([
        'chatbot_id' => $this->chatbot->id,
        'session_id' => 'sess-duration-1',
    ]);

    // Message 1 at T+0
    ChatbotHistory::query()->create([
        'chatbot_id'      => $this->chatbot->id,
        'conversation_id' => $conversation->id,
        'role'            => 'user',
        'message'         => 'Hello',
        'created_at'      => now(),
    ]);

    // Message 2 at T+60s (within 5-min window → counted)
    ChatbotHistory::query()->create([
        'chatbot_id'      => $this->chatbot->id,
        'conversation_id' => $conversation->id,
        'role'            => 'assistant',
        'message'         => 'Hi there',
        'created_at'      => now()->addSeconds(60),
    ]);

    // Message 3 at T+120s (within 5-min window → counted)
    ChatbotHistory::query()->create([
        'chatbot_id'      => $this->chatbot->id,
        'conversation_id' => $conversation->id,
        'role'            => 'user',
        'message'         => 'Thanks',
        'created_at'      => now()->addSeconds(120),
    ]);

    // Message 4 at T+2h (idle gap → NOT counted)
    ChatbotHistory::query()->create([
        'chatbot_id'      => $this->chatbot->id,
        'conversation_id' => $conversation->id,
        'role'            => 'user',
        'message'         => 'Back again',
        'created_at'      => now()->addHours(2),
    ]);

    // Message 5 at T+2h+30s (within 5-min window → counted)
    ChatbotHistory::query()->create([
        'chatbot_id'      => $this->chatbot->id,
        'conversation_id' => $conversation->id,
        'role'            => 'assistant',
        'message'         => 'Welcome back',
        'created_at'      => now()->addHours(2)->addSeconds(30),
    ]);

    $service = app(ChatbotAnalyticsService::class);
    $result = $service->averageConversationDuration([$this->chatbot->id]);

    // Active time: 60s + 60s + 30s = 150s = 2m 30s
    // The 2-hour idle gap is excluded
    expect($result)->toBe('2m 30s');
});

test('average conversation duration includes single-message conversations as zero', function () {
    // Single-message conversation (duration = 0)
    $conv1 = ChatbotConversation::query()->create([
        'chatbot_id' => $this->chatbot->id,
        'session_id' => 'sess-single',
    ]);

    ChatbotHistory::query()->create([
        'chatbot_id'      => $this->chatbot->id,
        'conversation_id' => $conv1->id,
        'role'            => 'user',
        'message'         => 'Hello',
        'created_at'      => now(),
    ]);

    // Multi-message conversation (duration = 120s)
    $conv2 = ChatbotConversation::query()->create([
        'chatbot_id' => $this->chatbot->id,
        'session_id' => 'sess-multi',
    ]);

    ChatbotHistory::query()->create([
        'chatbot_id'      => $this->chatbot->id,
        'conversation_id' => $conv2->id,
        'role'            => 'user',
        'message'         => 'Hello',
        'created_at'      => now(),
    ]);

    ChatbotHistory::query()->create([
        'chatbot_id'      => $this->chatbot->id,
        'conversation_id' => $conv2->id,
        'role'            => 'assistant',
        'message'         => 'Hi',
        'created_at'      => now()->addSeconds(120),
    ]);

    $service = app(ChatbotAnalyticsService::class);
    $result = $service->averageConversationDuration([$this->chatbot->id]);

    // Average: (0 + 120) / 2 = 60 seconds
    expect($result)->toBe('1m');
});

test('format duration displays correctly', function () {
    $service = app(ChatbotAnalyticsService::class);

    $reflection = new ReflectionMethod($service, 'formatDuration');
    $reflection->setAccessible(true);

    expect($reflection->invoke($service, 0))->toBe('0s')
        ->and($reflection->invoke($service, 33))->toBe('33s')
        ->and($reflection->invoke($service, 110))->toBe('1m 50s')
        ->and($reflection->invoke($service, 3600))->toBe('1h 0m')
        ->and($reflection->invoke($service, 3720))->toBe('1h 2m');
});

test('stats view data has expected keys', function () {
    $response = $this->actingAs($this->user)
        ->get(route('dashboard.chatbot.analytics.index'));

    $stats = $response->viewData('stats');

    expect($stats)->toHaveKeys([
        'total_conversations',
        'avg_rating',
        'avg_response_time',
        'avg_conversation_duration',
    ]);
});
