<?php

declare(strict_types=1);

use App\Extensions\Chatbot\System\Models\Chatbot;
use App\Extensions\Chatbot\System\Models\ChatbotConversation;
use App\Extensions\Chatbot\System\Models\ChatbotPageVisit;
use App\Models\User;
use Illuminate\Support\Str;

beforeEach(function () {
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
    $this->sessionId = md5(uniqid((string) mt_rand(), true));
});

test('page visit can be recorded via API', function () {
    $this->postJson(
        "/api/v2/chatbot/{$this->chatbot->uuid}/session/{$this->sessionId}/page-visit",
        [
            'page_url'   => 'https://example.com/products',
            'page_title' => 'Products - My Store',
        ]
    )->assertCreated()
        ->assertJsonStructure(['data' => ['id']]);

    $this->assertDatabaseHas('ext_chatbot_page_visits', [
        'chatbot_id' => $this->chatbot->id,
        'session_id' => $this->sessionId,
        'page_url'   => 'https://example.com/products',
        'page_title' => 'Products - My Store',
    ]);
});

test('page visit requires page_url', function () {
    $this->postJson(
        "/api/v2/chatbot/{$this->chatbot->uuid}/session/{$this->sessionId}/page-visit",
        [
            'page_title' => 'Some Page',
        ]
    )->assertUnprocessable()
        ->assertJsonValidationErrors(['page_url']);
});

test('recording a new page visit closes the previous one', function () {
    $this->postJson(
        "/api/v2/chatbot/{$this->chatbot->uuid}/session/{$this->sessionId}/page-visit",
        [
            'page_url'   => 'https://example.com/page-1',
            'page_title' => 'Page 1',
        ]
    )->assertCreated();

    $firstVisit = ChatbotPageVisit::query()->first();
    expect($firstVisit->left_at)->toBeNull();

    $this->postJson(
        "/api/v2/chatbot/{$this->chatbot->uuid}/session/{$this->sessionId}/page-visit",
        [
            'page_url'   => 'https://example.com/page-2',
            'page_title' => 'Page 2',
        ]
    )->assertCreated();

    $firstVisit->refresh();
    expect($firstVisit->left_at)->not->toBeNull();

    expect(ChatbotPageVisit::query()->count())->toBe(2);
});

test('leave page visit sets left_at on open visits', function () {
    $this->postJson(
        "/api/v2/chatbot/{$this->chatbot->uuid}/session/{$this->sessionId}/page-visit",
        [
            'page_url'   => 'https://example.com/about',
            'page_title' => 'About',
        ]
    )->assertCreated();

    $this->putJson(
        "/api/v2/chatbot/{$this->chatbot->uuid}/session/{$this->sessionId}/page-visit"
    )->assertSuccessful();

    $visit = ChatbotPageVisit::query()->first();
    expect($visit->left_at)->not->toBeNull();
});

test('duration accessor calculates difference in seconds', function () {
    $visit = ChatbotPageVisit::query()->create([
        'chatbot_id' => $this->chatbot->id,
        'session_id' => $this->sessionId,
        'page_url'   => 'https://example.com',
        'entered_at' => now()->subMinutes(5),
        'left_at'    => now(),
    ]);

    expect($visit->duration)->toBeGreaterThanOrEqual(299)
        ->and($visit->duration)->toBeLessThanOrEqual(301);
});

test('visited pages appear in conversation resource', function () {
    $conversation = ChatbotConversation::query()->create([
        'chatbot_id' => $this->chatbot->id,
        'session_id' => $this->sessionId,
    ]);

    ChatbotPageVisit::query()->create([
        'chatbot_id' => $this->chatbot->id,
        'session_id' => $this->sessionId,
        'page_url'   => 'https://example.com/home',
        'page_title' => 'Home',
        'entered_at' => now()->subMinutes(10),
        'left_at'    => now()->subMinutes(5),
    ]);

    ChatbotPageVisit::query()->create([
        'chatbot_id' => $this->chatbot->id,
        'session_id' => $this->sessionId,
        'page_url'   => 'https://example.com/pricing',
        'page_title' => 'Pricing',
        'entered_at' => now()->subMinutes(5),
        'left_at'    => now(),
    ]);

    $resource = new \App\Extensions\Chatbot\System\Http\Resources\Admin\ChatbotConversationResource($conversation->load('chatbot'));
    $data = $resource->toArray(request());

    expect($data['visited_pages'])->toHaveCount(2)
        ->and($data['visited_pages'][0]['page_title'])->toBe('Pricing')
        ->and($data['visited_pages'][1]['page_title'])->toBe('Home');
});
