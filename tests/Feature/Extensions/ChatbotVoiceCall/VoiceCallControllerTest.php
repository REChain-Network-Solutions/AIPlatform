<?php

declare(strict_types=1);

use App\Domains\Entity\BaseDriver;
use App\Domains\Entity\Enums\EntityEnum;
use App\Domains\Entity\Facades\Entity;
use App\Extensions\Chatbot\System\Models\Chatbot;
use App\Extensions\Chatbot\System\Models\ChatbotConversation;
use App\Extensions\Chatbot\System\Models\ChatbotHistory;
use App\Models\Plan;
use App\Models\Subscriptions;
use App\Models\User;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Str;

uses()->beforeEach(function () {
    if (! \App\Helpers\Classes\MarketplaceHelper::isRegistered('chatbot-voice-call')) {
        $this->markTestSkipped('ChatbotVoiceCall extension is not registered.');
    }
});

function mockEntityDriver(bool $hasCreditBalance = true): void
{
    $mockDriver = Mockery::mock(BaseDriver::class);
    $mockDriver->shouldReceive('forUser')->andReturnSelf();
    $mockDriver->shouldReceive('hasCreditBalance')->andReturn($hasCreditBalance);
    $mockDriver->shouldReceive('input')->andReturnSelf();
    $mockDriver->shouldReceive('calculateCredit')->andReturnSelf();
    $mockDriver->shouldReceive('decreaseCredit')->andReturn(true);

    Entity::shouldReceive('driver')->andReturn($mockDriver);
}

beforeEach(function () {
    $this->user = User::factory()->create();

    $this->chatbot = Chatbot::query()->create([
        'uuid'                => Str::uuid()->toString(),
        'user_id'             => $this->user->id,
        'title'               => 'Test Chatbot',
        'ai_model'            => 'gpt-4o',
        'ai_embedding_model'  => 'text-embedding-3-small',
        'voice_call_enabled'  => true,
        'voice_call_provider' => 'openai_realtime',
    ]);

    $this->conversation = ChatbotConversation::query()->create([
        'chatbot_id' => $this->chatbot->id,
        'session_id' => 'test-session-123',
    ]);

    $this->baseUrl = "api/v2/chatbot/{$this->chatbot->uuid}/session/test-session-123/voice-call";
});

test('start returns 403 when voice call is disabled', function () {
    $this->chatbot->update(['voice_call_enabled' => false]);

    $response = $this->postJson("{$this->baseUrl}/start");

    $response->assertForbidden()
        ->assertJsonPath('error', __('Voice call is not enabled for this chatbot.'));
});

test('start returns 403 when plan does not include ext_voice_call', function () {
    $plan = Plan::factory()->create([
        'open_ai_items' => ['ext_voice_call' => false],
        'plan_ai_tools' => [],
        'plan_features' => [],
    ]);

    $subscription = new Subscriptions;
    $subscription->user_id = $this->user->id;
    $subscription->plan_id = $plan->id;
    $subscription->stripe_id = 'sub_test_' . Str::random(10);
    $subscription->stripe_status = 'active';
    $subscription->name = 'default';
    $subscription->save();

    $response = $this->postJson("{$this->baseUrl}/start");

    $response->assertForbidden()
        ->assertJsonPath('error', __('Voice call is not available in your current plan.'));
});

test('start returns 403 when user has no credit balance', function () {
    mockEntityDriver(hasCreditBalance: false);

    $response = $this->postJson("{$this->baseUrl}/start");

    $response->assertForbidden()
        ->assertJsonPath('error', __('Insufficient credits for voice call.'));
});

test('start succeeds with valid credits and returns provider info', function () {
    mockEntityDriver(hasCreditBalance: true);

    Http::fake([
        'api.openai.com/*' => Http::response([
            'client_secret' => ['value' => 'test-ephemeral-key'],
        ]),
    ]);

    $response = $this->postJson("{$this->baseUrl}/start");

    $response->assertSuccessful()
        ->assertJsonPath('provider', 'openai_realtime')
        ->assertJsonPath('conversation_id', $this->conversation->id)
        ->assertJsonMissing(['remaining_seconds']);

    $this->assertDatabaseHas('ext_chatbot_histories', [
        'conversation_id' => $this->conversation->id,
        'role'            => 'voice-call-started',
    ]);
});

test('start returns remaining_seconds in demo mode', function () {
    config()->set('app.status', 'Demo');

    mockEntityDriver(hasCreditBalance: true);

    Http::fake([
        'api.openai.com/*' => Http::response([
            'client_secret' => ['value' => 'test-ephemeral-key'],
        ]),
    ]);

    $response = $this->postJson("{$this->baseUrl}/start");

    $response->assertSuccessful()
        ->assertJsonStructure(['remaining_seconds'])
        ->assertJsonPath('provider', 'openai_realtime');
});

test('start returns 429 in demo mode when limit exceeded', function () {
    config()->set('app.status', 'Demo');

    // Pre-fill demo usage for all possible IPs
    Cache::put('demo_ai_usage_seconds_127.0.0.1', 31, now()->endOfDay());

    $response = $this->postJson("{$this->baseUrl}/start");

    $response->assertStatus(429);
});

test('end deducts credits based on transcripts', function () {
    // Create voice-call-started marker
    ChatbotHistory::query()->create([
        'chatbot_id'      => $this->chatbot->id,
        'conversation_id' => $this->conversation->id,
        'role'            => 'voice-call-started',
        'message'         => 'Voice call started',
        'created_at'      => now()->subMinutes(2),
    ]);

    // Create transcript history entries
    ChatbotHistory::query()->create([
        'chatbot_id'      => $this->chatbot->id,
        'conversation_id' => $this->conversation->id,
        'role'            => 'voice-transcript-user',
        'message'         => 'Hello how are you',
        'created_at'      => now()->subMinute(),
    ]);

    ChatbotHistory::query()->create([
        'chatbot_id'      => $this->chatbot->id,
        'conversation_id' => $this->conversation->id,
        'role'            => 'voice-transcript-assistant',
        'message'         => 'I am doing well',
        'created_at'      => now()->subSeconds(30),
    ]);

    $mockDriver = Mockery::mock(BaseDriver::class);
    $mockDriver->shouldReceive('forUser')->andReturnSelf();
    $mockDriver->shouldReceive('input')->once()->andReturnSelf();
    $mockDriver->shouldReceive('calculateCredit')->once()->andReturnSelf();
    $mockDriver->shouldReceive('decreaseCredit')->once()->andReturn(true);

    Entity::shouldReceive('driver')
        ->with(EntityEnum::GPT_4_O_REALTIME_PREVIEW)
        ->andReturn($mockDriver);

    $response = $this->postJson("{$this->baseUrl}/end", ['duration' => 120]);

    $response->assertSuccessful()
        ->assertJsonPath('success', true);
});

test('end records demo usage seconds', function () {
    config()->set('app.status', 'Demo');

    $response = $this->postJson("{$this->baseUrl}/end", ['duration' => 10]);

    $response->assertSuccessful();
});

test('end creates voice-call-ended history entry', function () {
    $response = $this->postJson("{$this->baseUrl}/end");

    $response->assertSuccessful();

    $this->assertDatabaseHas('ext_chatbot_histories', [
        'chatbot_id'      => $this->chatbot->id,
        'conversation_id' => $this->conversation->id,
        'role'            => 'voice-call-ended',
    ]);
});

test('end returns 404 for invalid session', function () {
    $response = $this->postJson(
        "api/v2/chatbot/{$this->chatbot->uuid}/session/nonexistent-session/voice-call/end"
    );

    $response->assertNotFound();
});

test('transcript stores user message', function () {
    $response = $this->postJson("{$this->baseUrl}/transcript", [
        'role'    => 'user',
        'message' => 'Hello world',
    ]);

    $response->assertSuccessful()
        ->assertJsonPath('success', true);

    $this->assertDatabaseHas('ext_chatbot_histories', [
        'conversation_id' => $this->conversation->id,
        'role'            => 'voice-transcript-user',
        'message'         => 'Hello world',
    ]);
});

test('transcript stores assistant message', function () {
    $response = $this->postJson("{$this->baseUrl}/transcript", [
        'role'    => 'assistant',
        'message' => 'Hi there',
    ]);

    $response->assertSuccessful();

    $this->assertDatabaseHas('ext_chatbot_histories', [
        'conversation_id' => $this->conversation->id,
        'role'            => 'voice-transcript-assistant',
        'message'         => 'Hi there',
    ]);
});

test('transcript validates required fields', function () {
    $response = $this->postJson("{$this->baseUrl}/transcript", []);

    $response->assertUnprocessable()
        ->assertJsonValidationErrors(['role', 'message']);
});

test('start returns 403 when plan seconds limit is zero', function () {
    $plan = Plan::factory()->create([
        'open_ai_items'            => ['ext_voice_call' => true],
        'plan_ai_tools'            => ['ext_voice_call' => true],
        'plan_features'            => [],
        'voice_call_seconds_limit' => 0,
    ]);

    $subscription = new Subscriptions;
    $subscription->user_id = $this->user->id;
    $subscription->plan_id = $plan->id;
    $subscription->stripe_id = 'sub_test_' . Str::random(10);
    $subscription->stripe_status = 'active';
    $subscription->name = 'default';
    $subscription->save();

    mockEntityDriver(hasCreditBalance: true);

    $response = $this->postJson("{$this->baseUrl}/start");

    $response->assertForbidden()
        ->assertJsonPath('error', __('Voice call is not available in your current plan.'));
});

test('start returns 403 when monthly seconds limit exceeded', function () {
    $plan = Plan::factory()->create([
        'open_ai_items'            => ['ext_voice_call' => true],
        'plan_ai_tools'            => ['ext_voice_call' => true],
        'plan_features'            => [],
        'voice_call_seconds_limit' => 60,
    ]);

    $subscription = new Subscriptions;
    $subscription->user_id = $this->user->id;
    $subscription->plan_id = $plan->id;
    $subscription->stripe_id = 'sub_test_' . Str::random(10);
    $subscription->stripe_status = 'active';
    $subscription->name = 'default';
    $subscription->save();

    // Record 60 seconds of usage this month
    ChatbotHistory::query()->create([
        'chatbot_id'          => $this->chatbot->id,
        'conversation_id'     => $this->conversation->id,
        'role'                => 'voice-call-ended',
        'message'             => 'Voice call ended',
        'voice_call_duration' => 60,
        'created_at'          => now(),
    ]);

    mockEntityDriver(hasCreditBalance: true);

    $response = $this->postJson("{$this->baseUrl}/start");

    $response->assertForbidden()
        ->assertJsonPath('error', __('Monthly voice call limit reached.'));
});

test('start returns remaining_seconds when plan has seconds limit', function () {
    $plan = Plan::factory()->create([
        'open_ai_items'            => ['ext_voice_call' => true],
        'plan_ai_tools'            => ['ext_voice_call' => true],
        'plan_features'            => [],
        'voice_call_seconds_limit' => 120,
    ]);

    $subscription = new Subscriptions;
    $subscription->user_id = $this->user->id;
    $subscription->plan_id = $plan->id;
    $subscription->stripe_id = 'sub_test_' . Str::random(10);
    $subscription->stripe_status = 'active';
    $subscription->name = 'default';
    $subscription->save();

    // Record 30 seconds of usage this month
    ChatbotHistory::query()->create([
        'chatbot_id'          => $this->chatbot->id,
        'conversation_id'     => $this->conversation->id,
        'role'                => 'voice-call-ended',
        'message'             => 'Voice call ended',
        'voice_call_duration' => 30,
        'created_at'          => now(),
    ]);

    mockEntityDriver(hasCreditBalance: true);

    Http::fake([
        'api.openai.com/*' => Http::response([
            'client_secret' => ['value' => 'test-ephemeral-key'],
        ]),
    ]);

    $response = $this->postJson("{$this->baseUrl}/start");

    $response->assertSuccessful()
        ->assertJsonPath('remaining_seconds', 90)
        ->assertJsonPath('provider', 'openai_realtime');
});

test('end records voice call duration in history', function () {
    $response = $this->postJson("{$this->baseUrl}/end", ['duration' => 45]);

    $response->assertSuccessful();

    $this->assertDatabaseHas('ext_chatbot_histories', [
        'chatbot_id'          => $this->chatbot->id,
        'conversation_id'     => $this->conversation->id,
        'role'                => 'voice-call-ended',
        'voice_call_duration' => 45,
    ]);
});
