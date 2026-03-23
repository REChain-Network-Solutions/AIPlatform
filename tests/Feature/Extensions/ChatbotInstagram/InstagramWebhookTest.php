<?php

declare(strict_types=1);

use App\Extensions\Chatbot\System\Models\Chatbot;
use App\Extensions\Chatbot\System\Models\ChatbotChannel;
use App\Extensions\Chatbot\System\Models\ChatbotConversation;
use App\Extensions\ChatbotInstagram\System\Services\InstagramConversationService;
use App\Models\User;
use Illuminate\Support\Str;

uses()->beforeEach(function () {
    if (! \App\Helpers\Classes\MarketplaceHelper::isRegistered('chatbot-instagram')) {
        $this->markTestSkipped('ChatbotInstagram extension is not registered.');
    }
});

beforeEach(function () {
    $this->user = User::factory()->create();

    $this->chatbot = Chatbot::query()->create([
        'uuid'               => Str::uuid()->toString(),
        'user_id'            => $this->user->id,
        'title'              => 'Test Instagram Bot',
        'ai_model'           => 'gpt-4o',
        'ai_embedding_model' => 'text-embedding-3-small',
    ]);

    $this->channel = ChatbotChannel::query()->create([
        'user_id'      => $this->user->id,
        'chatbot_id'   => $this->chatbot->id,
        'channel'      => 'instagram',
        'credentials'  => [
            'instagram_id'  => '17841400123456789',
            'access_token'  => 'test-access-token',
            'verify_token'  => 'legacy-verify-token',
        ],
        'connected_at' => now(),
    ]);

    $this->conversation = ChatbotConversation::query()->create([
        'chatbot_id'  => $this->chatbot->id,
        'session_id'  => 'test-session-' . Str::random(8),
    ]);

    $this->globalUrl = 'api/v2/chatbot/webhook/instagram';
    $this->legacyUrl = "api/v2/chatbot/{$this->chatbot->id}/channel/{$this->channel->id}/instagram";
});

// ── Global Endpoint: GET Verification ────────────────────────────────

test('global GET verification succeeds with valid verify token', function () {
    setting(['INSTAGRAM_VERIFY_TOKEN' => 'my-global-token'])->save();

    $response = $this->get($this->globalUrl . '?' . http_build_query([
        'hub_mode'         => 'subscribe',
        'hub_verify_token' => 'my-global-token',
        'hub_challenge'    => 'challenge_abc123',
    ]));

    $response->assertOk();
    expect($response->getContent())->toBe('challenge_abc123');
});

test('global GET verification fails with invalid verify token', function () {
    setting(['INSTAGRAM_VERIFY_TOKEN' => 'my-global-token'])->save();

    $response = $this->get($this->globalUrl . '?' . http_build_query([
        'hub_mode'         => 'subscribe',
        'hub_verify_token' => 'wrong-token',
        'hub_challenge'    => 'challenge_abc123',
    ]));

    $response->assertStatus(403);
});

// ── Global Endpoint: POST Channel Resolution ─────────────────────────

test('global POST resolves channel from entry.0.id and processes webhook', function () {
    setting(['INSTAGRAM_APP_SECRET' => ''])->save();

    $this->mock(InstagramConversationService::class, function ($mock) {
        $mock->shouldReceive('setIpAddress')->andReturnSelf();
        $mock->shouldReceive('setChatbotId')->with($this->chatbot->id)->andReturnSelf();
        $mock->shouldReceive('setChannelId')->with($this->channel->id)->andReturnSelf();
        $mock->shouldReceive('setPayload')->andReturnSelf();
        $mock->shouldReceive('storeConversation')->andReturn($this->conversation);
        $mock->shouldReceive('getChatbot')->andReturn($this->chatbot);
        $mock->shouldReceive('insertMessage')->andReturn(null);
        $mock->shouldReceive('handle')->andReturn(null);
    });

    $response = $this->postJson($this->globalUrl, [
        'object' => 'instagram',
        'entry'  => [
            [
                'id'        => '17841400123456789',
                'time'      => time(),
                'messaging' => [
                    [
                        'sender'    => ['id' => '9876543210'],
                        'recipient' => ['id' => '17841400123456789'],
                        'message'   => [
                            'mid'  => 'mid.test123',
                            'text' => 'Hello from Instagram',
                        ],
                    ],
                ],
            ],
        ],
    ]);

    $response->assertOk()
        ->assertJsonPath('status', 'processed');
});

test('global POST returns 404 when instagram_id is not found', function () {
    setting(['INSTAGRAM_APP_SECRET' => ''])->save();

    $response = $this->postJson($this->globalUrl, [
        'object' => 'instagram',
        'entry'  => [
            [
                'id'        => 'unknown_instagram_id',
                'time'      => time(),
                'messaging' => [
                    [
                        'sender'  => ['id' => '111'],
                        'message' => ['mid' => 'mid.x', 'text' => 'hi'],
                    ],
                ],
            ],
        ],
    ]);

    $response->assertNotFound()
        ->assertJsonPath('error', 'Channel not found');
});

test('global POST returns 400 when entry ID is missing', function () {
    setting(['INSTAGRAM_APP_SECRET' => ''])->save();

    $response = $this->postJson($this->globalUrl, [
        'object' => 'instagram',
        'entry'  => [],
    ]);

    $response->assertStatus(400)
        ->assertJsonPath('error', 'Missing entry ID');
});

// ── Signature Verification ───────────────────────────────────────────

test('global POST with valid X-Hub-Signature-256 is accepted', function () {
    $appSecret = 'test-app-secret-key';
    setting(['INSTAGRAM_APP_SECRET' => $appSecret])->save();

    $this->mock(InstagramConversationService::class, function ($mock) {
        $mock->shouldReceive('setIpAddress')->andReturnSelf();
        $mock->shouldReceive('setChatbotId')->andReturnSelf();
        $mock->shouldReceive('setChannelId')->andReturnSelf();
        $mock->shouldReceive('setPayload')->andReturnSelf();
        $mock->shouldReceive('storeConversation')->andReturn($this->conversation);
        $mock->shouldReceive('getChatbot')->andReturn($this->chatbot);
        $mock->shouldReceive('insertMessage')->andReturn(null);
        $mock->shouldReceive('handle')->andReturn(null);
    });

    $payload = json_encode([
        'object' => 'instagram',
        'entry'  => [
            [
                'id'        => '17841400123456789',
                'time'      => time(),
                'messaging' => [
                    [
                        'sender'  => ['id' => '9876543210'],
                        'message' => ['mid' => 'mid.sig', 'text' => 'signed message'],
                    ],
                ],
            ],
        ],
    ]);

    $signature = 'sha256=' . hash_hmac('sha256', $payload, $appSecret);

    $response = $this->call(
        'POST',
        $this->globalUrl,
        [],
        [],
        [],
        [
            'HTTP_X-Hub-Signature-256' => $signature,
            'CONTENT_TYPE'             => 'application/json',
        ],
        $payload
    );

    $response->assertOk()
        ->assertJsonPath('status', 'processed');
});

test('global POST with invalid X-Hub-Signature-256 is rejected', function () {
    $appSecret = 'test-app-secret-key';
    setting(['INSTAGRAM_APP_SECRET' => $appSecret])->save();

    $payload = json_encode([
        'object' => 'instagram',
        'entry'  => [
            ['id' => '17841400123456789', 'messaging' => [['sender' => ['id' => '1'], 'message' => ['text' => 'hi']]]],
        ],
    ]);

    $response = $this->call(
        'POST',
        $this->globalUrl,
        [],
        [],
        [],
        [
            'HTTP_X-Hub-Signature-256' => 'sha256=invalid_signature_hash',
            'CONTENT_TYPE'             => 'application/json',
        ],
        $payload
    );

    $response->assertStatus(403)
        ->assertJsonPath('error', 'Invalid signature');
});

test('global POST without signature header is rejected when app secret is set', function () {
    setting(['INSTAGRAM_APP_SECRET' => 'some-secret'])->save();

    $response = $this->postJson($this->globalUrl, [
        'object' => 'instagram',
        'entry'  => [
            ['id' => '17841400123456789', 'messaging' => [['sender' => ['id' => '1'], 'message' => ['text' => 'hi']]]],
        ],
    ]);

    $response->assertStatus(403)
        ->assertJsonPath('error', 'Invalid signature');
});

// ── Legacy URL Backward Compatibility ────────────────────────────────

test('legacy URL GET verification still works', function () {
    $response = $this->get($this->legacyUrl . '?' . http_build_query([
        'hub_mode'         => 'subscribe',
        'hub_verify_token' => 'legacy-verify-token',
        'hub_challenge'    => 'legacy_challenge_123',
    ]));

    $response->assertOk();
    expect($response->getContent())->toBe('legacy_challenge_123');
});

test('legacy URL GET verification fails with wrong token', function () {
    $response = $this->get($this->legacyUrl . '?' . http_build_query([
        'hub_mode'         => 'subscribe',
        'hub_verify_token' => 'wrong-token',
        'hub_challenge'    => 'challenge',
    ]));

    $response->assertStatus(403);
});

// ── Multiple Channels ────────────────────────────────────────────────

test('correct channel is matched when multiple channels exist', function () {
    setting(['INSTAGRAM_APP_SECRET' => ''])->save();

    $otherChatbot = Chatbot::query()->create([
        'uuid'               => Str::uuid()->toString(),
        'user_id'            => $this->user->id,
        'title'              => 'Other Bot',
        'ai_model'           => 'gpt-4o',
        'ai_embedding_model' => 'text-embedding-3-small',
    ]);

    $otherChannel = ChatbotChannel::query()->create([
        'user_id'      => $this->user->id,
        'chatbot_id'   => $otherChatbot->id,
        'channel'      => 'instagram',
        'credentials'  => [
            'instagram_id'  => '17841400999999999',
            'access_token'  => 'other-access-token',
            'verify_token'  => 'other-verify-token',
        ],
        'connected_at' => now(),
    ]);

    $otherConversation = ChatbotConversation::query()->create([
        'chatbot_id' => $otherChatbot->id,
        'session_id' => 'test-session-other-' . Str::random(8),
    ]);

    $this->mock(InstagramConversationService::class, function ($mock) use ($otherChatbot, $otherChannel, $otherConversation) {
        $mock->shouldReceive('setIpAddress')->andReturnSelf();
        $mock->shouldReceive('setChatbotId')->with($otherChatbot->id)->andReturnSelf();
        $mock->shouldReceive('setChannelId')->with($otherChannel->id)->andReturnSelf();
        $mock->shouldReceive('setPayload')->andReturnSelf();
        $mock->shouldReceive('storeConversation')->andReturn($otherConversation);
        $mock->shouldReceive('getChatbot')->andReturn($otherChatbot);
        $mock->shouldReceive('insertMessage')->andReturn(null);
        $mock->shouldReceive('handle')->andReturn(null);
    });

    $response = $this->postJson($this->globalUrl, [
        'object' => 'instagram',
        'entry'  => [
            [
                'id'        => '17841400999999999',
                'time'      => time(),
                'messaging' => [
                    [
                        'sender'  => ['id' => '5555555'],
                        'message' => ['mid' => 'mid.multi', 'text' => 'Hello other bot'],
                    ],
                ],
            ],
        ],
    ]);

    $response->assertOk()
        ->assertJsonPath('status', 'processed');
});
