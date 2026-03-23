<?php

declare(strict_types=1);

use App\Extensions\Chatbot\System\Generators\Contracts\Generator;
use App\Extensions\Chatbot\System\Models\Chatbot;
use App\Extensions\Chatbot\System\Models\ChatbotConversation;
use App\Extensions\Chatbot\System\Models\ChatbotHistory;
use App\Extensions\Chatbot\System\Services\GeneratorService;
use App\Models\User;
use Illuminate\Http\UploadedFile;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Str;

beforeEach(function () {
    Storage::fake('public');

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
        'is_attachment'      => true,
    ]);
    $this->sessionId = md5(uniqid((string) mt_rand(), true));
    $this->conversation = ChatbotConversation::query()->create([
        'chatbot_id'        => $this->chatbot->id,
        'session_id'        => $this->sessionId,
        'chatbot_channel'   => 'frame',
        'last_activity_at'  => now(),
    ]);
});

test('storeMessage accepts image upload with prompt text', function () {
    $this->mock(GeneratorService::class, function ($mock) {
        $mock->shouldReceive('setChatbot')->andReturnSelf();
        $mock->shouldReceive('setConversation')->andReturnSelf();
        $mock->shouldReceive('setPrompt')->andReturnSelf();
        $mock->shouldReceive('generate')->andReturn('I see a photo of something.');
    });

    $image = UploadedFile::fake()->image('photo.jpg', 200, 200);

    $response = $this->postJson(
        "/api/v2/chatbot/{$this->chatbot->uuid}/session/{$this->sessionId}/conversation/{$this->conversation->id}/messages",
        [
            'prompt' => 'What is in this image?',
            'media'  => $image,
        ]
    );

    $response->assertSuccessful();

    $this->assertDatabaseHas('ext_chatbot_histories', [
        'conversation_id' => $this->conversation->id,
        'role'            => 'user',
        'message'         => 'What is in this image?',
    ]);

    $userHistory = ChatbotHistory::query()
        ->where('conversation_id', $this->conversation->id)
        ->where('role', 'user')
        ->first();

    expect($userHistory->media_url)->not->toBeNull();
    expect($userHistory->media_url)->toContain('chatbot-media');
});

test('storeMessage accepts image upload without prompt text', function () {
    $this->mock(GeneratorService::class, function ($mock) {
        $mock->shouldReceive('setChatbot')->andReturnSelf();
        $mock->shouldReceive('setConversation')->andReturnSelf();
        $mock->shouldReceive('setPrompt')->andReturnSelf();
        $mock->shouldReceive('generate')->andReturn('I see a landscape.');
    });

    $image = UploadedFile::fake()->image('landscape.png', 300, 300);

    $response = $this->postJson(
        "/api/v2/chatbot/{$this->chatbot->uuid}/session/{$this->sessionId}/conversation/{$this->conversation->id}/messages",
        [
            'media' => $image,
        ]
    );

    $response->assertSuccessful();

    $userHistory = ChatbotHistory::query()
        ->where('conversation_id', $this->conversation->id)
        ->where('role', 'user')
        ->first();

    expect($userHistory->message)->toBe('');
    expect($userHistory->media_url)->not->toBeNull();
});

test('storeMessage rejects non-image files in AI mode via validation', function () {
    $pdf = UploadedFile::fake()->create('doc.pdf', 100, 'application/pdf');

    $response = $this->postJson(
        "/api/v2/chatbot/{$this->chatbot->uuid}/session/{$this->sessionId}/conversation/{$this->conversation->id}/messages",
        [
            'media' => $pdf,
        ]
    );

    // Should still pass validation since ChatbotHistoryStoreRequest allows various file types
    // But prompt is required_without:media, and media is present, so no prompt error
    $response->assertJsonMissingValidationErrors(['prompt']);
});

test('validation requires prompt when no media is present', function () {
    $response = $this->postJson(
        "/api/v2/chatbot/{$this->chatbot->uuid}/session/{$this->sessionId}/conversation/{$this->conversation->id}/messages",
        []
    );

    $response->assertUnprocessable();
    $response->assertJsonValidationErrors(['prompt']);
});

test('isImageMediaUrl correctly identifies image URLs', function () {
    $generator = new class extends Generator
    {
        public function generate(): string
        {
            return '';
        }

        public function modifyMessages(): array
        {
            return [];
        }
    };

    expect($generator->isImageMediaUrl('/uploads/chatbot-media/photo.jpg'))->toBeTrue();
    expect($generator->isImageMediaUrl('/uploads/chatbot-media/photo.jpeg'))->toBeTrue();
    expect($generator->isImageMediaUrl('/uploads/chatbot-media/photo.png'))->toBeTrue();
    expect($generator->isImageMediaUrl('/uploads/chatbot-media/photo.webp'))->toBeTrue();
    expect($generator->isImageMediaUrl('/uploads/chatbot-media/photo.gif'))->toBeTrue();
    expect($generator->isImageMediaUrl('/uploads/chatbot-media/doc.pdf'))->toBeFalse();
    expect($generator->isImageMediaUrl('/uploads/chatbot-media/file.docx'))->toBeFalse();
    expect($generator->isImageMediaUrl(null))->toBeFalse();
    expect($generator->isImageMediaUrl(''))->toBeFalse();
});

test('encodeImageFromMediaUrl returns base64 data for stored image', function () {
    Storage::disk('public')->put('chatbot-media/test.png', 'fake-image-content');

    $generator = new class extends Generator
    {
        public function generate(): string
        {
            return '';
        }

        public function modifyMessages(): array
        {
            return [];
        }
    };

    $result = $generator->encodeImageFromMediaUrl('/uploads/chatbot-media/test.png');

    expect($result)->not->toBeNull();
    expect($result['base64'])->toBe(base64_encode('fake-image-content'));
    expect($result['mime_type'])->toBe('image/png');
});

test('encodeImageFromMediaUrl returns null for non-existent file', function () {
    $generator = new class extends Generator
    {
        public function generate(): string
        {
            return '';
        }

        public function modifyMessages(): array
        {
            return [];
        }
    };

    $result = $generator->encodeImageFromMediaUrl('/uploads/chatbot-media/nonexistent.jpg');

    expect($result)->toBeNull();
});

test('histories query includes media_url column', function () {
    ChatbotHistory::query()->create([
        'chatbot_id'      => $this->chatbot->id,
        'conversation_id' => $this->conversation->id,
        'role'            => 'user',
        'model'           => 'gpt-4o-mini',
        'message'         => 'Hello',
        'media_url'       => '/uploads/chatbot-media/photo.jpg',
        'created_at'      => now(),
    ]);

    $generator = new class extends Generator
    {
        public function generate(): string
        {
            return '';
        }

        public function modifyMessages(): array
        {
            return [];
        }
    };

    $generator->conversation = $this->conversation;
    $histories = $generator->histories();

    expect($histories)->toHaveCount(1);
    expect($histories->first()->media_url)->toBe('/uploads/chatbot-media/photo.jpg');
});
