<?php

declare(strict_types=1);

use App\Extensions\SocialMedia\System\Models\SocialMediaPlatform;
use App\Helpers\Classes\MarketplaceHelper;
use Illuminate\Support\Facades\Http;

beforeEach(function () {
    if (! MarketplaceHelper::isRegistered('social-media') || ! MarketplaceHelper::isRegistered('social-media-automation')) {
        $this->markTestSkipped('SocialMedia or SocialMediaAutomation extension is not registered.');
    }

    $this->user = loginAsUser();
});

it('returns posts for facebook platform via api', function () {
    $platform = SocialMediaPlatform::query()->create([
        'user_id'      => $this->user->id,
        'platform'     => 'facebook',
        'credentials'  => [
            'platform_id'  => 'page_123',
            'access_token' => 'fake-fb-token',
        ],
        'connected_at' => now(),
        'expires_at'   => now()->addMonth(),
    ]);

    Http::fake([
        '*graph.facebook.com*' => Http::response([
            'data' => [
                [
                    'id'           => 'post_1',
                    'message'      => 'Hello Facebook',
                    'full_picture' => 'https://example.com/img.jpg',
                    'type'         => 'photo',
                    'created_time' => '2024-01-01T12:00:00+0000',
                ],
            ],
        ]),
    ]);

    $response = $this->getJson(route('dashboard.user.social-media.automation.posts', ['platform_id' => $platform->id]));

    $response->assertSuccessful()
        ->assertJsonPath('status', 'success')
        ->assertJsonPath('data.0.platform_post_id', 'post_1')
        ->assertJsonPath('data.0.content', 'Hello Facebook')
        ->assertJsonPath('data.0.post_type', 'photo');
});

it('returns posts for x platform via api', function () {
    $platform = SocialMediaPlatform::query()->create([
        'user_id'      => $this->user->id,
        'platform'     => 'x',
        'credentials'  => [
            'platform_id'  => 'user_x_123',
            'access_token' => 'fake-x-token',
        ],
        'connected_at' => now(),
        'expires_at'   => now()->addMonth(),
    ]);

    Http::fake([
        '*api.twitter.com*' => Http::response([
            'data' => [
                [
                    'id'         => 'tweet_1',
                    'text'       => 'Hello X platform',
                    'created_at' => '2024-01-01T12:00:00.000Z',
                ],
                [
                    'id'          => 'tweet_2',
                    'text'        => 'Tweet with image',
                    'created_at'  => '2024-01-02T12:00:00.000Z',
                    'attachments' => ['media_keys' => ['media_key_1']],
                ],
            ],
            'includes' => [
                'media' => [
                    [
                        'media_key' => 'media_key_1',
                        'type'      => 'photo',
                        'url'       => 'https://example.com/tweet-img.jpg',
                    ],
                ],
            ],
        ]),
    ]);

    $response = $this->getJson(route('dashboard.user.social-media.automation.posts', ['platform_id' => $platform->id]));

    $response->assertSuccessful()
        ->assertJsonPath('status', 'success')
        ->assertJsonPath('data.0.platform_post_id', 'tweet_1')
        ->assertJsonPath('data.0.content', 'Hello X platform')
        ->assertJsonPath('data.0.post_type', 'post')
        ->assertJsonPath('data.1.platform_post_id', 'tweet_2')
        ->assertJsonPath('data.1.post_type', 'photo')
        ->assertJsonPath('data.1.image', 'https://example.com/tweet-img.jpg');
});

it('filters x posts by search term', function () {
    $platform = SocialMediaPlatform::query()->create([
        'user_id'      => $this->user->id,
        'platform'     => 'x',
        'credentials'  => [
            'platform_id'  => 'user_x_123',
            'access_token' => 'fake-x-token',
        ],
        'connected_at' => now(),
        'expires_at'   => now()->addMonth(),
    ]);

    Http::fake([
        '*api.twitter.com*' => Http::response([
            'data' => [
                ['id' => 'tweet_1', 'text' => 'Hello world', 'created_at' => '2024-01-01T12:00:00.000Z'],
                ['id' => 'tweet_2', 'text' => 'Laravel is great', 'created_at' => '2024-01-02T12:00:00.000Z'],
            ],
        ]),
    ]);

    $response = $this->getJson(route('dashboard.user.social-media.automation.posts', [
        'platform_id' => $platform->id,
        'search'      => 'laravel',
    ]));

    $response->assertSuccessful()
        ->assertJsonCount(1, 'data')
        ->assertJsonPath('data.0.platform_post_id', 'tweet_2');
});

it('returns error when platform does not belong to user', function () {
    $otherUser = \App\Models\User::factory()->create();

    $platform = SocialMediaPlatform::query()->create([
        'user_id'      => $otherUser->id,
        'platform'     => 'x',
        'credentials'  => ['platform_id' => 'user_x_123', 'access_token' => 'fake-token'],
        'connected_at' => now(),
        'expires_at'   => now()->addMonth(),
    ]);

    $response = $this->getJson(route('dashboard.user.social-media.automation.posts', ['platform_id' => $platform->id]));

    $response->assertSuccessful()
        ->assertJsonPath('status', 'error');
});
