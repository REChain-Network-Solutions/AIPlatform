<?php

declare(strict_types=1);

use App\Extensions\SocialMedia\System\Models\SocialMediaPlatform;
use App\Extensions\SocialMediaAutomation\System\Models\Automation;
use App\Extensions\SocialMediaAutomation\System\Models\AutomationAction;
use App\Extensions\SocialMediaAutomation\System\Models\AutomationLog;
use App\Extensions\SocialMediaAutomation\System\Models\PendingAutomation;
use App\Extensions\SocialMediaAutomation\System\Services\WebhookProcessor;
use App\Helpers\Classes\MarketplaceHelper;

beforeEach(function () {
    if (! MarketplaceHelper::isRegistered('social-media') || ! MarketplaceHelper::isRegistered('social-media-automation')) {
        $this->markTestSkipped('SocialMedia or SocialMediaAutomation extension is not registered.');
    }

    $user = loginAsUser();

    $this->platform = SocialMediaPlatform::query()->create([
        'user_id'     => $user->id,
        'platform'    => 'instagram',
        'credentials' => [
            'type'         => 'user',
            'id'           => 'ig_123',
            'platform_id'  => 'ig_123',
            'name'         => 'Test Account',
            'username'     => 'testaccount',
            'access_token' => 'fake-token-123',
        ],
        'connected_at' => now(),
        'expires_at'   => now()->addMonth(),
    ]);

    $this->automation = Automation::query()->create([
        'user_id'                   => $user->id,
        'social_media_platform_id'  => $this->platform->id,
        'name'                      => 'Test Automation',
        'status'                    => 'live',
        'trigger_target'            => 'all_posts',
        'keyword_mode'              => 'any',
        'include_keywords'          => [],
        'exclude_keywords'          => [],
        'enable_public_replies'     => false,
        'delay_seconds'             => 0,
    ]);

    AutomationAction::query()->create([
        'automation_id' => $this->automation->id,
        'type'          => 'text',
        'content'       => ['text' => 'Hello {first_name}!'],
        'order'         => 0,
    ]);
});

it('processes instagram webhook payload and creates pending automation', function () {
    $processor = app(WebhookProcessor::class);

    $payload = [
        'object' => 'instagram',
        'entry'  => [
            [
                'id'      => 'ig_123',
                'time'    => time(),
                'changes' => [
                    [
                        'field' => 'comments',
                        'value' => [
                            'id'    => 'comment_456',
                            'text'  => 'Great post!',
                            'from'  => [
                                'id'       => 'user_789',
                                'username' => 'johndoe',
                            ],
                            'media' => [
                                'id' => 'media_001',
                            ],
                        ],
                    ],
                ],
            ],
        ],
    ];

    $processor->processInstagramPayload($payload);

    $pending = PendingAutomation::query()
        ->where('automation_id', $this->automation->id)
        ->first();

    expect($pending)->not->toBeNull()
        ->and($pending->status)->toBe('pending')
        ->and($pending->comment_data['comment_id'])->toBe('comment_456');
});

it('processes facebook webhook payload for comment events', function () {
    $fbPlatform = SocialMediaPlatform::query()->create([
        'user_id'     => $this->platform->user_id,
        'platform'    => 'facebook',
        'credentials' => [
            'type'         => 'user',
            'platform_id'  => 'page_123',
            'name'         => 'Test Page',
            'access_token' => 'fb-token-123',
        ],
        'connected_at' => now(),
        'expires_at'   => now()->addMonth(),
    ]);

    $this->automation->update(['social_media_platform_id' => $fbPlatform->id]);

    $processor = app(WebhookProcessor::class);

    $payload = [
        'object' => 'page',
        'entry'  => [
            [
                'id'      => 'page_123',
                'time'    => time(),
                'changes' => [
                    [
                        'field' => 'feed',
                        'value' => [
                            'item'        => 'comment',
                            'post_id'     => 'post_abc',
                            'comment_id'  => 'comment_def',
                            'sender_id'   => 'user_ghi',
                            'sender_name' => 'Jane Doe',
                            'message'     => 'Awesome!',
                        ],
                    ],
                ],
            ],
        ],
    ];

    $processor->processFacebookPayload($payload);

    $pending = PendingAutomation::query()
        ->where('automation_id', $this->automation->id)
        ->first();

    expect($pending)->not->toBeNull()
        ->and($pending->comment_data['comment_id'])->toBe('comment_def')
        ->and($pending->comment_data['text'])->toBe('Awesome!');
});

it('ignores non-comment facebook webhook events', function () {
    $processor = app(WebhookProcessor::class);

    $payload = [
        'object' => 'page',
        'entry'  => [
            [
                'id'      => 'page_123',
                'changes' => [
                    [
                        'field' => 'feed',
                        'value' => [
                            'item'    => 'like',
                            'post_id' => 'post_abc',
                        ],
                    ],
                ],
            ],
        ],
    ];

    $processor->processFacebookPayload($payload);

    expect(AutomationLog::query()->where('automation_id', $this->automation->id)->count())->toBe(0);
});

it('ignores non-comment instagram webhook fields', function () {
    $processor = app(WebhookProcessor::class);

    $payload = [
        'object' => 'instagram',
        'entry'  => [
            [
                'id'      => 'page_123',
                'changes' => [
                    [
                        'field' => 'mentions',
                        'value' => ['text' => 'mentioned you'],
                    ],
                ],
            ],
        ],
    ];

    $processor->processInstagramPayload($payload);

    expect(AutomationLog::query()->where('automation_id', $this->automation->id)->count())->toBe(0);
});

it('processes x webhook payload for reply events', function () {
    $xPlatform = SocialMediaPlatform::query()->create([
        'user_id'     => $this->platform->user_id,
        'platform'    => 'x',
        'credentials' => [
            'type'         => 'user',
            'platform_id'  => 'x_user_123',
            'name'         => 'Test X Account',
            'access_token' => 'x-token-123',
        ],
        'connected_at' => now(),
        'expires_at'   => now()->addMonth(),
    ]);

    $this->automation->update(['social_media_platform_id' => $xPlatform->id]);

    $processor = app(WebhookProcessor::class);

    $payload = [
        'for_user_id'         => 'x_user_123',
        'tweet_create_events' => [
            [
                'id_str'                       => 'tweet_999',
                'text'                         => 'Nice tweet!',
                'in_reply_to_status_id_str'    => 'original_tweet_111',
                'user'                         => [
                    'id_str'      => 'commenter_222',
                    'screen_name' => 'replier_user',
                    'name'        => 'Reply User',
                ],
            ],
        ],
    ];

    $processor->processXPayload($payload);

    $pending = PendingAutomation::query()
        ->where('automation_id', $this->automation->id)
        ->first();

    expect($pending)->not->toBeNull()
        ->and($pending->comment_data['commenter_username'])->toBe('replier_user')
        ->and($pending->comment_data['text'])->toBe('Nice tweet!');
});

it('ignores x webhook events that are not replies', function () {
    $processor = app(WebhookProcessor::class);

    $payload = [
        'tweet_create_events' => [
            [
                'id_str' => 'tweet_999',
                'text'   => 'Just a new tweet',
                'user'   => [
                    'id_str'      => 'user_123',
                    'screen_name' => 'someuser',
                ],
            ],
        ],
    ];

    $processor->processXPayload($payload);

    expect(AutomationLog::query()->where('automation_id', $this->automation->id)->count())->toBe(0);
});

it('verifies instagram webhook signature correctly', function () {
    $processor = app(WebhookProcessor::class);

    $body = '{"test": true}';
    $secret = 'my-app-secret';
    $validSignature = 'sha256=' . hash_hmac('sha256', $body, $secret);

    $request = Request::create('/webhook', 'POST', content: $body);
    $request->headers->set('X-Hub-Signature-256', $validSignature);

    expect($processor->verifySignature($request, $secret))->toBeTrue();
});

it('rejects invalid webhook signature', function () {
    $processor = app(WebhookProcessor::class);

    $request = Request::create('/webhook', 'POST', content: '{"test": true}');
    $request->headers->set('X-Hub-Signature-256', 'sha256=invalid_hash');

    expect($processor->verifySignature($request, 'my-app-secret'))->toBeFalse();
});

it('generates valid x crc response', function () {
    $processor = app(WebhookProcessor::class);

    $crcToken = 'test-crc-token';
    $consumerSecret = 'test-consumer-secret';

    $response = $processor->generateXCrcResponse($crcToken, $consumerSecret);

    $expectedHash = hash_hmac('sha256', $crcToken, $consumerSecret, true);
    $expectedResponse = 'sha256=' . base64_encode($expectedHash);

    expect($response)->toBe($expectedResponse);
});

it('does not trigger automations for a different account', function () {
    $processor = app(WebhookProcessor::class);

    // Payload has account_id 'other_ig_account' which doesn't match our platform's 'ig_123'
    $payload = [
        'object' => 'instagram',
        'entry'  => [
            [
                'id'      => 'other_ig_account',
                'time'    => time(),
                'changes' => [
                    [
                        'field' => 'comments',
                        'value' => [
                            'id'    => 'comment_cross',
                            'text'  => 'Hello!',
                            'from'  => [
                                'id'       => 'user_cross',
                                'username' => 'crossuser',
                            ],
                            'media' => [
                                'id' => 'media_cross',
                            ],
                        ],
                    ],
                ],
            ],
        ],
    ];

    $processor->processInstagramPayload($payload);

    expect(PendingAutomation::query()->count())->toBe(0);
});
