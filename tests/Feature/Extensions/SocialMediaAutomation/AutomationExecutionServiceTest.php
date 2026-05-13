<?php

declare(strict_types=1);

use App\Extensions\SocialMedia\System\Models\SocialMediaPlatform;
use App\Extensions\SocialMediaAutomation\System\Models\Automation;
use App\Extensions\SocialMediaAutomation\System\Models\AutomationAction;
use App\Extensions\SocialMediaAutomation\System\Models\AutomationLog;
use App\Extensions\SocialMediaAutomation\System\Models\AutomationReply;
use App\Extensions\SocialMediaAutomation\System\Models\PendingAutomation;
use App\Extensions\SocialMediaAutomation\System\Services\AutomationExecutionService;
use App\Helpers\Classes\MarketplaceHelper;
use Illuminate\Support\Facades\Http;

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

    $this->service = app(AutomationExecutionService::class);
});

it('matches trigger for all_posts keyword mode any', function () {
    $automation = Automation::query()->create([
        'user_id'                   => $this->platform->user_id,
        'social_media_platform_id'  => $this->platform->id,
        'name'                      => 'Test',
        'status'                    => 'live',
        'trigger_target'            => 'all_posts',
        'keyword_mode'              => 'any',
        'include_keywords'          => [],
        'exclude_keywords'          => [],
        'enable_public_replies'     => false,
        'delay_seconds'             => 0,
    ]);

    expect($this->service->matchesTrigger($automation, [
        'text' => 'Any comment at all',
    ]))->toBeTrue();
});

it('matches trigger with specific keywords', function () {
    $automation = Automation::query()->create([
        'user_id'                   => $this->platform->user_id,
        'social_media_platform_id'  => $this->platform->id,
        'name'                      => 'Keyword Test',
        'status'                    => 'live',
        'trigger_target'            => 'all_posts',
        'keyword_mode'              => 'specific',
        'include_keywords'          => ['discount', 'price'],
        'exclude_keywords'          => ['spam'],
        'enable_public_replies'     => false,
        'delay_seconds'             => 0,
    ]);

    expect($this->service->matchesTrigger($automation, ['text' => 'What is the price?']))->toBeTrue()
        ->and($this->service->matchesTrigger($automation, ['text' => 'Give me a discount']))->toBeTrue()
        ->and($this->service->matchesTrigger($automation, ['text' => 'Hello there']))->toBeFalse()
        ->and($this->service->matchesTrigger($automation, ['text' => 'discount spam message']))->toBeFalse();
});

it('matches trigger for specific post only', function () {
    $automation = Automation::query()->create([
        'user_id'                   => $this->platform->user_id,
        'social_media_platform_id'  => $this->platform->id,
        'name'                      => 'Post Specific',
        'status'                    => 'live',
        'trigger_target'            => 'specific_post',
        'trigger_post_id'           => 'post_abc',
        'keyword_mode'              => 'any',
        'include_keywords'          => [],
        'exclude_keywords'          => [],
        'enable_public_replies'     => false,
        'delay_seconds'             => 0,
    ]);

    expect($this->service->matchesTrigger($automation, ['post_id' => 'post_abc', 'text' => 'Hello']))->toBeTrue()
        ->and($this->service->matchesTrigger($automation, ['post_id' => 'post_xyz', 'text' => 'Hello']))->toBeFalse();
});

it('executes actions and creates a log entry', function () {
    Http::fake();

    $automation = Automation::query()->create([
        'user_id'                   => $this->platform->user_id,
        'social_media_platform_id'  => $this->platform->id,
        'name'                      => 'DM Automation',
        'status'                    => 'live',
        'trigger_target'            => 'all_posts',
        'keyword_mode'              => 'any',
        'enable_public_replies'     => false,
        'delay_seconds'             => 0,
    ]);

    AutomationAction::query()->create([
        'automation_id' => $automation->id,
        'type'          => 'text',
        'content'       => ['text' => 'Hi {first_name}!'],
        'order'         => 0,
    ]);

    $this->service->executeActions($automation->load(['actions', 'replies', 'platform']), [
        'comment_id'          => 'c_123',
        'commenter_id'        => 'u_456',
        'commenter_username'  => 'johndoe',
        'commenter_name'      => 'John Doe',
        'text'                => 'Nice post!',
    ]);

    $log = AutomationLog::query()->where('automation_id', $automation->id)->first();
    expect($log)->not->toBeNull()
        ->and($log->status)->toBe('success')
        ->and($log->commenter_username)->toBe('johndoe')
        ->and($log->actions_executed)->toContain('text');
});

it('sends public reply when enabled', function () {
    Http::fake(['*' => Http::response(['success' => true])]);

    $automation = Automation::query()->create([
        'user_id'                   => $this->platform->user_id,
        'social_media_platform_id'  => $this->platform->id,
        'name'                      => 'Reply Automation',
        'status'                    => 'live',
        'trigger_target'            => 'all_posts',
        'keyword_mode'              => 'any',
        'enable_public_replies'     => true,
        'delay_seconds'             => 0,
    ]);

    AutomationReply::query()->create([
        'automation_id' => $automation->id,
        'content'       => 'Thanks for commenting, {username}!',
    ]);

    $this->service->executeActions($automation->load(['actions', 'replies', 'platform']), [
        'comment_id'          => 'c_123',
        'commenter_id'        => 'u_456',
        'commenter_username'  => 'johndoe',
        'text'                => 'Great!',
    ]);

    Http::assertSent(function ($request) {
        return str_contains($request->url(), '/replies')
            && str_contains($request->body(), 'johndoe');
    });
});

it('sends instagram dm with variable substitution', function () {
    Http::fake(['*' => Http::response(['message_id' => 'msg_123'])]);

    $automation = Automation::query()->create([
        'user_id'                   => $this->platform->user_id,
        'social_media_platform_id'  => $this->platform->id,
        'name'                      => 'DM Automation',
        'status'                    => 'live',
        'trigger_target'            => 'all_posts',
        'keyword_mode'              => 'any',
        'enable_public_replies'     => false,
        'delay_seconds'             => 0,
    ]);

    AutomationAction::query()->create([
        'automation_id' => $automation->id,
        'type'          => 'text',
        'content'       => ['text' => 'Hey {first_name}, thanks for saying "{comment_text}"!'],
        'order'         => 0,
    ]);

    $this->service->executeActions($automation->load(['actions', 'replies', 'platform']), [
        'comment_id'          => 'c_123',
        'commenter_id'        => 'u_456',
        'commenter_username'  => 'johndoe',
        'commenter_name'      => 'John Doe',
        'text'                => 'Love it',
    ]);

    Http::assertSent(function ($request) {
        return str_contains($request->url(), '/messages')
            && str_contains($request->body(), 'John')
            && str_contains($request->body(), 'Love it');
    });
});

it('sends facebook dm correctly', function () {
    Http::fake(['*' => Http::response(['message_id' => 'msg_456'])]);

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

    $this->service->sendDm($fbPlatform, ['comment_id' => 'recipient_789', 'commenter_id' => 'user_789'], [
        ['type' => 'text', 'content' => ['text' => 'Hello from Facebook!']],
    ]);

    Http::assertSent(function ($request) {
        return str_contains($request->url(), '/me/messages')
            && str_contains($request->body(), 'Hello from Facebook!');
    });
});

it('sends x dm correctly', function () {
    Http::fake(['*' => Http::response(['dm_event' => ['id' => 'dm_123']])]);

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

    $this->service->sendDm($xPlatform, ['comment_id' => 'tweet_456', 'commenter_id' => 'x_recipient_456'], [
        ['type' => 'text', 'content' => ['text' => 'Hello from X!']],
    ]);

    Http::assertSent(function ($request) {
        return str_contains($request->url(), '/dm_conversations/with/x_recipient_456/messages')
            && str_contains($request->body(), 'Hello from X!');
    });
});

it('handles button action in instagram dm', function () {
    Http::fake(['*' => Http::response(['message_id' => 'msg_btn'])]);

    $platform = SocialMediaPlatform::query()->create([
        'user_id'     => $this->platform->user_id,
        'platform'    => 'instagram',
        'credentials' => [
            'type'         => 'user',
            'id'           => 'ig_btn_test',
            'platform_id'  => 'ig_btn_test',
            'name'         => 'Button Test',
            'username'     => 'btntest',
            'access_token' => 'btn-token-123',
        ],
        'connected_at' => now(),
        'expires_at'   => now()->addMonth(),
    ]);

    $service = app(AutomationExecutionService::class);
    $service->sendDm($platform, ['comment_id' => 'recipient_btn', 'commenter_id' => 'user_btn'], [
        [
            'type'    => 'button',
            'content' => ['label' => 'Get Discount', 'url' => 'https://example.com/offer'],
        ],
    ]);

    Http::assertSent(function ($request) {
        return str_contains($request->url(), '/messages');
    });
});

it('handles quick replies action in instagram dm', function () {
    Http::fake(['*' => Http::response(['message_id' => 'msg_qr'])]);

    $this->service->sendDm($this->platform, ['comment_id' => 'recipient_123', 'commenter_id' => 'user_123'], [
        [
            'type'    => 'quick_replies',
            'content' => ['items' => ['Yes', 'No', 'Maybe']],
        ],
    ]);

    Http::assertSent(function ($request) {
        return str_contains($request->body(), 'quick_replies')
            && str_contains($request->body(), 'Yes')
            && str_contains($request->body(), 'No');
    });
});

it('logs failed api call and marks log as failed', function () {
    Http::fake(['*' => Http::response(['error' => 'Unauthorized'], 401)]);

    $automation = Automation::query()->create([
        'user_id'                   => $this->platform->user_id,
        'social_media_platform_id'  => $this->platform->id,
        'name'                      => 'Fail Test',
        'status'                    => 'live',
        'trigger_target'            => 'all_posts',
        'keyword_mode'              => 'any',
        'enable_public_replies'     => false,
        'delay_seconds'             => 0,
    ]);

    AutomationAction::query()->create([
        'automation_id' => $automation->id,
        'type'          => 'text',
        'content'       => ['text' => 'test message'],
        'order'         => 0,
    ]);

    $this->service->executeActions($automation->load(['actions', 'replies', 'platform']), [
        'comment_id'          => 'c_fail',
        'commenter_id'        => 'u_fail',
        'commenter_username'  => 'failuser',
        'text'                => 'trigger',
    ]);

    $log = AutomationLog::query()->where('automation_id', $automation->id)->first();
    expect($log)->not->toBeNull()
        ->and($log->status)->toBe('failed')
        ->and($log->error_message)->not->toBeNull();
});

it('skips dm when platform credentials missing access token', function () {
    Http::fake();

    $platformNoToken = SocialMediaPlatform::query()->create([
        'user_id'     => $this->platform->user_id,
        'platform'    => 'instagram',
        'credentials' => [
            'type' => 'user',
            'id'   => 'ig_no_token',
        ],
        'connected_at' => now(),
        'expires_at'   => now()->addMonth(),
    ]);

    $this->service->sendDm($platformNoToken, ['comment_id' => 'recipient_123', 'commenter_id' => 'user_123'], [
        ['type' => 'text', 'content' => ['text' => 'Should not send']],
    ]);

    Http::assertNothingSent();
});

it('processes comment event end to end', function () {
    Http::fake(['*' => Http::response(['success' => true])]);

    $automation = Automation::query()->create([
        'user_id'                   => $this->platform->user_id,
        'social_media_platform_id'  => $this->platform->id,
        'name'                      => 'E2E Test',
        'status'                    => 'live',
        'trigger_target'            => 'all_posts',
        'keyword_mode'              => 'specific',
        'include_keywords'          => ['info'],
        'exclude_keywords'          => [],
        'enable_public_replies'     => true,
        'delay_seconds'             => 0,
    ]);

    AutomationReply::query()->create([
        'automation_id' => $automation->id,
        'content'       => 'Check your DM, {username}!',
    ]);

    AutomationAction::query()->create([
        'automation_id' => $automation->id,
        'type'          => 'text',
        'content'       => ['text' => 'Here is the info you requested, {first_name}!'],
        'order'         => 0,
    ]);

    $commentData = [
        'account_id'           => 'ig_123',
        'post_id'              => 'post_123',
        'comment_id'           => 'c_e2e',
        'commenter_id'         => 'u_e2e',
        'commenter_username'   => 'curious_user',
        'commenter_name'       => 'Curious User',
        'text'                 => 'Can I get more info?',
    ];

    // Execute actions directly (simulating the job handler)
    $this->service->executeActions($automation->load(['actions', 'replies', 'platform']), $commentData);

    $log = AutomationLog::query()->where('automation_id', $automation->id)->first();
    expect($log)->not->toBeNull()
        ->and($log->status)->toBe('success')
        ->and($log->commenter_username)->toBe('curious_user');

    Http::assertSentCount(2);
});

it('creates pending automation when processing comment event', function () {
    $automation = Automation::query()->create([
        'user_id'                   => $this->platform->user_id,
        'social_media_platform_id'  => $this->platform->id,
        'name'                      => 'Dispatch Test',
        'status'                    => 'live',
        'trigger_target'            => 'all_posts',
        'keyword_mode'              => 'any',
        'enable_public_replies'     => false,
        'delay_seconds'             => 0,
    ]);

    AutomationAction::query()->create([
        'automation_id' => $automation->id,
        'type'          => 'text',
        'content'       => ['text' => 'Hello!'],
        'order'         => 0,
    ]);

    $this->service->processCommentEvent('instagram', [
        'account_id'           => 'ig_123',
        'comment_id'           => 'c_dispatch',
        'commenter_id'         => 'u_dispatch',
        'commenter_username'   => 'testuser',
        'text'                 => 'trigger comment',
    ]);

    $pending = PendingAutomation::query()
        ->where('automation_id', $automation->id)
        ->first();

    expect($pending)->not->toBeNull()
        ->and($pending->status)->toBe('pending')
        ->and($pending->comment_data['comment_id'])->toBe('c_dispatch');
});

it('sets correct execute_at with delay_seconds', function () {
    $automation = Automation::query()->create([
        'user_id'                   => $this->platform->user_id,
        'social_media_platform_id'  => $this->platform->id,
        'name'                      => 'Delay Test',
        'status'                    => 'live',
        'trigger_target'            => 'all_posts',
        'keyword_mode'              => 'any',
        'enable_public_replies'     => false,
        'delay_seconds'             => 60,
    ]);

    AutomationAction::query()->create([
        'automation_id' => $automation->id,
        'type'          => 'text',
        'content'       => ['text' => 'Hello!'],
        'order'         => 0,
    ]);

    $this->service->processCommentEvent('instagram', [
        'account_id'           => 'ig_123',
        'comment_id'           => 'c_delay',
        'commenter_id'         => 'u_delay',
        'commenter_username'   => 'testuser',
        'text'                 => 'trigger',
    ]);

    $pending = PendingAutomation::query()
        ->where('automation_id', $automation->id)
        ->first();

    expect($pending)->not->toBeNull()
        ->and($pending->execute_at->gt(now()->addSeconds(50)))->toBeTrue();
});

it('skips duplicate comment in executeActions', function () {
    Http::fake(['*' => Http::response(['success' => true])]);

    $automation = Automation::query()->create([
        'user_id'                   => $this->platform->user_id,
        'social_media_platform_id'  => $this->platform->id,
        'name'                      => 'Dedup Test',
        'status'                    => 'live',
        'trigger_target'            => 'all_posts',
        'keyword_mode'              => 'any',
        'enable_public_replies'     => false,
        'delay_seconds'             => 0,
    ]);

    AutomationAction::query()->create([
        'automation_id' => $automation->id,
        'type'          => 'text',
        'content'       => ['text' => 'Hello!'],
        'order'         => 0,
    ]);

    $commentData = [
        'comment_id'          => 'c_dedup',
        'commenter_id'        => 'u_dedup',
        'commenter_username'  => 'dedupuser',
        'text'                => 'test',
    ];

    // First call creates a log
    $this->service->executeActions($automation->load(['actions', 'replies', 'platform']), $commentData);
    expect(AutomationLog::query()->where('automation_id', $automation->id)->count())->toBe(1);

    // Second call with same comment_id is skipped
    $this->service->executeActions($automation->load(['actions', 'replies', 'platform']), $commentData);
    expect(AutomationLog::query()->where('automation_id', $automation->id)->count())->toBe(1);
});
