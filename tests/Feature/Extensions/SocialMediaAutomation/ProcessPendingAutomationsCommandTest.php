<?php

declare(strict_types=1);

use App\Extensions\SocialMedia\System\Models\SocialMediaPlatform;
use App\Extensions\SocialMediaAutomation\System\Models\Automation;
use App\Extensions\SocialMediaAutomation\System\Models\AutomationAction;
use App\Extensions\SocialMediaAutomation\System\Models\AutomationLog;
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

    $this->automation = Automation::query()->create([
        'user_id'                   => $user->id,
        'social_media_platform_id'  => $this->platform->id,
        'name'                      => 'Test Automation',
        'status'                    => 'live',
        'trigger_target'            => 'all_posts',
        'keyword_mode'              => 'any',
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

it('processes pending automations that are ready', function () {
    Http::fake(['*' => Http::response(['message_id' => 'msg_123'])]);

    PendingAutomation::query()->create([
        'automation_id' => $this->automation->id,
        'comment_data'  => [
            'comment_id'          => 'c_ready',
            'commenter_id'        => 'u_ready',
            'commenter_username'  => 'readyuser',
            'commenter_name'      => 'Ready User',
            'text'                => 'Hello!',
        ],
        'execute_at' => now()->subMinute(),
        'status'     => 'pending',
    ]);

    $this->artisan('social-media-automation:process-pending')
        ->assertSuccessful();

    $pending = PendingAutomation::query()->first();
    expect($pending->status)->toBe('completed');

    $log = AutomationLog::query()->where('automation_id', $this->automation->id)->first();
    expect($log)->not->toBeNull()
        ->and($log->status)->toBe('success');
});

it('skips pending automations that are not yet due', function () {
    PendingAutomation::query()->create([
        'automation_id' => $this->automation->id,
        'comment_data'  => [
            'comment_id'          => 'c_future',
            'commenter_id'        => 'u_future',
            'commenter_username'  => 'futureuser',
            'text'                => 'Hello!',
        ],
        'execute_at' => now()->addMinutes(5),
        'status'     => 'pending',
    ]);

    $this->artisan('social-media-automation:process-pending')
        ->assertSuccessful();

    $pending = PendingAutomation::query()->first();
    expect($pending->status)->toBe('pending');
});

it('marks pending automation as failed on error', function () {
    $this->mock(AutomationExecutionService::class, function ($mock) {
        $mock->shouldReceive('executeActions')
            ->once()
            ->andThrow(new RuntimeException('Test error'));
    });

    $pending = PendingAutomation::query()->create([
        'automation_id' => $this->automation->id,
        'comment_data'  => [
            'comment_id'          => 'c_fail',
            'commenter_id'        => 'u_fail',
            'commenter_username'  => 'failuser',
            'text'                => 'Hello!',
        ],
        'execute_at' => now()->subMinute(),
        'status'     => 'pending',
    ]);

    $this->artisan('social-media-automation:process-pending')
        ->assertSuccessful();

    $pending->refresh();
    expect($pending->status)->toBe('failed')
        ->and($pending->error_message)->toBe('Test error');
});

it('cleans up old completed and failed records', function () {
    $oldCompleted = PendingAutomation::query()->create([
        'automation_id' => $this->automation->id,
        'comment_data'  => ['comment_id' => 'c_old_completed'],
        'execute_at'    => now()->subDays(10),
        'status'        => 'completed',
    ]);
    PendingAutomation::query()->where('id', $oldCompleted->id)
        ->update(['updated_at' => now()->subDays(10)]);

    $oldFailed = PendingAutomation::query()->create([
        'automation_id' => $this->automation->id,
        'comment_data'  => ['comment_id' => 'c_old_failed'],
        'execute_at'    => now()->subDays(10),
        'status'        => 'failed',
    ]);
    PendingAutomation::query()->where('id', $oldFailed->id)
        ->update(['updated_at' => now()->subDays(10)]);

    PendingAutomation::query()->create([
        'automation_id' => $this->automation->id,
        'comment_data'  => ['comment_id' => 'c_recent'],
        'execute_at'    => now()->subDay(),
        'status'        => 'completed',
    ]);

    $this->artisan('social-media-automation:cleanup-pending')
        ->assertSuccessful();

    expect(PendingAutomation::query()->count())->toBe(1)
        ->and(PendingAutomation::query()->first()->comment_data['comment_id'])->toBe('c_recent');
});
