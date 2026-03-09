<?php

declare(strict_types=1);

use App\Extensions\SocialMedia\System\Enums\PlatformEnum;
use App\Extensions\SocialMedia\System\Enums\PostTypeEnum;
use App\Extensions\SocialMedia\System\Enums\StatusEnum;
use App\Extensions\SocialMedia\System\Models\SocialMediaPost;
use App\Helpers\Classes\MarketplaceHelper;

beforeEach(function () {
    if (! MarketplaceHelper::isRegistered('social-media')) {
        $this->markTestSkipped('SocialMedia extension is not registered.');
    }
});

it('has post_type in fillable array', function () {
    $post = new SocialMediaPost;

    expect($post->getFillable())->toContain('post_type');
});

it('casts post_type to PostTypeEnum', function () {
    $post = new SocialMediaPost;
    $casts = $post->getCasts();

    expect($casts)->toHaveKey('post_type')
        ->and($casts['post_type'])->toBe(PostTypeEnum::class);
});

it('can set post_type to post', function () {
    $post = new SocialMediaPost;
    $post->post_type = PostTypeEnum::Post;

    expect($post->post_type)->toBe(PostTypeEnum::Post);
});

it('can set post_type to story', function () {
    $post = new SocialMediaPost;
    $post->post_type = PostTypeEnum::Story;

    expect($post->post_type)->toBe(PostTypeEnum::Story)
        ->and($post->post_type->label())->toBe('Story');
});

it('includes post_type when filling attributes', function () {
    $post = new SocialMediaPost;
    $post->fill([
        'user_id'               => 1,
        'social_media_platform' => PlatformEnum::facebook->value,
        'post_type'             => 'story',
        'content'               => 'Test story content',
        'tone'                  => 'default',
        'status'                => StatusEnum::scheduled->value,
    ]);

    expect($post->getAttributes())->toHaveKey('post_type')
        ->and($post->getAttributes()['post_type'])->toBe('story');
});

it('validates story is supported for facebook platform', function () {
    expect(PostTypeEnum::platformSupportsStory(PlatformEnum::facebook))->toBeTrue();
});

it('validates story is supported for instagram platform', function () {
    expect(PostTypeEnum::platformSupportsStory(PlatformEnum::instagram))->toBeTrue();
});

it('validates story is supported for tiktok platform', function () {
    expect(PostTypeEnum::platformSupportsStory(PlatformEnum::tiktok))->toBeTrue();
});

it('validates story is not supported for linkedin platform', function () {
    expect(PostTypeEnum::platformSupportsStory(PlatformEnum::linkedin))->toBeFalse();
});

it('validates story is not supported for youtube platform', function () {
    expect(PostTypeEnum::platformSupportsStory(PlatformEnum::youtube))->toBeFalse();
});
