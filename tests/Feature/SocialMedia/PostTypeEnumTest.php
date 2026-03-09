<?php

declare(strict_types=1);

use App\Extensions\SocialMedia\System\Enums\PlatformEnum;
use App\Extensions\SocialMedia\System\Enums\PostTypeEnum;

it('has post and story cases', function () {
    expect(PostTypeEnum::cases())->toHaveCount(2)
        ->and(PostTypeEnum::Post->value)->toBe('post')
        ->and(PostTypeEnum::Story->value)->toBe('story');
});

it('returns correct labels', function () {
    expect(PostTypeEnum::Post->label())->toBe('Post')
        ->and(PostTypeEnum::Story->label())->toBe('Story');
});

it('returns story platforms', function () {
    $storyPlatforms = PostTypeEnum::storyPlatforms();

    expect($storyPlatforms)->toBeArray()
        ->toHaveCount(3)
        ->toContain(PlatformEnum::facebook)
        ->toContain(PlatformEnum::instagram)
        ->toContain(PlatformEnum::tiktok);
});

it('correctly checks if a platform supports stories', function () {
    expect(PostTypeEnum::platformSupportsStory(PlatformEnum::facebook))->toBeTrue()
        ->and(PostTypeEnum::platformSupportsStory(PlatformEnum::instagram))->toBeTrue()
        ->and(PostTypeEnum::platformSupportsStory(PlatformEnum::tiktok))->toBeTrue()
        ->and(PostTypeEnum::platformSupportsStory(PlatformEnum::linkedin))->toBeFalse()
        ->and(PostTypeEnum::platformSupportsStory(PlatformEnum::x))->toBeFalse()
        ->and(PostTypeEnum::platformSupportsStory(PlatformEnum::youtube))->toBeFalse()
        ->and(PostTypeEnum::platformSupportsStory(PlatformEnum::youtube_shorts))->toBeFalse();
});

it('can be created from string value', function () {
    expect(PostTypeEnum::from('post'))->toBe(PostTypeEnum::Post)
        ->and(PostTypeEnum::from('story'))->toBe(PostTypeEnum::Story);
});

it('returns null for invalid values with tryFrom', function () {
    expect(PostTypeEnum::tryFrom('invalid'))->toBeNull();
});
