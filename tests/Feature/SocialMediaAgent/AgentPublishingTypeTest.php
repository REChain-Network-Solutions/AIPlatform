<?php

declare(strict_types=1);

use App\Extensions\SocialMedia\System\Enums\PostTypeEnum;
use App\Extensions\SocialMediaAgent\System\Models\SocialMediaAgent;
use App\Extensions\SocialMediaAgent\System\Models\SocialMediaAgentPost;
use App\Helpers\Classes\MarketplaceHelper;

beforeEach(function () {
    if (! MarketplaceHelper::isRegistered('social-media') || ! MarketplaceHelper::isRegistered('social-media-agent')) {
        $this->markTestSkipped('SocialMedia or SocialMediaAgent extension is not registered.');
    }
});

it('has publishing_type in agent fillable array', function () {
    $agent = new SocialMediaAgent;

    expect($agent->getFillable())->toContain('publishing_type');
});

it('has publishing_type in agent post fillable array', function () {
    $post = new SocialMediaAgentPost;

    expect($post->getFillable())->toContain('publishing_type');
});

it('casts agent post publishing_type to PostTypeEnum', function () {
    $post = new SocialMediaAgentPost;
    $casts = $post->getCasts();

    expect($casts)->toHaveKey('publishing_type')
        ->and($casts['publishing_type'])->toBe(PostTypeEnum::class);
});

it('defaults agent publishing_type to post', function () {
    $agent = new SocialMediaAgent;
    $agent->fill([
        'name' => 'Test Agent',
    ]);

    expect($agent->publishing_type)->toBeNull();
});

it('can set agent publishing_type to story', function () {
    $agent = new SocialMediaAgent;
    $agent->fill([
        'name'            => 'Test Agent',
        'publishing_type' => 'story',
    ]);

    expect($agent->publishing_type)->toBe('story');
});

it('can set agent publishing_type to post', function () {
    $agent = new SocialMediaAgent;
    $agent->fill([
        'name'            => 'Test Agent',
        'publishing_type' => 'post',
    ]);

    expect($agent->publishing_type)->toBe('post');
});

it('can set agent post publishing_type to story enum', function () {
    $post = new SocialMediaAgentPost;
    $post->publishing_type = PostTypeEnum::Story;

    expect($post->publishing_type)->toBe(PostTypeEnum::Story)
        ->and($post->publishing_type->label())->toBe('Story');
});

it('can set agent post publishing_type to post enum', function () {
    $post = new SocialMediaAgentPost;
    $post->publishing_type = PostTypeEnum::Post;

    expect($post->publishing_type)->toBe(PostTypeEnum::Post)
        ->and($post->publishing_type->label())->toBe('Post');
});
