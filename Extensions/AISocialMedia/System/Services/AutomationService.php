<?php

namespace App\Extensions\AISocialMedia\System\Services;

use App\Extensions\AISocialMedia\System\Enums\Platform;
use App\Extensions\AISocialMedia\System\Models\AutomationPlatform;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Fluent;

class AutomationService
{
    public static function platforms(): array
    {
        return [
            '1' => new Fluent([
                'id'                       => '1',
                'key'                      => 'x',
                'name'                     => 'X',
                'content_character_length' => 280,
                'logo'                     => 'vendor/ai-social-media/images/x-logo.png',
                'has_setting'              => true,
                'service'	                 => new TwitterService,
                'setting'                  => AutomationPlatform::query()
                    ->where('platform', Platform::x->value)
                    ->where('user_id', Auth::id())->first(),
            ]),
            '2' => new Fluent([
                'id'                       => '2',
                'key'                      => 'linkedin',
                'name'                     => 'LinkedIn',
                'system'                   => true,
                'has_setting'              => true,
                'content_character_length' => 2900,
                'service'	                 => new LinkedInService,
                'setting'                  => AutomationPlatform::query()
                    ->where('platform', Platform::linkedin->value)
                    ->where('user_id', Auth::id())->first(),
                'logo'    => 'vendor/ai-social-media/images/linkedin-logo.png',
            ]),
            '3' => new Fluent([
                'id'                       => '3',
                'key'                      => 'instagram',
                'name'                     => 'Instagram',
                'has_setting'              => true,
                'content_character_length' => 2200,
                'logo'                     => 'vendor/ai-social-media/images/instagram-logo.png',
                'service'	                 => new InstagramService,
                'setting'                  => AutomationPlatform::query()
                    ->where('platform', Platform::instagram->value)
                    ->where('user_id', Auth::id())->first(),
            ]),
        ];
    }

    public static function find(int $id): ?Fluent
    {
        return self::platforms()[$id] ?? null;
    }

    public function findAutomationPlatform(int $id)
    {
        $platform = self::find($id);

        if (empty($platform)) {
            return new AutomationPlatform;
        }

        return $platform?->setting;
    }
}
