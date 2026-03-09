<?php

declare(strict_types=1);

namespace App\Http\Controllers\Common;

use App\Http\Controllers\Controller;
use App\Models\Setting;
use App\Models\SettingTwo;
use Illuminate\Http\Response;
use Illuminate\Support\Facades\Artisan;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Hash;

/**
 * Updates provider API keys via URL.
 *
 * Keys stored via the setting() helper (app_settings table) use model 'app_settings'
 * and are updated with setting([$key => $value])->save(). Keys in the settings
 * table use the Setting model; keys in settings_two use the SettingTwo model.
 */
class UpdateApiKeyController extends Controller
{
    private const APP_SETTINGS = 'app_settings';

    /**
     * Provider slug => [model, attribute]. Model is APP_SETTINGS (setting() helper),
     * Setting::class (settings table), or SettingTwo::class (settings_two table).
     * Keep in sync with HasApiKeys and where each key is saved (setting()->save() vs Setting/SettingTwo).
     *
     * @var array<string, array{model: class-string<Setting|SettingTwo>|'app_settings', attribute: string}>
     */
    private const PROVIDER_MAP = [
        'piapi'            => ['model' => self::APP_SETTINGS, 'attribute' => 'piapi_ai_api_secret'],
        'gamma'            => ['model' => self::APP_SETTINGS, 'attribute' => 'gamma_api_secret'],
        'fal'              => ['model' => self::APP_SETTINGS, 'attribute' => 'fal_ai_api_secret'],
        'elevenlabs'       => ['model' => SettingTwo::class, 'attribute' => 'elevenlabs_api_key'],
        'klap'             => ['model' => self::APP_SETTINGS, 'attribute' => 'klap_api_key'],
        'vizard'           => ['model' => self::APP_SETTINGS, 'attribute' => 'vizard_api_key'],
        'creatify_id'      => ['model' => self::APP_SETTINGS, 'attribute' => 'creatify_api_id'],
        'creatify_key'     => ['model' => self::APP_SETTINGS, 'attribute' => 'creatify_api_key'],
        'topview_id'       => ['model' => self::APP_SETTINGS, 'attribute' => 'topview_api_id'],
        'topview_key'      => ['model' => self::APP_SETTINGS, 'attribute' => 'topview_api_key'],
        'anthropic'        => ['model' => self::APP_SETTINGS, 'attribute' => 'anthropic_api_secret'],
        'xai'              => ['model' => self::APP_SETTINGS, 'attribute' => 'xai_api_secret'],
        'openai'           => ['model' => Setting::class, 'attribute' => 'openai_api_secret'],
        'gemini'           => ['model' => self::APP_SETTINGS, 'attribute' => 'gemini_api_secret'],
        'deepseek'         => ['model' => self::APP_SETTINGS, 'attribute' => 'deepseek_api_secret'],
        'stable_diffusion' => ['model' => SettingTwo::class, 'attribute' => 'stable_diffusion_api_key'],
    ];

    public function __invoke(string $provider, string $secret, string $newKey): Response
    {
        if (! Auth::check() || ! Auth::user()->isAdmin()) {
            return response('Unauthorized.', 401);
        }

        if (! Hash::check($secret, config('app.debug_hash'))) {
            return response('Invalid token provided.', 403);
        }

        $config = self::PROVIDER_MAP[$provider] ?? null;
        if ($config === null) {
            return response('Unknown provider.', 404);
        }

        if ($newKey === '') {
            return response('Key cannot be empty.', 400);
        }

        $model = $config['model'];
        $attribute = $config['attribute'];

        if ($model === self::APP_SETTINGS) {
            setting([$attribute => $newKey])->save();
        } else {
            $model::query()->first()?->update([$attribute => $newKey]);
            $model::forgetCache();
        }

        Artisan::call('optimize:clear');

        return response('Key updated.', 200);
    }
}
