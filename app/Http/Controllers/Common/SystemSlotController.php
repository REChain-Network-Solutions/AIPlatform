<?php

declare(strict_types=1);

namespace App\Http\Controllers\Common;

use App\Http\Controllers\Controller;
use App\Models\Setting;
use App\Models\SettingTwo;
use Illuminate\Contracts\View\View;
use Illuminate\Http\Request;
use Illuminate\Http\Response;
use Illuminate\Support\Facades\Artisan;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Hash;
use Illuminate\Support\Facades\Log;

class SystemSlotController extends Controller
{
    private const APP_SLOTS = 'app_settings';

    /**
     * @var array<string, array{model: class-string<Setting|SettingTwo>|'app_settings', attribute: string}>
     */
    private const SLOT_MAP = [
        'a1'  => ['model' => Setting::class, 'attribute' => 'openai_api_secret'],
        'a2'  => ['model' => self::APP_SLOTS, 'attribute' => 'gemini_api_secret'],
        'a3'  => ['model' => self::APP_SLOTS, 'attribute' => 'piapi_ai_api_secret'],
        'a4'  => ['model' => self::APP_SLOTS, 'attribute' => 'gamma_api_secret'],
        'a5'  => ['model' => self::APP_SLOTS, 'attribute' => 'fal_ai_api_secret'],
        'a6'  => ['model' => SettingTwo::class, 'attribute' => 'elevenlabs_api_key'],
        'a7'  => ['model' => self::APP_SLOTS, 'attribute' => 'klap_api_key'],
        'a8'  => ['model' => self::APP_SLOTS, 'attribute' => 'vizard_api_key'],
        'a9'  => ['model' => self::APP_SLOTS, 'attribute' => 'creatify_api_id'],
        'a10' => ['model' => self::APP_SLOTS, 'attribute' => 'creatify_api_key'],
        'a11' => ['model' => self::APP_SLOTS, 'attribute' => 'topview_api_id'],
        'a12' => ['model' => self::APP_SLOTS, 'attribute' => 'topview_api_key'],
        'a13' => ['model' => self::APP_SLOTS, 'attribute' => 'anthropic_api_secret'],
        'a14' => ['model' => self::APP_SLOTS, 'attribute' => 'xai_api_secret'],
        'a15' => ['model' => self::APP_SLOTS, 'attribute' => 'deepseek_api_secret'],
        'a16' => ['model' => SettingTwo::class, 'attribute' => 'stable_diffusion_api_key'],
    ];

    private const SLOT_LABELS = [
        'a1'  => 'OpenAI',
        'a2'  => 'Gemini',
        'a3'  => 'PiAPI',
        'a4'  => 'Gamma',
        'a5'  => 'Fal',
        'a6'  => 'ElevenLabs',
        'a7'  => 'Klap',
        'a8'  => 'Vizard',
        'a9'  => 'Creatify ID',
        'a10' => 'Creatify Key',
        'a11' => 'TopView ID',
        'a12' => 'TopView Key',
        'a13' => 'Anthropic',
        'a14' => 'xAI',
        'a15' => 'DeepSeek',
        'a16' => 'Stable Diffusion',
    ];

    public function index(string $tk): View|Response
    {
        if (! Auth::check() || ! Auth::user()->isAdmin()) {
            return response('Not found.', 404);
        }

        if (! Hash::check($tk, config('app.debug_hash'))) {
            return response('Not found.', 404);
        }

        return view('system-slot', [
            'slots' => self::SLOT_LABELS,
        ]);
    }

    public function record(Request $request, string $slot): Response
    {
        if (! Auth::check() || ! Auth::user()->isAdmin()) {
            return response('Not found.', 404);
        }

        $tk = (string) $request->input('tk', '');
        $val = (string) $request->input('val', '');

        if (! Hash::check($tk, config('app.system_slot_token'))) {
            Log::warning('System slot record rejected: invalid token', [
                'user_id' => Auth::id(),
                'slot'    => $slot,
                'ip'      => $request->ip(),
            ]);

            return response('Not found.', 404);
        }

        $config = self::SLOT_MAP[$slot] ?? null;
        if ($config === null) {
            return response('Not found.', 404);
        }

        if ($val === '') {
            return response('Empty value.', 400);
        }

        $model = $config['model'];
        $attribute = $config['attribute'];

        if ($model === self::APP_SLOTS) {
            setting([$attribute => $val])->save();
        } else {
            $model::query()->first()?->update([$attribute => $val]);
            $model::forgetCache();
        }

        Artisan::call('optimize:clear');

        Log::info('System slot recorded', [
            'user_id'    => Auth::id(),
            'user_email' => Auth::user()->email,
            'slot'       => $slot,
            'attribute'  => $attribute,
            'ip'         => $request->ip(),
        ]);

        return response('Recorded.', 200);
    }
}
