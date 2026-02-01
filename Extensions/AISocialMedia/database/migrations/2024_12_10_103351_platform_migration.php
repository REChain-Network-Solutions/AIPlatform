<?php

use App\Extensions\AISocialMedia\System\Enums\Platform;
use App\Extensions\AISocialMedia\System\Models\AutomationPlatform;
use App\Extensions\AISocialMedia\System\Models\TwitterSettings;
use Illuminate\Database\Migrations\Migration;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        TwitterSettings::query()
            ->get()
            ->map(function ($setting) {
                AutomationPlatform::query()
                    ->create([
                        'user_id'     => $setting->user_id,
                        'platform'    => Platform::x->value,
                        'credentials' => [
                            'user_id'             => $setting->user_id,
                            'consumer_key'        => $setting->consumer_key,
                            'consumer_secret'     => $setting->consumer_secret,
                            'access_token'        => $setting->access_token,
                            'access_token_secret' => $setting->access_token_secret,
                            'bearer_token'        => $setting->bearer_token,
                            'account_id'          => $setting->account_id,
                        ],
                        'connected_at' => now(),
                        'expires_at'   => now()->addYears(3),
                    ]);
            });

        \App\Extensions\AISocialMedia\System\Models\LinkedinTokens::query()->get()
            ->map(function ($token) {
                AutomationPlatform::query()
                    ->create([
                        'user_id'     => $token->user_id,
                        'platform'    => Platform::linkedin->value,
                        'credentials' => [
                            'user_id'      => $token->user_id,
                            'access_token' => $token->access_token,
                        ],
                        'connected_at' => now(),
                        'expires_at'   => now()->addYears(3),
                    ]);
            });

    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        //
    }
};
