<?php

namespace Database\Seeders;

use App\Extensions\AIAgent\System\Models\AIAgentAvatar;
use Illuminate\Database\Seeder;

class AIAgentAvatarSeeder extends Seeder
{
    public function run(): void
    {
        if (AIAgentAvatar::query()->whereNull('user_id')->exists()) {
            return;
        }

        foreach (range(1, 5) as $n) {
            AIAgentAvatar::query()->create([
                'user_id' => null,
                'avatar'  => "vendor/ai-agent/images/avatars/avatar-{$n}.png",
            ]);
        }
    }
}
