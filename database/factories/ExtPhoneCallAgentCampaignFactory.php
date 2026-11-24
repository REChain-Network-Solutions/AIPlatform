<?php

namespace Database\Factories;

use App\Extensions\PhoneCallAgent\System\Models\ExtPhoneCallAgent;
use App\Extensions\PhoneCallAgent\System\Models\ExtPhoneCallAgentCampaign;
use App\Models\User;
use Illuminate\Database\Eloquent\Factories\Factory;

class ExtPhoneCallAgentCampaignFactory extends Factory
{
    protected $model = ExtPhoneCallAgentCampaign::class;

    public function definition(): array
    {
        return [
            'uuid'             => fake()->uuid(),
            'agent_id'         => ExtPhoneCallAgent::factory(),
            'user_id'          => User::factory(),
            'name'             => fake()->words(3, true),
            'status'           => 'pending',
            'scheduled_at'     => null,
            'total_recipients' => 0,
            'completed_count'  => 0,
            'failed_count'     => 0,
        ];
    }
}
