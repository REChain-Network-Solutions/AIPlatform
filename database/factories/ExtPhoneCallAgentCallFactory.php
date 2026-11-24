<?php

namespace Database\Factories;

use App\Extensions\PhoneCallAgent\System\Enums\CallStatusEnum;
use App\Extensions\PhoneCallAgent\System\Enums\DirectionEnum;
use App\Extensions\PhoneCallAgent\System\Models\ExtPhoneCallAgent;
use App\Extensions\PhoneCallAgent\System\Models\ExtPhoneCallAgentCall;
use App\Models\User;
use Illuminate\Database\Eloquent\Factories\Factory;

class ExtPhoneCallAgentCallFactory extends Factory
{
    protected $model = ExtPhoneCallAgentCall::class;

    public function definition(): array
    {
        return [
            'uuid'          => fake()->uuid(),
            'agent_id'      => ExtPhoneCallAgent::factory(),
            'user_id'       => User::factory(),
            'call_sid'      => 'CA' . fake()->regexify('[A-Za-z0-9]{32}'),
            'caller_number' => fake()->e164PhoneNumber(),
            'called_number' => fake()->e164PhoneNumber(),
            'direction'     => DirectionEnum::Inbound,
            'status'        => CallStatusEnum::Ringing,
            'duration'      => 0,
            'started_at'    => now(),
            'ended_at'      => null,
        ];
    }

    public function outbound(): static
    {
        return $this->state(fn (array $attributes): array => [
            'direction' => DirectionEnum::Outbound,
        ]);
    }
}
