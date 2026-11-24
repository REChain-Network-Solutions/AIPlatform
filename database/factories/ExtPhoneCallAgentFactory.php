<?php

namespace Database\Factories;

use App\Extensions\PhoneCallAgent\System\Enums\ProviderEnum;
use App\Extensions\PhoneCallAgent\System\Models\ExtPhoneCallAgent;
use App\Models\User;
use Illuminate\Database\Eloquent\Factories\Factory;

class ExtPhoneCallAgentFactory extends Factory
{
    protected $model = ExtPhoneCallAgent::class;

    public function definition(): array
    {
        return [
            'uuid'                  => fake()->uuid(),
            'user_id'               => User::factory(),
            'agent_id'              => null,
            'title'                 => fake()->words(3, true),
            'welcome_message'       => fake()->sentence(),
            'instructions'          => fake()->paragraph(),
            'language'              => 'en',
            'ai_model'              => null,
            'voice_id'              => null,
            'provider'              => ProviderEnum::Elevenlabs,
            'max_call_duration'     => 300,
            'silence_timeout'       => 30,
            'active'                => true,
            'is_favorite'           => false,
            'booking_enabled'       => false,
            'booking_provider'      => null,
            'booking_event_type_id' => null,
            'booking_api_key'       => null,
            'booking_timezone'      => null,
        ];
    }
}
