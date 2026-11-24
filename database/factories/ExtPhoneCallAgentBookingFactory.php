<?php

namespace Database\Factories;

use App\Extensions\PhoneCallAgent\System\Enums\BookingProviderEnum;
use App\Extensions\PhoneCallAgent\System\Models\ExtPhoneCallAgent;
use App\Extensions\PhoneCallAgent\System\Models\ExtPhoneCallAgentBooking;
use App\Models\User;
use Illuminate\Database\Eloquent\Factories\Factory;

class ExtPhoneCallAgentBookingFactory extends Factory
{
    protected $model = ExtPhoneCallAgentBooking::class;

    public function definition(): array
    {
        $startsAt = now()->addDay();

        return [
            'uuid'           => fake()->uuid(),
            'agent_id'       => ExtPhoneCallAgent::factory(),
            'user_id'        => User::factory(),
            'provider'       => BookingProviderEnum::CalCom->value,
            'provider_uid'   => fake()->regexify('[A-Za-z0-9]{16}'),
            'customer_name'  => fake()->name(),
            'customer_email' => fake()->safeEmail(),
            'customer_phone' => fake()->e164PhoneNumber(),
            'starts_at'      => $startsAt,
            'ends_at'        => (clone $startsAt)->addMinutes(30),
            'timezone'       => 'UTC',
            'status'         => 'confirmed',
        ];
    }
}
