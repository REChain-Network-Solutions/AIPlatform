<?php

namespace Database\Factories;

use App\Extensions\MarketingBot\System\Models\Telegram\TelegramGroupSubscriber;
use Illuminate\Database\Eloquent\Factories\Factory;

/**
 * @extends Factory<TelegramGroupSubscriber>
 */
class TelegramGroupSubscriberFactory extends Factory
{
    /** @var class-string<TelegramGroupSubscriber> */
    protected $model = TelegramGroupSubscriber::class;

    /**
     * Define the model's default state.
     *
     * @return array<string, mixed>
     */
    public function definition(): array
    {
        return [
            'user_id'      => null,
            'name'         => $this->faker->name(),
            'username'     => $this->faker->unique()->userName(),
            'phone'        => null,
            'status'       => true,
            'is_bot'       => false,
            'is_admin'     => false,
            'is_blacklist' => false,
        ];
    }
}
