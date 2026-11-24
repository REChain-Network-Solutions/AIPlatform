<?php

namespace Database\Factories;

use App\Extensions\MarketingBot\System\Models\Telegram\TelegramGroup;
use Illuminate\Database\Eloquent\Factories\Factory;

/**
 * @extends Factory<TelegramGroup>
 */
class TelegramGroupFactory extends Factory
{
    /** @var class-string<TelegramGroup> */
    protected $model = TelegramGroup::class;

    /**
     * @return array<string, mixed>
     */
    public function definition(): array
    {
        return [
            'user_id'  => null,
            'name'     => $this->faker->company(),
            'group_id' => $this->faker->unique()->numerify('##########'),
            'status'   => true,
        ];
    }
}
