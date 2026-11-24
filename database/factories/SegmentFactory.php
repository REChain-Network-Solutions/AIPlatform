<?php

namespace Database\Factories;

use App\Extensions\MarketingBot\System\Models\Whatsapp\Segment;
use Illuminate\Database\Eloquent\Factories\Factory;

/**
 * @extends Factory<Segment>
 */
class SegmentFactory extends Factory
{
    /** @var class-string<Segment> */
    protected $model = Segment::class;

    /**
     * @return array<string, mixed>
     */
    public function definition(): array
    {
        return [
            'user_id' => null,
            'name'    => $this->faker->word(),
            'status'  => true,
        ];
    }
}
